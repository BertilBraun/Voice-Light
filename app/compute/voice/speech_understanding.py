from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final
from uuid import uuid4

from app.compute.voice.interfaces import (
    FinalizedSpeechTurn,
    SpeechUnderstandingSession,
    Transcriber,
    TranscriptionSession,
    TurnPredictionObservation,
    TurnPredictionProvider,
    TurnPredictionSource,
)
from app.compute.voice.predictive import TranscriptRevisionTracker
from app.compute.voice.schemas import (
    CapturedAudioChunk,
    CausalSource,
    FutureActivityEvidence,
    InteractionPrediction,
    OverlapDisposition,
    OverlapDispositionEvidence,
    OverlapDispositionProbability,
    PlaybackCondition,
    SpeechUnderstandingAbstainedEvent,
    SpeechUnderstandingComponent,
    SpeechUnderstandingDegradedEvent,
    SpeechUnderstandingEvent,
    SpeechUnderstandingStatus,
    SpeechUnderstandingStatusEvent,
    TraceStamp,
    TranscriptRevision,
    TurnEventEvidence,
    TurnEventKind,
    TurnEventProbability,
    YieldEvidence,
)

DEFAULT_OPTIONAL_PREDICTOR_QUEUE_SIZE: Final = 8
DEFAULT_REDUCER_MAXIMUM_PLAYBACK_CONDITIONS: Final = 128
DEFAULT_REDUCER_MAXIMUM_EVIDENCE_GROUPS: Final = 64
DEFAULT_REDUCER_MAXIMUM_SAMPLE_LAG: Final = 1_600


@dataclass(frozen=True)
class _PredictionWork:
    chunk: CapturedAudioChunk
    transcript_revision: TranscriptRevision | None


@dataclass
class _EvidenceGroup:
    stream_epoch: int
    turn_epoch: int
    observed_through_input_sample: int
    yield_evidence: YieldEvidence | None = None
    future_activity: FutureActivityEvidence | None = None
    turn_event: TurnEventEvidence | None = None
    overlap: OverlapDispositionEvidence | None = None


@dataclass(frozen=True)
class _PlaybackConditionRecord:
    condition: PlaybackCondition
    stream_epoch: int
    turn_epoch: int
    observed_through_input_sample: int


class InteractionPredictionReducer:
    """Policy compatibility reducer for complete evidence emitted by legacy predictors."""

    def __init__(
        self,
        maximum_playback_conditions: int = DEFAULT_REDUCER_MAXIMUM_PLAYBACK_CONDITIONS,
        maximum_evidence_groups: int = DEFAULT_REDUCER_MAXIMUM_EVIDENCE_GROUPS,
        maximum_sample_lag: int = DEFAULT_REDUCER_MAXIMUM_SAMPLE_LAG,
    ) -> None:
        if maximum_playback_conditions <= 0:
            raise ValueError("Reducer playback-condition capacity must be positive.")
        if maximum_evidence_groups <= 0:
            raise ValueError("Reducer evidence-group capacity must be positive.")
        if maximum_sample_lag < 0:
            raise ValueError("Reducer maximum sample lag cannot be negative.")
        self.maximum_playback_conditions = maximum_playback_conditions
        self.maximum_evidence_groups = maximum_evidence_groups
        self.maximum_sample_lag = maximum_sample_lag
        self._playback_conditions: OrderedDict[str, _PlaybackConditionRecord] = OrderedDict()
        self._groups: OrderedDict[str, _EvidenceGroup] = OrderedDict()
        self._stream_epoch: int | None = None
        self._turn_epoch: int | None = None
        self._minimum_observed_through_input_sample = 0

    @property
    def retained_playback_condition_count(self) -> int:
        return len(self._playback_conditions)

    @property
    def pending_evidence_group_count(self) -> int:
        return len(self._groups)

    def observe_audio_chunk(self, chunk: CapturedAudioChunk) -> None:
        condition = chunk.playback_condition
        self._stream_epoch = chunk.stream_epoch
        self._turn_epoch = chunk.turn_epoch
        self._minimum_observed_through_input_sample = max(
            0,
            chunk.end_input_sample - self.maximum_sample_lag,
        )
        self._prune_stale_state()
        self._playback_conditions[condition.event_id] = _PlaybackConditionRecord(
            condition=condition,
            stream_epoch=chunk.stream_epoch,
            turn_epoch=chunk.turn_epoch,
            observed_through_input_sample=chunk.end_input_sample,
        )
        self._playback_conditions.move_to_end(condition.event_id)
        while len(self._playback_conditions) > self.maximum_playback_conditions:
            self._playback_conditions.popitem(last=False)

    def reduce(self, event: SpeechUnderstandingEvent) -> InteractionPrediction | None:
        group = self._group_for_event(event)
        if group is None:
            return None
        match event:
            case YieldEvidence():
                group.yield_evidence = event
            case FutureActivityEvidence():
                group.future_activity = event
            case TurnEventEvidence():
                group.turn_event = event
            case OverlapDispositionEvidence():
                group.overlap = event
            case _:
                return None
        return self._complete_prediction(event.evidence_group_id)

    def reset_turn(self) -> None:
        self._groups.clear()
        self._playback_conditions.clear()
        self._stream_epoch = None
        self._turn_epoch = None
        self._minimum_observed_through_input_sample = 0

    def _group_for_event(
        self,
        event: SpeechUnderstandingEvent,
    ) -> _EvidenceGroup | None:
        match event:
            case (
                YieldEvidence()
                | FutureActivityEvidence()
                | TurnEventEvidence()
                | OverlapDispositionEvidence()
            ):
                pass
            case _:
                return None
        stamp = event.stamp
        if (
            stamp.stream_epoch != self._stream_epoch
            or stamp.turn_epoch != self._turn_epoch
            or stamp.observed_through_input_sample < self._minimum_observed_through_input_sample
        ):
            return None
        group = self._groups.get(event.evidence_group_id)
        if group is None:
            group = _EvidenceGroup(
                stream_epoch=stamp.stream_epoch,
                turn_epoch=stamp.turn_epoch,
                observed_through_input_sample=stamp.observed_through_input_sample,
            )
            self._groups[event.evidence_group_id] = group
            while len(self._groups) > self.maximum_evidence_groups:
                self._groups.popitem(last=False)
            return group
        if (
            stamp.stream_epoch != group.stream_epoch
            or stamp.turn_epoch != group.turn_epoch
            or stamp.observed_through_input_sample != group.observed_through_input_sample
        ):
            raise ValueError("Evidence siblings must share one causal observation.")
        self._groups.move_to_end(event.evidence_group_id)
        return group

    def _complete_prediction(self, evidence_group_id: str) -> InteractionPrediction | None:
        group = self._groups[evidence_group_id]
        if (
            group.yield_evidence is None
            or group.future_activity is None
            or group.turn_event is None
            or group.overlap is None
        ):
            return None
        del self._groups[evidence_group_id]
        yield_evidence = group.yield_evidence
        playback_event_id = yield_evidence.stamp.conditioned_playback_event_id
        if playback_event_id is None:
            raise ValueError("Legacy prediction evidence must cite its playback condition.")
        playback_record = self._playback_conditions.pop(playback_event_id, None)
        if yield_evidence.p_user_speech is None or group.overlap.p_user_interruption is None:
            return None
        backchannel_probability = next(
            (
                probability.probability
                for probability in group.turn_event.probabilities
                if probability.event is TurnEventKind.BACKCHANNEL
            ),
            None,
        )
        if backchannel_probability is None:
            return None
        if playback_record is None:
            return None
        return InteractionPrediction(
            stamp=yield_evidence.stamp,
            p_user_speech=yield_evidence.p_user_speech,
            p_user_yield=yield_evidence.p_user_yield,
            p_user_backchannel=backchannel_probability,
            p_user_interruption=group.overlap.p_user_interruption,
            future_user_activity_horizons=group.future_activity.horizons,
            assistant_playback_state=playback_record.condition.state,
            confidence=min(
                yield_evidence.confidence,
                group.future_activity.confidence,
                group.turn_event.confidence,
                group.overlap.confidence,
            ),
        )

    def _prune_stale_state(self) -> None:
        stale_playback_event_ids = tuple(
            event_id
            for event_id, record in self._playback_conditions.items()
            if record.stream_epoch != self._stream_epoch
            or record.turn_epoch != self._turn_epoch
            or record.observed_through_input_sample < self._minimum_observed_through_input_sample
        )
        for event_id in stale_playback_event_ids:
            del self._playback_conditions[event_id]
        stale_evidence_group_ids = tuple(
            group_id
            for group_id, group in self._groups.items()
            if group.stream_epoch != self._stream_epoch
            or group.turn_epoch != self._turn_epoch
            or group.observed_through_input_sample < self._minimum_observed_through_input_sample
        )
        for group_id in stale_evidence_group_ids:
            del self._groups[group_id]


class CompositeSpeechUnderstandingProvider:
    def __init__(
        self,
        transcriber: Transcriber,
        turn_prediction_provider: TurnPredictionProvider | None,
        asr_model_name: str | None,
        asr_model_revision: str | None,
        optional_predictor_queue_size: int = DEFAULT_OPTIONAL_PREDICTOR_QUEUE_SIZE,
    ) -> None:
        if optional_predictor_queue_size <= 0:
            raise ValueError("Optional predictor queue size must be positive.")
        self.transcriber = transcriber
        self.turn_prediction_provider = turn_prediction_provider
        self.asr_model_name = asr_model_name
        self.asr_model_revision = asr_model_revision
        self.optional_predictor_queue_size = optional_predictor_queue_size
        self.closed = False

    def create_session(self, stream_epoch: int) -> SpeechUnderstandingSession:
        if self.closed:
            raise RuntimeError("Cannot create a session from a closed speech provider.")
        if stream_epoch < 1:
            raise ValueError("Speech stream epochs start at one.")
        prediction_source = (
            None
            if self.turn_prediction_provider is None
            else self.turn_prediction_provider.create_session()
        )
        return CompositeSpeechUnderstandingSession(
            transcriber=self.transcriber,
            prediction_source=prediction_source,
            stream_epoch=stream_epoch,
            asr_model_name=self.asr_model_name,
            asr_model_revision=self.asr_model_revision,
            optional_predictor_queue_size=self.optional_predictor_queue_size,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.turn_prediction_provider is not None:
            self.turn_prediction_provider.close()
        self.transcriber.close()


class SingleSessionTurnPredictionProvider:
    """Migration adapter for one already-created standalone prediction source."""

    def __init__(self, source: TurnPredictionSource) -> None:
        self.source = source
        self.claimed = False

    def create_session(self) -> TurnPredictionSource:
        if self.claimed:
            raise RuntimeError("The provisional prediction source already has a session.")
        self.claimed = True
        return self.source

    def close(self) -> None:
        return


class CompositeSpeechUnderstandingSession:
    def __init__(
        self,
        transcriber: Transcriber,
        prediction_source: TurnPredictionSource | None,
        stream_epoch: int,
        asr_model_name: str | None,
        asr_model_revision: str | None,
        optional_predictor_queue_size: int,
    ) -> None:
        self.transcriber = transcriber
        self.prediction_source = prediction_source
        self._stream_epoch = stream_epoch
        self._turn_epoch = 1
        self.asr_model_name = asr_model_name
        self.asr_model_revision = asr_model_revision
        self.transcription: TranscriptionSession = transcriber.start_session()
        self.transcript_revisions = TranscriptRevisionTracker()
        self.inference_step = 0
        self.latest_chunk: CapturedAudioChunk | None = None
        self.last_sequence_number: int | None = None
        self.last_input_end_sample: int | None = None
        self.event_queue: asyncio.Queue[SpeechUnderstandingEvent | None] = asyncio.Queue()
        self.prediction_queue: asyncio.Queue[_PredictionWork | None] = asyncio.Queue(
            maxsize=optional_predictor_queue_size
        )
        self.prediction_task: asyncio.Task[None] | None = None
        self.prediction_in_flight = False
        self.optional_predictor_degraded = False
        self.dropped_prediction_observations = 0
        self.active_status_emitted = False
        self.interaction_observation_started = False
        self.closed = False

    @property
    def stream_epoch(self) -> int:
        return self._stream_epoch

    @property
    def turn_epoch(self) -> int:
        return self._turn_epoch

    async def add_audio(self, chunk: CapturedAudioChunk) -> None:
        self._validate_chunk(chunk)
        self.inference_step += 1
        self.latest_chunk = chunk
        self.last_sequence_number = chunk.sequence_number
        self.last_input_end_sample = chunk.end_input_sample
        if not self.active_status_emitted:
            self.event_queue.put_nowait(
                self._status_event(SpeechUnderstandingStatus.ACTIVE, self._turn_epoch)
            )
            self.active_status_emitted = True
        try:
            partial_text = await self.transcription.add_audio(chunk.pcm16)
        except Exception:
            await self._stop_optional_predictor()
            raise
        if partial_text:
            previous_revision = self.transcript_revisions.latest
            revision = self.transcript_revisions.update(
                text=partial_text,
                chunk=chunk,
                inference_step=self.inference_step,
                observed_through_input_sample=chunk.end_input_sample,
                model_name=self.asr_model_name,
                model_revision=self.asr_model_revision,
            )
            if revision is not None and revision is not previous_revision:
                self.event_queue.put_nowait(revision)
        self._enqueue_prediction(chunk)
        await asyncio.sleep(0)

    async def events(self) -> AsyncIterator[SpeechUnderstandingEvent]:
        while True:
            event = await self.event_queue.get()
            if event is None:
                return
            yield event

    def drain_events(self) -> tuple[SpeechUnderstandingEvent, ...]:
        events: list[SpeechUnderstandingEvent] = []
        while not self.event_queue.empty():
            event = self.event_queue.get_nowait()
            if event is None:
                self.event_queue.put_nowait(None)
                break
            events.append(event)
        return tuple(events)

    async def finalize_turn(self) -> FinalizedSpeechTurn:
        if self.closed:
            raise RuntimeError("Cannot finalize a closed speech-understanding session.")
        finalized_turn_epoch = self._turn_epoch
        stale_prediction_count = self.prediction_queue.qsize() + int(self.prediction_in_flight)
        if stale_prediction_count:
            self.dropped_prediction_observations += stale_prediction_count
            assert self.latest_chunk is not None
            self.event_queue.put_nowait(
                SpeechUnderstandingDegradedEvent(
                    stamp=self._chunk_stamp(
                        self.latest_chunk,
                        source=CausalSource.TURN_ADAPTER,
                        model_name=None,
                        model_revision=None,
                        conditioned_transcript_revision=self.transcript_revisions.latest,
                    ),
                    component=SpeechUnderstandingComponent.STANDALONE_TURN_DETECTOR,
                    reason=("Optional detector observations became stale when the turn finalized."),
                    dropped_observation_count=self.dropped_prediction_observations,
                )
            )
        await self._stop_optional_predictor()
        while not self.prediction_queue.empty():
            self.prediction_queue.get_nowait()
        final_text = await self.transcription.finish()
        latest_revision = self.transcript_revisions.latest
        latest_chunk = self.latest_chunk
        if latest_chunk is not None and final_text.strip():
            previous_revision = latest_revision
            latest_revision = self.transcript_revisions.update(
                text=final_text,
                chunk=latest_chunk,
                inference_step=self.inference_step,
                observed_through_input_sample=latest_chunk.end_input_sample,
                model_name=self.asr_model_name,
                model_revision=self.asr_model_revision,
            )
            if latest_revision is not None and latest_revision is not previous_revision:
                self.event_queue.put_nowait(latest_revision)
        self.event_queue.put_nowait(
            self._status_event(
                SpeechUnderstandingStatus.TURN_FINALIZED,
                finalized_turn_epoch,
            )
        )
        await self.transcription.close()
        self._turn_epoch += 1
        self.transcription = self.transcriber.start_session()
        self.transcript_revisions.reset_turn()
        self.latest_chunk = None
        self.last_sequence_number = None
        self.last_input_end_sample = None
        self.active_status_emitted = False
        self.interaction_observation_started = False
        return FinalizedSpeechTurn(
            text=final_text,
            transcript_revision=latest_revision,
        )

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        optional_error: Exception | None = None
        try:
            await self._stop_optional_predictor()
        except Exception as error:
            optional_error = error
        try:
            await self.transcription.close()
        finally:
            if self.prediction_source is not None:
                with contextlib.suppress(Exception):
                    await self.prediction_source.close()
            self.event_queue.put_nowait(
                self._status_event(SpeechUnderstandingStatus.CLOSED, self._turn_epoch)
            )
            self.event_queue.put_nowait(None)
        if optional_error is not None:
            raise optional_error

    def _validate_chunk(self, chunk: CapturedAudioChunk) -> None:
        if self.closed:
            raise RuntimeError("Cannot add audio to a closed speech-understanding session.")
        if chunk.stream_epoch != self._stream_epoch:
            raise ValueError("Captured audio belongs to a stale speech stream epoch.")
        if chunk.turn_epoch != self._turn_epoch:
            raise ValueError("Captured audio belongs to a stale speech turn epoch.")
        if (
            self.last_sequence_number is not None
            and chunk.sequence_number <= self.last_sequence_number
        ):
            raise ValueError("Captured audio sequence numbers must increase.")
        if (
            self.last_input_end_sample is not None
            and chunk.start_input_sample < self.last_input_end_sample
        ):
            raise ValueError("Captured audio sample ranges must not overlap.")

    def _enqueue_prediction(self, chunk: CapturedAudioChunk) -> None:
        if self.prediction_source is None or self.optional_predictor_degraded:
            return
        if not self.interaction_observation_started:
            if chunk.silero_evidence.is_speech:
                self.interaction_observation_started = True
            return
        if self.prediction_task is None:
            self.prediction_task = asyncio.create_task(self._run_optional_predictor())
        work = _PredictionWork(
            chunk=chunk,
            transcript_revision=self.transcript_revisions.latest,
        )
        if self.prediction_queue.full():
            dropped_observation_count = (
                self.prediction_queue.qsize() + int(self.prediction_in_flight) + 1
            )
            self.dropped_prediction_observations += dropped_observation_count
            self.optional_predictor_degraded = True
            while not self.prediction_queue.empty():
                self.prediction_queue.get_nowait()
            if self.prediction_task is not None and not self.prediction_task.done():
                self.prediction_task.cancel()
            self.prediction_in_flight = False
            self.event_queue.put_nowait(
                SpeechUnderstandingDegradedEvent(
                    stamp=self._chunk_stamp(
                        chunk,
                        source=CausalSource.TURN_ADAPTER,
                        model_name=None,
                        model_revision=None,
                        conditioned_transcript_revision=None,
                    ),
                    component=SpeechUnderstandingComponent.STANDALONE_TURN_DETECTOR,
                    reason=(
                        "Optional detector queue overflowed; continuity was lost and the detector "
                        "was disabled for the conversation."
                    ),
                    dropped_observation_count=self.dropped_prediction_observations,
                )
            )
            return
        self.prediction_queue.put_nowait(work)

    async def _run_optional_predictor(self) -> None:
        assert self.prediction_source is not None
        while True:
            work = await self.prediction_queue.get()
            if work is None:
                return
            observation = TurnPredictionObservation(
                audio_chunk=work.chunk,
                transcript_revision=work.transcript_revision,
            )
            self.prediction_in_flight = True
            try:
                prediction = await self.prediction_source.predict(observation)
                if self.optional_predictor_degraded:
                    return
                if prediction is None:
                    if self._is_current(work.chunk):
                        self.event_queue.put_nowait(
                            SpeechUnderstandingAbstainedEvent(
                                stamp=self._chunk_stamp(
                                    work.chunk,
                                    source=CausalSource.TURN_ADAPTER,
                                    model_name=None,
                                    model_revision=None,
                                    conditioned_transcript_revision=work.transcript_revision,
                                ),
                                component=(SpeechUnderstandingComponent.STANDALONE_TURN_DETECTOR),
                                reason="The optional detector emitted no evidence.",
                            )
                        )
                    continue
                self._validate_prediction(prediction, work)
                if not self._is_current(work.chunk):
                    continue
                for event in _prediction_events(prediction):
                    self.event_queue.put_nowait(event)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.optional_predictor_degraded = True
                if self._is_current(work.chunk):
                    self.event_queue.put_nowait(
                        SpeechUnderstandingDegradedEvent(
                            stamp=self._chunk_stamp(
                                work.chunk,
                                source=CausalSource.TURN_ADAPTER,
                                model_name=None,
                                model_revision=None,
                                conditioned_transcript_revision=work.transcript_revision,
                            ),
                            component=SpeechUnderstandingComponent.STANDALONE_TURN_DETECTOR,
                            reason=str(error),
                            dropped_observation_count=self.dropped_prediction_observations,
                        )
                    )
                while not self.prediction_queue.empty():
                    self.prediction_queue.get_nowait()
                return
            finally:
                self.prediction_in_flight = False

    async def _stop_optional_predictor(self) -> None:
        task = self.prediction_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.prediction_task = None

    def _validate_prediction(
        self,
        prediction: InteractionPrediction,
        work: _PredictionWork,
    ) -> None:
        stamp = prediction.stamp
        chunk = work.chunk
        if stamp.stream_epoch != chunk.stream_epoch or stamp.turn_epoch != chunk.turn_epoch:
            raise ValueError("Turn prediction epochs do not match the observed audio.")
        if stamp.observation_id != _observation_id(chunk):
            raise ValueError("Turn prediction must identify its audio observation.")
        if (
            stamp.input_start_sample != chunk.start_input_sample
            or stamp.input_end_sample != chunk.end_input_sample
        ):
            raise ValueError("Turn prediction sample range does not match its observation.")
        conditioned_revision_id = stamp.conditioned_transcript_revision_id
        if conditioned_revision_id is not None:
            revision = work.transcript_revision
            if revision is None or conditioned_revision_id != revision.revision_id:
                raise ValueError("Lexical prediction cites an unavailable transcript revision.")
            if revision.stamp.event_id not in stamp.parent_event_ids:
                raise ValueError("Lexical prediction must cite its consumed transcript revision.")
        conditioned_playback_event_id = stamp.conditioned_playback_event_id
        if conditioned_playback_event_id != chunk.playback_condition.event_id:
            raise ValueError("Turn prediction must cite its playback-condition input.")

    def _is_current(self, chunk: CapturedAudioChunk) -> bool:
        return (
            not self.closed
            and chunk.stream_epoch == self._stream_epoch
            and chunk.turn_epoch == self._turn_epoch
        )

    def _chunk_stamp(
        self,
        chunk: CapturedAudioChunk,
        source: CausalSource,
        model_name: str | None,
        model_revision: str | None,
        conditioned_transcript_revision: TranscriptRevision | None,
    ) -> TraceStamp:
        return TraceStamp(
            event_id=str(uuid4()),
            parent_event_ids=(
                ()
                if conditioned_transcript_revision is None
                else (conditioned_transcript_revision.stamp.event_id,)
            ),
            stream_epoch=chunk.stream_epoch,
            turn_epoch=chunk.turn_epoch,
            inference_step=self.inference_step,
            observation_id=_observation_id(chunk),
            observation_monotonic_time_ns=chunk.monotonic_observation_time_ns,
            emission_monotonic_time_ns=time.perf_counter_ns(),
            encoder_frame_start=None,
            encoder_frame_end=None,
            input_start_sample=chunk.start_input_sample,
            input_end_sample=chunk.end_input_sample,
            observed_through_input_sample=chunk.end_input_sample,
            input_sample_position=chunk.end_input_sample,
            output_sample_position=chunk.playback_condition.latest_output_sample_position,
            conditioned_transcript_revision_id=(
                None
                if conditioned_transcript_revision is None
                else conditioned_transcript_revision.revision_id
            ),
            conditioned_playback_event_id=chunk.playback_condition.event_id,
            source=source,
            model_name=model_name,
            model_revision=model_revision,
        )

    def _status_event(
        self,
        status: SpeechUnderstandingStatus,
        turn_epoch: int,
    ) -> SpeechUnderstandingStatusEvent:
        chunk = self.latest_chunk
        if chunk is None:
            stamp = TraceStamp(
                event_id=str(uuid4()),
                parent_event_ids=(),
                stream_epoch=self._stream_epoch,
                turn_epoch=turn_epoch,
                inference_step=self.inference_step,
                observation_id=f"lifecycle:{self._stream_epoch}:{turn_epoch}:{status}",
                observation_monotonic_time_ns=time.perf_counter_ns(),
                emission_monotonic_time_ns=time.perf_counter_ns(),
                encoder_frame_start=None,
                encoder_frame_end=None,
                input_start_sample=self.last_input_end_sample or 0,
                input_end_sample=self.last_input_end_sample or 0,
                observed_through_input_sample=self.last_input_end_sample or 0,
                input_sample_position=self.last_input_end_sample or 0,
                output_sample_position=None,
                conditioned_transcript_revision_id=None,
                conditioned_playback_event_id=None,
                source=CausalSource.FLOOR_POLICY,
                model_name=None,
                model_revision=None,
            )
        else:
            stamp = self._chunk_stamp(
                chunk,
                source=CausalSource.FLOOR_POLICY,
                model_name=None,
                model_revision=None,
                conditioned_transcript_revision=None,
            )
        return SpeechUnderstandingStatusEvent(stamp=stamp, status=status)


def _prediction_events(
    prediction: InteractionPrediction,
) -> tuple[
    FutureActivityEvidence,
    TurnEventEvidence,
    OverlapDispositionEvidence,
    YieldEvidence,
]:
    evidence_group_id = prediction.stamp.event_id
    future_stamp = _sibling_stamp(prediction.stamp, evidence_group_id)
    turn_event_stamp = _sibling_stamp(prediction.stamp, evidence_group_id)
    overlap_stamp = _sibling_stamp(prediction.stamp, evidence_group_id)
    yield_stamp = _sibling_stamp(prediction.stamp, evidence_group_id)
    return (
        FutureActivityEvidence(
            stamp=future_stamp,
            evidence_group_id=evidence_group_id,
            horizons=prediction.future_user_activity_horizons,
            confidence=prediction.confidence,
        ),
        TurnEventEvidence(
            stamp=turn_event_stamp,
            evidence_group_id=evidence_group_id,
            probabilities=(
                TurnEventProbability(
                    event=TurnEventKind.BACKCHANNEL,
                    probability=prediction.p_user_backchannel,
                ),
            ),
            confidence=prediction.confidence,
        ),
        OverlapDispositionEvidence(
            stamp=overlap_stamp,
            evidence_group_id=evidence_group_id,
            probabilities=(
                OverlapDispositionProbability(
                    disposition=OverlapDisposition.FLOOR_TAKING,
                    probability=prediction.p_user_interruption,
                ),
            ),
            p_user_interruption=prediction.p_user_interruption,
            confidence=prediction.confidence,
        ),
        YieldEvidence(
            stamp=yield_stamp,
            evidence_group_id=evidence_group_id,
            p_user_yield=prediction.p_user_yield,
            p_user_speech=prediction.p_user_speech,
            confidence=prediction.confidence,
        ),
    )


def _sibling_stamp(stamp: TraceStamp, evidence_group_id: str) -> TraceStamp:
    return stamp.model_copy(
        update={
            "event_id": str(uuid4()),
            "parent_event_ids": (*stamp.parent_event_ids, evidence_group_id),
            "emission_monotonic_time_ns": time.perf_counter_ns(),
        }
    )


def _observation_id(chunk: CapturedAudioChunk) -> str:
    return f"audio:{chunk.stream_epoch}:{chunk.sequence_number}"

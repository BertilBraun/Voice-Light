from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest

from app.compute.voice.interfaces import (
    TranscriptionSession,
    TurnPredictionObservation,
    TurnPredictionSource,
)
from app.compute.voice.schemas import (
    CapturedAudioChunk,
    CausalSource,
    InteractionPrediction,
    PlaybackCondition,
    PlaybackConditionAuthority,
    PlaybackState,
    SileroEvidence,
    SpeechUnderstandingDegradedEvent,
    TraceStamp,
    TranscriptRevision,
    YieldEvidence,
)
from app.compute.voice.speech_understanding import (
    CompositeSpeechUnderstandingProvider,
    InteractionPredictionReducer,
    SingleSessionTurnPredictionProvider,
)


class RecordingTranscriptionSession:
    def __init__(self, partial_text: str = "hello", fail_add: bool = False) -> None:
        self.partial_text = partial_text
        self.fail_add = fail_add
        self.audio: list[bytes] = []
        self.finished = False
        self.closed = False

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        if self.fail_add:
            raise RuntimeError("synthetic ASR failure")
        self.audio.append(pcm_bytes)
        return self.partial_text

    async def finish(self) -> str:
        self.finished = True
        return self.partial_text

    async def close(self) -> None:
        self.closed = True


class RecordingTranscriber:
    def __init__(
        self,
        session_factory: Callable[[], RecordingTranscriptionSession],
    ) -> None:
        self.session_factory = session_factory
        self.sessions: list[RecordingTranscriptionSession] = []
        self.closed = False

    def start_session(self) -> TranscriptionSession:
        session = self.session_factory()
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


class SourceProvider:
    def __init__(self, source: TurnPredictionSource) -> None:
        self.source = source
        self.closed = False

    def create_session(self) -> TurnPredictionSource:
        return self.source

    def close(self) -> None:
        self.closed = True


class RecordingPredictionSource:
    def __init__(self, condition_on_transcript: bool) -> None:
        self.condition_on_transcript = condition_on_transcript
        self.observations: list[TurnPredictionObservation] = []
        self.closed = False

    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction:
        self.observations.append(observation)
        chunk = observation.audio_chunk
        revision = observation.transcript_revision if self.condition_on_transcript else None
        return create_prediction(chunk, revision)

    async def close(self) -> None:
        self.closed = True


class FailingPredictionSource:
    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction | None:
        del observation
        raise RuntimeError("synthetic optional detector failure")

    async def close(self) -> None:
        return


class BlockingPredictionSource:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False
        self.observation_count = 0

    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction:
        self.observation_count += 1
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return create_prediction(observation.audio_chunk, observation.transcript_revision)

    async def close(self) -> None:
        return


def test_provider_owns_persistent_resources_and_sessions_own_turn_state() -> None:
    async def exercise() -> None:
        transcriber = RecordingTranscriber(RecordingTranscriptionSession)
        source = RecordingPredictionSource(condition_on_transcript=False)
        source_provider = SourceProvider(source)
        provider = CompositeSpeechUnderstandingProvider(
            transcriber=transcriber,
            turn_prediction_provider=source_provider,
            asr_model_name="test-asr",
            asr_model_revision="1",
        )
        session = provider.create_session(stream_epoch=3)

        await session.add_audio(create_chunk(sequence_number=0, stream_epoch=3, turn_epoch=1))
        finalized = await session.finalize_turn()
        await session.add_audio(create_chunk(sequence_number=1, stream_epoch=3, turn_epoch=2))
        await session.close()
        provider.close()

        assert finalized.text == "hello"
        assert len(transcriber.sessions) == 2
        assert all(item.closed for item in transcriber.sessions)
        assert source.closed is True
        assert source_provider.closed is True
        assert transcriber.closed is True

    asyncio.run(exercise())


@pytest.mark.parametrize("condition_on_transcript", (False, True))
def test_transcript_and_prediction_evidence_are_sibling_events_with_causal_inputs(
    condition_on_transcript: bool,
) -> None:
    async def exercise() -> None:
        source = RecordingPredictionSource(condition_on_transcript=condition_on_transcript)
        provider = create_provider(source)
        session = provider.create_session(stream_epoch=1)
        chunk = create_chunk(
            sequence_number=1,
            stream_epoch=1,
            turn_epoch=1,
            playback_state=PlaybackState.SPEAKING,
            assistant_audible=True,
        )

        await session.add_audio(create_chunk(sequence_number=0, stream_epoch=1, turn_epoch=1))
        await session.add_audio(chunk)
        events = session.drain_events()
        transcripts = tuple(event for event in events if isinstance(event, TranscriptRevision))
        transcript = transcripts[-1]
        yields = tuple(event for event in events if isinstance(event, YieldEvidence))

        assert len(yields) == 1
        evidence = yields[0]
        assert evidence.stamp.observation_id == "audio:1:1"
        assert evidence.stamp.input_start_sample == chunk.start_input_sample
        assert evidence.stamp.input_end_sample == chunk.end_input_sample
        assert evidence.stamp.conditioned_playback_event_id == chunk.playback_condition.event_id
        expected_revision_id = transcript.revision_id if condition_on_transcript else None
        assert evidence.stamp.conditioned_transcript_revision_id == expected_revision_id
        assert (
            transcript.stamp.event_id in evidence.stamp.parent_event_ids
        ) is condition_on_transcript
        assert source.observations[0].audio_chunk.playback_condition.state is PlaybackState.SPEAKING

        await session.close()

    asyncio.run(exercise())


def test_optional_predictor_failure_degrades_without_killing_asr() -> None:
    async def exercise() -> None:
        provider = create_provider(FailingPredictionSource())
        session = provider.create_session(stream_epoch=1)

        await session.add_audio(create_chunk(sequence_number=0, stream_epoch=1, turn_epoch=1))
        await session.add_audio(create_chunk(sequence_number=1, stream_epoch=1, turn_epoch=1))
        events = session.drain_events()
        finalized = await session.finalize_turn()

        assert any(isinstance(event, TranscriptRevision) for event in events)
        degraded = next(
            event for event in events if isinstance(event, SpeechUnderstandingDegradedEvent)
        )
        assert "synthetic optional detector failure" in degraded.reason
        assert finalized.text == "hello"
        await session.close()

    asyncio.run(exercise())


def test_late_optional_result_is_cancelled_and_old_turn_audio_is_rejected() -> None:
    async def exercise() -> None:
        source = BlockingPredictionSource()
        provider = create_provider(source)
        session = provider.create_session(stream_epoch=1)
        old_chunk = create_chunk(sequence_number=1, stream_epoch=1, turn_epoch=1)

        await session.add_audio(create_chunk(sequence_number=0, stream_epoch=1, turn_epoch=1))
        await session.add_audio(old_chunk)
        await asyncio.wait_for(source.started.wait(), timeout=1.0)
        await session.finalize_turn()

        assert source.cancelled is True
        assert not any(isinstance(event, YieldEvidence) for event in session.drain_events())
        with pytest.raises(ValueError, match="stale speech turn epoch"):
            await session.add_audio(old_chunk)
        await session.close()

    asyncio.run(exercise())


def test_optional_predictor_queue_gap_disables_detector_without_stalling_asr() -> None:
    async def exercise() -> None:
        source = BlockingPredictionSource()
        transcriber = RecordingTranscriber(RecordingTranscriptionSession)
        provider = CompositeSpeechUnderstandingProvider(
            transcriber=transcriber,
            turn_prediction_provider=SingleSessionTurnPredictionProvider(source),
            asr_model_name="test-asr",
            asr_model_revision="1",
            optional_predictor_queue_size=1,
        )
        session = provider.create_session(stream_epoch=1)

        for sequence_number in range(4):
            await session.add_audio(
                create_chunk(
                    sequence_number=sequence_number,
                    stream_epoch=1,
                    turn_epoch=1,
                )
            )
        await session.add_audio(create_chunk(sequence_number=4, stream_epoch=1, turn_epoch=1))

        events = session.drain_events()
        degraded = tuple(
            event for event in events if isinstance(event, SpeechUnderstandingDegradedEvent)
        )
        assert len(transcriber.sessions[0].audio) == 5
        assert degraded[-1].dropped_observation_count >= 1
        assert "disabled for the conversation" in degraded[-1].reason
        assert source.cancelled is True
        assert source.observation_count == 1
        assert not any(isinstance(event, YieldEvidence) for event in events)
        await session.close()

    asyncio.run(exercise())


def test_prediction_reducer_bounds_and_prunes_causal_state() -> None:
    reducer = InteractionPredictionReducer(
        maximum_playback_conditions=2,
        maximum_evidence_groups=2,
        maximum_sample_lag=320,
    )
    first_chunk = create_chunk(sequence_number=0, stream_epoch=1, turn_epoch=1)
    second_chunk = create_chunk(sequence_number=1, stream_epoch=1, turn_epoch=1)
    third_chunk = create_chunk(sequence_number=2, stream_epoch=1, turn_epoch=1)

    reducer.observe_audio_chunk(first_chunk)
    reducer.observe_audio_chunk(second_chunk)
    reducer.observe_audio_chunk(third_chunk)

    assert reducer.retained_playback_condition_count == 2

    for group_number in range(3):
        prediction = create_prediction(third_chunk, transcript_revision=None)
        reducer.reduce(
            YieldEvidence(
                stamp=prediction.stamp,
                evidence_group_id=f"incomplete-{group_number}",
                p_user_yield=prediction.p_user_yield,
                p_user_speech=prediction.p_user_speech,
                confidence=prediction.confidence,
            )
        )

    assert reducer.pending_evidence_group_count == 2

    far_chunk = create_chunk(sequence_number=5, stream_epoch=1, turn_epoch=1)
    reducer.observe_audio_chunk(far_chunk)

    assert reducer.pending_evidence_group_count == 0
    assert reducer.retained_playback_condition_count == 1

    next_turn_chunk = create_chunk(sequence_number=6, stream_epoch=1, turn_epoch=2)
    reducer.observe_audio_chunk(next_turn_chunk)

    assert reducer.retained_playback_condition_count == 1
    reducer.reset_turn()
    assert reducer.pending_evidence_group_count == 0
    assert reducer.retained_playback_condition_count == 0


def test_prediction_reducer_releases_completed_group_playback_condition() -> None:
    async def exercise() -> None:
        source = RecordingPredictionSource(condition_on_transcript=False)
        provider = create_provider(source)
        session = provider.create_session(stream_epoch=1)
        first_chunk = create_chunk(sequence_number=0, stream_epoch=1, turn_epoch=1)
        second_chunk = create_chunk(sequence_number=1, stream_epoch=1, turn_epoch=1)
        reducer = InteractionPredictionReducer()
        reducer.observe_audio_chunk(first_chunk)
        reducer.observe_audio_chunk(second_chunk)

        await session.add_audio(first_chunk)
        await session.add_audio(second_chunk)
        predictions = tuple(
            prediction
            for event in session.drain_events()
            if (prediction := reducer.reduce(event)) is not None
        )

        assert len(predictions) == 1
        assert reducer.pending_evidence_group_count == 0
        assert reducer.retained_playback_condition_count == 1
        await session.close()

    asyncio.run(exercise())


def test_mandatory_asr_failure_remains_fatal() -> None:
    async def exercise() -> None:
        transcriber = RecordingTranscriber(lambda: RecordingTranscriptionSession(fail_add=True))
        provider = CompositeSpeechUnderstandingProvider(
            transcriber=transcriber,
            turn_prediction_provider=None,
            asr_model_name="test-asr",
            asr_model_revision="1",
        )
        session = provider.create_session(stream_epoch=1)

        with pytest.raises(RuntimeError, match="synthetic ASR failure"):
            await session.add_audio(create_chunk(sequence_number=0, stream_epoch=1, turn_epoch=1))
        await session.close()

    asyncio.run(exercise())


def create_provider(
    source: TurnPredictionSource,
) -> CompositeSpeechUnderstandingProvider:
    return CompositeSpeechUnderstandingProvider(
        transcriber=RecordingTranscriber(RecordingTranscriptionSession),
        turn_prediction_provider=SingleSessionTurnPredictionProvider(source),
        asr_model_name="test-asr",
        asr_model_revision="1",
    )


def create_chunk(
    sequence_number: int,
    stream_epoch: int,
    turn_epoch: int,
    playback_state: PlaybackState = PlaybackState.IDLE,
    assistant_audible: bool = False,
) -> CapturedAudioChunk:
    start_sample = sequence_number * 320
    observation_time_ns = sequence_number + 1
    return CapturedAudioChunk(
        pcm16=b"\x01\x00" * 320,
        sequence_number=sequence_number,
        start_input_sample=start_sample,
        end_input_sample=start_sample + 320,
        monotonic_observation_time_ns=observation_time_ns,
        stream_epoch=stream_epoch,
        turn_epoch=turn_epoch,
        silero_evidence=SileroEvidence(
            is_speech=True,
            monotonic_time_ns=observation_time_ns,
        ),
        playback_condition=PlaybackCondition(
            event_id=str(uuid4()),
            generation_id=7 if playback_state is PlaybackState.SPEAKING else None,
            state=playback_state,
            assistant_audible=assistant_audible,
            latest_output_sample_position=640,
            latest_source_sample_position=320,
            output_sample_rate=48_000,
            monotonic_time_ns=observation_time_ns,
            authority=PlaybackConditionAuthority.SERVER_ESTIMATED,
        ),
    )


def create_prediction(
    chunk: CapturedAudioChunk,
    transcript_revision: TranscriptRevision | None,
) -> InteractionPrediction:
    return InteractionPrediction(
        stamp=TraceStamp(
            event_id=str(uuid4()),
            parent_event_ids=(
                () if transcript_revision is None else (transcript_revision.stamp.event_id,)
            ),
            stream_epoch=chunk.stream_epoch,
            turn_epoch=chunk.turn_epoch,
            inference_step=chunk.sequence_number,
            observation_id=f"audio:{chunk.stream_epoch}:{chunk.sequence_number}",
            observation_monotonic_time_ns=chunk.monotonic_observation_time_ns,
            emission_monotonic_time_ns=chunk.monotonic_observation_time_ns,
            encoder_frame_start=None,
            encoder_frame_end=None,
            input_start_sample=chunk.start_input_sample,
            input_end_sample=chunk.end_input_sample,
            observed_through_input_sample=chunk.end_input_sample,
            input_sample_position=chunk.end_input_sample,
            output_sample_position=chunk.playback_condition.latest_output_sample_position,
            conditioned_transcript_revision_id=(
                None if transcript_revision is None else transcript_revision.revision_id
            ),
            conditioned_playback_event_id=chunk.playback_condition.event_id,
            source=CausalSource.TURN_ADAPTER,
            model_name="test-detector",
            model_revision="1",
        ),
        p_user_speech=0.2,
        p_user_yield=0.8,
        p_user_backchannel=0.1,
        p_user_interruption=0.05,
        future_user_activity_horizons=(),
        assistant_playback_state=chunk.playback_condition.state,
        confidence=0.9,
    )

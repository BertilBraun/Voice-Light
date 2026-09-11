from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.compute.voice.predictive import (
    CandidateInvalidationReason,
    CandidateOutput,
    CandidateReleaseGate,
    PredictiveMetrics,
    ReleasedAudioChunk,
    ReleasedTextDelta,
    ReleasedWordBoundary,
    TranscriptRevisionTracker,
    candidate_final_invalidation_reason,
    candidate_revision_invalidation_reason,
)
from app.compute.voice.schemas import (
    CapturedAudioChunk,
    CausalSource,
    PlaybackCondition,
    PlaybackConditionAuthority,
    PlaybackState,
    SileroEvidence,
    TranscriptRevision,
)


class InMemoryPlaybackSink:
    def __init__(self) -> None:
        self.outputs: list[CandidateOutput] = []

    async def send(self, output: CandidateOutput) -> None:
        self.outputs.append(output)


def test_release_gate_hides_then_releases_candidate_outputs_in_order() -> None:
    async def exercise_gate() -> None:
        sink = InMemoryPlaybackSink()
        gate = CandidateReleaseGate(sink=sink, released=False)
        buffered_outputs: tuple[CandidateOutput, ...] = (
            ReleasedTextDelta(generation_id=7, text="Hello "),
            ReleasedWordBoundary(generation_id=7, text_offset=6, start_sample=0),
            ReleasedAudioChunk(
                generation_id=7,
                sequence_number=0,
                start_sample=0,
                pcm_bytes=b"\x01\x00\x02\x00",
            ),
        )
        for output in buffered_outputs:
            await gate.publish(output)

        assert sink.outputs == []
        assert gate.outputs == buffered_outputs
        assert gate.buffered_pcm_sample_count == 2

        await gate.release()
        trailing_output = ReleasedTextDelta(generation_id=7, text="world")
        await gate.publish(trailing_output)

        assert sink.outputs == [*buffered_outputs, trailing_output]
        assert gate.first_released_pcm_at is not None

    asyncio.run(exercise_gate())


def test_discarded_release_gate_never_reaches_sink() -> None:
    async def exercise_gate() -> None:
        sink = InMemoryPlaybackSink()
        gate = CandidateReleaseGate(sink=sink, released=False)
        await gate.publish(ReleasedTextDelta(generation_id=3, text="stale"))
        await gate.discard()
        await gate.publish(ReleasedTextDelta(generation_id=3, text="later"))

        assert sink.outputs == []

    asyncio.run(exercise_gate())


def test_transcript_revision_retains_stable_prefix_across_volatile_changes() -> None:
    tracker = TranscriptRevisionTracker()

    first = update_transcript(tracker, "book a", sequence_number=0)
    anchored = update_transcript(tracker, "book a", sequence_number=1)
    timing_only = update_transcript(tracker, "book a", sequence_number=2)
    extended = update_transcript(tracker, "book a table", sequence_number=3)
    revised_suffix = update_transcript(tracker, "book a train", sequence_number=4)

    assert first is not None
    assert anchored is not None
    assert timing_only is anchored
    assert anchored.stable_prefix == "book a"
    assert extended is not None
    assert extended.stable_prefix == "book a"
    assert extended.volatile_suffix == " table"
    assert revised_suffix is not None
    assert revised_suffix.stable_prefix == "book a "
    assert revised_suffix.volatile_suffix == "train"
    assert revised_suffix.supersedes_revision_id == extended.revision_id


def update_transcript(
    tracker: TranscriptRevisionTracker,
    text: str,
    sequence_number: int,
) -> TranscriptRevision | None:
    start_sample = sequence_number * 320
    chunk = CapturedAudioChunk(
        pcm16=b"\x00\x00" * 320,
        sequence_number=sequence_number,
        start_input_sample=start_sample,
        end_input_sample=start_sample + 320,
        monotonic_observation_time_ns=sequence_number,
        stream_epoch=1,
        turn_epoch=1,
        silero_evidence=SileroEvidence(
            is_speech=True,
            monotonic_time_ns=sequence_number,
        ),
        playback_condition=PlaybackCondition(
            event_id=str(uuid4()),
            generation_id=None,
            state=PlaybackState.IDLE,
            assistant_audible=False,
            latest_output_sample_position=0,
            latest_source_sample_position=0,
            output_sample_rate=None,
            monotonic_time_ns=sequence_number,
            authority=PlaybackConditionAuthority.SERVER_ESTIMATED,
        ),
    )
    return tracker.update(
        text=text,
        chunk=chunk,
        inference_step=sequence_number,
        observed_through_input_sample=chunk.end_input_sample,
        model_name="test-asr",
        model_revision="1",
    )


def test_final_candidate_validation_is_conservative() -> None:
    assert (
        candidate_final_invalidation_reason(
            stable_prefix="What time is it",
            prompted_text="What time is it",
            final_text="What time is it?",
        )
        is None
    )
    assert (
        candidate_final_invalidation_reason(
            stable_prefix="What time is it",
            prompted_text="What time is it",
            final_text="What time is it in Tokyo?",
        )
        is CandidateInvalidationReason.MATERIAL_REQUEST_CHANGE
    )
    assert (
        candidate_final_invalidation_reason(
            stable_prefix="book a flight",
            prompted_text="book a flight",
            final_text="cancel the flight",
        )
        is CandidateInvalidationReason.FINAL_PREFIX_CHANGED
    )


@pytest.mark.parametrize(
    "final_text",
    (
        "what TIME is it",
        "  What   time is it?  ",
        "What time is it...",
    ),
)
def test_final_candidate_survives_presentation_only_revision(final_text: str) -> None:
    assert (
        candidate_final_invalidation_reason(
            stable_prefix="What time is it",
            prompted_text="What time is it",
            final_text=final_text,
        )
        is None
    )


def test_final_candidate_survives_asr_apostrophe_revision() -> None:
    assert (
        candidate_final_invalidation_reason(
            stable_prefix="dont stop",
            prompted_text="dont stop",
            final_text="don't stop!",
        )
        is None
    )


def test_vad_candidate_uses_combined_transcript_across_prefix_resegmentation() -> None:
    assert (
        candidate_revision_invalidation_reason(
            source=CausalSource.SILERO_VAD,
            stable_prefix="book a",
            prompted_text="book a table",
            revised_stable_prefix="book a ",
            revised_volatile_suffix="table",
        )
        is None
    )
    assert (
        candidate_revision_invalidation_reason(
            source=CausalSource.SILERO_VAD,
            stable_prefix="book a",
            prompted_text="book a table",
            revised_stable_prefix="book a ",
            revised_volatile_suffix="train",
        )
        is CandidateInvalidationReason.TRANSCRIPT_SUPERSEDED
    )


@pytest.mark.parametrize(
    ("prompted_text", "revised_stable_prefix", "revised_volatile_suffix"),
    (
        ("what time is it", "What time is it", "?"),
        ("Hello New York", "hello ", "new york."),
        ("temperature in  New York", "temperature in ", "New York"),
    ),
)
def test_vad_candidate_survives_nonsemantic_transcript_revision(
    prompted_text: str,
    revised_stable_prefix: str,
    revised_volatile_suffix: str,
) -> None:
    assert (
        candidate_revision_invalidation_reason(
            source=CausalSource.SILERO_PENDING_SILENCE,
            stable_prefix="",
            prompted_text=prompted_text,
            revised_stable_prefix=revised_stable_prefix,
            revised_volatile_suffix=revised_volatile_suffix,
        )
        is None
    )


def test_trained_candidate_remains_anchored_to_stable_prefix() -> None:
    assert (
        candidate_revision_invalidation_reason(
            source=CausalSource.TURN_ADAPTER,
            stable_prefix="book a ",
            prompted_text="book a",
            revised_stable_prefix="book ",
            revised_volatile_suffix="a table",
        )
        is CandidateInvalidationReason.STABLE_PREFIX_REVISED
    )


@pytest.mark.parametrize(
    ("revised_stable_prefix", "revised_volatile_suffix"),
    (
        ("Book a table", "?"),
        ("book  a ", "table."),
        ("book a table...", ""),
    ),
)
def test_trained_candidate_survives_presentation_only_revision(
    revised_stable_prefix: str,
    revised_volatile_suffix: str,
) -> None:
    assert (
        candidate_revision_invalidation_reason(
            source=CausalSource.TURN_ADAPTER,
            stable_prefix="book a table",
            prompted_text="book a table",
            revised_stable_prefix=revised_stable_prefix,
            revised_volatile_suffix=revised_volatile_suffix,
        )
        is None
    )


@pytest.mark.parametrize(
    ("revised_stable_prefix", "revised_volatile_suffix"),
    (
        ("book a table", " tomorrow"),
        ("book a ", "train"),
    ),
)
def test_trained_candidate_rejects_lexical_revision(
    revised_stable_prefix: str,
    revised_volatile_suffix: str,
) -> None:
    assert (
        candidate_revision_invalidation_reason(
            source=CausalSource.TURN_ADAPTER,
            stable_prefix="book a table",
            prompted_text="book a table",
            revised_stable_prefix=revised_stable_prefix,
            revised_volatile_suffix=revised_volatile_suffix,
        )
        is not None
    )


def test_predictive_metrics_report_hits_waste_and_latency_buckets() -> None:
    metrics = PredictiveMetrics()
    metrics.record_candidate_created()
    metrics.record_invalidation(
        CandidateInvalidationReason.USER_ACTIVITY_RESUMED,
        qwen_tokens=4,
        tts_samples=480,
    )
    metrics.record_first_playback(
        commit_at=1.0,
        playback_at=1.2,
        true_end_at=None,
        had_candidate=False,
        followed_invalidation=True,
    )
    metrics.record_first_release(commit_at=1.0, release_at=1.15)
    metrics.record_promoted_candidate_timing(
        speculation_started_at=0.8,
        first_endpoint_at=0.85,
        final_endpoint_at=0.95,
        committed_at=1.0,
        first_qwen_word_at=0.9,
        first_tts_pcm_at=1.1,
        qwen_word_ready_at_commit=True,
        tts_pcm_ready_at_commit=False,
    )

    report = metrics.report()

    assert report.candidate_hit_rate == 0.0
    assert report.invalidation_rate == 1.0
    assert report.wasted_qwen_tokens == 4
    assert report.wasted_tts_samples == 480
    assert report.candidate_start_relative_to_first_endpoint_p50_ms == pytest.approx(-50.0)
    assert report.candidate_start_relative_to_final_endpoint_p50_ms == pytest.approx(-150.0)
    assert report.candidate_start_to_commit_p50_ms == pytest.approx(200.0)
    assert report.candidate_start_to_first_qwen_word_p50_ms == pytest.approx(100.0)
    assert report.candidate_start_to_first_tts_pcm_p50_ms == pytest.approx(300.0)
    assert report.qwen_word_ready_at_commit_rate == 1.0
    assert report.tts_pcm_ready_at_commit_rate == 0.0
    assert report.commit_to_first_played_audio_p50_ms == pytest.approx(200.0)
    assert report.commit_to_first_released_pcm_p50_ms == pytest.approx(150.0)
    assert report.commit_to_first_released_pcm_p90_ms == pytest.approx(150.0)
    assert report.commit_to_first_released_pcm_p95_ms == pytest.approx(150.0)
    assert report.no_candidate_latency_p50_ms == pytest.approx(200.0)
    assert report.post_invalidation_latency_p50_ms == pytest.approx(200.0)

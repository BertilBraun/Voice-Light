from __future__ import annotations

import asyncio

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

    first = tracker.update("book a", input_sample_position=320)
    anchored = tracker.update("book a", input_sample_position=640)
    extended = tracker.update("book a table", input_sample_position=960)
    revised_suffix = tracker.update("book a train", input_sample_position=1_280)

    assert first is not None
    assert anchored is not None
    assert anchored.stable_prefix == "book a"
    assert extended is not None
    assert extended.stable_prefix == "book a"
    assert extended.volatile_suffix == " table"
    assert revised_suffix is not None
    assert revised_suffix.stable_prefix == "book a "
    assert revised_suffix.volatile_suffix == "train"
    assert revised_suffix.supersedes_revision_id == extended.revision_id


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

    report = metrics.report()

    assert report.candidate_hit_rate == 0.0
    assert report.invalidation_rate == 1.0
    assert report.wasted_qwen_tokens == 4
    assert report.wasted_tts_samples == 480
    assert report.commit_to_first_played_audio_p50_ms == pytest.approx(200.0)
    assert report.commit_to_first_released_pcm_p50_ms == pytest.approx(150.0)
    assert report.commit_to_first_released_pcm_p90_ms == pytest.approx(150.0)
    assert report.commit_to_first_released_pcm_p95_ms == pytest.approx(150.0)
    assert report.no_candidate_latency_p50_ms == pytest.approx(200.0)
    assert report.post_invalidation_latency_p50_ms == pytest.approx(200.0)

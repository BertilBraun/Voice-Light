from __future__ import annotations

import wave
from array import array
from pathlib import Path

import numpy as np
import pytest

from app.local.analyses.end_of_turn.detectors.naive_vad import (
    run_naive_vad_floor,
    run_naive_vad_floor_full_recording,
)
from app.local.synchronization_review.activity_optimization import StableOffsetEstimate
from app.local.synchronization_review.audit import (
    _anomaly_score,
    _audit_kind,
    _audit_window,
)
from app.local.synchronization_review.models import (
    SynchronizationAuditKind,
    SynchronizationAuditWindow,
    SynchronizationEvidenceSource,
    SynchronizationWindowEstimate,
)
from app.local.synchronization_review.optimized_alignment import (
    ActivityOffsetAnalysis,
    ActivityWindowAnalysis,
    overlapping_window_analyses,
)
from app.shared.audio.wav import iter_mono_wave_audio_chunks


def test_full_recording_vad_reads_activity_after_legacy_three_minute_cap(
    tmp_path: Path,
) -> None:
    sample_rate = 1_000
    wave_path = tmp_path / "long.wav"
    samples = array("h", [0] * (sample_rate * 190))
    for index in range(sample_rate * 185, sample_rate * 186):
        samples[index] = 10_000
    _write_wave(wave_path=wave_path, sample_rate=sample_rate, samples=samples)

    capped = run_naive_vad_floor(
        wave_path=wave_path,
        result_name="capped",
        description="capped",
    )
    full = run_naive_vad_floor_full_recording(
        wave_path=wave_path,
        result_name="full",
        description="full",
    )

    assert not capped.speech_segments
    assert full.speech_segments[-1].start_seconds == pytest.approx(185.0, abs=0.04)
    assert full.speech_segments[-1].end_seconds == pytest.approx(186.0, abs=0.03)


def test_wave_chunk_iterator_preserves_entire_recording(tmp_path: Path) -> None:
    sample_rate = 100
    wave_path = tmp_path / "chunks.wav"
    samples = array("h", range(1_050))
    _write_wave(wave_path=wave_path, sample_rate=sample_rate, samples=samples)

    chunks = tuple(
        iter_mono_wave_audio_chunks(
            wave_path=wave_path,
            chunk_duration_seconds=3.0,
        )
    )

    assert [chunk.frame_count for chunk in chunks] == [300, 300, 300, 150]
    assert sum(chunk.frame_count for chunk in chunks) == len(samples)


def test_overlapping_windows_use_sixty_second_hop() -> None:
    random_generator = np.random.default_rng(seed=31)
    frame_count = round(600.0 / 0.03)
    speaker1 = random_generator.random(frame_count) > 0.7
    speaker2 = random_generator.random(frame_count) > 0.7

    windows = overlapping_window_analyses(
        speaker1_mask=speaker1,
        speaker2_mask=speaker2,
    )

    assert [window.start_seconds for window in windows] == [
        0.0,
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
        360.0,
        420.0,
    ]
    assert all(window.end_seconds - window.start_seconds == 180.0 for window in windows)


def test_audit_rejects_maximum_lag_boundary_even_with_transcript_agreement() -> None:
    activity_window = _activity_window(start_seconds=0.0, shift_seconds=12.0)
    transcript_windows = tuple(
        SynchronizationWindowEstimate(
            source=source,
            start_seconds=0.0,
            end_seconds=180.0,
            estimated_b_shift_seconds=12.0,
            bad_state_improvement=0.2,
            overlap_reduction=0.1,
            silence_reduction=0.1,
            meaningful=True,
        )
        for source in (
            SynchronizationEvidenceSource.PARAKEET,
            SynchronizationEvidenceSource.CANARY,
        )
    )

    audited = _audit_window(
        window=activity_window,
        all_activity_windows=(activity_window,),
        transcript_windows=transcript_windows,
    )

    assert audited.maximum_lag_boundary
    assert audited.confidence_score == 0.0


@pytest.mark.parametrize(
    (
        "persistence_window_count",
        "shift_seconds",
        "temporal_range_seconds",
        "expected_kind",
    ),
    (
        (1, 6.0, 0.0, SynchronizationAuditKind.UNCERTAIN),
        (3, 2.0, 0.4, SynchronizationAuditKind.STABLE_OFFSET),
        (3, 4.5, 4.8, SynchronizationAuditKind.TEMPORAL_CHANGE),
    ),
)
def test_audit_kind_requires_persistent_non_boundary_evidence(
    persistence_window_count: int,
    shift_seconds: float,
    temporal_range_seconds: float,
    expected_kind: SynchronizationAuditKind,
) -> None:
    strongest_window = _scored_audit_window(
        persistence_window_count=persistence_window_count,
        shift_seconds=shift_seconds,
    )

    kind = _audit_kind(
        strongest_window=strongest_window,
        temporal_shift_range_seconds=temporal_range_seconds,
    )
    score = _anomaly_score(
        kind=kind,
        strongest_window=strongest_window,
        temporal_shift_range_seconds=temporal_range_seconds,
    )

    assert kind is expected_kind
    if expected_kind is SynchronizationAuditKind.UNCERTAIN:
        assert score < 0.2
    else:
        assert score >= 0.5


def _activity_window(start_seconds: float, shift_seconds: float) -> ActivityWindowAnalysis:
    return ActivityWindowAnalysis(
        start_seconds=start_seconds,
        end_seconds=start_seconds + 180.0,
        analysis=ActivityOffsetAnalysis(
            estimate=StableOffsetEstimate(
                shift_seconds=shift_seconds,
                point_loss=0.1,
                neighborhood_loss=0.1,
                zero_shift_neighborhood_loss=0.2,
                improvement_over_zero=0.1,
                competing_margin=0.1,
                basin_width_seconds=0.3,
            ),
            overlap_reduction=0.1,
            dual_silence_reduction=0.1,
            accepted=True,
        ),
    )


def _scored_audit_window(
    persistence_window_count: int,
    shift_seconds: float,
) -> SynchronizationAuditWindow:
    return SynchronizationAuditWindow(
        start_seconds=360.0,
        end_seconds=540.0,
        estimated_b_shift_seconds=shift_seconds,
        confidence_score=0.9,
        bad_state_improvement=0.1,
        competing_margin=0.01,
        basin_width_seconds=0.12,
        persistence_window_count=persistence_window_count,
        agreeing_transcript_sources=(
            SynchronizationEvidenceSource.PARAKEET,
            SynchronizationEvidenceSource.CANARY,
        ),
        accepted=True,
        maximum_lag_boundary=False,
    )


def _write_wave(wave_path: Path, sample_rate: int, samples: array[int]) -> None:
    with wave.open(str(wave_path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(samples.tobytes())

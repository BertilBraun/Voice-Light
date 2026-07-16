from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.local.main import app
from app.local.synchronization_review.models import (
    OffsetPattern,
    SynchronizationEvidenceSource,
    SynchronizationWindowEstimate,
)
from app.local.synchronization_review.service import (
    SourceMetrics,
    SpeechSpan,
    TimelineMetrics,
    classify_offset_pattern,
    default_gain_for_active_rms,
    estimated_active_rms_dbfs,
    timeline_metrics,
)


@pytest.fixture(scope="module")
def synchronization_review_assets() -> Iterator[tuple[str, str]]:
    with TestClient(app) as client:
        page_response = client.get("/analyses/synchronization-review")
        script_response = client.get("/pages/synchronization-review/app.js")
    assert page_response.status_code == 200
    assert script_response.status_code == 200
    yield page_response.text, script_response.text


@pytest.mark.parametrize(
    "required_text",
    (
        "Track synchronization review",
        "Slide speaker B relative to speaker A",
        "B earlier",
        "B later",
        "Estimated shift by minute",
        "Speaker A gain",
        "Speaker B gain",
        "Reset automatic gains",
    ),
)
def test_synchronization_review_page_exposes_alignment_controls(
    required_text: str,
    synchronization_review_assets: tuple[str, str],
) -> None:
    page, _ = synchronization_review_assets
    assert required_text in page


@pytest.mark.parametrize(
    "required_script",
    (
        "MAXIMUM_SHIFT_SECONDS = 12",
        "timelineSeconds - state.bShiftSeconds",
        "candidate.estimated_b_shift_seconds",
        "estimate.estimated_b_shift_seconds",
        "trimmed=true",
        "setupAudioGraph",
        "ensureMediaReady",
        "startActivePlayers",
        "candidate.speaker1_gain.default_gain",
        "createDynamicsCompressor",
    ),
)
def test_synchronization_review_script_uses_shared_shifted_timeline(
    required_script: str,
    synchronization_review_assets: tuple[str, str],
) -> None:
    _, script = synchronization_review_assets
    assert required_script in script


def test_timeline_metrics_recovers_speaker_b_delay() -> None:
    speaker1_spans = tuple(
        SpeechSpan(start_seconds=float(start), end_seconds=float(start + 5))
        for start in range(0, 180, 10)
    )
    speaker2_spans = tuple(
        SpeechSpan(start_seconds=float(start + 8), end_seconds=float(start + 13))
        for start in range(0, 170, 10)
    )

    metrics = timeline_metrics(
        speaker1_spans=speaker1_spans,
        speaker2_spans=speaker2_spans,
        duration_seconds=180.0,
    )

    assert metrics.best_lag_seconds == pytest.approx(-3.0)
    assert metrics.overlap_reduction > 0.1
    assert metrics.silence_reduction > 0.1


def test_offset_pattern_is_constant_when_sources_and_minutes_agree() -> None:
    source_metrics = (
        _source_metrics(SynchronizationEvidenceSource.CONVERSATION_ANNOTATION, -6.7),
        _source_metrics(SynchronizationEvidenceSource.PARAKEET, -6.8),
        _source_metrics(SynchronizationEvidenceSource.CANARY, -6.8),
    )
    windows = (
        _window(0.0, -6.7),
        _window(60.0, -6.0),
        _window(120.0, -7.4),
    )

    pattern = classify_offset_pattern(
        meaningful_sources=source_metrics,
        meaningful_windows=windows,
    )

    assert pattern is OffsetPattern.CONSTANT


def test_offset_pattern_is_variable_when_a_middle_minute_needs_no_shift() -> None:
    source_metrics = (
        _source_metrics(SynchronizationEvidenceSource.CONVERSATION_ANNOTATION, -4.5),
        _source_metrics(SynchronizationEvidenceSource.PARAKEET, -4.2),
        _source_metrics(SynchronizationEvidenceSource.CANARY, -4.5),
    )
    windows = (
        _window(0.0, -4.3),
        _window(60.0, -0.5),
        _window(120.0, -4.1),
    )

    pattern = classify_offset_pattern(
        meaningful_sources=source_metrics,
        meaningful_windows=windows,
    )

    assert pattern is OffsetPattern.VARIABLE


def test_gain_normalization_targets_active_speech_level() -> None:
    active_rms_dbfs = estimated_active_rms_dbfs(
        rms_dbfs=-40.0,
        speech_ratio=0.1,
    )

    assert active_rms_dbfs == pytest.approx(-30.0)
    assert default_gain_for_active_rms(active_rms_dbfs) == pytest.approx(10**0.5)


def test_gain_normalization_caps_extremely_quiet_tracks() -> None:
    assert default_gain_for_active_rms(-80.0) == pytest.approx(12.0)


def _source_metrics(
    source: SynchronizationEvidenceSource,
    lag_seconds: float,
) -> SourceMetrics:
    return SourceMetrics(
        source=source,
        metrics=TimelineMetrics(
            overlap_ratio=0.2,
            silence_ratio=0.2,
            overlap_silence_cycle_count=12,
            best_lag_seconds=lag_seconds,
            bad_state_improvement=0.2,
            overlap_reduction=0.1,
            silence_reduction=0.1,
        ),
    )


def _window(
    start_seconds: float,
    lag_seconds: float,
) -> SynchronizationWindowEstimate:
    return SynchronizationWindowEstimate(
        start_seconds=start_seconds,
        end_seconds=start_seconds + 60.0,
        estimated_b_shift_seconds=lag_seconds,
        bad_state_improvement=0.1,
        overlap_reduction=0.05,
        silence_reduction=0.05,
        meaningful=True,
    )

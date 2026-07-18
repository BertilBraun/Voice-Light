from __future__ import annotations

import math
import wave
from array import array
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.local.main import app
from app.local.synchronization_review.calibration import (
    ReviewedAlignment,
    is_unresolved_alignment,
    reviewed_alignment,
)
from app.local.synchronization_review.models import (
    AlignmentEstimateOrigin,
    GainMeasurementBasis,
    OffsetPattern,
    SynchronizationEvidenceSource,
    SynchronizationReviewRequest,
    SynchronizationWindowEstimate,
)
from app.local.synchronization_review.service import (
    SourceMetrics,
    SpeechSpan,
    TimelineMetrics,
    classify_offset_pattern,
    default_gain_for_active_rms,
    estimated_active_rms_dbfs,
    offset_confidence_score,
    recommended_shift_estimate,
    speech_only_gain_normalization,
    timeline_metrics,
)
from app.shared.quality import AnnotationSpan


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
        "Estimated shift by recording window",
        "Speaker A gain",
        "Speaker B gain",
        "Reset automatic gains",
        "Use recommended estimate",
        "Save current offset as reviewed",
        "Scrub full recording",
        "three-minute playback window",
        "Hide reviewed",
        "Lowest confidence first",
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
        "trimmed=${!state.fullRecordingMode}",
        "setupAudioGraph",
        "decodeAudioData",
        "createBufferSource",
        "scheduleTrack",
        "startBufferPlayback",
        "gainNormalization.speaker1_gain.default_gain",
        "createDynamicsCompressor",
        "fetchSpeechGains",
        "/api/synchronization-review/gain/",
        "Speech-only RMS",
        "manually reviewed",
        "alignment_estimate_origin",
        "alignment unresolved",
        "offset confidence",
        "offset_confidence_score",
        "/api/synchronization-review/reviews/",
        "saveCurrentOffsetAsReviewed",
        "toggleFullRecordingMode",
        "/api/synchronization-review/audio-window/",
        "candidateComparator",
        "offset_candidate_count",
        "audio_activity",
    ),
)
def test_synchronization_review_script_uses_shared_shifted_timeline(
    required_script: str,
    synchronization_review_assets: tuple[str, str],
) -> None:
    _, script = synchronization_review_assets
    assert required_script in script


def test_synchronization_review_does_not_seek_streaming_media_elements(
    synchronization_review_assets: tuple[str, str],
) -> None:
    _, script = synchronization_review_assets
    assert "createMediaElementSource" not in script
    assert ".currentTime =" not in script


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
        _source_window(0.0, -4.2, SynchronizationEvidenceSource.PARAKEET),
        _source_window(60.0, -0.4, SynchronizationEvidenceSource.PARAKEET),
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


def test_speech_only_gain_excludes_long_track_silence(tmp_path: Path) -> None:
    sample_rate = 8_000
    wave_path = tmp_path / "speech-with-silence.wav"
    samples = array("h", [0] * (sample_rate * 10))
    for index in range(sample_rate, sample_rate * 2):
        samples[index] = 10_000
    with wave.open(str(wave_path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(samples.tobytes())

    normalization = speech_only_gain_normalization(
        wave_path=wave_path,
        speech_segments=(AnnotationSpan(start_seconds=1.0, end_seconds=2.0, text="spoken term"),),
    )

    expected_rms_dbfs = 20.0 * math.log10(10_000 / 32_767)
    assert normalization.measurement_basis is GainMeasurementBasis.ANNOTATED_SPEECH
    assert normalization.measured_speech_duration_seconds == pytest.approx(1.0)
    assert normalization.measured_rms_dbfs == pytest.approx(expected_rms_dbfs)
    assert normalization.default_gain == pytest.approx(
        default_gain_for_active_rms(expected_rms_dbfs)
    )


@pytest.mark.parametrize(
    ("offset_pattern", "full_recording_shift", "first_window_shift", "expected"),
    (
        (OffsetPattern.CONSTANT, -6.8, -6.7, -6.8),
        (OffsetPattern.VARIABLE, -9.5, -4.2, -4.2),
        (OffsetPattern.UNCERTAIN, -5.64, -4.9, -5.6),
    ),
)
def test_recommended_shift_separates_static_estimate_from_variable_review_anchor(
    offset_pattern: OffsetPattern,
    full_recording_shift: float,
    first_window_shift: float,
    expected: float,
) -> None:
    recommended = recommended_shift_estimate(
        offset_pattern=offset_pattern,
        full_recording_estimated_shift=full_recording_shift,
        window_estimates=(_window(0.0, first_window_shift),),
    )

    assert recommended == pytest.approx(expected)


@pytest.mark.parametrize(
    ("external_id", "expected_shift"),
    (
        ("pmt_284", -6.8),
        ("pmt_205", -10.0),
        ("pmt_236", -4.8),
        ("pmt_008", -4.2),
        ("pmt_132", -3.2),
        ("pmt_161", -3.2),
        ("pmt_226", -2.5),
        ("pmt_180", -3.6),
        ("pmt_315", -2.6),
        ("pmt_095", -2.9),
        ("pmt_192", -4.0),
        ("pmt_256", -3.6),
        ("pmt_217", -4.2),
        ("pmt_237", -2.4),
        ("pmt_087", -1.6),
        ("pmt_225", -2.8),
        ("pmt_219", -1.4),
        ("pmt_318", -0.4),
        ("pmt_144", -2.0),
        ("pmt_306", 0.8),
        ("pmt_246", -1.2),
        ("pmt_214", -1.6),
        ("pmt_310", -1.6),
        ("pmt_245", -1.2),
        ("pmt_222", -2.6),
        ("pmt_324", -2.4),
        ("pmt_298", -1.2),
        ("pmt_007", -2.0),
    ),
)
def test_reviewed_alignment_uses_manual_reference(
    external_id: str,
    expected_shift: float,
) -> None:
    alignment = reviewed_alignment(external_id=external_id)

    assert alignment is not None
    assert alignment.speaker2_shift_seconds == pytest.approx(expected_shift)


def test_unreviewed_alignment_keeps_model_prediction() -> None:
    assert reviewed_alignment(external_id="pmt_999") is None


def test_pmt_326_is_marked_unresolved() -> None:
    assert reviewed_alignment(external_id="pmt_326") is None
    assert is_unresolved_alignment(external_id="pmt_326")


def test_stored_review_overrides_static_alignment_state() -> None:
    stored = ReviewedAlignment(
        external_id="pmt_326",
        speaker2_shift_seconds=-3.4,
    )

    alignment = reviewed_alignment(
        external_id="pmt_326",
        stored_alignments=(stored,),
    )

    assert alignment is stored


@pytest.mark.parametrize("shift_seconds", (-12.1, 12.1))
def test_review_request_rejects_shift_outside_slider_range(
    shift_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        SynchronizationReviewRequest(speaker2_shift_seconds=shift_seconds)


def test_reviewed_offset_confidence_is_certain() -> None:
    confidence = offset_confidence_score(
        estimate_origin=AlignmentEstimateOrigin.REVIEWED,
        predicted_shift=-4.0,
        full_recording_estimated_shift=-3.7,
        source_metrics=(_source_metrics(SynchronizationEvidenceSource.PARAKEET, -3.7),),
        meaningful_windows=(_window(0.0, -10.3),),
    )

    assert confidence == pytest.approx(1.0)


def test_unresolved_offset_confidence_is_zero() -> None:
    confidence = offset_confidence_score(
        estimate_origin=AlignmentEstimateOrigin.UNRESOLVED,
        predicted_shift=10.2,
        full_recording_estimated_shift=-3.0,
        source_metrics=(_source_metrics(SynchronizationEvidenceSource.PARAKEET, -3.0),),
        meaningful_windows=(_window(0.0, 10.2),),
    )

    assert confidence == pytest.approx(0.0)


def test_predicted_offset_confidence_penalizes_disagreement() -> None:
    agreed = offset_confidence_score(
        estimate_origin=AlignmentEstimateOrigin.PREDICTED,
        predicted_shift=-3.0,
        full_recording_estimated_shift=-3.0,
        source_metrics=(
            _source_metrics(SynchronizationEvidenceSource.CONVERSATION_ANNOTATION, -3.0),
            _source_metrics(SynchronizationEvidenceSource.PARAKEET, -3.1),
            _source_metrics(SynchronizationEvidenceSource.CANARY, -3.0),
        ),
        meaningful_windows=(_window(0.0, -3.0), _window(60.0, -3.1)),
    )
    disputed = offset_confidence_score(
        estimate_origin=AlignmentEstimateOrigin.PREDICTED,
        predicted_shift=-9.0,
        full_recording_estimated_shift=-3.0,
        source_metrics=(
            _source_metrics(SynchronizationEvidenceSource.CONVERSATION_ANNOTATION, -2.0),
            _source_metrics(SynchronizationEvidenceSource.PARAKEET, -6.0),
            _source_metrics(SynchronizationEvidenceSource.CANARY, 1.0),
        ),
        meaningful_windows=(_window(0.0, -9.0), _window(60.0, 2.0)),
    )

    assert agreed > disputed


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
    return _source_window(
        start_seconds=start_seconds,
        lag_seconds=lag_seconds,
        source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
    )


def _source_window(
    start_seconds: float,
    lag_seconds: float,
    source: SynchronizationEvidenceSource,
) -> SynchronizationWindowEstimate:
    return SynchronizationWindowEstimate(
        source=source,
        start_seconds=start_seconds,
        end_seconds=start_seconds + 60.0,
        estimated_b_shift_seconds=lag_seconds,
        bad_state_improvement=0.1,
        overlap_reduction=0.05,
        silence_reduction=0.05,
        meaningful=True,
    )

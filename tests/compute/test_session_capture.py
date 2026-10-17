from __future__ import annotations

import os
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.compute.voice.client_metadata import VoiceClientMetadata
from app.compute.voice.conversation import ConversationMessage, ConversationRole
from app.compute.voice.overlap import OverlapMetricsReport
from app.compute.voice.playback import PlaybackMetricsReport
from app.compute.voice.predictive import PredictiveMetricsReport
from app.compute.voice.session_capture import (
    SessionCaptureConfig,
    SessionCaptureRecord,
    SessionCaptureStore,
    SessionClosureReason,
)


def test_capture_configuration_is_optional() -> None:
    assert SessionCaptureConfig.from_environment({}) is None


@pytest.mark.parametrize(
    "environment",
    (
        {"VOICE_LIGHT_SESSION_CAPTURE_DIRECTORY": ""},
        {
            "VOICE_LIGHT_SESSION_CAPTURE_DIRECTORY": "captures",
            "VOICE_LIGHT_SESSION_CAPTURE_RETENTION_DAYS": "0",
        },
        {
            "VOICE_LIGHT_SESSION_CAPTURE_DIRECTORY": "captures",
            "VOICE_LIGHT_SESSION_CAPTURE_RETENTION_DAYS": "invalid",
        },
    ),
)
def test_capture_configuration_rejects_invalid_values(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        SessionCaptureConfig.from_environment(environment)


def test_capture_store_writes_reconstructable_audio_and_json(tmp_path: Path) -> None:
    store = SessionCaptureStore(SessionCaptureConfig(tmp_path, retention_days=30))
    pcm16 = b"\x01\x00\x02\x00\x03\x00"
    record = _capture_record()

    store.save(record, pcm16)

    audio_path = tmp_path / record.audio_file
    with wave.open(str(audio_path), "rb") as audio_file:
        assert audio_file.getnchannels() == 1
        assert audio_file.getsampwidth() == 2
        assert audio_file.getframerate() == 16_000
        assert audio_file.readframes(audio_file.getnframes()) == pcm16
    stored_record = SessionCaptureRecord.model_validate_json(
        (tmp_path / "voice-session-session-1.json").read_text(encoding="utf-8")
    )
    assert stored_record == record


def test_capture_store_prunes_expired_session_files(tmp_path: Path) -> None:
    expired_path = tmp_path / "voice-session-expired.wav"
    expired_path.write_bytes(b"old")
    expired_timestamp = (_capture_record().closed_at - timedelta(days=31)).timestamp()
    os.utime(expired_path, (expired_timestamp, expired_timestamp))
    unrelated_path = tmp_path / "keep.txt"
    unrelated_path.write_text("keep", encoding="utf-8")

    SessionCaptureStore(SessionCaptureConfig(tmp_path, retention_days=30)).save(
        _capture_record(), b"\x00\x00"
    )

    assert not expired_path.exists()
    assert unrelated_path.exists()


def _capture_record() -> SessionCaptureRecord:
    connected_at = datetime(2026, 10, 17, 10, 0, tzinfo=UTC)
    return SessionCaptureRecord(
        session_id="session-1",
        request_id="request-1",
        connected_at=connected_at,
        closed_at=connected_at + timedelta(seconds=2),
        duration_seconds=2.0,
        closure_reason=SessionClosureReason.CLIENT_STOPPED,
        local_time_zone="Europe/Berlin",
        client=VoiceClientMetadata(
            origin="https://voice.bertil-braun.de",
            user_agent="test-browser",
            client_address_hash="abcdef1234567890",
        ),
        input_sample_rate=16_000,
        input_sample_count=3,
        audio_file="voice-session-session-1.wav",
        conversation=(
            ConversationMessage(role=ConversationRole.USER, content="Hello"),
            ConversationMessage(role=ConversationRole.ASSISTANT, content="Hi"),
        ),
        tools=(),
        generations=(),
        predictive_metrics=PredictiveMetricsReport(
            candidate_count=0,
            candidate_hit_rate=0.0,
            invalidation_rate=0.0,
            invalidations=(),
            stale_candidate_escape_rate=0.0,
            commit_to_first_played_audio_p50_ms=None,
            commit_to_first_played_audio_p90_ms=None,
            commit_to_first_played_audio_p95_ms=None,
            commit_to_first_released_pcm_p50_ms=None,
            commit_to_first_released_pcm_p90_ms=None,
            commit_to_first_released_pcm_p95_ms=None,
            true_end_to_first_played_audio_p50_ms=None,
            true_end_to_first_played_audio_p90_ms=None,
            true_end_to_first_played_audio_p95_ms=None,
            hidden_work_seconds=0.0,
            hidden_qwen_tokens=0,
            hidden_tts_samples=0,
            wasted_qwen_tokens=0,
            wasted_tts_samples=0,
            candidate_start_relative_to_first_endpoint_p50_ms=None,
            candidate_start_relative_to_final_endpoint_p50_ms=None,
            candidate_start_to_commit_p50_ms=None,
            candidate_start_to_first_qwen_word_p50_ms=None,
            candidate_start_to_first_tts_pcm_p50_ms=None,
            qwen_word_ready_at_commit_rate=0.0,
            tts_pcm_ready_at_commit_rate=0.0,
            no_candidate_latency_p50_ms=None,
            post_invalidation_latency_p50_ms=None,
        ),
        playback_metrics=PlaybackMetricsReport(
            command_count=0,
            acknowledgement_count=0,
            duplicate_acknowledgement_count=0,
            stale_acknowledgement_count=0,
            command_to_acknowledgement_p95_ms=None,
            rendered_output_sample_estimate_error_p95=None,
            maximum_buffered_source_sample_count=0,
            underrun_count=0,
            discarded_source_sample_count=0,
            replayed_source_sample_count=0,
            skipped_source_sample_count=0,
        ),
        overlap_metrics=OverlapMetricsReport(
            overlap_count=0,
            cooperative_overlap_count=0,
            competitive_overlap_count=0,
            user_onset_to_duck_p95_ms=None,
            user_onset_to_pause_p95_ms=None,
            explicit_stop_p95_ms=None,
            user_onset_to_resume_p95_ms=None,
            cooperative_overlap_p95_ms=None,
            competitive_overlap_p95_ms=None,
            paused_buffer_age_p95_ms=None,
            generated_source_sample_count=0,
        ),
    )

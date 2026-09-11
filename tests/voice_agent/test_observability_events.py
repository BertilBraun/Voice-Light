from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.compute.voice.schemas import (
    AssistantLatencyEvent,
    CausalSource,
    InteractionActionDebugEvent,
    InteractionPolicyDebugEvent,
    OverlapResolutionKind,
    PlaybackCommandAction,
    SpeechUnderstandingDebugEvent,
    TurnAdapterStatus,
    TurnPredictionDisposition,
)


def test_speech_debug_serialization_distinguishes_heartbeat_from_model_observation() -> None:
    heartbeat = SpeechUnderstandingDebugEvent(
        adapter_status=TurnAdapterStatus.ACTIVE,
        silero_speech=False,
        assistant_audible=True,
        user_yield_probability=None,
        turn_completion_probability=None,
        floor_take_probability=None,
        non_floor_feedback_probability=None,
        inference_latency_ms=None,
        observed_audio_time_ms=160,
    )
    observation = heartbeat.model_copy(
        update={
            "user_yield_probability": 0.8,
            "turn_completion_probability": 0.7,
            "floor_take_probability": 0.2,
            "non_floor_feedback_probability": 0.3,
            "inference_latency_ms": 22.0,
            "prediction_disposition": TurnPredictionDisposition.REJECTED_SUPERSEDED,
        }
    )

    assert heartbeat.model_dump(mode="json")["turn_completion_probability"] is None
    assert observation.model_dump(mode="json")["user_yield_probability"] == pytest.approx(0.8)
    assert observation.model_dump(mode="json")["turn_completion_probability"] == pytest.approx(0.7)
    assert observation.model_dump(mode="json")["prediction_disposition"] == "rejected_superseded"


def test_assistant_latency_serializes_endpoint_and_commit_readiness() -> None:
    event = AssistantLatencyEvent(
        generation_id=4,
        turn_commit_causal_source=CausalSource.TURN_ADAPTER,
        first_vad_endpoint_to_turn_commit_ms=1_480.0,
        final_vad_endpoint_to_turn_commit_ms=480.0,
        final_vad_endpoint_to_first_audio_send_ms=990.0,
        asr_finalization_ms=80.0,
        candidate_resolution_ms=90.0,
        speculation_start_relative_to_first_vad_endpoint_ms=4.0,
        speculation_start_relative_to_final_vad_endpoint_ms=-120.0,
        speculation_start_to_turn_commit_ms=420.0,
        speculation_start_to_first_qwen_word_ms=64.0,
        speculation_start_to_first_tts_pcm_ms=494.0,
        first_qwen_word_ready_at_commit=True,
        turn_commit_to_playback_ms=820.0,
        turn_commit_to_first_audio_send_ms=510.0,
        generation_to_first_word_ms=64.0,
        tts_first_word_to_first_pcm_ms=430.0,
        first_audio_send_to_playback_ms=310.0,
        speculative_candidate_promoted=True,
        speculative_hidden_work_ms=420.0,
        prepared_qwen_token_count=18,
        prepared_word_count=3,
        first_tts_pcm_ready_at_commit=False,
        buffered_audio_ms=0.0,
    )

    payload = event.model_dump(mode="json")

    assert payload["first_vad_endpoint_to_turn_commit_ms"] == pytest.approx(1_480.0)
    assert payload["turn_commit_causal_source"] == "turn_adapter"
    assert payload["final_vad_endpoint_to_turn_commit_ms"] == pytest.approx(480.0)
    assert payload["final_vad_endpoint_to_first_audio_send_ms"] == pytest.approx(990.0)
    assert payload["asr_finalization_ms"] == pytest.approx(80.0)
    assert payload["candidate_resolution_ms"] == pytest.approx(90.0)
    assert payload["speculation_start_relative_to_first_vad_endpoint_ms"] == pytest.approx(4.0)
    assert payload["speculation_start_relative_to_final_vad_endpoint_ms"] == pytest.approx(-120.0)
    assert payload["speculation_start_to_turn_commit_ms"] == pytest.approx(420.0)
    assert payload["speculation_start_to_first_qwen_word_ms"] == pytest.approx(64.0)
    assert payload["speculation_start_to_first_tts_pcm_ms"] == pytest.approx(494.0)
    assert payload["first_qwen_word_ready_at_commit"] is True
    assert payload["turn_commit_to_first_audio_send_ms"] == pytest.approx(510.0)
    assert payload["speculative_hidden_work_ms"] == pytest.approx(420.0)
    assert payload["prepared_qwen_token_count"] == 18
    assert payload["prepared_word_count"] == 3
    assert payload["first_tts_pcm_ready_at_commit"] is False


def test_assistant_latency_allows_missing_vad_endpoints_and_candidate() -> None:
    event = AssistantLatencyEvent(
        generation_id=1,
        turn_commit_causal_source=CausalSource.SILERO_VAD,
        first_vad_endpoint_to_turn_commit_ms=None,
        final_vad_endpoint_to_turn_commit_ms=None,
        final_vad_endpoint_to_first_audio_send_ms=None,
        asr_finalization_ms=40.0,
        candidate_resolution_ms=None,
        speculation_start_relative_to_first_vad_endpoint_ms=None,
        speculation_start_relative_to_final_vad_endpoint_ms=None,
        speculation_start_to_turn_commit_ms=None,
        speculation_start_to_first_qwen_word_ms=None,
        speculation_start_to_first_tts_pcm_ms=None,
        first_qwen_word_ready_at_commit=False,
        turn_commit_to_playback_ms=800.0,
        turn_commit_to_first_audio_send_ms=500.0,
        generation_to_first_word_ms=60.0,
        tts_first_word_to_first_pcm_ms=420.0,
        first_audio_send_to_playback_ms=300.0,
        speculative_candidate_promoted=False,
        speculative_hidden_work_ms=0.0,
        prepared_qwen_token_count=0,
        prepared_word_count=0,
        first_tts_pcm_ready_at_commit=False,
        buffered_audio_ms=0.0,
    )

    assert event.first_vad_endpoint_to_turn_commit_ms is None
    assert event.final_vad_endpoint_to_turn_commit_ms is None
    assert event.final_vad_endpoint_to_first_audio_send_ms is None
    assert event.candidate_resolution_ms is None


def test_latency_events_reject_negative_durations() -> None:
    with pytest.raises(ValidationError):
        InteractionActionDebugEvent(
            action=PlaybackCommandAction.RESUME,
            onset_to_acknowledgement_ms=-0.1,
        )


def test_interaction_policy_serializes_prediction_availability() -> None:
    event = InteractionPolicyDebugEvent(
        decision=OverlapResolutionKind.NON_FLOOR_TAKING,
        reason="predicted_non_floor_feedback",
        decision_latency_ms=190.0,
        first_applicable_prediction_latency_ms=170.0,
        causal_source=CausalSource.TURN_ADAPTER,
    )

    assert event.model_dump(mode="json")["first_applicable_prediction_latency_ms"] == pytest.approx(
        170.0
    )

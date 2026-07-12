from __future__ import annotations

from pathlib import Path

from app.analyses.asr.models import AsrModelMode
from app.analyses.asr.service import (
    filtered_remote_dependencies,
    remote_model_ids_for_selected_modes,
    transcription_result_from_cached_asr,
)
from app.asr.schemas import AsrModelId, AsrRuntimeStats, AsrTranscriptResult, TimestampedWord
from app.asr.transcript import SpeakerTrack, Word


def test_transcription_result_from_cached_asr_preserves_words_and_runtime() -> None:
    transcription = transcription_result_from_cached_asr(
        audio_path=Path("sample.wav"),
        speaker_track=SpeakerTrack.SPEAKER1,
        audio_duration_seconds=2.0,
        transcript=AsrTranscriptResult(
            model_id=AsrModelId.WHISPERX,
            text="hello world",
            words=(
                TimestampedWord(text="hello", start_seconds=0.0, end_seconds=0.5),
                TimestampedWord(text="world", start_seconds=0.6, end_seconds=1.0),
            ),
            processing_time_seconds=1.5,
            runtime=AsrRuntimeStats(
                processing_time_seconds=1.5,
                model_loading_time_seconds=3.0,
                inference_time_seconds=1.2,
                real_time_factor=0.75,
                peak_gpu_memory_mb=1000.0,
                package_versions={"torch": "test"},
            ),
        ),
    )

    assert transcription.model_name == "whisperx_large_v3"
    assert transcription.words == (
        Word(text="hello", start_seconds=0.0, end_seconds=0.5),
        Word(text="world", start_seconds=0.6, end_seconds=1.0),
    )
    assert transcription.package_versions == {"torch": "test"}
    assert transcription.model_loading_time_seconds == 3.0
    assert transcription.inference_time_seconds == 1.2
    assert transcription.real_time_factor == 0.75
    assert transcription.peak_gpu_memory_mb == 1000.0


def test_remote_model_ids_expand_merged_consensus_dependencies() -> None:
    assert remote_model_ids_for_selected_modes((AsrModelMode.MERGED_CONSENSUS,)) == (
        AsrModelId.PARAKEET_TDT,
        AsrModelId.WHISPERX,
        AsrModelId.CANARY,
        AsrModelId.NEMOTRON_3_5,
    )


def test_remote_model_ids_expand_parakeet_canary_consensus_dependencies() -> None:
    assert remote_model_ids_for_selected_modes((AsrModelMode.PARAKEET_CANARY_CONSENSUS,)) == (
        AsrModelId.PARAKEET_TDT,
        AsrModelId.CANARY,
    )


def test_remote_model_ids_expand_filtered_remote_dependency() -> None:
    assert remote_model_ids_for_selected_modes((AsrModelMode.PARAKEET_TDT_CROSSTALK_FILTERED,)) == (
        AsrModelId.PARAKEET_TDT,
    )


def test_remote_model_ids_expand_filtered_union_dependencies() -> None:
    assert remote_model_ids_for_selected_modes(
        (AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED,)
    ) == (
        AsrModelId.PARAKEET_TDT,
        AsrModelId.CANARY,
    )


def test_filtered_remote_dependencies_expand_before_derived_merge() -> None:
    assert filtered_remote_dependencies(
        (
            AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED,
            AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED,
        )
    ) == (
        AsrModelMode.PARAKEET_TDT,
        AsrModelMode.WHISPERX,
        AsrModelMode.CANARY,
        AsrModelMode.NEMOTRON_3_5,
    )


def test_remote_model_ids_do_not_duplicate_explicit_dependencies() -> None:
    assert remote_model_ids_for_selected_modes(
        (
            AsrModelMode.PARAKEET_TDT,
            AsrModelMode.MERGED_CONSENSUS,
            AsrModelMode.WHISPERX,
        )
    ) == (
        AsrModelId.PARAKEET_TDT,
        AsrModelId.WHISPERX,
        AsrModelId.CANARY,
        AsrModelId.NEMOTRON_3_5,
    )

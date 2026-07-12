from __future__ import annotations

import tempfile
import time
from pathlib import Path

from app.analyses.asr.catalog import (
    available_asr_models,
    is_filtered_model,
    is_remote_model,
    remote_dependencies_for_model_mode,
    requires_merged_consensus,
    requires_parakeet_canary_union,
    source_mode_for_filtered_model,
)
from app.analyses.asr.crosstalk import (
    CrosstalkFilterConfig,
    CrosstalkFramePower,
    filter_crosstalk_words,
    load_crosstalk_frame_power,
)
from app.analyses.asr.merger import (
    TranscriptToMerge,
    merged_consensus_transcription,
    parakeet_canary_consensus_transcription,
)
from app.analyses.asr.models import (
    AsrAnalysisResponse,
    AsrModelInfo,
    AsrModelMode,
    AsrModelRun,
)
from app.asr.models.parsing import (
    CANARY_IDENTIFIER,
    NEMOTRON_3_5_IDENTIFIER,
    PARAKEET_IDENTIFIER,
    WHISPERX_IDENTIFIER,
)
from app.asr.schemas import AsrModelId, AsrTranscriptResult, TimestampedWord
from app.asr.service import AsrTranscriptCache, RemoteAsrClientFactory, cached_asr_transcripts
from app.asr_quality.metrics import evaluate_file
from app.asr_quality.schemas import (
    FileMetrics,
    ReferenceTranscript,
    SpeakerTrack,
    TranscriptionResult,
    Word,
)
from app.audio.wav import capped_wave_bytes
from app.data.sessions import SpeakerName, session_audio_path
from app.quality.audio import load_audio

CROSSTALK_FILTER_CONFIG = CrosstalkFilterConfig()


def analyze_asr(
    session_id: str,
    speaker_track: SpeakerTrack,
    selected_models: tuple[AsrModelMode, ...],
    reference_words: tuple[Word, ...],
    cache: AsrTranscriptCache,
    remote_client_factory: RemoteAsrClientFactory,
) -> AsrAnalysisResponse:
    source_path = session_audio_path(
        identifier=session_id,
        speaker_name=speaker_name_for_track(speaker_track),
    )
    other_source_path = session_audio_path(
        identifier=session_id,
        speaker_name=other_speaker_name_for_track(speaker_track),
    )
    capped_audio_path = write_capped_analysis_audio(source_path)
    try:
        analyzed_duration_seconds = load_audio(capped_audio_path).metadata.duration_seconds
        model_infos = available_asr_models()
        requested_model_ids = remote_model_ids_for_selected_modes(selected_models)
        cached_response = cached_asr_transcripts(
            audio_path=capped_audio_path,
            requested_models=requested_model_ids,
            cache=cache,
            remote_client_factory=remote_client_factory,
        )
        dependency_runs = tuple(
            asr_model_run(
                model_info=model_info_for_mode(
                    model_infos=model_infos,
                    model_mode=AsrModelMode(model_id.value),
                ),
                transcript=transcript_for_model(
                    model_id=model_id,
                    results=cached_response.results,
                ),
                audio_path=capped_audio_path,
                speaker_track=speaker_track,
                audio_duration_seconds=analyzed_duration_seconds,
                reference_words=reference_words,
            )
            for model_id in requested_model_ids
        )
        derived_runs: tuple[AsrModelRun, ...] = ()
        if requires_merged_consensus(selected_models):
            derived_runs += (
                merged_asr_model_run(
                    model_info=model_info_for_mode(
                        model_infos=model_infos,
                        model_mode=AsrModelMode.MERGED_CONSENSUS,
                    ),
                    source_runs=dependency_runs,
                    audio_path=capped_audio_path,
                    speaker_track=speaker_track,
                    audio_duration_seconds=analyzed_duration_seconds,
                    reference_words=reference_words,
                ),
            )
        if requires_parakeet_canary_union(selected_models):
            derived_runs += (
                parakeet_canary_consensus_model_run(
                    model_info=model_info_for_mode(
                        model_infos=model_infos,
                        model_mode=AsrModelMode.PARAKEET_CANARY_CONSENSUS,
                    ),
                    parakeet=source_run_for_mode(
                        source_runs=dependency_runs,
                        model_mode=AsrModelMode.PARAKEET_TDT,
                    ),
                    canary=source_run_for_mode(
                        source_runs=dependency_runs,
                        model_mode=AsrModelMode.CANARY,
                    ),
                    audio_path=capped_audio_path,
                    speaker_track=speaker_track,
                    audio_duration_seconds=analyzed_duration_seconds,
                    reference_words=reference_words,
                ),
            )
        source_runs = (*dependency_runs, *derived_runs)
        frame_power_pair = (
            load_crosstalk_frame_power(
                target_audio_path=source_path,
                other_audio_path=other_source_path,
                config=CROSSTALK_FILTER_CONFIG,
            )
            if any(is_filtered_model(model_mode) for model_mode in selected_models)
            else None
        )
        runs = selected_model_runs(
            selected_models=selected_models,
            source_runs=source_runs,
            model_infos=model_infos,
            frame_power_pair=frame_power_pair,
            audio_path=capped_audio_path,
            speaker_track=speaker_track,
            audio_duration_seconds=analyzed_duration_seconds,
            reference_words=reference_words,
        )
        return AsrAnalysisResponse(
            session_id=session_id,
            speaker_track=speaker_track,
            audio_url=f"/api/audio/{session_id}/{speaker_name_for_track(speaker_track).value}",
            analyzed_duration_seconds=analyzed_duration_seconds,
            reference_word_count=len(reference_words),
            runs=runs,
        )
    finally:
        capped_audio_path.unlink(missing_ok=True)


def remote_model_ids_for_selected_modes(
    selected_models: tuple[AsrModelMode, ...],
) -> tuple[AsrModelId, ...]:
    remote_model_modes = tuple(
        model_mode for model_mode in selected_models if is_remote_model(model_mode)
    )
    for model_mode in selected_models:
        remote_model_modes = unique_model_modes(
            (*remote_model_modes, *remote_dependencies_for_model_mode(model_mode))
        )
    if not remote_model_modes:
        raise ValueError("Select at least one remote ASR model or ASR consensus.")
    return tuple(AsrModelId(model_mode.value) for model_mode in remote_model_modes)


def unique_model_modes(model_modes: tuple[AsrModelMode, ...]) -> tuple[AsrModelMode, ...]:
    unique_modes: list[AsrModelMode] = []
    for model_mode in model_modes:
        if model_mode not in unique_modes:
            unique_modes.append(model_mode)
    return tuple(unique_modes)


def speaker_name_for_track(speaker_track: SpeakerTrack) -> SpeakerName:
    match speaker_track:
        case SpeakerTrack.SPEAKER1:
            return SpeakerName.SPEAKER1
        case SpeakerTrack.SPEAKER2:
            return SpeakerName.SPEAKER2


def other_speaker_name_for_track(speaker_track: SpeakerTrack) -> SpeakerName:
    match speaker_track:
        case SpeakerTrack.SPEAKER1:
            return SpeakerName.SPEAKER2
        case SpeakerTrack.SPEAKER2:
            return SpeakerName.SPEAKER1


def transcript_for_model(
    model_id: AsrModelId,
    results: tuple[AsrTranscriptResult, ...],
) -> AsrTranscriptResult:
    for result in results:
        if result.model_id == model_id:
            return result
    raise ValueError(f"Missing ASR transcript for model: {model_id.value}")


def asr_model_run(
    model_info: AsrModelInfo,
    transcript: AsrTranscriptResult,
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    reference_words: tuple[Word, ...],
) -> AsrModelRun:
    transcription = transcription_result_from_cached_asr(
        audio_path=audio_path,
        speaker_track=speaker_track,
        audio_duration_seconds=audio_duration_seconds,
        transcript=transcript,
    )
    return AsrModelRun(
        model=model_info,
        transcription=transcription,
        metrics=metrics_for_model(
            audio_path=audio_path,
            reference_words=reference_words,
            transcription=transcription,
        ),
    )


def merged_asr_model_run(
    model_info: AsrModelInfo,
    source_runs: tuple[AsrModelRun, ...],
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    reference_words: tuple[Word, ...],
) -> AsrModelRun:
    if len(source_runs) < 2:
        raise ValueError("Merged ASR consensus requires Parakeet and WhisperX transcripts.")
    transcription = merged_consensus_transcription(
        audio_path=str(audio_path),
        speaker_track=speaker_track,
        audio_duration_seconds=audio_duration_seconds,
        transcripts=tuple(
            TranscriptToMerge(
                model_name=source_run.transcription.model_name,
                words=source_run.transcription.words,
            )
            for source_run in source_runs
        ),
    )
    return AsrModelRun(
        model=model_info,
        transcription=transcription,
        metrics=metrics_for_model(
            audio_path=audio_path,
            reference_words=reference_words,
            transcription=transcription,
        ),
    )


def parakeet_canary_consensus_model_run(
    model_info: AsrModelInfo,
    parakeet: AsrModelRun,
    canary: AsrModelRun,
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    reference_words: tuple[Word, ...],
) -> AsrModelRun:
    transcription = parakeet_canary_consensus_transcription(
        audio_path=str(audio_path),
        speaker_track=speaker_track,
        audio_duration_seconds=audio_duration_seconds,
        parakeet=TranscriptToMerge(
            model_name=parakeet.transcription.model_name,
            words=parakeet.transcription.words,
        ),
        canary=TranscriptToMerge(
            model_name=canary.transcription.model_name,
            words=canary.transcription.words,
        ),
    )
    return AsrModelRun(
        model=model_info,
        transcription=transcription,
        metrics=metrics_for_model(
            audio_path=audio_path,
            reference_words=reference_words,
            transcription=transcription,
        ),
    )


def source_run_for_mode(
    source_runs: tuple[AsrModelRun, ...], model_mode: AsrModelMode
) -> AsrModelRun:
    for source_run in source_runs:
        if source_run.transcription.model_name == model_mode.value:
            return source_run
    raise ValueError(f"Missing ASR transcript for model: {model_mode.value}")


def model_info_for_mode(
    model_infos: tuple[AsrModelInfo, ...], model_mode: AsrModelMode
) -> AsrModelInfo:
    for model_info in model_infos:
        if model_info.mode == model_mode:
            return model_info
    raise ValueError(f"Missing ASR model information: {model_mode.value}")


def selected_model_runs(
    selected_models: tuple[AsrModelMode, ...],
    source_runs: tuple[AsrModelRun, ...],
    model_infos: tuple[AsrModelInfo, ...],
    frame_power_pair: CrosstalkFramePower | None,
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    reference_words: tuple[Word, ...],
) -> tuple[AsrModelRun, ...]:
    runs: list[AsrModelRun] = []
    for model_mode in selected_models:
        if not is_filtered_model(model_mode):
            runs.append(source_run_for_mode(source_runs=source_runs, model_mode=model_mode))
            continue
        assert frame_power_pair is not None
        runs.append(
            crosstalk_filtered_model_run(
                model_info=model_info_for_mode(
                    model_infos=model_infos,
                    model_mode=model_mode,
                ),
                source_run=source_run_for_mode(
                    source_runs=source_runs,
                    model_mode=source_mode_for_filtered_model(model_mode),
                ),
                frame_power_pair=frame_power_pair,
                audio_path=audio_path,
                speaker_track=speaker_track,
                audio_duration_seconds=audio_duration_seconds,
                reference_words=reference_words,
            )
        )
    return tuple(runs)


def crosstalk_filtered_model_run(
    model_info: AsrModelInfo,
    source_run: AsrModelRun,
    frame_power_pair: CrosstalkFramePower,
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    reference_words: tuple[Word, ...],
) -> AsrModelRun:
    filter_start = time.perf_counter()
    source_transcription = source_run.transcription
    words = filter_crosstalk_words(
        words=source_transcription.words,
        frame_power_pair=frame_power_pair,
        config=CROSSTALK_FILTER_CONFIG,
    )
    processing_time_seconds = time.perf_counter() - filter_start
    transcription = TranscriptionResult(
        model_name=model_info.mode.value,
        audio_path=str(audio_path),
        track=speaker_track,
        audio_duration_seconds=audio_duration_seconds,
        processing_time_seconds=processing_time_seconds,
        words=words,
        raw_output={
            "source_model": source_transcription.model_name,
            "source_word_count": len(source_transcription.words),
            "word_count": len(words),
            "removed_word_count": len(source_transcription.words) - len(words),
            "rolling_radius_seconds": CROSSTALK_FILTER_CONFIG.rolling_radius_seconds,
            "quiet_below_baseline_db": CROSSTALK_FILTER_CONFIG.quiet_below_baseline_db,
            "other_channel_dominance_db": (CROSSTALK_FILTER_CONFIG.other_channel_dominance_db),
        },
        model_identifier=(f"{source_transcription.model_identifier}/crosstalk-filter-v1"),
        package_versions={},
        model_loading_time_seconds=0.0,
        inference_time_seconds=processing_time_seconds,
        real_time_factor=processing_time_seconds / audio_duration_seconds
        if audio_duration_seconds > 0.0
        else None,
        peak_gpu_memory_mb=None,
        error=source_transcription.error,
    )
    return AsrModelRun(
        model=model_info,
        transcription=transcription,
        metrics=metrics_for_model(
            audio_path=audio_path,
            reference_words=reference_words,
            transcription=transcription,
        ),
    )


def metrics_for_model(
    audio_path: Path,
    reference_words: tuple[Word, ...],
    transcription: TranscriptionResult,
) -> FileMetrics | None:
    if not reference_words:
        return None
    return evaluate_file(
        ReferenceTranscript(audio_path=str(audio_path), words=reference_words),
        transcription,
    )


def transcription_result_from_cached_asr(
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    transcript: AsrTranscriptResult,
) -> TranscriptionResult:
    processing_time_seconds = (
        transcript.processing_time_seconds
        if transcript.processing_time_seconds is not None
        else 0.0
    )
    runtime = transcript.runtime
    return TranscriptionResult(
        model_name=transcript.model_id.value,
        audio_path=str(audio_path),
        track=speaker_track,
        audio_duration_seconds=audio_duration_seconds,
        processing_time_seconds=processing_time_seconds,
        words=tuple(word_from_timestamped_word(word) for word in transcript.words),
        raw_output={
            "text": transcript.text,
            "word_count": len(transcript.words),
            "cache_source": "database",
        },
        model_identifier=model_identifier_for_model_id(transcript.model_id),
        package_versions=runtime.package_versions if runtime is not None else {},
        model_loading_time_seconds=runtime.model_loading_time_seconds
        if runtime is not None
        else None,
        inference_time_seconds=runtime.inference_time_seconds if runtime is not None else None,
        real_time_factor=runtime.real_time_factor if runtime is not None else None,
        peak_gpu_memory_mb=runtime.peak_gpu_memory_mb if runtime is not None else None,
        error=transcript.error,
    )


def model_identifier_for_model_id(model_id: AsrModelId) -> str:
    match model_id:
        case AsrModelId.PARAKEET_TDT:
            return PARAKEET_IDENTIFIER
        case AsrModelId.WHISPERX:
            return WHISPERX_IDENTIFIER
        case AsrModelId.CANARY:
            return CANARY_IDENTIFIER
        case AsrModelId.NEMOTRON_3_5:
            return NEMOTRON_3_5_IDENTIFIER


def word_from_timestamped_word(word: TimestampedWord) -> Word:
    return Word(
        text=word.text,
        start_seconds=word.start_seconds,
        end_seconds=word.end_seconds,
        confidence=word.confidence,
    )


def write_capped_analysis_audio(source_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"{source_path.stem}_asr_analysis_",
        suffix=".wav",
        delete=False,
    ) as output_file:
        output_file.write(capped_wave_bytes(source_path))
        return Path(output_file.name)

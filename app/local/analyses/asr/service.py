from __future__ import annotations

import tempfile
import time
from pathlib import Path

from app.local.analyses.asr.catalog import (
    REMOTE_ASR_MODEL_MODES,
    available_asr_models,
    filtered_mode_for_remote_model,
    is_filtered_model,
    is_remote_model,
    remote_dependencies_for_model_mode,
)
from app.local.analyses.asr.crosstalk import (
    CrosstalkFilterConfig,
    CrosstalkFramePower,
    filter_crosstalk_words,
    load_crosstalk_frame_power,
)
from app.local.analyses.asr.merger import (
    TranscriptToMerge,
    filtered_merged_consensus_transcription,
    filtered_parakeet_canary_union_transcription,
    merged_consensus_transcription,
    parakeet_canary_consensus_transcription,
)
from app.local.analyses.asr.metric_models import FileMetrics
from app.local.analyses.asr.metrics import evaluate_file
from app.local.analyses.asr.models import (
    AsrAnalysisResponse,
    AsrModelInfo,
    AsrModelMode,
    AsrModelRun,
)
from app.local.asr.service import AsrTranscriptCache, RemoteAsrClientFactory, cached_asr_transcripts
from app.local.asr.transcript import ReferenceTranscript, SpeakerTrack, TranscriptionResult, Word
from app.local.data.sessions import SpeakerName, session_audio_path
from app.shared.asr import (
    CANARY_IDENTIFIER,
    NEMOTRON_3_5_IDENTIFIER,
    PARAKEET_IDENTIFIER,
    WHISPER_IDENTIFIER,
    AsrModelId,
    AsrTranscriptResult,
    TimestampedWord,
)
from app.shared.audio import load_audio
from app.shared.audio.wav import capped_wave_bytes
from app.shared.storage.local import LocalStorageBackend

CROSSTALK_FILTER_CONFIG = CrosstalkFilterConfig()


def analyze_asr(
    session_id: str,
    speaker_track: SpeakerTrack,
    selected_models: tuple[AsrModelMode, ...],
    reference_words: tuple[Word, ...],
    cache: AsrTranscriptCache,
    remote_client_factory: RemoteAsrClientFactory,
    source_audio_path: Path | None = None,
    other_audio_path: Path | None = None,
    prepared_audio_path: Path | None = None,
) -> AsrAnalysisResponse:
    resolved_source_path = source_audio_path or session_audio_path(
        identifier=session_id, speaker_name=speaker_name_for_track(speaker_track)
    )
    resolved_other_source_path = other_audio_path or session_audio_path(
        identifier=session_id, speaker_name=other_speaker_name_for_track(speaker_track)
    )
    analysis_audio_path = prepared_audio_path or write_capped_analysis_audio(resolved_source_path)
    owns_analysis_audio = prepared_audio_path is None
    try:
        analyzed_duration_seconds = load_audio(
            LocalStorageBackend(), analysis_audio_path.as_posix()
        ).metadata.duration_seconds
        model_infos = available_asr_models()
        requested_model_ids = remote_model_ids_for_selected_modes(selected_models)
        cached_response = cached_asr_transcripts(
            audio_path=analysis_audio_path,
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
                audio_path=analysis_audio_path,
                speaker_track=speaker_track,
                audio_duration_seconds=analyzed_duration_seconds,
                reference_words=reference_words,
            )
            for model_id in requested_model_ids
        )
        frame_power_pair = (
            load_crosstalk_frame_power(
                target_audio_path=resolved_source_path,
                other_audio_path=resolved_other_source_path,
                config=CROSSTALK_FILTER_CONFIG,
            )
            if any(is_filtered_model(model_mode) for model_mode in selected_models)
            else None
        )
        filtered_dependency_runs = filtered_remote_model_runs(
            selected_models=selected_models,
            dependency_runs=dependency_runs,
            model_infos=model_infos,
            frame_power_pair=frame_power_pair,
            audio_path=analysis_audio_path,
            speaker_track=speaker_track,
            audio_duration_seconds=analyzed_duration_seconds,
            reference_words=reference_words,
        )
        derived_runs: tuple[AsrModelRun, ...] = ()
        if AsrModelMode.MERGED_CONSENSUS in selected_models:
            derived_runs += (
                merged_asr_model_run(
                    model_info=model_info_for_mode(
                        model_infos=model_infos,
                        model_mode=AsrModelMode.MERGED_CONSENSUS,
                    ),
                    source_runs=dependency_runs,
                    audio_path=analysis_audio_path,
                    speaker_track=speaker_track,
                    audio_duration_seconds=analyzed_duration_seconds,
                    reference_words=reference_words,
                ),
            )
        if AsrModelMode.PARAKEET_CANARY_CONSENSUS in selected_models:
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
                    audio_path=analysis_audio_path,
                    speaker_track=speaker_track,
                    audio_duration_seconds=analyzed_duration_seconds,
                    reference_words=reference_words,
                ),
            )
        if AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED in selected_models:
            derived_runs += (
                merged_asr_model_run(
                    model_info=model_info_for_mode(
                        model_infos=model_infos,
                        model_mode=AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED,
                    ),
                    source_runs=filtered_dependency_runs,
                    audio_path=analysis_audio_path,
                    speaker_track=speaker_track,
                    audio_duration_seconds=analyzed_duration_seconds,
                    reference_words=reference_words,
                ),
            )
        if AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED in selected_models:
            derived_runs += (
                parakeet_canary_consensus_model_run(
                    model_info=model_info_for_mode(
                        model_infos=model_infos,
                        model_mode=AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED,
                    ),
                    parakeet=source_run_for_mode(
                        source_runs=filtered_dependency_runs,
                        model_mode=AsrModelMode.PARAKEET_TDT_CROSSTALK_FILTERED,
                    ),
                    canary=source_run_for_mode(
                        source_runs=filtered_dependency_runs,
                        model_mode=AsrModelMode.CANARY_CROSSTALK_FILTERED,
                    ),
                    audio_path=analysis_audio_path,
                    speaker_track=speaker_track,
                    audio_duration_seconds=analyzed_duration_seconds,
                    reference_words=reference_words,
                ),
            )
        source_runs = (*dependency_runs, *filtered_dependency_runs, *derived_runs)
        runs = selected_model_runs(
            selected_models=selected_models,
            source_runs=source_runs,
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
        if owns_analysis_audio:
            analysis_audio_path.unlink(missing_ok=True)


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
    return tuple(
        AsrModelId(model_mode.value)
        for model_mode in REMOTE_ASR_MODEL_MODES
        if model_mode in remote_model_modes
    )


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
        raise ValueError("Merged ASR consensus requires at least two source transcripts.")
    transcripts = tuple(
        TranscriptToMerge(
            model_name=source_run.transcription.model_name,
            words=source_run.transcription.words,
        )
        for source_run in source_runs
    )
    match model_info.mode:
        case AsrModelMode.MERGED_CONSENSUS:
            transcription = merged_consensus_transcription(
                audio_path=str(audio_path),
                speaker_track=speaker_track,
                audio_duration_seconds=audio_duration_seconds,
                transcripts=transcripts,
            )
        case AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED:
            transcription = filtered_merged_consensus_transcription(
                audio_path=str(audio_path),
                speaker_track=speaker_track,
                audio_duration_seconds=audio_duration_seconds,
                transcripts=transcripts,
            )
        case _:
            raise ValueError(f"Unsupported merged ASR model mode: {model_info.mode.value}")
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
    parakeet_transcript = TranscriptToMerge(
        model_name=parakeet.transcription.model_name,
        words=parakeet.transcription.words,
    )
    canary_transcript = TranscriptToMerge(
        model_name=canary.transcription.model_name,
        words=canary.transcription.words,
    )
    match model_info.mode:
        case AsrModelMode.PARAKEET_CANARY_CONSENSUS:
            transcription = parakeet_canary_consensus_transcription(
                audio_path=str(audio_path),
                speaker_track=speaker_track,
                audio_duration_seconds=audio_duration_seconds,
                parakeet=parakeet_transcript,
                canary=canary_transcript,
            )
        case AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED:
            transcription = filtered_parakeet_canary_union_transcription(
                audio_path=str(audio_path),
                speaker_track=speaker_track,
                audio_duration_seconds=audio_duration_seconds,
                parakeet=parakeet_transcript,
                canary=canary_transcript,
            )
        case _:
            raise ValueError(f"Unsupported Parakeet-Canary model mode: {model_info.mode.value}")
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
) -> tuple[AsrModelRun, ...]:
    return tuple(
        source_run_for_mode(source_runs=source_runs, model_mode=model_mode)
        for model_mode in selected_models
    )


def filtered_remote_model_runs(
    selected_models: tuple[AsrModelMode, ...],
    dependency_runs: tuple[AsrModelRun, ...],
    model_infos: tuple[AsrModelInfo, ...],
    frame_power_pair: CrosstalkFramePower | None,
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    reference_words: tuple[Word, ...],
) -> tuple[AsrModelRun, ...]:
    filtered_remote_modes = filtered_remote_dependencies(selected_models)
    if not filtered_remote_modes:
        return ()
    assert frame_power_pair is not None
    return tuple(
        crosstalk_filtered_model_run(
            model_info=model_info_for_mode(
                model_infos=model_infos,
                model_mode=filtered_mode_for_remote_model(model_mode),
            ),
            source_run=source_run_for_mode(
                source_runs=dependency_runs,
                model_mode=model_mode,
            ),
            frame_power_pair=frame_power_pair,
            audio_path=audio_path,
            speaker_track=speaker_track,
            audio_duration_seconds=audio_duration_seconds,
            reference_words=reference_words,
        )
        for model_mode in filtered_remote_modes
    )


def filtered_remote_dependencies(
    selected_models: tuple[AsrModelMode, ...],
) -> tuple[AsrModelMode, ...]:
    model_modes: tuple[AsrModelMode, ...] = ()
    for selected_model in selected_models:
        if is_filtered_model(selected_model):
            model_modes = unique_model_modes(
                (*model_modes, *remote_dependencies_for_model_mode(selected_model))
            )
    return tuple(model_mode for model_mode in REMOTE_ASR_MODEL_MODES if model_mode in model_modes)


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
            return WHISPER_IDENTIFIER
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

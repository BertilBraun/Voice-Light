from __future__ import annotations

import tempfile
from pathlib import Path

from app.analyses.asr.merger import (
    TranscriptToMerge,
    merged_consensus_transcription,
)
from app.analyses.asr.models import (
    AsrAnalysisResponse,
    AsrModelInfo,
    AsrModelMode,
    AsrModelRun,
)
from app.asr.models.parsing import PARAKEET_IDENTIFIER, WHISPERX_IDENTIFIER
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

REMOTE_ASR_MODEL_MODES = (AsrModelMode.PARAKEET_TDT, AsrModelMode.WHISPERX)


def available_asr_models() -> tuple[AsrModelInfo, ...]:
    return (
        AsrModelInfo(
            mode=AsrModelMode.PARAKEET_TDT,
            label="Parakeet TDT 0.6B v3",
            description="NVIDIA Parakeet word-timestamp ASR via Transformers.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.WHISPERX,
            label="WhisperX large-v3",
            description="Faster-Whisper transcription plus WhisperX word alignment.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.MERGED_CONSENSUS,
            label="Merged ASR consensus",
            description="Local alignment merge of Parakeet and WhisperX timestamped words.",
        ),
    )


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
    capped_audio_path = write_capped_analysis_audio(source_path)
    try:
        analyzed_duration_seconds = load_audio(capped_audio_path).metadata.duration_seconds
        model_info_by_id = {model.mode: model for model in available_asr_models()}
        requested_model_ids = remote_model_ids_for_selected_modes(selected_models)
        cached_response = cached_asr_transcripts(
            audio_path=capped_audio_path,
            requested_models=requested_model_ids,
            cache=cache,
            remote_client_factory=remote_client_factory,
        )
        dependency_runs = tuple(
            asr_model_run(
                model_info=model_info_by_id[AsrModelMode(model_id.value)],
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
        runs = tuple(
            run
            for run in dependency_runs
            if AsrModelMode(run.transcription.model_name) in selected_models
        )
        if AsrModelMode.MERGED_CONSENSUS in selected_models:
            runs += (
                merged_asr_model_run(
                    model_info=model_info_by_id[AsrModelMode.MERGED_CONSENSUS],
                    source_runs=dependency_runs,
                    audio_path=capped_audio_path,
                    speaker_track=speaker_track,
                    audio_duration_seconds=analyzed_duration_seconds,
                    reference_words=reference_words,
                ),
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
    if AsrModelMode.MERGED_CONSENSUS in selected_models:
        remote_model_modes = unique_model_modes((*remote_model_modes, *REMOTE_ASR_MODEL_MODES))
    if not remote_model_modes:
        raise ValueError("Select at least one remote ASR model or merged ASR consensus.")
    return tuple(AsrModelId(model_mode.value) for model_mode in remote_model_modes)


def unique_model_modes(model_modes: tuple[AsrModelMode, ...]) -> tuple[AsrModelMode, ...]:
    unique_modes: list[AsrModelMode] = []
    for model_mode in model_modes:
        if model_mode not in unique_modes:
            unique_modes.append(model_mode)
    return tuple(unique_modes)


def is_remote_model(model_mode: AsrModelMode) -> bool:
    return model_mode in REMOTE_ASR_MODEL_MODES


def speaker_name_for_track(speaker_track: SpeakerTrack) -> SpeakerName:
    match speaker_track:
        case SpeakerTrack.SPEAKER1:
            return SpeakerName.SPEAKER1
        case SpeakerTrack.SPEAKER2:
            return SpeakerName.SPEAKER2


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

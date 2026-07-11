from __future__ import annotations

import tempfile
from pathlib import Path

from app.analyses.asr.models import (
    AsrAnalysisResponse,
    AsrModelInfo,
    AsrModelMode,
    AsrModelRun,
)
from app.analyses.asr.runners import model_identifier_for_mode
from app.asr.schemas import AsrModelId, AsrTranscriptResult, TimestampedWord
from app.asr.service import AsrTranscriptCache, RemoteAsrClientFactory, cached_asr_transcripts
from app.asr_quality.metrics import evaluate_file
from app.asr_quality.schemas import ReferenceTranscript, SpeakerTrack, TranscriptionResult, Word
from app.audio.wav import capped_wave_bytes
from app.data.sessions import SpeakerName, session_audio_path
from app.quality.audio import load_audio


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
        model_info_by_mode = {model.mode: model for model in available_asr_models()}
        cached_response = cached_asr_transcripts(
            audio_path=capped_audio_path,
            requested_models=tuple(model_id_for_mode(model_mode) for model_mode in selected_models),
            cache=cache,
            remote_client_factory=remote_client_factory,
        )
        runs: list[AsrModelRun] = []
        for model_mode in selected_models:
            transcript = transcript_for_model(
                model_id=model_id_for_mode(model_mode),
                results=cached_response.results,
            )
            transcription = transcription_result_from_cached_asr(
                audio_path=capped_audio_path,
                speaker_track=speaker_track,
                audio_duration_seconds=analyzed_duration_seconds,
                transcript=transcript,
            )
            metrics = (
                evaluate_file(
                    ReferenceTranscript(audio_path=str(capped_audio_path), words=reference_words),
                    transcription,
                )
                if reference_words
                else None
            )
            runs.append(
                AsrModelRun(
                    model=model_info_by_mode[model_mode],
                    transcription=transcription,
                    metrics=metrics,
                )
            )
        return AsrAnalysisResponse(
            session_id=session_id,
            speaker_track=speaker_track,
            audio_url=f"/api/audio/{session_id}/{speaker_name_for_track(speaker_track).value}",
            analyzed_duration_seconds=analyzed_duration_seconds,
            reference_word_count=len(reference_words),
            runs=tuple(runs),
        )
    finally:
        capped_audio_path.unlink(missing_ok=True)


def speaker_name_for_track(speaker_track: SpeakerTrack) -> SpeakerName:
    match speaker_track:
        case SpeakerTrack.SPEAKER1:
            return SpeakerName.SPEAKER1
        case SpeakerTrack.SPEAKER2:
            return SpeakerName.SPEAKER2


def model_id_for_mode(model_mode: AsrModelMode) -> AsrModelId:
    return AsrModelId(model_mode.value)


def transcript_for_model(
    model_id: AsrModelId,
    results: tuple[AsrTranscriptResult, ...],
) -> AsrTranscriptResult:
    for result in results:
        if result.model_id == model_id:
            return result
    raise ValueError(f"Missing ASR transcript for model: {model_id.value}")


def transcription_result_from_cached_asr(
    audio_path: Path,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    transcript: AsrTranscriptResult,
) -> TranscriptionResult:
    model_mode = AsrModelMode(transcript.model_id.value)
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
        model_identifier=model_identifier_for_mode(model_mode),
        package_versions=runtime.package_versions if runtime is not None else {},
        model_loading_time_seconds=runtime.model_loading_time_seconds
        if runtime is not None
        else None,
        inference_time_seconds=runtime.inference_time_seconds if runtime is not None else None,
        real_time_factor=runtime.real_time_factor if runtime is not None else None,
        peak_gpu_memory_mb=runtime.peak_gpu_memory_mb if runtime is not None else None,
        error=transcript.error,
    )


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

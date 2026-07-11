from __future__ import annotations

import tempfile
from pathlib import Path

from app.analyses.asr.models import (
    AsrAnalysisResponse,
    AsrModelInfo,
    AsrModelMode,
    AsrModelRun,
    AsrTranscriberContext,
)
from app.analyses.asr.runners import transcribe_with_model
from app.asr_quality.metrics import evaluate_file
from app.asr_quality.schemas import ReferenceTranscript, SpeakerTrack, Word
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
) -> AsrAnalysisResponse:
    source_path = session_audio_path(
        identifier=session_id,
        speaker_name=speaker_name_for_track(speaker_track),
    )
    capped_audio_path = write_capped_analysis_audio(source_path)
    try:
        analyzed_duration_seconds = load_audio(capped_audio_path).metadata.duration_seconds
        model_info_by_mode = {model.mode: model for model in available_asr_models()}
        runs: list[AsrModelRun] = []
        for model_mode in selected_models:
            context = AsrTranscriberContext(
                audio_path=str(capped_audio_path),
                speaker_track=speaker_track,
                audio_duration_seconds=analyzed_duration_seconds,
            )
            transcription = transcribe_with_model(model_mode, context)
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


def write_capped_analysis_audio(source_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"{source_path.stem}_asr_analysis_",
        suffix=".wav",
        delete=False,
    ) as output_file:
        output_file.write(capped_wave_bytes(source_path))
        return Path(output_file.name)

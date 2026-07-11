from __future__ import annotations

from pathlib import Path
from threading import Lock

from faster_whisper import WhisperModel

from app.asr.models.base import cuda_device, load_time_seconds, timestamped_word_from_word
from app.asr.models.text_alignment import WhisperxTextAligner
from app.asr.schemas import AsrModelId, TimestampedWord


class WhisperxAsrModel:
    model_id = AsrModelId.WHISPERX
    package_names = ("torch", "whisperx", "faster-whisper")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.device = cuda_device()
        self.model = WhisperModel("large-v3", device=self.device, compute_type="float16")
        self.aligner = WhisperxTextAligner(device=self.device)

    def transcribe(self, audio_path: Path) -> tuple[TimestampedWord, ...]:
        with self.inference_lock:
            segments, transcription_info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=False,
                vad_filter=False,
            )
            transcription_segments = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ]
            language_code = str(transcription_info.language)
            words = self.aligner.align_segments(
                audio_path=audio_path,
                segments=transcription_segments,
                language_code=language_code,
            )
        return tuple(timestamped_word_from_word(word) for word in words)

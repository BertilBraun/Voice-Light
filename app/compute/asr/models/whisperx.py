from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from app.compute.asr.models.base import ModelTranscription, cuda_device, load_time_seconds
from app.shared.asr import WHISPER_IDENTIFIER, WHISPER_REVISION, AsrModelId, TimestampedWord


class WhisperxAsrModel:
    model_id = AsrModelId.WHISPERX
    package_names = ("torch", "faster-whisper")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.device = cuda_device()
        model_path = snapshot_download(
            WHISPER_IDENTIFIER,
            revision=WHISPER_REVISION,
        )
        self.model = WhisperModel(model_path, device=self.device, compute_type="float16")

    def transcribe(self, audio_path: Path) -> ModelTranscription:
        queue_start = time.perf_counter()
        with self.inference_lock:
            inference_queue_time_seconds = time.perf_counter() - queue_start
            model_execution_start = time.perf_counter()
            segments, transcription_info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,
            )
            words = [
                TimestampedWord(
                    text=word.word,
                    start_seconds=word.start,
                    end_seconds=word.end,
                    confidence=word.probability,
                )
                for segment in segments
                for word in (segment.words or ())
            ]
            del transcription_info
            model_execution_time_seconds = time.perf_counter() - model_execution_start
        return ModelTranscription(
            words=tuple(words),
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=0.0,
            model_execution_time_seconds=model_execution_time_seconds,
        )

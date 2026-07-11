from __future__ import annotations

from pathlib import Path
from threading import Lock

from app.asr.models.base import LoadedAsrModel, TimedTranscription, timed_transcription
from app.asr.models.parakeet import ParakeetAsrModel
from app.asr.models.whisperx import WhisperxAsrModel
from app.asr.schemas import AsrModelId


class AsrModelCache:
    def __init__(self) -> None:
        self.load_lock = Lock()
        self.loaded_models: dict[AsrModelId, LoadedAsrModel] = {}

    def transcribe(self, model_id: AsrModelId, audio_path: Path) -> TimedTranscription:
        model = self.loaded_model(model_id=model_id)
        return timed_transcription(model=model, audio_path=audio_path)

    def loaded_model(self, model_id: AsrModelId) -> LoadedAsrModel:
        with self.load_lock:
            cached_model = self.loaded_models.get(model_id)
            if cached_model is not None:
                return cached_model
            loaded_model = load_asr_model(model_id=model_id)
            self.loaded_models[model_id] = loaded_model
            return loaded_model


def load_asr_model(model_id: AsrModelId) -> LoadedAsrModel:
    match model_id:
        case AsrModelId.PARAKEET_TDT:
            return ParakeetAsrModel()
        case AsrModelId.WHISPERX:
            return WhisperxAsrModel()

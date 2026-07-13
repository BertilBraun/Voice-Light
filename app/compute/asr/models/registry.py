from __future__ import annotations

from pathlib import Path
from threading import Lock

from app.compute.asr.models.base import LoadedAsrModel, TimedTranscription, timed_transcription
from app.shared.asr import AsrModelId


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
            from app.compute.asr.models.parakeet import ParakeetAsrModel

            return ParakeetAsrModel()
        case AsrModelId.WHISPERX:
            from app.compute.asr.models.whisperx import WhisperxAsrModel

            return WhisperxAsrModel()
        case AsrModelId.CANARY:
            from app.compute.asr.models.canary import CanaryAsrModel

            return CanaryAsrModel()
        case AsrModelId.NEMOTRON_3_5:
            from app.compute.asr.models.nemotron import NemotronAsrModel

            return NemotronAsrModel()

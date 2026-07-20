from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path
from threading import Lock

import torch

from app.compute.asr.models.base import (
    BatchInferenceExecutor,
    LoadedAsrModel,
    PreparedAsrAudio,
    TimedTranscription,
    prepare_asr_audio,
    timed_transcription,
)
from app.shared.asr import AsrModelId

DEFAULT_MAX_LOADED_MODELS = 2
AsrModelLoader = Callable[[AsrModelId, BatchInferenceExecutor], LoadedAsrModel]


def load_asr_model(
    model_id: AsrModelId,
    inference_executor: BatchInferenceExecutor,
) -> LoadedAsrModel:
    match model_id:
        case AsrModelId.PARAKEET_TDT:
            from app.compute.asr.models.parakeet import ParakeetAsrModel

            return ParakeetAsrModel(inference_executor)
        case AsrModelId.WHISPERX:
            from app.compute.asr.models.whisperx import WhisperxAsrModel

            return WhisperxAsrModel(inference_executor)
        case AsrModelId.CANARY:
            from app.compute.asr.models.canary import CanaryAsrModel

            return CanaryAsrModel(inference_executor)
        case AsrModelId.NEMOTRON_3_5:
            from app.compute.asr.models.nemotron import NemotronAsrModel

            return NemotronAsrModel(inference_executor)


class AsrModelCache:
    def __init__(
        self,
        max_loaded_models: int = DEFAULT_MAX_LOADED_MODELS,
        model_loader: AsrModelLoader = load_asr_model,
    ) -> None:
        if max_loaded_models <= 0:
            raise ValueError("Maximum loaded ASR models must be positive.")
        self.max_loaded_models = max_loaded_models
        self.model_loader = model_loader
        self.model_lock = Lock()
        self.inference_executor = BatchInferenceExecutor()
        self.loaded_models: dict[AsrModelId, LoadedAsrModel] = {}

    def transcribe(self, model_id: AsrModelId, audio_path: Path) -> TimedTranscription:
        return self.transcribe_prepared(
            model_id=model_id,
            audio=prepare_asr_audio(audio_path),
        )

    def transcribe_prepared(
        self,
        model_id: AsrModelId,
        audio: PreparedAsrAudio,
    ) -> TimedTranscription:
        with self.model_lock:
            model = self._loaded_model(model_id=model_id)
            return timed_transcription(model=model, audio=audio)

    def _loaded_model(self, model_id: AsrModelId) -> LoadedAsrModel:
        cached_model = self.loaded_models.pop(model_id, None)
        if cached_model is not None:
            self.loaded_models[model_id] = cached_model
            return cached_model
        if len(self.loaded_models) == self.max_loaded_models:
            self.loaded_models.pop(next(iter(self.loaded_models)))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        loaded_model = self.model_loader(model_id, self.inference_executor)
        self.loaded_models[model_id] = loaded_model
        return loaded_model

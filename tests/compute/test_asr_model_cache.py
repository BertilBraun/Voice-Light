from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.compute.asr.models.base import (
    BatchInferenceExecutor,
    LoadedAsrModel,
    ModelTranscription,
    PreparedAsrAudio,
)
from app.compute.asr.models.registry import AsrModelCache
from app.shared.asr import AsrModelId


@dataclass(frozen=True)
class StubAsrModel:
    model_id: AsrModelId

    @property
    def package_names(self) -> tuple[str, ...]:
        return ()

    @property
    def model_loading_time_seconds(self) -> float:
        return 0.0

    def transcribe(self, audio: PreparedAsrAudio) -> ModelTranscription:
        del audio
        return ModelTranscription(
            words=(),
            language_estimate=None,
            inference_queue_time_seconds=0.0,
            audio_loading_time_seconds=0.0,
            model_execution_time_seconds=0.0,
        )


def test_model_cache_evicts_least_recently_used_model(tmp_path: Path) -> None:
    loaded_model_ids: list[AsrModelId] = []

    def load_model(
        model_id: AsrModelId,
        inference_executor: BatchInferenceExecutor,
    ) -> LoadedAsrModel:
        del inference_executor
        loaded_model_ids.append(model_id)
        return StubAsrModel(model_id=model_id)

    model_cache = AsrModelCache(max_loaded_models=3, model_loader=load_model)
    audio = PreparedAsrAudio(
        path=tmp_path / "audio.wav",
        samples=np.zeros(160, dtype=np.float32),
        sample_rate=16_000,
        duration_seconds=0.01,
        loading_time_seconds=0.0,
    )

    for model_id in (
        AsrModelId.WHISPERX,
        AsrModelId.PARAKEET_TDT,
        AsrModelId.CANARY,
        AsrModelId.WHISPERX,
        AsrModelId.NEMOTRON_3_5,
        AsrModelId.PARAKEET_TDT,
    ):
        model_cache.transcribe_prepared(model_id=model_id, audio=audio)

    assert loaded_model_ids == [
        AsrModelId.WHISPERX,
        AsrModelId.PARAKEET_TDT,
        AsrModelId.CANARY,
        AsrModelId.NEMOTRON_3_5,
        AsrModelId.PARAKEET_TDT,
    ]

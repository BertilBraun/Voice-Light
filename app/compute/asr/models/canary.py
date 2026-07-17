from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import cast

import nemo.collections.asr as nemo_asr
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from app.compute.asr.models.base import ModelTranscription, cuda_device, load_time_seconds
from app.compute.asr.models.parsing import words_from_nemo_timestamps
from app.shared.asr import CANARY_IDENTIFIER, CANARY_REVISION, AsrModelId

CANARY_MODEL_FILENAME = "canary-1b-v2.nemo"


class CanaryAsrModel:
    model_id = AsrModelId.CANARY
    package_names = ("torch", "nemo_toolkit")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.device = cuda_device()
        model_path = hf_hub_download(
            repo_id=CANARY_IDENTIFIER,
            filename=CANARY_MODEL_FILENAME,
            revision=CANARY_REVISION,
        )
        self.model = cast(
            nemo_asr.models.ASRModel,
            nemo_asr.models.ASRModel.restore_from(
                restore_path=model_path,
                map_location=torch.device("cpu"),
            ),
        )
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: Path) -> ModelTranscription:
        queue_start = time.perf_counter()
        with self.inference_lock:
            inference_queue_time_seconds = time.perf_counter() - queue_start
            model_execution_start = time.perf_counter()
            hypotheses: list[Hypothesis] = self.model.transcribe(
                [str(audio_path)],
                source_lang="en",
                target_lang="en",
                timestamps=True,
            )
            model_execution_time_seconds = time.perf_counter() - model_execution_start
        return ModelTranscription(
            words=tuple(words_from_nemo_timestamps(hypotheses[0].timestamp["word"])),
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=0.0,
            model_execution_time_seconds=model_execution_time_seconds,
        )

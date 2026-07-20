from __future__ import annotations

from threading import Lock
from typing import cast

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from numpy.typing import NDArray

from app.compute.asr.chunking import (
    CANARY_INFERENCE_BATCH_SIZE,
    canary_audio_chunks,
    canary_chunk_samples,
    global_canary_chunk_words,
)
from app.compute.asr.models.base import (
    BatchInferenceExecutor,
    BatchModelOutput,
    ModelTranscription,
    PreparedAsrAudio,
    cuda_device,
    inference_batches,
    load_time_seconds,
)
from app.compute.asr.models.parsing import words_from_nemo_timestamps
from app.shared.asr import CANARY_IDENTIFIER, CANARY_REVISION, AsrModelId, TimestampedWord

CANARY_MODEL_FILENAME = "canary-1b-v2.nemo"


class CanaryAsrModel:
    model_id = AsrModelId.CANARY
    package_names = ("torch", "nemo_toolkit")

    def __init__(self, inference_executor: BatchInferenceExecutor) -> None:
        self.inference_executor = inference_executor
        self.model_state_lock = Lock()
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
        self.model.half()
        self.model.eval()

    def transcribe(self, audio: PreparedAsrAudio) -> ModelTranscription:
        chunks = canary_audio_chunks(audio.duration_seconds)
        audio_loading_time_seconds = audio.loading_time_seconds
        words: list[TimestampedWord] = []
        inference_queue_time_seconds = 0.0
        model_execution_time_seconds = 0.0
        for batch in inference_batches(chunks, CANARY_INFERENCE_BATCH_SIZE):
            result = self.inference_executor.execute(
                tuple(
                    canary_chunk_samples(
                        audio=audio.samples,
                        sample_rate=audio.sample_rate,
                        chunk=chunk,
                    )
                    for chunk in batch
                ),
                self,
            )
            inference_queue_time_seconds += result.queue_time_seconds
            model_execution_time_seconds += result.execution_time_seconds
            for chunk, output in zip(batch, result.outputs, strict=True):
                words.extend(
                    global_canary_chunk_words(
                        chunk=chunk,
                        words=output.words,
                    )
                )
        return ModelTranscription(
            words=tuple(words),
            language_estimate=None,
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=audio_loading_time_seconds,
            model_execution_time_seconds=model_execution_time_seconds,
        )

    def transcribe_batch(
        self,
        audio_chunks: tuple[NDArray[np.float32], ...],
    ) -> tuple[BatchModelOutput, ...]:
        with self.model_state_lock:
            hypotheses: list[Hypothesis] = self.model.transcribe(
                list(audio_chunks),
                batch_size=len(audio_chunks),
                source_lang="en",
                target_lang="en",
                timestamps=True,
            )
        if len(hypotheses) != len(audio_chunks):
            raise ValueError("Canary did not return one hypothesis per audio chunk.")
        return tuple(
            BatchModelOutput(
                words=tuple(words_from_nemo_timestamps(hypothesis.timestamp["word"])),
                language_estimate=None,
            )
            for hypothesis in hypotheses
        )

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import cast

import nemo.collections.asr as nemo_asr
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from app.compute.asr.chunking import (
    CANARY_INFERENCE_BATCH_SIZE,
    CanaryAudioChunk,
    canary_audio_chunks,
    global_canary_chunk_words,
)
from app.compute.asr.models.base import (
    BatchInferenceExecutor,
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

    def transcribe(self, audio: PreparedAsrAudio) -> ModelTranscription:
        chunks = canary_audio_chunks(audio.duration_seconds)
        audio_loading_time_seconds = audio.loading_time_seconds
        words: list[TimestampedWord] = []
        inference_queue_time_seconds = 0.0
        model_execution_time_seconds = 0.0
        with tempfile.TemporaryDirectory(prefix="voice-light-canary-") as directory_name:
            directory = Path(directory_name)
            for batch_index, batch in enumerate(
                inference_batches(chunks, CANARY_INFERENCE_BATCH_SIZE)
            ):
                chunk_paths = tuple(
                    directory / f"batch-{batch_index}-chunk-{chunk_index}.flac"
                    for chunk_index in range(len(batch))
                )
                try:
                    chunk_loading_start = time.perf_counter()
                    for chunk, chunk_path in zip(batch, chunk_paths, strict=True):
                        write_canary_audio_chunk(
                            source_path=audio.path,
                            output_path=chunk_path,
                            chunk=chunk,
                        )
                    audio_loading_time_seconds += time.perf_counter() - chunk_loading_start
                    result = self.inference_executor.execute(
                        chunk_paths,
                        self,
                    )
                    inference_queue_time_seconds += result.queue_time_seconds
                    model_execution_time_seconds += result.execution_time_seconds
                    for chunk, chunk_words in zip(batch, result.outputs, strict=True):
                        words.extend(
                            global_canary_chunk_words(
                                chunk=chunk,
                                words=chunk_words,
                            )
                        )
                finally:
                    for chunk_path in chunk_paths:
                        chunk_path.unlink(missing_ok=True)
        return ModelTranscription(
            words=tuple(words),
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=audio_loading_time_seconds,
            model_execution_time_seconds=model_execution_time_seconds,
        )

    def transcribe_batch(
        self,
        chunk_paths: tuple[Path, ...],
    ) -> tuple[tuple[TimestampedWord, ...], ...]:
        hypotheses: list[Hypothesis] = self.model.transcribe(
            [str(chunk_path) for chunk_path in chunk_paths],
            source_lang="en",
            target_lang="en",
            timestamps=True,
        )
        if len(hypotheses) != len(chunk_paths):
            raise ValueError("Canary did not return one hypothesis per audio chunk.")
        return tuple(
            tuple(words_from_nemo_timestamps(hypothesis.timestamp["word"]))
            for hypothesis in hypotheses
        )


def write_canary_audio_chunk(
    source_path: Path,
    output_path: Path,
    chunk: CanaryAudioChunk,
) -> None:
    subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(chunk.start_seconds),
            "-i",
            str(source_path),
            "-t",
            str(chunk.duration_seconds),
            "-map",
            "0:a:0",
            "-c:a",
            "flac",
            "-compression_level",
            "0",
            str(output_path),
        ),
        check=True,
    )

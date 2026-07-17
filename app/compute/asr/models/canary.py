from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import cast

import nemo.collections.asr as nemo_asr
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from app.compute.asr.chunking import (
    CanaryAudioChunk,
    canary_audio_chunks,
    global_canary_chunk_words,
)
from app.compute.asr.models.base import ModelTranscription, cuda_device, load_time_seconds
from app.compute.asr.models.parsing import words_from_nemo_timestamps
from app.shared.asr import CANARY_IDENTIFIER, CANARY_REVISION, AsrModelId, TimestampedWord
from app.shared.audio import probe_local_audio_metadata
from app.shared.audio.transport import CANONICAL_MODEL_AUDIO_SPEC

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
            audio_loading_start = time.perf_counter()
            metadata = probe_local_audio_metadata(audio_path)
            chunks = canary_audio_chunks(metadata.duration_seconds)
            audio_loading_time_seconds = time.perf_counter() - audio_loading_start
            words: list[TimestampedWord] = []
            model_execution_time_seconds = 0.0
            with tempfile.TemporaryDirectory(prefix="voice-light-canary-") as directory_name:
                chunk_path = Path(directory_name) / "chunk.flac"
                for chunk in chunks:
                    chunk_loading_start = time.perf_counter()
                    write_canary_audio_chunk(
                        source_path=audio_path,
                        output_path=chunk_path,
                        chunk=chunk,
                    )
                    audio_loading_time_seconds += time.perf_counter() - chunk_loading_start
                    try:
                        model_execution_start = time.perf_counter()
                        hypotheses: list[Hypothesis] = self.model.transcribe(
                            [str(chunk_path)],
                            source_lang="en",
                            target_lang="en",
                            timestamps=True,
                        )
                        model_execution_time_seconds += time.perf_counter() - model_execution_start
                        words.extend(
                            global_canary_chunk_words(
                                chunk=chunk,
                                words=tuple(
                                    words_from_nemo_timestamps(
                                        hypotheses[0].timestamp["word"],
                                    )
                                ),
                            )
                        )
                    finally:
                        chunk_path.unlink(missing_ok=True)
        return ModelTranscription(
            words=tuple(words),
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=audio_loading_time_seconds,
            model_execution_time_seconds=model_execution_time_seconds,
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
            "-ac",
            str(CANONICAL_MODEL_AUDIO_SPEC.channels),
            "-ar",
            str(CANONICAL_MODEL_AUDIO_SPEC.sample_rate),
            "-c:a",
            "flac",
            "-compression_level",
            "0",
            str(output_path),
        ),
        check=True,
    )

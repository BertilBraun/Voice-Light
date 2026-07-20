from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from app.compute.asr.chunking import canary_audio_chunks, canary_chunk_samples
from app.compute.asr.models.base import BatchInferenceExecutor, prepare_asr_audio
from app.compute.asr.models.canary import CanaryAsrModel


@dataclass(frozen=True)
class BatchBenchmarkResult:
    dtype: str
    batch_size: int
    chunk_count: int
    audio_duration_seconds: float
    execution_time_seconds: float
    real_time_factor: float
    peak_gpu_memory_mb: float
    word_count: int
    word_text_sha256: str
    transcript_sha256: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Canary generation throughput across audio chunk batch sizes."
    )
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--chunk-count", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument(
        "--dtypes",
        choices=("float32", "float16", "bfloat16"),
        nargs="+",
        default=("float32",),
    )
    return parser.parse_args()


def benchmark_batch_size(
    model: CanaryAsrModel,
    audio_chunks: tuple[NDArray[np.float32], ...],
    audio_duration_seconds: float,
    batch_size: int,
    dtype: str,
) -> BatchBenchmarkResult:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    words: list[dict[str, str | float]] = []
    execution_start = time.perf_counter()
    for start_index in range(0, len(audio_chunks), batch_size):
        outputs = model.transcribe_batch(audio_chunks[start_index : start_index + batch_size])
        words.extend(word.model_dump(mode="json") for output in outputs for word in output.words)
    torch.cuda.synchronize()
    execution_time_seconds = time.perf_counter() - execution_start
    serialized_words = json.dumps(words, sort_keys=True).encode("utf-8")
    word_text = " ".join(str(word["text"]) for word in words)
    serialized_word_text = word_text.encode("utf-8")
    return BatchBenchmarkResult(
        dtype=dtype,
        batch_size=batch_size,
        chunk_count=len(audio_chunks),
        audio_duration_seconds=audio_duration_seconds,
        execution_time_seconds=execution_time_seconds,
        real_time_factor=execution_time_seconds / audio_duration_seconds,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1_048_576,
        word_count=len(words),
        word_text_sha256=hashlib.sha256(serialized_word_text).hexdigest(),
        transcript_sha256=hashlib.sha256(serialized_words).hexdigest(),
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.chunk_count <= 0:
        raise ValueError("Chunk count must be positive.")
    if not arguments.audio_path.is_file():
        raise ValueError(f"Audio file not found: {arguments.audio_path}")
    model = CanaryAsrModel(BatchInferenceExecutor())
    audio = prepare_asr_audio(arguments.audio_path)
    chunks = canary_audio_chunks(audio.duration_seconds)[: arguments.chunk_count]
    audio_chunks = tuple(
        canary_chunk_samples(
            audio=audio.samples,
            sample_rate=audio.sample_rate,
            chunk=chunk,
        )
        for chunk in chunks
    )
    audio_duration_seconds = sum(chunk.duration_seconds for chunk in chunks)
    results: list[BatchBenchmarkResult] = []
    for dtype in arguments.dtypes:
        match dtype:
            case "float32":
                torch_dtype = torch.float32
            case "float16":
                torch_dtype = torch.float16
            case "bfloat16":
                torch_dtype = torch.bfloat16
            case _:
                raise ValueError(f"Unsupported benchmark dtype: {dtype}")
        model.model.to(dtype=torch_dtype)
        model.transcribe_batch(audio_chunks[:1])
        results.extend(
            benchmark_batch_size(
                model=model,
                audio_chunks=audio_chunks,
                audio_duration_seconds=audio_duration_seconds,
                batch_size=batch_size,
                dtype=dtype,
            )
            for batch_size in arguments.batch_sizes
        )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()

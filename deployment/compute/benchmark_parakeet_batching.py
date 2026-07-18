from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from app.compute.asr.chunking import (
    ParakeetAudioChunk,
    global_chunk_words,
    parakeet_audio_chunks,
)
from app.compute.asr.models.base import BatchInferenceExecutor, prepare_asr_audio
from app.compute.asr.models.parakeet import ParakeetAsrModel
from app.compute.asr.models.parsing import words_from_parakeet_timestamps
from app.shared.asr import TimestampedWord


@dataclass(frozen=True)
class BatchBenchmarkResult:
    batch_size: int
    chunk_count: int
    audio_duration_seconds: float
    execution_time_seconds: float
    real_time_factor: float
    peak_gpu_memory_mb: float
    word_count: int
    transcript_sha256: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Parakeet generation throughput across audio chunk batch sizes."
    )
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--chunk-count", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    return parser.parse_args()


def benchmark_batch_size(
    model: ParakeetAsrModel,
    chunks: tuple[ParakeetAudioChunk, ...],
    batch_size: int,
) -> BatchBenchmarkResult:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    words: list[TimestampedWord] = []
    execution_start = time.perf_counter()
    for start_index in range(0, len(chunks), batch_size):
        batch = chunks[start_index : start_index + batch_size]
        inputs = model.processor(
            [chunk.samples for chunk in batch],
            sampling_rate=model.sample_rate,
        )
        inputs.to(model.model.device, dtype=model.model.dtype)
        output = model.model.generate(**inputs, return_dict_in_generate=True)
        _decoded_output, decoded_timestamps = model.processor.decode(
            output.sequences,
            durations=output.durations,
            skip_special_tokens=True,
        )
        if not isinstance(decoded_timestamps, list) or len(decoded_timestamps) != len(batch):
            raise ValueError("Parakeet did not return one timestamp sequence per audio chunk.")
        for chunk, chunk_timestamps in zip(batch, decoded_timestamps, strict=True):
            words.extend(
                global_chunk_words(
                    chunk=chunk,
                    words=tuple(words_from_parakeet_timestamps(chunk_timestamps)),
                )
            )
    torch.cuda.synchronize()
    execution_time_seconds = time.perf_counter() - execution_start
    audio_duration_seconds = sum(len(chunk.samples) / model.sample_rate for chunk in chunks)
    serialized_words = json.dumps(
        [word.model_dump(mode="json") for word in words],
        sort_keys=True,
    ).encode("utf-8")
    return BatchBenchmarkResult(
        batch_size=batch_size,
        chunk_count=len(chunks),
        audio_duration_seconds=audio_duration_seconds,
        execution_time_seconds=execution_time_seconds,
        real_time_factor=execution_time_seconds / audio_duration_seconds,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1_048_576,
        word_count=len(words),
        transcript_sha256=hashlib.sha256(serialized_words).hexdigest(),
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.chunk_count <= 0:
        raise ValueError("Chunk count must be positive.")
    if not arguments.audio_path.is_file():
        raise ValueError(f"Audio file not found: {arguments.audio_path}")
    model = ParakeetAsrModel(BatchInferenceExecutor())
    audio = prepare_asr_audio(arguments.audio_path)
    chunks = parakeet_audio_chunks(audio=audio.samples, sample_rate=audio.sample_rate)[
        : arguments.chunk_count
    ]
    if not chunks:
        raise ValueError("Audio file contains no benchmark chunks.")
    benchmark_batch_size(model=model, chunks=chunks[:1], batch_size=1)
    results = tuple(
        benchmark_batch_size(model=model, chunks=chunks, batch_size=batch_size)
        for batch_size in arguments.batch_sizes
    )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()

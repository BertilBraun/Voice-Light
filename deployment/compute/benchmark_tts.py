from __future__ import annotations

import argparse
import asyncio
import re
import time
from collections.abc import Sequence

from app.compute.voice.interfaces import (
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
)
from app.compute.voice.kyutai_tts import KyutaiSpeechSynthesizer


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=(
            "Hello! This benchmark measures how quickly the voice agent starts speaking and how "
            "fast it produces a complete response."
        ),
    )
    parser.add_argument("--runs", type=int, default=3)
    options = parser.parse_args(arguments)
    if options.runs < 1:
        raise ValueError("--runs must be at least 1.")
    asyncio.run(run_benchmark(options.text, options.runs))


async def run_benchmark(text: str, runs: int) -> None:
    load_started = time.perf_counter()
    synthesizer = await asyncio.to_thread(KyutaiSpeechSynthesizer)
    print(f"model_load_seconds={time.perf_counter() - load_started:.3f}")

    words = tuple(
        SynthesisWord(match.group(), match.start(), match.end())
        for match in re.finditer(r"\S+", text)
    )
    if not words:
        raise ValueError("--text must contain at least one non-whitespace word.")

    for run_index in range(runs):
        session = synthesizer.start_session()
        started = time.perf_counter()
        first_chunk_seconds: float | None = None
        sample_count = 0
        boundary_count = 0
        for word in words:
            await session.add_word(word)
        await session.finish_input()
        async for event in session.stream_events():
            match event:
                case SynthesizedAudioChunk():
                    if first_chunk_seconds is None:
                        first_chunk_seconds = time.perf_counter() - started
                    sample_count += len(event.pcm_bytes) // 2
                case SynthesizedWordBoundary():
                    boundary_count += 1
        await session.cancel()
        if first_chunk_seconds is None or sample_count == 0:
            raise RuntimeError("Kyutai TTS produced no audio.")
        total_seconds = time.perf_counter() - started
        audio_duration_seconds = sample_count / synthesizer.sample_rate
        real_time_factor = total_seconds / audio_duration_seconds
        print(
            f"run={run_index + 1} first_chunk_seconds={first_chunk_seconds:.3f} "
            f"total_seconds={total_seconds:.3f} "
            f"audio_duration_seconds={audio_duration_seconds:.3f} "
            f"real_time_factor={real_time_factor:.3f} boundaries={boundary_count}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from pocket_tts import TTSModel


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

    load_started = time.perf_counter()
    model = TTSModel.load_model(language="english")
    voice_state = model.get_state_for_audio_prompt("alba")
    print(f"model_load_seconds={time.perf_counter() - load_started:.3f}")

    for run_index in range(options.runs):
        started = time.perf_counter()
        first_chunk_seconds: float | None = None
        sample_count = 0
        for audio_chunk in model.generate_audio_stream(voice_state, options.text):
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - started
            sample_count += audio_chunk.numel()
        total_seconds = time.perf_counter() - started
        audio_duration_seconds = sample_count / model.sample_rate
        real_time_factor = total_seconds / audio_duration_seconds
        print(
            f"run={run_index + 1} first_chunk_seconds={first_chunk_seconds:.3f} "
            f"total_seconds={total_seconds:.3f} "
            f"audio_duration_seconds={audio_duration_seconds:.3f} "
            f"real_time_factor={real_time_factor:.3f}"
        )


if __name__ == "__main__":
    main()

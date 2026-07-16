from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from app.compute.voice.interfaces import SynthesisWord, SynthesizedAudioChunk
from app.compute.voice.model_constants import (
    KYUTAI_TTS_MODEL_NAME,
    KYUTAI_TTS_MODEL_REVISION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
)
from app.compute.voice.nemotron_worker import MODEL_NAME, MODEL_REVISION
from app.compute.voice.tts_selection import (
    SpeechSynthesisBackend,
    SpeechSynthesisSettings,
    create_speech_synthesizer,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true")
    options = parser.parse_args(arguments)
    speech_synthesis_settings = SpeechSynthesisSettings.from_environment(
        os.environ,
        REPOSITORY_ROOT,
    )
    validate_nvidia_gpu()
    validate_imports()
    if options.download_models:
        download_required_models(speech_synthesis_settings)
        smoke_test_tts(speech_synthesis_settings)
    print("Compute environment validation passed.")


def validate_nvidia_gpu() -> None:
    subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 4090" not in device_name:
        raise RuntimeError(f"Expected an RTX 4090, found {device_name}.")
    memory_bytes = torch.cuda.get_device_properties(0).total_memory
    if memory_bytes < 22 * 1024**3:
        raise RuntimeError("The GPU exposes less than 22 GiB of VRAM.")
    if torch.version.cuda is None:
        raise RuntimeError("The installed PyTorch build has no CUDA runtime.")
    print(f"CUDA validation passed: {device_name}, PyTorch CUDA {torch.version.cuda}.")


def validate_imports() -> None:
    for module_name in (
        "av",
        "faster_whisper",
        "librosa",
        "nemo.collections.asr",
        "moshi",
        "soundfile",
        "transformers",
    ):
        importlib.import_module(module_name)
    print("Runtime import smoke tests passed.")


def download_required_models(settings: SpeechSynthesisSettings) -> None:
    required_models = [
        (MODEL_NAME, MODEL_REVISION),
        (LANGUAGE_MODEL_NAME, LANGUAGE_MODEL_REVISION),
    ]
    if settings.backend is SpeechSynthesisBackend.KYUTAI:
        required_models.append((KYUTAI_TTS_MODEL_NAME, KYUTAI_TTS_MODEL_REVISION))
    for repository_id, revision in required_models:
        snapshot_download(repository_id, revision=revision)
        print(f"Cached {repository_id} at revision {revision}.")


def smoke_test_tts(settings: SpeechSynthesisSettings) -> None:
    asyncio.run(_smoke_test_tts(settings))


async def _smoke_test_tts(settings: SpeechSynthesisSettings) -> None:
    synthesizer = await asyncio.to_thread(create_speech_synthesizer, settings)
    try:
        session = synthesizer.start_session()
        await session.add_word(SynthesisWord(text="Voice", text_start=0, text_end=5))
        await session.add_word(SynthesisWord(text="Light", text_start=6, text_end=11))
        await session.add_word(SynthesisWord(text="is", text_start=12, text_end=14))
        await session.add_word(SynthesisWord(text="ready.", text_start=15, text_end=21))
        await session.finish_input()
        sample_count = 0
        async for event in session.stream_events():
            match event:
                case SynthesizedAudioChunk():
                    sample_count += len(event.pcm_bytes) // 2
                case _:
                    pass
        await session.cancel()
        if sample_count == 0:
            raise RuntimeError(f"{settings.backend} produced no smoke-test audio.")
    finally:
        await asyncio.to_thread(synthesizer.close)
    print(f"{settings.backend} word streaming and decoder smoke test passed.")


if __name__ == "__main__":
    main()

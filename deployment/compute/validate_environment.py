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
    NEMOTRON_ASR_MODEL_NAME,
    NEMOTRON_ASR_MODEL_REVISION,
    SEARCH_SUMMARIZER_MODEL_NAME,
    SEARCH_SUMMARIZER_MODEL_REVISION,
)
from app.compute.voice.qwen_config import (
    QwenModelConfiguration,
    language_model_configuration_from_environment,
)
from app.compute.voice.tts_selection import (
    SpeechSynthesisBackend,
    SpeechSynthesisSettings,
    create_speech_synthesizer,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VLLM_PYTHON_PATH = REPOSITORY_ROOT / "deployment/compute/vllm/.venv/bin/python"
MINIMUM_GPU_MEMORY_BYTES = 10 * 1024**3


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
    validate_vllm_environment()
    if options.download_models:
        language_model_configuration = language_model_configuration_from_environment(os.environ)
        download_required_models(speech_synthesis_settings, language_model_configuration)
        smoke_test_tts(speech_synthesis_settings)
    print("Compute environment validation passed.")


def validate_nvidia_gpu() -> None:
    subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")
    device_name = torch.cuda.get_device_name(0)
    memory_bytes = torch.cuda.get_device_properties(0).total_memory
    cuda_version = torch.version.cuda
    validate_gpu_properties(device_name, memory_bytes, cuda_version)
    print(f"CUDA validation passed: {device_name}, PyTorch CUDA {cuda_version}.")


def validate_gpu_properties(
    device_name: str,
    memory_bytes: int,
    cuda_version: str | None,
) -> None:
    if memory_bytes < MINIMUM_GPU_MEMORY_BYTES:
        memory_gib = memory_bytes / 1024**3
        raise RuntimeError(
            f"{device_name} exposes {memory_gib:.1f} GiB of VRAM; at least 10 GiB is required."
        )
    if cuda_version is None:
        raise RuntimeError("The installed PyTorch build has no CUDA runtime.")


def validate_imports() -> None:
    for module_name in (
        "av",
        "faster_whisper",
        "librosa",
        "nemo.collections.asr",
        "moshi",
        "peft",
        "soundfile",
        "transformers",
    ):
        importlib.import_module(module_name)
    print("Runtime import smoke tests passed.")


def validate_vllm_environment() -> None:
    subprocess.run(
        [VLLM_PYTHON_PATH, "-c", "import vllm"],
        check=True,
    )
    print("vLLM runtime import smoke test passed.")


def download_required_models(
    settings: SpeechSynthesisSettings,
    language_model_configuration: QwenModelConfiguration,
) -> None:
    required_models = [
        (NEMOTRON_ASR_MODEL_NAME, NEMOTRON_ASR_MODEL_REVISION),
        (
            language_model_configuration.model_name,
            language_model_configuration.model_revision,
        ),
        (SEARCH_SUMMARIZER_MODEL_NAME, SEARCH_SUMMARIZER_MODEL_REVISION),
    ]
    adapter = language_model_configuration.adapter
    if adapter is not None:
        required_models.append((adapter.repository_id, adapter.revision))
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

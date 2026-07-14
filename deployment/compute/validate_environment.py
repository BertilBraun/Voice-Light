from __future__ import annotations

import argparse
import importlib
import subprocess
from collections.abc import Sequence

import torch
from huggingface_hub import snapshot_download
from pocket_tts import TTSModel

from app.compute.voice.models import LANGUAGE_MODEL_NAME, LANGUAGE_MODEL_REVISION
from app.compute.voice.nemotron_worker import MODEL_NAME, MODEL_REVISION


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true")
    options = parser.parse_args(arguments)
    validate_nvidia_gpu()
    validate_imports()
    if options.download_models:
        download_required_models()
        smoke_test_tts()
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
        "pocket_tts",
        "soundfile",
        "transformers",
    ):
        importlib.import_module(module_name)
    print("Runtime import smoke tests passed.")


def download_required_models() -> None:
    for repository_id, revision in (
        (MODEL_NAME, MODEL_REVISION),
        (LANGUAGE_MODEL_NAME, LANGUAGE_MODEL_REVISION),
    ):
        snapshot_download(repository_id, revision=revision)
        print(f"Cached {repository_id} at revision {revision}.")


def smoke_test_tts() -> None:
    model = TTSModel.load_model(language="english")
    voice_state = model.get_state_for_audio_prompt("alba")
    stream = model.generate_audio_stream(voice_state, "Voice Light is ready.")
    sample_count = sum(audio_chunk.numel() for audio_chunk in stream)
    if sample_count == 0:
        raise RuntimeError("Pocket TTS produced an empty smoke-test chunk.")
    print("Pocket TTS model and streaming decoder smoke test passed.")


if __name__ == "__main__":
    main()

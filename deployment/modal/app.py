from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from fastapi import FastAPI

import modal

APPLICATION_NAME: Final = "VoiceLightAgent"
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
REMOTE_REPOSITORY_ROOT: Final = PurePosixPath("/opt/voice-light")
ADAPTER_CHECKPOINT: Final = (
    REPOSITORY_ROOT
    / ".cache"
    / "rtx3090-node-final-backup-20260829"
    / "voice-light-human-finetune-v1"
    / "adapter-best.pt"
)
REMOTE_ADAPTER_CHECKPOINT: Final = PurePosixPath("/opt/voice-light-artifacts/adapter-best.pt")
MODEL_CACHE_MOUNT: Final = PurePosixPath("/model-cache")
RUNTIME_CACHE_MOUNT: Final = PurePosixPath("/runtime-cache")


@dataclass(frozen=True)
class ModalDeploymentConfiguration:
    gpu: str = "L40S"
    scaledown_window_seconds: int = 1_200
    startup_timeout_seconds: int = 1_800
    function_timeout_seconds: int = 86_400
    secret_name: str = "voice-light-compute"
    model_cache_volume_name: str = "voice-light-agent-model-cache"
    runtime_cache_volume_name: str = "voice-light-runtime-cache"

    def environment(self) -> dict[str, str]:
        hugging_face_cache = MODEL_CACHE_MOUNT / "huggingface"
        return {
            "HF_HOME": str(hugging_face_cache),
            "HF_HUB_CACHE": str(hugging_face_cache / "hub"),
            "PYTHONPATH": str(REMOTE_REPOSITORY_ROOT),
            "TORCH_HOME": str(MODEL_CACHE_MOUNT / "torch"),
            "VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS": "1",
            "VOICE_LIGHT_COMPUTE_LOG_DIR": str(RUNTIME_CACHE_MOUNT / "logs"),
            "VOICE_LIGHT_DATASET_AUDIO_CACHE_DIR": str(RUNTIME_CACHE_MOUNT / "dataset-audio"),
            "VOICE_LIGHT_EAGER_MODEL_LOADING": "true",
            "VOICE_LIGHT_FLOOR_TAKE_THRESHOLD": "0.82",
            "VOICE_LIGHT_NON_FLOOR_FEEDBACK_THRESHOLD": "0.82",
            "VOICE_LIGHT_OVERLAP_CLASSIFICATION_DEADLINE_MS": "500",
            "VOICE_LIGHT_TTS_BACKEND": "kyutai",
            "VOICE_LIGHT_TURN_ADAPTER_CHECKPOINT": str(REMOTE_ADAPTER_CHECKPOINT),
            "VOICE_LIGHT_VOICE_STACK_ENABLED": "true",
            "XDG_CACHE_HOME": str(RUNTIME_CACHE_MOUNT),
        }


configuration = ModalDeploymentConfiguration()

if modal.is_local() and not ADAPTER_CHECKPOINT.is_file():
    raise ValueError(f"Turn-taking adapter checkpoint is missing: {ADAPTER_CHECKPOINT}")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "ca-certificates",
        "clang",
        "espeak-ng",
        "ffmpeg",
        "git",
        "git-lfs",
        "libsndfile1",
        "libsndfile1-dev",
        "libsox-dev",
        "pkg-config",
        "sox",
    )
    .uv_sync(".", extras=["compute"], extra_options="--no-group dev")
    .add_local_dir(
        str(REPOSITORY_ROOT / "deployment" / "compute" / "vllm"),
        remote_path=str(REMOTE_REPOSITORY_ROOT / "deployment" / "compute" / "vllm"),
        ignore=[".venv", "__pycache__"],
        copy=True,
    )
    .run_commands(
        "/.uv/uv sync "
        f"--project={REMOTE_REPOSITORY_ROOT / 'deployment' / 'compute' / 'vllm'} "
        "--frozen --no-install-project --compile-bytecode"
    )
    .add_local_dir(
        str(REPOSITORY_ROOT / "app"),
        remote_path=str(REMOTE_REPOSITORY_ROOT / "app"),
        ignore=["__pycache__", "*.pyc"],
        copy=True,
    )
    .add_local_file(
        str(ADAPTER_CHECKPOINT),
        remote_path=str(REMOTE_ADAPTER_CHECKPOINT),
        copy=True,
    )
    .env(configuration.environment())
    .workdir(str(REMOTE_REPOSITORY_ROOT))
)

with image.imports():
    from app.compute.main import create_app_from_environment

app = modal.App(APPLICATION_NAME)
compute_secret = modal.Secret.from_name(configuration.secret_name)
model_cache = modal.Volume.from_name(
    configuration.model_cache_volume_name,
    create_if_missing=True,
)
runtime_cache = modal.Volume.from_name(
    configuration.runtime_cache_volume_name,
    create_if_missing=True,
)


@app.function(
    image=image,
    gpu=configuration.gpu,
    max_containers=1,
    min_containers=0,
    scaledown_window=configuration.scaledown_window_seconds,
    startup_timeout=configuration.startup_timeout_seconds,
    timeout=configuration.function_timeout_seconds,
    secrets=[compute_secret],
    volumes={
        str(MODEL_CACHE_MOUNT): model_cache,
        str(RUNTIME_CACHE_MOUNT): runtime_cache,
    },
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def voice_light() -> FastAPI:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return create_app_from_environment()

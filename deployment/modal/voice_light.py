from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from fastapi import FastAPI

import modal
from app.compute.voice.model_constants import (
    SEARCH_SUMMARIZER_MODEL_NAME,
    SEARCH_SUMMARIZER_MODEL_REVISION,
)
from app.shared.model_constants import NEMOTRON_ASR_MODEL_NAME, NEMOTRON_ASR_MODEL_REVISION

APPLICATION_NAME: Final = "VoiceLightAgent"
REMOTE_REPOSITORY_ROOT: Final = PurePosixPath("/opt/voice-light")


def repository_root(module_path: Path, local: bool) -> Path:
    if not local:
        return Path(REMOTE_REPOSITORY_ROOT)
    return module_path.parents[2]


REPOSITORY_ROOT: Final = repository_root(Path(__file__).resolve(), modal.is_local())
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
MERGED_LANGUAGE_MODEL_NAME: Final = "Qwen/Qwen3-4B-Instruct-2507"
MERGED_LANGUAGE_MODEL_REVISION: Final = "cdbee75f17c01a7cc42f958dc650907174af0554"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRepository:
    repository_id: str
    revision: str
    allowed_files: tuple[str, ...] | None = None


MODEL_REPOSITORIES: Final = (
    ModelRepository(
        repository_id=NEMOTRON_ASR_MODEL_NAME,
        revision=NEMOTRON_ASR_MODEL_REVISION,
    ),
    ModelRepository(
        repository_id=MERGED_LANGUAGE_MODEL_NAME,
        revision=MERGED_LANGUAGE_MODEL_REVISION,
    ),
    ModelRepository(
        repository_id=SEARCH_SUMMARIZER_MODEL_NAME,
        revision=SEARCH_SUMMARIZER_MODEL_REVISION,
    ),
    ModelRepository(
        repository_id="kyutai/tts-1.6b-en_fr",
        revision="f65439609986c392cb12df63938abcc550c3fb15",
    ),
    ModelRepository(
        repository_id="kyutai/tts-voices",
        revision="323332d33f997de8394f24a193e1a76df720e01a",
        allowed_files=(
            "expresso/ex03-ex01_happy_001_channel1_334s.wav",
            "expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors",
        ),
    ),
)


@dataclass(frozen=True)
class ModalDeploymentConfiguration:
    gpu_options: tuple[str, ...] = ("A10", "L40S")
    compute_regions: tuple[str, ...] = ("eu",)
    routing_region: str = "eu-west"
    scaledown_window_seconds: int = 120
    startup_timeout_seconds: int = 1_800
    function_timeout_seconds: int = 86_400
    compute_secret_name: str = "voice-light-compute"
    search_secret_name: str = "voice-light-search"
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
            "VOICE_LIGHT_MERGED_LANGUAGE_MODEL_NAME": MERGED_LANGUAGE_MODEL_NAME,
            "VOICE_LIGHT_MERGED_LANGUAGE_MODEL_REVISION": MERGED_LANGUAGE_MODEL_REVISION,
            "VOICE_LIGHT_NON_FLOOR_FEEDBACK_THRESHOLD": "0.82",
            "VOICE_LIGHT_OVERLAP_CLASSIFICATION_DEADLINE_MS": "500",
            "VOICE_LIGHT_OVERLAP_FINALIZATION_GRACE_MS": "120",
            "VOICE_LIGHT_OVERLAP_PREDICTION_SETTLE_MS": "80",
            "VOICE_LIGHT_PENDING_SILENCE_SPECULATION_MS": "80",
            "VOICE_LIGHT_MAXIMUM_TRANSPORT_AHEAD_MS": "1200",
            "VOICE_LIGHT_MINIMUM_NORMAL_TURN_COMMIT_SILENCE_MS": "240",
            "VOICE_LIGHT_SPECULATIVE_MINIMUM_CONFIDENCE": "0.60",
            "VOICE_LIGHT_SPECULATIVE_TURN_COMPLETION_THRESHOLD": "0.55",
            "VOICE_LIGHT_SPECULATIVE_YIELD_THRESHOLD": "0.55",
            "VOICE_LIGHT_TRANSCRIPT_FREE_FLOOR_TAKE_DEADLINE_MS": "900",
            "VOICE_LIGHT_QWEN_ENFORCE_EAGER": "true",
            "VOICE_LIGHT_QWEN_BACKEND": "transformers",
            "VOICE_LIGHT_SHARE_LANGUAGE_MODEL_FOR_SEARCH": "false",
            "VOICE_LIGHT_TTS_BACKEND": "kyutai",
            "VOICE_LIGHT_TURN_ADAPTER_CHECKPOINT": str(REMOTE_ADAPTER_CHECKPOINT),
            "VOICE_LIGHT_VAD_ENDPOINT_CONFIDENCE": "0.70",
            "VOICE_LIGHT_VAD_ENDPOINT_YIELD_PROBABILITY": "0.70",
            "VOICE_LIGHT_VAD_SPECULATION_DEBOUNCE_MS": "0",
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
    .add_local_file(
        str(REPOSITORY_ROOT / "deployment" / "compute" / "smoke_test_tool_use.py"),
        remote_path=str(
            REMOTE_REPOSITORY_ROOT / "deployment" / "compute" / "smoke_test_tool_use.py"
        ),
        copy=True,
    )
    .add_local_file(
        str(REPOSITORY_ROOT / "deployment" / "modal" / "smoke_search_provider.py"),
        remote_path=str(
            REMOTE_REPOSITORY_ROOT / "deployment" / "modal" / "smoke_search_provider.py"
        ),
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
    from huggingface_hub import snapshot_download

    from app.compute.config import ComputeSettings
    from app.compute.main import create_compute_app_for_runtime, create_compute_runtime
    from app.compute.runtime import ComputeRuntime
    from app.compute.telemetry import configure_logging

app = modal.App(APPLICATION_NAME)
compute_secret = modal.Secret.from_name(configuration.compute_secret_name)
search_secret = modal.Secret.from_name(configuration.search_secret_name)
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
    timeout=configuration.startup_timeout_seconds,
    secrets=[compute_secret],
    volumes={str(MODEL_CACHE_MOUNT): model_cache},
)
def cache_models() -> None:
    for repository in MODEL_REPOSITORIES:
        snapshot_download(
            repository.repository_id,
            revision=repository.revision,
            allow_patterns=repository.allowed_files,
        )
    model_cache.commit()


@app.function(
    image=image,
    env={"HF_HUB_OFFLINE": "1"},
    gpu="L40S",
    timeout=configuration.startup_timeout_seconds,
    secrets=[compute_secret],
    volumes={str(MODEL_CACHE_MOUNT): model_cache},
)
def smoke_tool_use() -> None:
    from deployment.compute.smoke_test_tool_use import main

    main()


@app.function(
    image=image,
    timeout=60,
    secrets=[search_secret],
)
def smoke_search_provider() -> None:
    from deployment.modal.smoke_search_provider import (
        configured_tavily_provider,
        measure_search_provider,
    )

    provider = configured_tavily_provider(os.environ, configuration.search_secret_name)
    result = asyncio.run(measure_search_provider(provider))
    print(result.model_dump_json())


@app.cls(
    image=image,
    env={"HF_HUB_OFFLINE": "1"},
    gpu=list(configuration.gpu_options),
    region=list(configuration.compute_regions),
    routing_region=configuration.routing_region,
    max_containers=1,
    min_containers=0,
    scaledown_window=configuration.scaledown_window_seconds,
    startup_timeout=configuration.startup_timeout_seconds,
    timeout=configuration.function_timeout_seconds,
    secrets=[compute_secret, search_secret],
    volumes={
        str(MODEL_CACHE_MOUNT): model_cache,
        str(RUNTIME_CACHE_MOUNT): runtime_cache,
    },
)
@modal.concurrent(max_inputs=1)
class VoiceLightEurope:
    settings: ComputeSettings
    runtime: ComputeRuntime

    @modal.enter()
    def load_models(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.settings = ComputeSettings.from_environment(os.environ)
        configure_logging(self.settings.log_directory)
        logger.info(
            "Modal container placement: cloud=%s region=%s task_id=%s",
            os.environ["MODAL_CLOUD_PROVIDER"],
            os.environ["MODAL_REGION"],
            os.environ["MODAL_TASK_ID"],
        )
        self.runtime = create_compute_runtime(self.settings)
        asyncio.run(self._load_models())

    async def _load_models(self) -> None:
        self.runtime.start_loading()
        await self.runtime.wait_until_loading_complete()

    @modal.asgi_app(label="voicelightagent-voice-light")
    def voice_light(self) -> FastAPI:
        return create_compute_app_for_runtime(self.settings, self.runtime)

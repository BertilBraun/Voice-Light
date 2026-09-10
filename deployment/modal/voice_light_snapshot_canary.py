from __future__ import annotations

import asyncio
import os
from typing import Final

from fastapi import FastAPI

import modal
from deployment.modal.voice_light import (
    MODEL_CACHE_MOUNT,
    RUNTIME_CACHE_MOUNT,
    compute_secret,
    configuration,
    image,
    model_cache,
    runtime_cache,
    search_secret,
)

with image.imports():
    from app.compute.config import ComputeSettings
    from app.compute.main import create_compute_app_for_runtime, create_compute_runtime
    from app.compute.runtime import ComputeRuntime
    from app.compute.telemetry import configure_logging

CANARY_APPLICATION_NAME: Final = "VoiceLightSnapshotCanary"
CANARY_GPU: Final = "A10"
CANARY_ENDPOINT_LABEL: Final = "voicelightagent-voice-light-snapshot-canary"

app = modal.App(CANARY_APPLICATION_NAME)


@app.cls(
    image=image,
    env={"HF_HUB_OFFLINE": "1"},
    gpu=CANARY_GPU,
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
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class VoiceLightSnapshotCanary:
    settings: ComputeSettings
    runtime: ComputeRuntime

    @modal.enter(snap=True)
    def load_and_warm_models(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.settings = ComputeSettings.from_environment(os.environ)
        configure_logging(self.settings.log_directory)
        self.runtime = create_compute_runtime(self.settings)
        asyncio.run(self._load_and_warm_models())

    async def _load_and_warm_models(self) -> None:
        self.runtime.start_loading()
        await self.runtime.wait_until_loading_complete()
        if not self.runtime.voice_ready:
            raise RuntimeError("The complete voice stack must be ready before snapshot capture.")

    @modal.enter(snap=False)
    def verify_restored_models(self) -> None:
        if not self.runtime.voice_ready:
            raise RuntimeError("The restored voice stack is not ready.")

    @modal.asgi_app(label=CANARY_ENDPOINT_LABEL)
    def voice_light(self) -> FastAPI:
        return create_compute_app_for_runtime(self.settings, self.runtime)

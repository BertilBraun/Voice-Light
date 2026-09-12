from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.compute.voice.subprocess_tts import (
    SubprocessSpeechSynthesizer,
    SubprocessTtsConfiguration,
)
from app.compute.voice.worker_device import CudaWorkerDevice

KYUTAI_TTS_PYTHON_PATH: Final = Path(sys.executable)
KYUTAI_TTS_GENERATION_PROGRESS_TIMEOUT_SECONDS: Final = 5.0
KYUTAI_TTS_CANCEL_TIMEOUT_SECONDS: Final = 2.0
KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS: Final = 5.0
KYUTAI_TTS_WORKER_START_TIMEOUT_SECONDS: Final = 180.0
KYUTAI_TTS_WATCHDOG_POLL_SECONDS: Final = 0.1
KYUTAI_TTS_DEFAULT_CODEBOOK_COUNT: Final = 32


@dataclass(frozen=True)
class KyutaiTtsConfiguration:
    python_path: Path = KYUTAI_TTS_PYTHON_PATH
    codebook_count: int = KYUTAI_TTS_DEFAULT_CODEBOOK_COUNT
    cuda_device: CudaWorkerDevice | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.codebook_count <= KYUTAI_TTS_DEFAULT_CODEBOOK_COUNT:
            raise ValueError("Kyutai TTS codebook count must be between 1 and 32.")


DEFAULT_KYUTAI_TTS_CONFIGURATION: Final = KyutaiTtsConfiguration()


class KyutaiSpeechSynthesizer(SubprocessSpeechSynthesizer):
    def __init__(
        self,
        configuration: KyutaiTtsConfiguration | None = None,
    ) -> None:
        if configuration is None:
            configuration = KyutaiTtsConfiguration(
                cuda_device=CudaWorkerDevice.from_environment(
                    os.environ,
                    "VOICE_LIGHT_TTS_CUDA_DEVICE",
                )
            )
        super().__init__(
            SubprocessTtsConfiguration(
                provider_name="Kyutai TTS",
                python_path=configuration.python_path,
                module_name="app.compute.voice.kyutai_tts_worker",
                module_arguments=("--codebook-count", str(configuration.codebook_count)),
                generation_progress_timeout_seconds=(
                    KYUTAI_TTS_GENERATION_PROGRESS_TIMEOUT_SECONDS
                ),
                cancel_timeout_seconds=KYUTAI_TTS_CANCEL_TIMEOUT_SECONDS,
                worker_stop_timeout_seconds=KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS,
                worker_start_timeout_seconds=KYUTAI_TTS_WORKER_START_TIMEOUT_SECONDS,
                watchdog_poll_seconds=KYUTAI_TTS_WATCHDOG_POLL_SECONDS,
                cuda_device=configuration.cuda_device,
            )
        )

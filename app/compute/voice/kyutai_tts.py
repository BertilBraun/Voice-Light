from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from app.compute.voice.subprocess_tts import (
    SubprocessSpeechSynthesizer,
    SubprocessTtsConfiguration,
)

KYUTAI_TTS_PYTHON_PATH: Final = Path(sys.executable)
KYUTAI_TTS_GENERATION_PROGRESS_TIMEOUT_SECONDS: Final = 5.0
KYUTAI_TTS_CANCEL_TIMEOUT_SECONDS: Final = 2.0
KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS: Final = 5.0
KYUTAI_TTS_WORKER_START_TIMEOUT_SECONDS: Final = 180.0
KYUTAI_TTS_WATCHDOG_POLL_SECONDS: Final = 0.1


class KyutaiSpeechSynthesizer(SubprocessSpeechSynthesizer):
    def __init__(self, python_path: Path = KYUTAI_TTS_PYTHON_PATH) -> None:
        super().__init__(
            SubprocessTtsConfiguration(
                provider_name="Kyutai TTS",
                python_path=python_path,
                module_name="app.compute.voice.kyutai_tts_worker",
                module_arguments=(),
                generation_progress_timeout_seconds=(
                    KYUTAI_TTS_GENERATION_PROGRESS_TIMEOUT_SECONDS
                ),
                cancel_timeout_seconds=KYUTAI_TTS_CANCEL_TIMEOUT_SECONDS,
                worker_stop_timeout_seconds=KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS,
                worker_start_timeout_seconds=KYUTAI_TTS_WORKER_START_TIMEOUT_SECONDS,
                watchdog_poll_seconds=KYUTAI_TTS_WATCHDOG_POLL_SECONDS,
            )
        )

from __future__ import annotations

from pathlib import Path

from app.compute.voice.subprocess_tts import (
    SubprocessSpeechSynthesizer,
    SubprocessTtsConfiguration,
)

VOXTREAM_GENERATION_PROGRESS_TIMEOUT_SECONDS = 5.0
VOXTREAM_CANCEL_TIMEOUT_SECONDS = 2.0
VOXTREAM_WORKER_STOP_TIMEOUT_SECONDS = 5.0
VOXTREAM_WORKER_START_TIMEOUT_SECONDS = 300.0
VOXTREAM_WATCHDOG_POLL_SECONDS = 0.1


class VoxtreamSpeechSynthesizer(SubprocessSpeechSynthesizer):
    def __init__(
        self,
        python_path: Path,
        config_path: Path,
        prompt_audio_path: Path,
        compile_model: bool,
        cache_prompt_in_memory: bool,
    ) -> None:
        super().__init__(
            SubprocessTtsConfiguration(
                provider_name="VoXtream2 TTS",
                python_path=python_path,
                module_name="app.compute.voice.voxtream_tts_worker",
                module_arguments=(
                    "--config",
                    str(config_path),
                    "--prompt-audio",
                    str(prompt_audio_path),
                    "--compile" if compile_model else "--no-compile",
                    (
                        "--cache-prompt-in-memory"
                        if cache_prompt_in_memory
                        else "--no-cache-prompt-in-memory"
                    ),
                ),
                generation_progress_timeout_seconds=(VOXTREAM_GENERATION_PROGRESS_TIMEOUT_SECONDS),
                cancel_timeout_seconds=VOXTREAM_CANCEL_TIMEOUT_SECONDS,
                worker_stop_timeout_seconds=VOXTREAM_WORKER_STOP_TIMEOUT_SECONDS,
                worker_start_timeout_seconds=VOXTREAM_WORKER_START_TIMEOUT_SECONDS,
                watchdog_poll_seconds=VOXTREAM_WATCHDOG_POLL_SECONDS,
            )
        )

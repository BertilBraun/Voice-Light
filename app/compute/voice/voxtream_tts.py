from __future__ import annotations

from pathlib import Path

from app.compute.voice.subprocess_tts import (
    SubprocessSpeechSynthesizer,
    SubprocessTtsConfiguration,
)
from app.compute.voice.voxtream_speaking_rate import FinalPhraseSlowdown

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
        speaking_rate_config_path: Path | None,
        final_phrase_slowdown: FinalPhraseSlowdown | None,
    ) -> None:
        super().__init__(
            SubprocessTtsConfiguration(
                provider_name="VoXtream2 TTS",
                python_path=python_path,
                module_name="app.compute.voice.voxtream_tts_worker",
                module_arguments=_voxtream_worker_arguments(
                    config_path=config_path,
                    prompt_audio_path=prompt_audio_path,
                    compile_model=compile_model,
                    cache_prompt_in_memory=cache_prompt_in_memory,
                    speaking_rate_config_path=speaking_rate_config_path,
                    final_phrase_slowdown=final_phrase_slowdown,
                ),
                generation_progress_timeout_seconds=(VOXTREAM_GENERATION_PROGRESS_TIMEOUT_SECONDS),
                cancel_timeout_seconds=VOXTREAM_CANCEL_TIMEOUT_SECONDS,
                worker_stop_timeout_seconds=VOXTREAM_WORKER_STOP_TIMEOUT_SECONDS,
                worker_start_timeout_seconds=VOXTREAM_WORKER_START_TIMEOUT_SECONDS,
                watchdog_poll_seconds=VOXTREAM_WATCHDOG_POLL_SECONDS,
            )
        )


def _voxtream_worker_arguments(
    config_path: Path,
    prompt_audio_path: Path,
    compile_model: bool,
    cache_prompt_in_memory: bool,
    speaking_rate_config_path: Path | None,
    final_phrase_slowdown: FinalPhraseSlowdown | None,
) -> tuple[str, ...]:
    if (speaking_rate_config_path is None) != (final_phrase_slowdown is None):
        raise ValueError(
            "VoXtream speaking-rate configuration and final slowdown must be enabled together."
        )
    module_arguments = [
        "--config",
        str(config_path),
        "--prompt-audio",
        str(prompt_audio_path),
        "--compile" if compile_model else "--no-compile",
        ("--cache-prompt-in-memory" if cache_prompt_in_memory else "--no-cache-prompt-in-memory"),
    ]
    if speaking_rate_config_path is not None and final_phrase_slowdown is not None:
        module_arguments.extend(
            (
                "--speaking-rate-config",
                str(speaking_rate_config_path),
                "--final-slowdown-syllables-per-second",
                str(final_phrase_slowdown.syllables_per_second),
                "--final-slowdown-word-count",
                str(final_phrase_slowdown.word_count),
            )
        )
    return tuple(module_arguments)

from __future__ import annotations

from pathlib import Path

import torch

from app.compute.voice.voxtream_prompt_cache import (
    VoxtreamPreparedPrompt,
    VoxtreamPromptCache,
    VoxtreamPromptCacheKey,
)


def test_voxtream_prompt_cache_prepares_once_and_returns_independent_tensors() -> None:
    cache = VoxtreamPromptCache()
    key = VoxtreamPromptCacheKey(
        prompt_audio_path=Path("voice.wav"),
        enhance_prompt=False,
        apply_vad=False,
    )
    preparation_count = 0

    def prepare() -> VoxtreamPreparedPrompt:
        nonlocal preparation_count
        preparation_count += 1
        return _prepared_prompt(10)

    first = cache.get_or_prepare(key, prepare)
    first[0].add_(100)
    second = cache.get_or_prepare(key, prepare)

    assert preparation_count == 1
    assert first[0].tolist() == [110]
    assert second[0].tolist() == [10]
    assert first[0].data_ptr() != second[0].data_ptr()


def test_voxtream_prompt_cache_replaces_entry_when_prompt_options_change() -> None:
    cache = VoxtreamPromptCache()
    first_key = VoxtreamPromptCacheKey(
        prompt_audio_path=Path("voice.wav"),
        enhance_prompt=False,
        apply_vad=False,
    )
    second_key = VoxtreamPromptCacheKey(
        prompt_audio_path=Path("voice.wav"),
        enhance_prompt=True,
        apply_vad=False,
    )

    first = cache.get_or_prepare(first_key, lambda: _prepared_prompt(10))
    second = cache.get_or_prepare(second_key, lambda: _prepared_prompt(20))

    assert first[0].tolist() == [10]
    assert second[0].tolist() == [20]


def _prepared_prompt(value: int) -> VoxtreamPreparedPrompt:
    return (
        torch.tensor([value]),
        torch.tensor([value + 1]),
        torch.tensor([value + 2]),
        torch.tensor([value + 3]),
        torch.tensor([value + 4]),
    )

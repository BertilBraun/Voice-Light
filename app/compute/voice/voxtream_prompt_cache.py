from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

VoxtreamPreparedPrompt = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class VoxtreamPromptCacheKey:
    prompt_audio_path: Path
    enhance_prompt: bool
    apply_vad: bool


class VoxtreamPromptCache:
    def __init__(self) -> None:
        self._key: VoxtreamPromptCacheKey | None = None
        self._prepared_prompt: VoxtreamPreparedPrompt | None = None

    def get_or_prepare(
        self,
        key: VoxtreamPromptCacheKey,
        prepare: Callable[[], VoxtreamPreparedPrompt],
    ) -> VoxtreamPreparedPrompt:
        if key != self._key:
            self._prepared_prompt = prepare()
            self._key = key
        prepared_prompt = self._prepared_prompt
        assert prepared_prompt is not None
        return tuple(tensor.clone() for tensor in prepared_prompt)

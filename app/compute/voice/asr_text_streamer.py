from __future__ import annotations

from typing import cast

import torch
from transformers import TextIteratorStreamer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class CumulativeAsrTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

    def put(self, value: torch.Tensor) -> None:
        if value.ndim > 1 and value.shape[0] > 1:
            raise ValueError("Cumulative ASR streaming only supports batch size one.")
        if value.ndim > 1:
            value = value[0]
        token_ids = cast(list[int], value.tolist())
        self.token_cache.extend(token_ids)
        cumulative_text = cast(
            str,
            self.tokenizer.decode(
                self.token_cache,
                skip_special_tokens=True,
            ),
        )
        self.on_finalized_text(cumulative_text)

    def end(self) -> None:
        self.next_tokens_are_prompt = True
        self.on_finalized_text("", stream_end=True)

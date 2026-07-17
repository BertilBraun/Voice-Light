from __future__ import annotations

from typing import cast

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from app.compute.voice.asr_text_streamer import CumulativeAsrTextIteratorStreamer


class FragmentTokenizer:
    def __init__(self, fragments: dict[int, str]) -> None:
        self.fragments = fragments

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True
        return "".join(self.fragments[token_id] for token_id in token_ids)


def test_cumulative_streamer_exposes_trailing_word_before_stream_end() -> None:
    tokenizer = cast(
        PreTrainedTokenizerBase,
        FragmentTokenizer(
            {
                1: "What ",
                2: "is ",
                3: "France",
            }
        ),
    )
    streamer = CumulativeAsrTextIteratorStreamer(tokenizer)

    streamer.put(torch.tensor([1, 2]))
    assert next(streamer) == "What is "

    streamer.put(torch.tensor([3]))
    assert next(streamer) == "What is France"

    streamer.end()
    assert next(streamer) == ""

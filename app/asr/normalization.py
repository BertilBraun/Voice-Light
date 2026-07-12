from __future__ import annotations

import re
import unicodedata

from app.asr.transcript import Word

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*-?")


def normalize_token(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", text).lower()
    tokens = TOKEN_PATTERN.findall(lowered)
    if not tokens:
        return ""
    return tokens[0]


def normalized_tokens(words: tuple[Word, ...]) -> list[str]:
    return [token for token in (normalize_token(word.text) for word in words) if token]


def normalized_word_pairs(words: tuple[Word, ...]) -> list[tuple[Word, str]]:
    pairs: list[tuple[Word, str]] = []
    for word in words:
        token = normalize_token(word.text)
        if token:
            pairs.append((word, token))
    return pairs


def is_partial_token(token: str) -> bool:
    return token.endswith("-") and len(token) > 1

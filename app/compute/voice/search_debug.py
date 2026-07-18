from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchDebugResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchDebugTrace:
    query: str
    results: tuple[SearchDebugResult, ...]
    summarizer_system_prompt: str
    summarizer_user_prompt: str
    summary: str
    provider_duration_ms: float
    summarizer_duration_ms: float
    total_duration_ms: float


@dataclass(frozen=True)
class SearchToolOutput:
    result: str
    debug_trace: SearchDebugTrace

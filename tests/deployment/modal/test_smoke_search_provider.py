from __future__ import annotations

import asyncio

import pytest

from app.compute.voice.search import SearchResult
from deployment.modal.smoke_search_provider import (
    configured_tavily_provider,
    measure_search_provider,
)


class RecordingSearchProvider:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.requests: list[tuple[str, int]] = []
        self.closed = False

    async def search(self, query: str, result_limit: int) -> tuple[SearchResult, ...]:
        self.requests.append((query, result_limit))
        return self.results

    async def close(self) -> None:
        self.closed = True


def test_missing_modal_search_secret_fails_with_exact_configuration_name() -> None:
    with pytest.raises(
        ValueError,
        match="voice-light-compute.*VOICE_LIGHT_TAVILY_API_KEY",
    ):
        configured_tavily_provider({}, "voice-light-compute")


def test_search_smoke_reports_only_safe_provider_measurements() -> None:
    provider = RecordingSearchProvider(
        (
            SearchResult(title="Weather", url="https://example.com", snippet="Cloudy"),
            SearchResult(title="Forecast", url="https://example.org", snippet="Cool"),
        )
    )

    result = asyncio.run(measure_search_provider(provider))
    payload = result.model_dump(mode="json")

    assert payload["configured"] is True
    assert payload["result_count"] == 2
    assert float(payload["provider_latency_ms"]) >= 0.0
    assert set(payload) == {"configured", "result_count", "provider_latency_ms"}
    assert provider.requests == [("current weather in London", 2)]
    assert provider.closed is True


def test_search_smoke_fails_when_provider_returns_no_results() -> None:
    provider = RecordingSearchProvider(())

    with pytest.raises(RuntimeError, match="returned no results"):
        asyncio.run(measure_search_provider(provider))

    assert provider.closed is True

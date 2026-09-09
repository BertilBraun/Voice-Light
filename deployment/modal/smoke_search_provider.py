from __future__ import annotations

import time
from collections.abc import Mapping

from pydantic import Field

from app.compute.voice.search import (
    MAXIMUM_SEARCH_RESULTS,
    TAVILY_SEARCH_API_KEY_ENVIRONMENT_VARIABLE,
    SearchProvider,
    TavilySearchProvider,
)
from app.shared.base_model import FrozenBaseModel

DEFAULT_SEARCH_SMOKE_QUERY = "current weather in London"
DEFAULT_SEARCH_SMOKE_RESULT_LIMIT = 2


class SearchProviderSmokeResult(FrozenBaseModel):
    configured: bool
    result_count: int = Field(ge=1, le=MAXIMUM_SEARCH_RESULTS)
    provider_latency_ms: float = Field(ge=0.0)


def configured_tavily_provider(
    environment: Mapping[str, str],
    secret_name: str,
) -> TavilySearchProvider:
    api_key = environment.get(TAVILY_SEARCH_API_KEY_ENVIRONMENT_VARIABLE, "").strip()
    if not api_key:
        raise ValueError(
            f"Modal secret {secret_name!r} must contain "
            f"{TAVILY_SEARCH_API_KEY_ENVIRONMENT_VARIABLE}."
        )
    return TavilySearchProvider(api_key)


async def measure_search_provider(
    provider: SearchProvider,
    query: str = DEFAULT_SEARCH_SMOKE_QUERY,
    result_limit: int = DEFAULT_SEARCH_SMOKE_RESULT_LIMIT,
) -> SearchProviderSmokeResult:
    started_at = time.perf_counter()
    try:
        results = await provider.search(query, result_limit)
    finally:
        await provider.close()
    provider_latency_ms = (time.perf_counter() - started_at) * 1_000
    if not results:
        raise RuntimeError("Tavily search smoke returned no results.")
    return SearchProviderSmokeResult(
        configured=True,
        result_count=len(results),
        provider_latency_ms=provider_latency_ms,
    )

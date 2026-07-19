from __future__ import annotations

import pytest

from deployment.compute.benchmark_qwen_workers import percentile, search_cases


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [
        (0.50, 20.0),
        (0.90, 30.0),
        (0.95, 30.0),
    ],
)
def test_percentile_uses_nearest_rank(quantile: float, expected: float) -> None:
    assert percentile((10.0, 20.0, 30.0), quantile) == expected


def test_percentile_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile((), 0.50)


def test_search_benchmark_cases_are_bounded_and_distinct() -> None:
    cases = search_cases()

    assert len(cases) == 3
    assert len({query for query, _ in cases}) == len(cases)
    assert all(results for _, results in cases)

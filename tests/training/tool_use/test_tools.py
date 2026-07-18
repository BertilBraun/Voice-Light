from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.training.tool_use.tools import (
    calculate_expression,
    seeded_time,
    seeded_time_before_deadline,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("7 * 8", "56"),
        ("(18 * 60 - (14 * 60 + 30)) / 60", "3.5"),
        ("2 ** 8", "256"),
        ("-5 + 2.25", "-2.75"),
    ),
)
def test_calculate_expression(expression: str, expected: str) -> None:
    assert calculate_expression(expression) == expected


@pytest.mark.parametrize(
    "expression",
    (
        "",
        "open('secret')",
        "__import__('os')",
        "value.attribute",
        "2 ** 100",
        "1 / 0",
        "True + 1",
    ),
)
def test_calculate_expression_rejects_unsafe_or_invalid_input(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate_expression(expression)


def test_seeded_time_is_stable_and_timezone_aware() -> None:
    reference_time = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    first = seeded_time(91, reference_time)
    assert first == seeded_time(91, reference_time)
    assert first != seeded_time(92, reference_time)
    assert "T" in first
    assert first[-6] in ("+", "-")

    sampled_times = tuple(
        datetime.fromisoformat(seeded_time(random_seed, reference_time))
        for random_seed in range(200)
    )
    assert len(set(sampled_times)) == len(sampled_times)
    assert all(
        reference_time - timedelta(days=365) <= sample.astimezone(UTC) <= reference_time
        for sample in sampled_times
    )


def test_seeded_time_requires_timezone_aware_reference() -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        seeded_time(91, datetime(2026, 7, 18, 12, 0))


def test_seeded_time_before_deadline_is_varied_and_compatible() -> None:
    reference_time = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    deadline_minutes = 20 * 60 + 30
    sampled_times = tuple(
        datetime.fromisoformat(
            seeded_time_before_deadline(
                random_seed,
                reference_time,
                deadline_minutes,
            )
        )
        for random_seed in range(200)
    )

    assert len(set(sampled_times)) == len(sampled_times)
    assert all(
        15 <= deadline_minutes - (sample.hour * 60 + sample.minute) <= 360
        for sample in sampled_times
    )
    assert all(
        reference_time - timedelta(days=365) <= sample.astimezone(UTC) <= reference_time
        for sample in sampled_times
    )

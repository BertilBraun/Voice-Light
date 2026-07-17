from __future__ import annotations

import pytest

from app.training.tool_use.tools import calculate_expression, seeded_time


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
    assert seeded_time(91) == seeded_time(91)
    assert seeded_time(91) != seeded_time(92)
    assert "T" in seeded_time(91)
    assert seeded_time(91)[-6] in ("+", "-")

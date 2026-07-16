from __future__ import annotations

import pytest

from deployment.compute.benchmark_llm import (
    DEFAULT_SMALL_MODEL_NAME,
    DEFAULT_SMALL_MODEL_REVISION,
    PromptStyle,
    _benchmark_cases,
    _default_revision,
    _system_prompt,
)


def test_benchmark_cases_cover_single_and_multi_turn_conversation() -> None:
    cases = _benchmark_cases()

    assert any(len(benchmark_case.messages) == 1 for benchmark_case in cases)
    assert any(len(benchmark_case.messages) > 1 for benchmark_case in cases)
    assert len({benchmark_case.name for benchmark_case in cases}) == len(cases)


@pytest.mark.parametrize("prompt_style", tuple(PromptStyle))
def test_prompt_styles_are_nonempty(prompt_style: PromptStyle) -> None:
    assert _system_prompt(prompt_style).strip()


def test_small_model_has_a_pinned_default_revision() -> None:
    assert _default_revision(DEFAULT_SMALL_MODEL_NAME) == DEFAULT_SMALL_MODEL_REVISION


def test_unknown_model_requires_an_explicit_revision() -> None:
    with pytest.raises(ValueError, match="--revision is required"):
        _default_revision("example/unknown-model")

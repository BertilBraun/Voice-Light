from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.completion_dataset import SyntheticEnglishSpeechPrompt
from app.local.synthetic_generation.completion_prompts import (
    PromptGeneratorProvenance,
    SyntheticSpeechPromptSet,
    validate_prompt_set_id,
)


def test_prompt_set_rejects_duplicate_text() -> None:
    first = _prompt("prompt_00001")
    second = first.model_copy(update={"prompt_id": "prompt_00002"})

    with pytest.raises(ValidationError, match="texts must be unique"):
        SyntheticSpeechPromptSet(
            set_id="duplicate_test",
            provenance=_provenance(),
            prompts=(first, second),
        )


def test_prompt_requires_long_spoken_text() -> None:
    values = _prompt("prompt_00001").model_dump()
    values["text"] = "This is much too short."
    with pytest.raises(ValidationError, match="at least 45 words"):
        SyntheticEnglishSpeechPrompt.model_validate(values)


def test_prompt_rejects_language_field_because_corpus_is_english_only() -> None:
    values = _prompt("prompt_00001").model_dump()
    values["language"] = "Spanish"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SyntheticEnglishSpeechPrompt.model_validate(values)


def test_prompt_set_id_is_validated_before_generation() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        validate_prompt_set_id("completion-pilot-20260826")

    assert validate_prompt_set_id("completion_pilot_20260826") == ("completion_pilot_20260826")


def _prompt(prompt_id: str) -> SyntheticEnglishSpeechPrompt:
    return SyntheticEnglishSpeechPrompt(
        prompt_id=prompt_id,
        text=(
            "I reviewed the schedule carefully before calling the team, and after comparing "
            "the available trains with the meeting times, I realized we should leave much "
            "earlier than planned. There is still enough time for breakfast near the station, "
            "but we should confirm the platform before everyone arrives tomorrow morning."
        ),
        voice_instruction="Warm adult voice with a natural pace and thoughtful pauses.",
        topic="travel planning",
        seed=7,
    )


def _provenance() -> PromptGeneratorProvenance:
    return PromptGeneratorProvenance(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="a" * 40,
        runtime_version="5.0.0",
        seed=7,
        requested_prompt_count=2,
    )

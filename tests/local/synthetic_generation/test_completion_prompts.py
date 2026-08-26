from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.completion_dataset import SyntheticEnglishSpeechPrompt
from app.local.synthetic_generation.completion_prompts import (
    PromptDeliveryProfile,
    PromptGeneratorProvenance,
    SpeechPromptDraft,
    SyntheticSpeechPromptSet,
    apply_prompt_draft_profile,
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


@pytest.mark.parametrize("prohibited_phrase", ["breathy", "hushed", "whispering"])
def test_prompt_rejects_noise_prone_voice_instruction(prohibited_phrase: str) -> None:
    values = _prompt("prompt_00001").model_dump()
    values["voice_instruction"] = f"Use a {prohibited_phrase} intimate delivery."

    with pytest.raises(ValidationError, match="clean voiced speech"):
        SyntheticEnglishSpeechPrompt.model_validate(values)


def test_prompt_set_id_is_validated_before_generation() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        validate_prompt_set_id("completion-pilot-20260826")

    assert validate_prompt_set_id("completion_pilot_20260826") == ("completion_pilot_20260826")


def test_brisk_profile_rejects_low_energy_draft() -> None:
    draft = SpeechPromptDraft(
        text=" ".join("word" for _ in range(100)),
        voice_instruction="A calm voice speaking at 220 words per minute.",
        topic="an energetic update",
    )

    with pytest.raises(ValueError, match="low-energy phrases"):
        apply_prompt_draft_profile(draft, PromptDeliveryProfile.BRISK_ENGAGED, seed=7)


def test_brisk_profile_accepts_explicit_fast_delivery() -> None:
    draft = SpeechPromptDraft(
        text=" ".join("word" for _ in range(100)),
        voice_instruction="An upbeat projected voice speaking at 225 words per minute.",
        topic="a lively neighborhood event",
    )

    profiled_draft = apply_prompt_draft_profile(
        draft,
        PromptDeliveryProfile.BRISK_ENGAGED,
        seed=7,
    )

    assert profiled_draft == draft


def test_brisk_profile_adds_deterministic_rate() -> None:
    draft = SpeechPromptDraft(
        text=" ".join("word" for _ in range(100)),
        voice_instruction="An upbeat projected voice with lively pacing.",
        topic="a lively neighborhood event",
    )

    profiled_draft = apply_prompt_draft_profile(
        draft,
        PromptDeliveryProfile.BRISK_ENGAGED,
        seed=7,
    )

    assert "207 words per minute" in profiled_draft.voice_instruction


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
        delivery_profile=PromptDeliveryProfile.BALANCED,
    )

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from app.local.synthetic_generation.completion_dataset import SyntheticEnglishSpeechPrompt
from app.local.synthetic_generation.models import SyntheticModel

PromptSetId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
prompt_set_id_adapter = TypeAdapter(PromptSetId)


class PromptDeliveryProfile(StrEnum):
    BALANCED = "balanced"
    BRISK_ENGAGED = "brisk_engaged"


class SpeechPromptDraft(SyntheticModel):
    text: str = Field(min_length=1)
    voice_instruction: str = Field(min_length=1)
    topic: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_word_count(self) -> SpeechPromptDraft:
        if not 55 <= len(self.text.split()) <= 140:
            raise ValueError("Prompt drafts require 55 to 140 words.")
        return self


class SpeechPromptDraftBatch(SyntheticModel):
    prompts: tuple[SpeechPromptDraft, ...] = Field(min_length=1)


class PromptGeneratorProvenance(SyntheticModel):
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str
    seed: int = Field(ge=0)
    requested_prompt_count: int = Field(gt=0)
    delivery_profile: PromptDeliveryProfile


class SyntheticSpeechPromptSet(SyntheticModel):
    schema_version: str = "voice-light-synthetic-speech-prompts-v1"
    set_id: PromptSetId
    provenance: PromptGeneratorProvenance
    prompts: tuple[SyntheticEnglishSpeechPrompt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_prompts(self) -> SyntheticSpeechPromptSet:
        prompt_ids = tuple(prompt.prompt_id for prompt in self.prompts)
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("Synthetic speech prompt IDs must be unique.")
        normalized_texts = tuple(" ".join(prompt.text.lower().split()) for prompt in self.prompts)
        if len(normalized_texts) != len(set(normalized_texts)):
            raise ValueError("Synthetic speech prompt texts must be unique.")
        return self


def apply_prompt_draft_profile(
    draft: SpeechPromptDraft,
    delivery_profile: PromptDeliveryProfile,
    seed: int,
) -> SpeechPromptDraft:
    match delivery_profile:
        case PromptDeliveryProfile.BALANCED:
            if len(draft.text.split()) > 115:
                raise ValueError("Balanced prompt drafts require 55 to 115 words.")
            return draft
        case PromptDeliveryProfile.BRISK_ENGAGED:
            word_count = len(draft.text.split())
            if not 75 <= word_count <= 130:
                raise ValueError("Brisk engaged prompt drafts require 75 to 130 words.")
            normalized_description = f"{draft.topic} {draft.voice_instruction}".casefold()
            prohibited_phrases = (
                "calm",
                "contemplat",
                "deliberate",
                "intimate",
                "quiet",
                "reflect",
                "slow",
                "soft",
                "subdued",
            )
            matches = tuple(
                phrase for phrase in prohibited_phrases if phrase in normalized_description
            )
            if matches:
                raise ValueError(
                    f"Brisk engaged prompt drafts contain low-energy phrases: {', '.join(matches)}."
                )
            maximum_rate_for_twenty_seconds = min(240, word_count * 3)
            rate_match = re.search(
                r"\b(\d{3})(?: words per minute| wpm)\b",
                normalized_description,
            )
            if rate_match is not None:
                words_per_minute = int(rate_match.group(1))
                if not 200 <= words_per_minute <= maximum_rate_for_twenty_seconds:
                    raise ValueError(
                        "Brisk engaged voice instructions require 200 to 240 words per minute "
                        "without planning less than 20 seconds of speech."
                    )
                return draft
            words_per_minute = 200 + seed % (maximum_rate_for_twenty_seconds - 199)
            return draft.model_copy(
                update={
                    "voice_instruction": (
                        f"{draft.voice_instruction.rstrip()} Maintain a consistently brisk rate "
                        f"of {words_per_minute} words per minute without rushing or slurring."
                    )
                }
            )


def validate_prompt_set_id(value: str) -> str:
    return prompt_set_id_adapter.validate_python(value)

from __future__ import annotations

from pydantic import Field, model_validator

from app.local.synthetic_generation.completion_dataset import SyntheticSpeechPrompt
from app.local.synthetic_generation.models import SyntheticModel


class SpeechPromptDraft(SyntheticModel):
    text: str = Field(min_length=1)
    voice_instruction: str = Field(min_length=1)
    topic: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_word_count(self) -> SpeechPromptDraft:
        if not 55 <= len(self.text.split()) <= 115:
            raise ValueError("Prompt drafts require 55 to 115 words.")
        return self


class SpeechPromptDraftBatch(SyntheticModel):
    prompts: tuple[SpeechPromptDraft, ...] = Field(min_length=1)


class PromptGeneratorProvenance(SyntheticModel):
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str
    seed: int = Field(ge=0)
    requested_prompt_count: int = Field(gt=0)


class SyntheticSpeechPromptSet(SyntheticModel):
    schema_version: str = "voice-light-synthetic-speech-prompts-v1"
    set_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    provenance: PromptGeneratorProvenance
    prompts: tuple[SyntheticSpeechPrompt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_prompts(self) -> SyntheticSpeechPromptSet:
        prompt_ids = tuple(prompt.prompt_id for prompt in self.prompts)
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("Synthetic speech prompt IDs must be unique.")
        normalized_texts = tuple(" ".join(prompt.text.lower().split()) for prompt in self.prompts)
        if len(normalized_texts) != len(set(normalized_texts)):
            raise ValueError("Synthetic speech prompt texts must be unique.")
        return self

"""Schema models for synthetic conversation specifications.

These models describe intended conversations only. They do not generate, store,
or transmit audio or text.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel


class SpeakerSpecification(FrozenBaseModel):
    """Content and delivery direction for one synthetic speaker."""

    speaker_id: str = Field(min_length=1, pattern=r"^speaker_[1-9][0-9]*$")
    text: str = Field(min_length=1)
    delivery_instruction: str | None = Field(default=None, min_length=1)


class BackchannelPlacement(FrozenBaseModel):
    """Place a brief listener response after an anchor phrase."""

    kind: Literal["backchannel"] = "backchannel"
    anchor_text: str = Field(min_length=1)
    delay_milliseconds: int = Field(default=300, ge=0)


class InterruptionPlacement(FrozenBaseModel):
    """Place a response in relation to an anchor phrase."""

    kind: Literal["interruption"] = "interruption"
    anchor_text: str = Field(min_length=1)
    position: Literal["before_last_word", "at_anchor_start", "at_anchor_end"] = "before_last_word"
    lead_milliseconds: int = Field(default=50, ge=0)


class CompletionPlacement(FrozenBaseModel):
    """Place a response after the first speaker completes a turn."""

    kind: Literal["completion"] = "completion"
    pause: Literal["short", "medium", "long"] = "short"
    minimum_delay_milliseconds: int | None = Field(default=None, ge=0)
    maximum_delay_milliseconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_delay_range(self) -> CompletionPlacement:
        if (self.minimum_delay_milliseconds is None) != (self.maximum_delay_milliseconds is None):
            raise ValueError("Set both delay bounds or neither delay bound.")
        if (
            self.minimum_delay_milliseconds is not None
            and self.maximum_delay_milliseconds is not None
            and self.minimum_delay_milliseconds > self.maximum_delay_milliseconds
        ):
            raise ValueError(
                "minimum_delay_milliseconds must not exceed maximum_delay_milliseconds."
            )
        return self


class InternalPausePlacement(FrozenBaseModel):
    """Select an internal pause in the first speaker's turn."""

    kind: Literal["internal_pause"] = "internal_pause"
    anchor_text: str = Field(min_length=1)
    pause: Literal["short", "medium", "long"] = "medium"


Placement = Annotated[
    BackchannelPlacement | InterruptionPlacement | CompletionPlacement | InternalPausePlacement,
    Field(discriminator="kind"),
]


class SyntheticConversationCase(FrozenBaseModel):
    """A two-speaker conversation case with an optional placement objective."""

    case_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    first_speaker: SpeakerSpecification
    second_speaker: SpeakerSpecification
    placement: Placement | None = None
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_distinct_speakers(self) -> SyntheticConversationCase:
        if self.first_speaker.speaker_id == self.second_speaker.speaker_id:
            raise ValueError("Conversation speakers must have distinct speaker_id values.")
        return self


class SyntheticConversationSet(FrozenBaseModel):
    """A named collection of uniquely identified synthetic conversation cases."""

    set_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    cases: tuple[SyntheticConversationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> SyntheticConversationSet:
        case_ids = tuple(conversation_case.case_id for conversation_case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Synthetic conversation case IDs must be unique.")
        return self

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from app.shared.base_model import FrozenBaseModel


class SyntheticModel(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpeakerRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class TurnBoundary(StrEnum):
    NONE = "none"
    CONTINUATION = "continuation"
    COMPLETION = "completion"


class InterruptionOutcome(StrEnum):
    ATTEMPT = "attempt"
    FLOOR_TAKE = "floor_take"


class SpeakerSpecification(SyntheticModel):
    speaker_id: str = Field(min_length=1, pattern=r"^speaker_[1-9][0-9]*$")
    role: SpeakerRole
    voice_id: str = Field(min_length=1)
    language: str = Field(min_length=2)
    voice_prompt_path: Path | None = None


class TimedEvent(SyntheticModel):
    event_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    speaker_id: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> TimedEvent:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Timeline event end must follow its start.")
        return self


class SpeechEvent(TimedEvent):
    kind: Literal["speech"] = "speech"
    text: str = Field(min_length=1)
    boundary_after: TurnBoundary = TurnBoundary.NONE
    delivery_instruction: str | None = Field(default=None, min_length=1)
    contains_disfluency: bool = False


class PauseEvent(TimedEvent):
    kind: Literal["pause"] = "pause"
    continuation_event_id: str


class BackchannelEvent(TimedEvent):
    kind: Literal["backchannel"] = "backchannel"
    text: str = Field(min_length=1)
    delivery_instruction: str | None = Field(default=None, min_length=1)


class InterruptionEvent(TimedEvent):
    kind: Literal["interruption"] = "interruption"
    text: str = Field(min_length=1)
    interrupted_event_id: str
    outcome: InterruptionOutcome
    delivery_instruction: str | None = Field(default=None, min_length=1)


TimelineEvent = Annotated[
    SpeechEvent | PauseEvent | BackchannelEvent | InterruptionEvent,
    Field(discriminator="kind"),
]


class SyntheticConversationPlan(SyntheticModel):
    schema_version: Literal["voice-light-synthetic-plan-v1"] = "voice-light-synthetic-plan-v1"
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    seed: int = Field(ge=0)
    duration_seconds: float = Field(gt=0.0)
    speakers: tuple[SpeakerSpecification, SpeakerSpecification]
    events: tuple[TimelineEvent, ...] = Field(min_length=1)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_timeline(self) -> SyntheticConversationPlan:
        roles = {speaker.role for speaker in self.speakers}
        if roles != {SpeakerRole.USER, SpeakerRole.ASSISTANT}:
            raise ValueError("A plan requires one user and one assistant speaker.")
        speaker_ids = {speaker.speaker_id for speaker in self.speakers}
        if len(speaker_ids) != 2:
            raise ValueError("A plan requires two distinct speaker IDs.")
        event_ids = {event.event_id for event in self.events}
        if len(event_ids) != len(self.events):
            raise ValueError("Timeline event IDs must be unique.")
        for event in self.events:
            if event.speaker_id not in speaker_ids:
                raise ValueError(f"Unknown speaker ID in event {event.event_id}.")
            if event.end_seconds > self.duration_seconds:
                raise ValueError(f"Event {event.event_id} exceeds the plan duration.")
            match event:
                case PauseEvent(continuation_event_id=continuation_event_id):
                    if continuation_event_id not in event_ids:
                        raise ValueError(
                            f"Pause {event.event_id} has an unknown continuation event."
                        )
                case InterruptionEvent(interrupted_event_id=interrupted_event_id):
                    if interrupted_event_id not in event_ids:
                        raise ValueError(
                            f"Interruption {event.event_id} has an unknown target event."
                        )
                case _:
                    pass
        return self

    def speaker_for_role(self, role: SpeakerRole) -> SpeakerSpecification:
        return next(speaker for speaker in self.speakers if speaker.role is role)


# Compatibility specifications retained for the imported exploratory generator.
class SpeakerPrompt(SyntheticModel):
    speaker_id: str = Field(min_length=1, pattern=r"^speaker_[1-9][0-9]*$")
    text: str = Field(min_length=1)
    delivery_instruction: str | None = Field(default=None, min_length=1)


class CompletionPlacement(SyntheticModel):
    kind: Literal["completion"] = "completion"
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


class SyntheticConversationCase(SyntheticModel):
    case_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    first_speaker: SpeakerPrompt
    second_speaker: SpeakerPrompt
    placement: CompletionPlacement | None = None
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_distinct_speakers(self) -> SyntheticConversationCase:
        if self.first_speaker.speaker_id == self.second_speaker.speaker_id:
            raise ValueError("Conversation speakers must have distinct speaker_id values.")
        return self


class SyntheticConversationSet(SyntheticModel):
    set_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    cases: tuple[SyntheticConversationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> SyntheticConversationSet:
        case_ids = tuple(conversation_case.case_id for conversation_case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Synthetic conversation case IDs must be unique.")
        return self

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from app.training.tool_use.scenario import ToolName
from app.training.tool_use.schema import SpeechStyle, ToolUseBaseModel


class TeacherChatMessage(ToolUseBaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GeneratedUserTurn(ToolUseBaseModel):
    text: str = Field(min_length=1, max_length=500)
    speech_style: SpeechStyle


class SearchCall(ToolUseBaseModel):
    kind: Literal["search"] = "search"
    query: str = Field(min_length=1, max_length=240)


class CalculateCall(ToolUseBaseModel):
    kind: Literal["calculate"] = "calculate"
    expression: str = Field(min_length=1, max_length=160)


class GetTimeCall(ToolUseBaseModel):
    kind: Literal["get_time"] = "get_time"


GeneratedToolCall: TypeAlias = Annotated[
    SearchCall | CalculateCall | GetTimeCall,
    Field(discriminator="kind"),
]


class FinalResponseStep(ToolUseBaseModel):
    kind: Literal["final_response"] = "final_response"
    audible_text: str = Field(min_length=1, max_length=600)


class ToolActionStep(ToolUseBaseModel):
    kind: Literal["tool_action"] = "tool_action"
    audible_text: str = Field(min_length=1, max_length=120)
    call: GeneratedToolCall


AssistantStep: TypeAlias = Annotated[
    FinalResponseStep | ToolActionStep,
    Field(discriminator="kind"),
]


class AssistantStepEnvelope(ToolUseBaseModel):
    step: AssistantStep


class FinalResponseStepEnvelope(ToolUseBaseModel):
    step: FinalResponseStep


class SyntheticSearchResult(ToolUseBaseModel):
    content: str = Field(min_length=1, max_length=1200)


class GeneratedUserTurnEnvelope(ToolUseBaseModel):
    turn: GeneratedUserTurn


class SyntheticSearchResultEnvelope(ToolUseBaseModel):
    result: SyntheticSearchResult


def generated_call_tool_name(call: GeneratedToolCall) -> ToolName:
    match call:
        case SearchCall():
            return ToolName.SEARCH
        case CalculateCall():
            return ToolName.CALCULATE
        case GetTimeCall():
            return ToolName.GET_TIME


class AssistantStepValidation(ToolUseBaseModel):
    expected_tool_name: ToolName | None
    maximum_bridge_words: int = Field(default=8, ge=1)

    def validate_step(self, step: AssistantStep) -> None:
        match (self.expected_tool_name, step):
            case (None, FinalResponseStep()):
                return
            case (None, ToolActionStep()):
                raise ValueError("Teacher emitted a tool call when a final response was required.")
            case (ToolName() as expected_tool_name, ToolActionStep(call=call)):
                actual_tool_name = generated_call_tool_name(call)
                if actual_tool_name is not expected_tool_name:
                    raise ValueError(
                        f"Expected {expected_tool_name.value}, got {actual_tool_name.value}."
                    )
                word_count = len(step.audible_text.split())
                if word_count > self.maximum_bridge_words:
                    raise ValueError(
                        f"Tool bridge has {word_count} words; maximum is "
                        f"{self.maximum_bridge_words}."
                    )
            case (ToolName() as expected_tool_name, FinalResponseStep()):
                raise ValueError(
                    f"Teacher ended the response before required {expected_tool_name.value} call."
                )

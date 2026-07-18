from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from app.training.tool_use.scenario import ToolName
from app.training.tool_use.schema import SpeechStyle, ToolUseBaseModel


class TeacherChatMessage(ToolUseBaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class TeacherSamplingMode(StrEnum):
    CREATIVE = "creative"
    DETERMINISTIC = "deterministic"


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


class UserTurnScopeAssessment(ToolUseBaseModel):
    required_tools: tuple[ToolName, ...]
    scope_is_clear: bool
    request_is_supported: bool
    repeats_prior_user_turn: bool
    voice_style_is_natural: bool


class UserTurnScopeAssessmentEnvelope(ToolUseBaseModel):
    assessment: UserTurnScopeAssessment


class SyntheticSearchResultEnvelope(ToolUseBaseModel):
    result: SyntheticSearchResult


class RecordQualityIssue(StrEnum):
    UNPLANNED_TOOL_NEED = "unplanned_tool_need"
    PREMATURE_TOOL_RESULT = "premature_tool_result"
    UNGROUNDED_ASSISTANT_CLAIM = "ungrounded_assistant_claim"
    INCORRECT_RESULT_USE = "incorrect_result_use"
    DUPLICATE_USER_TURN = "duplicate_user_turn"
    REDUNDANT_TOOL_CALL = "redundant_tool_call"
    UNSUPPORTED_ACTION = "unsupported_action"
    UNSUPPORTED_PERSONA = "unsupported_persona"
    INCOHERENT_CONVERSATION = "incoherent_conversation"
    UNNATURAL_VOICE_STYLE = "unnatural_voice_style"
    INAPPROPRIATE_TONE = "inappropriate_tone"


class RecordQualityAssessment(ToolUseBaseModel):
    tool_plan_is_sufficient: bool
    no_premature_tool_results: bool
    assistant_claims_are_grounded: bool
    tool_results_are_used_correctly: bool
    user_turns_are_distinct: bool
    sequential_calls_are_justified: bool
    no_unsupported_action_commitments: bool
    conversation_is_coherent: bool
    voice_style_is_natural: bool
    tone_is_appropriate: bool
    issues: tuple[RecordQualityIssue, ...]


class RecordQualityAssessmentEnvelope(ToolUseBaseModel):
    assessment: RecordQualityAssessment


class GroundingVerdict(StrEnum):
    GROUNDED = "grounded"
    UNSUPPORTED = "unsupported"


class AssistantGroundingFinding(ToolUseBaseModel):
    message_id: str
    verdict: GroundingVerdict
    unsupported_excerpts: tuple[str, ...]


class RecordGroundingAssessment(ToolUseBaseModel):
    findings: tuple[AssistantGroundingFinding, ...]


class RecordGroundingAssessmentEnvelope(ToolUseBaseModel):
    assessment: RecordGroundingAssessment


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

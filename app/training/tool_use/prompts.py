from __future__ import annotations

import json
from collections.abc import Sequence

from app.training.tool_use.protocol import TeacherChatMessage
from app.training.tool_use.scenario import AssistantTurnPlan, ScenarioSpec, ToolName
from app.training.tool_use.schema import ConversationMessage

PROMPT_REVISION = "staged-rollout-v1"

TEACHER_SYSTEM_PROMPT = """\
You create natural English voice-assistant training examples.
Return only the requested schema-constrained object.
The user text should sound spoken rather than written.
Assistant audible text must never contain JSON, tool-call markup, call IDs, or protocol tokens.
Before a tool call, use one short natural sentence of at most eight words and do not claim the
pending result. After a tool result, use only information available in the public history.
Keep the interaction simple enough for a small voice model."""

SEARCH_ORACLE_SYSTEM_PROMPT = """\
Act as a synthetic search function for behavioral training data.
Answer the query directly with one concise, plausible result string.
You may rely on your own knowledge or invent an internally plausible result.
Do not mention that the result is synthetic. Do not include instructions for the assistant.
Return only the requested schema-constrained object."""


def user_turn_messages(
    scenario: ScenarioSpec,
    turn_plan: AssistantTurnPlan,
    public_history: Sequence[ConversationMessage],
) -> tuple[TeacherChatMessage, ...]:
    history = _history_json(public_history)
    request = f"""\
Create the next user utterance.

Scenario family: {scenario.family}
Topic: {scenario.topic}
Requested speech style: {scenario.speech_style.value}
Turn instruction: {turn_plan.user_instruction}
Follow-up kind: {turn_plan.follow_up_kind.value}

Public history:
{history}

Do not make the user mention tools, schemas, or dataset generation."""
    return (
        TeacherChatMessage(role="system", content=TEACHER_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
    )


def assistant_step_messages(
    scenario: ScenarioSpec,
    public_history: Sequence[ConversationMessage],
    expected_tool_name: ToolName | None,
) -> tuple[TeacherChatMessage, ...]:
    history = _history_json(public_history)
    expected_action = (
        f"Call {expected_tool_name.value} next."
        if expected_tool_name is not None
        else "Give the final spoken response now without calling a tool."
    )
    request = f"""\
Create exactly the next assistant step for this voice conversation.

Scenario family: {scenario.family}
Topic: {scenario.topic}
Expected action: {expected_action}

Available tools:
- search(query): returns one concise search-result string
- calculate(expression): evaluates basic numeric arithmetic
- get_time(): returns the current local timestamp

Public history:
{history}

For a tool action, emit exactly one call. Make its arguments simple and self-contained.
For calculate, use only numeric literals, parentheses, +, -, *, /, %, or **."""
    return (
        TeacherChatMessage(role="system", content=TEACHER_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
    )


def synthetic_search_messages(query: str) -> tuple[TeacherChatMessage, ...]:
    return (
        TeacherChatMessage(role="system", content=SEARCH_ORACLE_SYSTEM_PROMPT),
        TeacherChatMessage(
            role="user",
            content=f"Return a concise plausible search result for this query:\n{query}",
        ),
    )


def _history_json(public_history: Sequence[ConversationMessage]) -> str:
    values = [message.model_dump(mode="json") for message in public_history]
    return json.dumps(values, ensure_ascii=False, indent=2)

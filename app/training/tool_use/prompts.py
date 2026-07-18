from __future__ import annotations

import json
from collections.abc import Sequence

from app.training.tool_use.protocol import TeacherChatMessage
from app.training.tool_use.scenario import (
    AssistantTurnPlan,
    ScenarioSpec,
    ToolName,
    UtteranceForm,
)
from app.training.tool_use.schema import ConversationMessage, ToolResultMessage

PROMPT_REVISION = "natural-staged-rollout-v2"

TEACHER_SYSTEM_PROMPT = """\
You create natural English voice-assistant training examples.
Return only the requested schema-constrained object.
User text should sound like one moment in a real conversation, not a benchmark question. Use
contractions, fragments, reactions, context, and occasional self-correction where requested.
Do not make every user turn a grammatical question.
Assistant audible text must never contain JSON, tool-call markup, call IDs, or protocol tokens.
Before a tool call, use at most eight natural spoken words and do not claim the pending result.
Avoid stock service phrases such as "Let me check that for you", "I would be happy to", and "Now I
will". For sequential calls, connect the returned fact to the next action rather than restarting.
After a tool result, use only information available in the public history. A timeout, failure, or
empty result supplies no answer: acknowledge it briefly and never infer the missing result.
Match the user's register. Prefer contractions and concise conversational replies. Do not restate
the whole request or end with a generic offer to help. Keep the interaction simple enough for a
small voice model."""

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
Requested utterance form: {turn_plan.utterance_form.value}
Utterance-form guidance: {_utterance_form_guidance(turn_plan.utterance_form)}
Turn instruction: {turn_plan.user_instruction}
Follow-up kind: {turn_plan.follow_up_kind.value}

Public history:
{history}

Do not make the user mention tools, schemas, or dataset generation. Do not turn a fragment,
reaction, contextual statement, or self-repair into a polished standalone question."""
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
    transition_guidance = _assistant_transition_guidance(
        public_history=public_history,
        expected_tool_name=expected_tool_name,
    )
    request = f"""\
Create exactly the next assistant step for this voice conversation.

Scenario family: {scenario.family}
Topic: {scenario.topic}
Expected action: {expected_action}
Style for this step: {transition_guidance}

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


def synthetic_search_messages(
    query: str,
    public_history: Sequence[ConversationMessage],
) -> tuple[TeacherChatMessage, ...]:
    return (
        TeacherChatMessage(role="system", content=SEARCH_ORACLE_SYSTEM_PROMPT),
        TeacherChatMessage(
            role="user",
            content=(
                "Return a concise plausible search result for this query. Stay consistent with "
                f"the supplied conversation and earlier results.\n\nQuery:\n{query}\n\n"
                f"Conversation context:\n{_history_json(public_history)}"
            ),
        ),
    )


def _utterance_form_guidance(utterance_form: UtteranceForm) -> str:
    match utterance_form:
        case UtteranceForm.DIRECT_QUESTION:
            return "Use one concise spoken question, without formal wording."
        case UtteranceForm.REQUEST:
            return "Phrase it as a casual request or imperative, not a question."
        case UtteranceForm.CONTEXT_FIRST:
            return "Lead with a personal detail or situation, then continue naturally."
        case UtteranceForm.FRAGMENT:
            return "Use a clear conversational fragment or elliptical continuation."
        case UtteranceForm.REACTION:
            return "Begin with a brief genuine reaction, then continue the thought."
        case UtteranceForm.SELF_REPAIR:
            return "Include a natural false start, correction, or mid-utterance repair."


def _assistant_transition_guidance(
    public_history: Sequence[ConversationMessage],
    expected_tool_name: ToolName | None,
) -> str:
    if expected_tool_name is None:
        return (
            "Reply like a conversational partner, usually in one short sentence. Do not use a "
            "customer-service preamble or closing."
        )
    if public_history:
        match public_history[-1]:
            case ToolResultMessage():
                return (
                    "This is a sequential call. Briefly connect the returned fact to the next "
                    "action. A short fragment or contraction is good. Do not restart with "
                    "'Let me' or 'Now I will'."
                )
            case _:
                pass
    return (
        "Use a brief acknowledgement or conversational transition. Avoid narrating your role and "
        "avoid 'Let me check that for you'."
    )


def _history_json(public_history: Sequence[ConversationMessage]) -> str:
    values = [message.model_dump(mode="json") for message in public_history]
    return json.dumps(values, ensure_ascii=False, indent=2)

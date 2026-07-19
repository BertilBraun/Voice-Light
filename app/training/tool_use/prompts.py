from __future__ import annotations

import json
from collections.abc import Sequence

from app.training.tool_use.protocol import TeacherChatMessage
from app.training.tool_use.scenario import (
    AssistantResponseMode,
    AssistantTurnPlan,
    ScenarioGenerationMode,
    ScenarioSpec,
    ToolName,
    UtteranceForm,
)
from app.training.tool_use.schema import (
    AssistantMessage,
    ConversationMessage,
    ToolResultMessage,
    UserMessage,
)

PROMPT_REVISION = "teacher-led-conversations-v20"

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
After calculate, state the exact returned digits directly. Do not convert units or perform another
mental calculation unless the exact result is also stated and the conversion is explicitly
grounded.
Match the user's register. Prefer contractions and concise conversational replies. Do not restate
the whole request or end with a generic offer to help. Keep the interaction simple enough for a
small voice model. The assistant has no body, possessions, location, personal tastes, desires, or
social plans. It may discuss the user's preferences but must not pretend it can drink coffee,
attend an event, own something, or join a physical activity. Every user utterance and assistant
response must stay under 100 spoken words. Final replies should usually be one to three short
sentences.
Do not blindly validate procrastination, giving up, unhealthy choices, or other counterproductive
behavior. Acknowledge how the user feels, then prefer a healthy, responsible option or a small
achievable compromise. Stay practical and warm without shaming, lecturing, or turning ordinary
choices into a safety warning."""

SEARCH_ORACLE_SYSTEM_PROMPT = """\
Act as a synthetic search function for behavioral training data.
Answer only the exact query with one concise, plausible result string.
You may rely on your own knowledge or invent an internally plausible result.
Correct a false premise or category mismatch explicitly. For example, if a query asks who sang an
instrumental theme, say that it has no singer and identify the composer only if useful.
Do not add adjacent facts, answer a broader conversation request, or anticipate a later search or
calculation. If the query requests one number, return that number with only enough context to
identify it.
Do not mention that the result is synthetic. Do not include instructions for the assistant.
Return only the requested schema-constrained object."""

QUALITY_AUDITOR_SYSTEM_PROMPT = """\
Audit a synthetic voice-tool conversation conservatively. Do not rewrite it.
Reject any record where an ordinary user turn assigned no tool actually requires current,
external, or calculated information. A search clarification turn is the narrow exception: the
assistant must ask the planned question without calling, and the user's answer must trigger the
search immediately. Also reject where the assistant states a pending result before its
tool result; where an assistant claim is unsupported by earlier user text or tool results; where a
result is misread, transformed incorrectly, or contradicted; where a user turn repeats; or where a
sequential call is redundant because an earlier result already answered it.
Reject promises or claims to play media, book, buy, message, navigate, control a device, or perform
another action that no recorded tool actually executed. This applies only to assistant claims,
not to a user's own plans, an acknowledgement, or a suggestion. Reject role/category changes such
as turning a composer into a singer or a count into a total. Reject claims that time is sufficient,
someone is on time, or a deadline works when the necessary departure, appointment, or duration is
missing.
Reject assistant claims of physical participation, possessions, location, personal tastes,
desires, or social plans. For example, the assistant must not say it wants coffee or will join the
user. Label this unsupported_persona.
Treat prices, schedules, opening hours, current people or roles, local events, availability, and
the present date or time as unsupported unless a preceding user message or tool result supplies
them. Only the stable_knowledge family may answer one simple timeless common-knowledge question
from model knowledge; that exception never covers changing or location-specific facts.
Also reject broken conversational logic, rude or dismissive wording, and clearly assistant-like or
benchmark-like dialogue. Reject openings that provide no clear assistant task but are answered as
if a hidden list, route plan, playlist, queue, or earlier conversation exists.
Reject an answer that ignores an explicit requested count or returns several alternatives when
the user requested exactly one. Reject tool bridges phrased as questions or requests for missing
information while simultaneously issuing a call.
Reject materially underdeveloped answers to open-ended advice, editing, explanation, or
brainstorming requests. Concise factual answers are valid and should not be padded.
Reject advice that blindly endorses procrastination, giving up, an explicitly unhealthy choice,
or another counterproductive behavior when a practical healthier or responsible option is clear.
A brief rest or treat can be reasonable, but the assistant should offer a balanced choice or small
achievable step instead of repeatedly encouraging avoidance. Label this counterproductive_advice.
Set every quality boolean independently and list every applicable issue. Use ungrounded_claim when
assistant_claims_are_grounded is false; use unsupported_action only for an assistant claiming or
promising an unavailable external action. Return only the requested schema-constrained object."""

GROUNDING_AUDITOR_SYSTEM_PROMPT = """\
Audit every assistant message in a synthetic conversation separately.
For each assistant message ID, decide whether all of its audible content is grounded. Return one
finding for every assistant message, in conversation order, without omitting bridges.

Grounding sources are limited to earlier user messages and earlier completed tool results. A tool
result is evidence for this behavioral dataset; do not fact-check the tool result against your own
knowledge. The stable_knowledge family has one narrow exception for its first answer: one simple,
timeless common-knowledge fact may come from model knowledge.

Grounded content includes brief acknowledgements, creative text explicitly requested by the user,
opinions based only on user-supplied tradeoffs, and exact restatements or filtering of prior
results. Unsupported content includes any new external fact, comparative claim, schedule, role,
event detail, product or place claim, or confirmation not present in the sources. It also includes
pretending a hidden list, route plan, playlist, queue, or external action exists. Examples:
- "Buses are less packed" is unsupported unless an earlier source says so.
- Confirming a famous play detail is unsupported if the search result did not mention that detail.
- "Direct routes only" can acknowledge a stated drafting constraint, but "I'll update the routes"
  is unsupported unless a tool actually did that.

Also mark a message unsupported when it claims to satisfy the latest instruction but violates an
explicit output count or constraint. A result containing two requested items does not authorize a
third item.

Set verdict to unsupported and copy only the shortest offending audible excerpts when any part is
unsupported. Otherwise use grounded with no excerpts. Return only the schema-constrained object."""

USER_SCOPE_AUDITOR_SYSTEM_PROMPT = """\
Classify what information a candidate user utterance actually requires from the assistant.
Use search for current, changing, local, private, or externally verified facts not already in the
public history. Stable common knowledge does not require search. Use calculate for requested exact
arithmetic. Use get_time only when the present date or time is needed and not supplied by the user.
Numeric comparisons, threshold checks, totals, and unit conversions require calculate even when
their input number is already in the public history.
Any requested elapsed time, remaining time, duration, or date difference derived from the current
timestamp requires get_time followed by calculate. Any requested arithmetic using a number that
must first be searched requires search followed by calculate.
Interpret ISO timestamps in 24-hour time: hour 23 is 11 PM, never 11 AM.
List repeated tools when separate sequential calls are genuinely required. A dependent two-search
request should list search twice; two facts answerable by one query should list it once.
Judge the candidate itself, not the intended scenario. Mark scope unclear when essential operands,
references, deadlines, locations, or requested facts are missing. Detect repetition against prior
user turns. A remaining-time request with a twelve-hour deadline is unclear unless AM or PM is
explicit. Mark request_is_supported false only when the user asks the assistant to perform an
action outside search, calculate, and get_time, including media playback, booking, purchasing,
messaging, navigation, device control, and file operations. A statement about the user's own
plans is supported. Return only the requested schema-constrained object."""

TEACHER_LED_SYSTEM_PROMPT = """\
Create one step of a coherent, natural English voice-assistant conversation.
The conversation should feel improvised rather than like a benchmark or a scripted demonstration.
Maintain the subject, entities, quantities, corrections, and pronoun references established in
the audible history. Short fragments, reactions, and self-corrections are welcome when natural.
Do not mention datasets, prompts, schemas, tools, function calls, or synthetic generation.
Return only the requested schema-constrained object."""


def user_turn_messages(
    scenario: ScenarioSpec,
    turn_plan: AssistantTurnPlan,
    public_history: Sequence[ConversationMessage],
) -> tuple[TeacherChatMessage, ...]:
    history = _history_json(public_history)
    planned_tools = tuple(step.tool_name.value for step in turn_plan.tool_steps)
    if turn_plan.response_mode is AssistantResponseMode.CLARIFY_SEARCH:
        tool_scope = (
            "The user wants an external lookup but must leave exactly one essential detail "
            "ambiguous. The request should ultimately require search, but the assistant must ask "
            "one clarifying question before any call."
        )
    elif planned_tools:
        tool_scope = (
            "This user turn must naturally require exactly these tools, in order: "
            f"{', '.join(planned_tools)}. Do not introduce any additional information need."
        )
    elif scenario.family == "stable_knowledge" and not public_history:
        tool_scope = (
            "No tool call is allowed. Ask at most one simple timeless common-knowledge question. "
            "Do not ask about anything current, changing, local, privately held, or uncertain."
        )
    else:
        tool_scope = (
            "No tool call is allowed after this user turn. The utterance must be answerable from "
            "existing public history or ordinary non-factual conversation. It must not ask for "
            "current time, dates, recent information, prices, schedules, opening hours, "
            "availability, calculations, numeric comparisons, unit conversions, or facts absent "
            "from the history. Do not ask whether a number exceeds a threshold or restate a "
            "quantity in different units."
        )
    request = f"""\
Create the next user utterance.

Scenario family: {scenario.family}
Topic: {scenario.topic}
Requested speech style: {scenario.speech_style.value}
Requested utterance form: {turn_plan.utterance_form.value}
Utterance-form guidance: {_utterance_form_guidance(turn_plan.utterance_form)}
Turn instruction: {turn_plan.user_instruction}
Follow-up kind: {turn_plan.follow_up_kind.value}
Tool scope: {tool_scope}

Public history:
{history}

Do not make the user mention tools, schemas, or dataset generation. Do not turn a fragment,
reaction, contextual statement, or self-repair into a polished standalone question.
Never repeat or closely paraphrase an earlier user utterance."""
    return (
        TeacherChatMessage(role="system", content=TEACHER_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
    )


def assistant_step_messages(
    scenario: ScenarioSpec,
    public_history: Sequence[ConversationMessage],
    expected_tool_name: ToolName | None,
    later_tool_names: Sequence[ToolName] = (),
    response_mode: AssistantResponseMode = AssistantResponseMode.ANSWER,
) -> tuple[TeacherChatMessage, ...]:
    history = _history_json(public_history)
    match response_mode:
        case AssistantResponseMode.CLARIFY_SEARCH:
            expected_action = (
                "Ask exactly one concise question for the missing search constraint. Do not call "
                "a tool or answer the lookup."
            )
        case AssistantResponseMode.ANSWER:
            expected_action = (
                f"Call {expected_tool_name.value} next."
                if expected_tool_name is not None
                else "Give the final spoken response now without calling a tool."
            )
    transition_guidance = _assistant_transition_guidance(
        public_history=public_history,
        expected_tool_name=expected_tool_name,
        scenario_family=scenario.family,
        response_mode=response_mode,
    )
    sequence_guidance = _tool_sequence_guidance(
        expected_tool_name,
        later_tool_names,
        response_mode,
    )
    request = f"""\
Create exactly the next assistant step for this voice conversation.

Scenario family: {scenario.family}
Topic: {scenario.topic}
Expected action: {expected_action}
Style for this step: {transition_guidance}
Tool-sequence constraint: {sequence_guidance}

Available tools:
- search(query): returns one concise search-result string
- calculate(expression): evaluates basic numeric arithmetic
- get_time(): returns the current local timestamp

Public history:
{history}

For a tool action, emit exactly one call. Make its arguments simple and self-contained.
For calculate, use only numeric literals, parentheses, +, -, *, /, %, or **.
Never put the pending result in audible_text, even when you already know how to answer.
For a final response, every factual claim must be supported by the user text or a preceding tool
result. Do not fill missing current facts from your own knowledge. If the latest result is from
calculate, say its exact returned digits directly and do not silently convert it. ISO timestamps
use 24-hour time. When calculating minutes to a deadline, preserve PM hours and subtract the
timestamp's hour and minute from the deadline's hour and minute."""
    return (
        TeacherChatMessage(role="system", content=TEACHER_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
    )


def teacher_led_user_turn_messages(
    scenario: ScenarioSpec,
    turn_index: int,
    public_history: Sequence[ConversationMessage],
) -> tuple[TeacherChatMessage, ...]:
    turn_number = turn_index + 1
    total_turns = len(scenario.turns)
    if turn_index == 0:
        turn_guidance = (
            "Open with a concrete, casual reason for asking and a clear request. The request "
            "should naturally need current or externally verified information."
        )
    else:
        turn_guidance = (
            "Continue as the same user in direct response to the assistant's latest audible "
            "answer. Follow up, compare, narrow, correct, or refer back elliptically. Preserve "
            "what the conversation was comparing even when the user does not repeat the metric. "
            "Do not change to an unrelated topic. If the assistant asked whether it should look "
            "something up and the user wants that, answer clearly and continue the request."
        )
    request = f"""\
Create only the next user utterance.

Loose conversation brief:
{scenario.topic}

This is user turn {turn_number} of {total_turns}.
Requested speech style: {scenario.speech_style.value}
{turn_guidance}

Across the whole conversation, allow several factual lookups and, when natural, one dependent
comparison or calculation. Do not force a lookup into every turn. Do not supply the external facts
the user is asking for.

Audible conversation so far:
{_audible_history_json(public_history)}
"""
    return (
        TeacherChatMessage(role="system", content=TEACHER_LED_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
    )


def teacher_led_assistant_step_messages(
    scenario: ScenarioSpec,
    public_history: Sequence[ConversationMessage],
) -> tuple[TeacherChatMessage, ...]:
    request = f"""\
Create only the next assistant step.

Loose conversation brief:
{scenario.topic}

Choose the action yourself from the conversation:
- Give a final spoken response when the request is answerable from the conversation.
- Call search immediately for a current, changing, local, or externally verified fact that is not
  already present.
- Call calculate for exact arithmetic, including a comparison derived from known numbers.
- Call get_time when the current local date or time is required.

Available tools:
- search(query): returns one concise search-result string
- calculate(expression): evaluates basic numeric arithmetic
- get_time(): returns the current local timestamp

For a tool action, emit exactly one call with a short natural spoken bridge. Never merely promise
to search, check, calculate, or look something up in a final response. If the user's request is
clear, do not ask permission before calling. After a tool result, use that result directly or make
another call when it is genuinely needed. Continue through as many dependent calls as the request
requires, then give the final spoken response. Preserve the metric, entity, and correction implied
by short follow-ups. Do not answer a pending lookup from memory.

Complete conversation history:
{_history_json(public_history)}
"""
    return (
        TeacherChatMessage(role="system", content=TEACHER_LED_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
    )


def user_turn_scope_messages(
    public_history: Sequence[ConversationMessage],
    candidate_text: str,
) -> tuple[TeacherChatMessage, ...]:
    request = f"""\
Classify this candidate user turn.

Public history before the candidate:
{_history_json(public_history)}

Candidate user utterance:
{candidate_text}

Return required_tools in the order they would need to run. If no tool is needed, return an empty
list."""
    return (
        TeacherChatMessage(role="system", content=USER_SCOPE_AUDITOR_SYSTEM_PROMPT),
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
                "the supplied conversation and earlier results, but answer the exact query only. "
                "Do not include information reserved for another call.\n\n"
                f"Query:\n{query}\n\n"
                f"Conversation context:\n{_history_json(public_history)}"
            ),
        ),
    )


def record_quality_messages(
    scenario: ScenarioSpec,
    public_history: Sequence[ConversationMessage],
) -> tuple[TeacherChatMessage, ...]:
    planned_turns = (
        "The teacher freely chose each user turn and assistant action. Judge tool need from the "
        "actual conversation rather than from a predetermined plan."
        if scenario.generation_mode is ScenarioGenerationMode.TEACHER_LED
        else json.dumps(
            [
                {
                    "turn_index": index,
                    "user_instruction": turn.user_instruction,
                    "planned_tools": [step.tool_name.value for step in turn.tool_steps],
                    "planned_outcomes": [step.outcome.value for step in turn.tool_steps],
                    "response_mode": turn.response_mode.value,
                }
                for index, turn in enumerate(scenario.turns)
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    request = f"""\
Audit this completed candidate record.

Scenario family: {scenario.family}
Topic: {scenario.topic}
Generation guidance:
{planned_turns}

Candidate public conversation:
{_history_json(public_history)}

The plan is not evidence for an answer. Only earlier user messages and completed tool results are
public evidence. A no-tool turn that requests a new external fact is invalid even if the assistant
hedges or asks a follow-up. The only valid no-call external request is a turn explicitly marked
clarify_search, and it must lead to a search immediately after the user supplies the missing
detail. A bridge before a call may acknowledge the task but may not state that call's result.
Search-result wording matters:
do not infer a different role or category from a bare name. Missing schedule details cannot
support claims about having enough time."""
    return (
        TeacherChatMessage(role="system", content=QUALITY_AUDITOR_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
    )


def record_grounding_messages(
    scenario: ScenarioSpec,
    public_history: Sequence[ConversationMessage],
) -> tuple[TeacherChatMessage, ...]:
    assistant_message_ids = [
        message.message_id for message in public_history if message.kind == "assistant"
    ]
    request = f"""\
Audit the assistant messages in this completed candidate.

Scenario family: {scenario.family}
Required assistant message IDs, in order:
{json.dumps(assistant_message_ids, ensure_ascii=False, indent=2)}

Public conversation:
{_history_json(public_history)}

Return exactly one finding per required assistant message ID, in the same order."""
    return (
        TeacherChatMessage(role="system", content=GROUNDING_AUDITOR_SYSTEM_PROMPT),
        TeacherChatMessage(role="user", content=request),
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
    scenario_family: str,
    response_mode: AssistantResponseMode,
) -> str:
    if response_mode is AssistantResponseMode.CLARIFY_SEARCH:
        return (
            "Sound curious and practical. Ask only for the single missing detail, without a "
            "customer-service preamble."
        )
    if expected_tool_name is None:
        if scenario_family == "long_mixed":
            return (
                "Answer with enough substance to match the conversational turn. For advice, "
                "editing, explanation, or brainstorming, aim for roughly half to the full length "
                "of the user's latest utterance. Do not pad a simple factual answer."
            )
        return (
            "Reply like a conversational partner. For advice, editing, explanation, or "
            "brainstorming, aim for roughly half to the full length of the user's latest "
            "utterance. Do not pad a simple factual answer or use a customer-service closing."
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


def _tool_sequence_guidance(
    expected_tool_name: ToolName | None,
    later_tool_names: Sequence[ToolName],
    response_mode: AssistantResponseMode,
) -> str:
    if response_mode is not AssistantResponseMode.ANSWER:
        return (
            "A later search is planned. Do not call it during this step and do not invent its "
            "result."
        )
    if expected_tool_name is None:
        if later_tool_names:
            return (
                "No call is allowed yet. This is a setup turn before a later tool request. "
                "Acknowledge the user's context without calculating, inferring a time gap, "
                "supplying current facts, or prematurely answering the later request."
            )
        return "No call is allowed. Use only evidence already present in public history."
    if not later_tool_names:
        return "This is the only remaining call for this user turn."
    later_names = ", ".join(tool_name.value for tool_name in later_tool_names)
    return (
        f"Later calls are still required in this order: {later_names}. Keep this call atomic. "
        "For search, request only the fact needed at this step and do not request facts reserved "
        "for later calls."
    )


def _history_json(public_history: Sequence[ConversationMessage]) -> str:
    values = [message.model_dump(mode="json") for message in public_history]
    return json.dumps(values, ensure_ascii=False, indent=2)


def _audible_history_json(public_history: Sequence[ConversationMessage]) -> str:
    values: list[dict[str, str]] = []
    for message in public_history:
        match message:
            case UserMessage(text=text):
                values.append({"role": "user", "content": text})
            case AssistantMessage(audible_text=audible_text):
                values.append({"role": "assistant", "content": audible_text})
            case _:
                pass
    return json.dumps(values, ensure_ascii=False, indent=2)

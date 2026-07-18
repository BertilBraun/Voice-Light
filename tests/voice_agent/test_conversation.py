from __future__ import annotations

from app.compute.voice.conversation import (
    ModelAssistantMessage,
    ModelConversationTurn,
    ModelTurnIdentity,
    ModelUserMessage,
    PrivateModelContext,
)
from app.compute.voice.tools import (
    SearchArguments,
    SearchToolCallFunction,
    ToolCall,
    ToolName,
    ToolSuccess,
)


def test_private_context_stages_call_without_exposing_dangling_message() -> None:
    context = PrivateModelContext()
    turn = _weather_turn()
    context.commit_turn(turn)

    stage = turn.stage_tool_call(
        invocation_id=41,
        audible_text_start=0,
        audible_text_end=len("Let me check that."),
        message=_assistant_call(),
    )

    assert stage.identity == turn.identity
    assert context.snapshot() == (ModelUserMessage(content="What is the weather in London?"),)


def test_private_context_atomically_commits_call_and_exact_result() -> None:
    context = PrivateModelContext()
    turn = _weather_turn()
    context.commit_turn(turn)
    turn.stage_tool_call(
        invocation_id=41,
        audible_text_start=0,
        audible_text_end=len("Let me check that."),
        message=_assistant_call(),
    )
    outcome = _weather_outcome()

    exchange = turn.commit_tool_result(outcome)

    assert exchange.tool_message.tool_call_id == "qwen-41-tool-1"
    assert exchange.tool_message.outcome.call_id == "qwen-41-tool-1"
    assert context.snapshot() == (
        ModelUserMessage(content="What is the weather in London?"),
        _assistant_call(),
        exchange.tool_message,
    )


def test_later_turn_keeps_tool_exchange_and_audible_continuation_in_order() -> None:
    context = PrivateModelContext()
    weather_turn = _weather_turn()
    context.commit_turn(weather_turn)
    weather_turn.stage_tool_call(
        invocation_id=41,
        audible_text_start=0,
        audible_text_end=len("Let me check that."),
        message=_assistant_call(),
    )
    exchange = weather_turn.commit_tool_result(_weather_outcome())
    weather_turn.update_audible_assistant_content(
        "Let me check that. It is 12 degrees and lightly cloudy in London."
    )
    later_turn = ModelConversationTurn(
        identity=ModelTurnIdentity(
            assistant_generation_id=12,
            user_turn_id=2,
            transcript_revision_id=8,
        ),
        user_message=ModelUserMessage(content="Should I take a coat?"),
    )
    context.commit_turn(later_turn)

    assert context.snapshot() == (
        ModelUserMessage(content="What is the weather in London?"),
        _assistant_call(),
        exchange.tool_message,
        ModelAssistantMessage(content="It is 12 degrees and lightly cloudy in London."),
        ModelUserMessage(content="Should I take a coat?"),
    )


def test_discarded_staged_call_leaves_only_confirmed_audible_bridge() -> None:
    context = PrivateModelContext()
    turn = _weather_turn()
    context.commit_turn(turn)
    turn.stage_tool_call(
        invocation_id=41,
        audible_text_start=0,
        audible_text_end=len("Let me check that."),
        message=_assistant_call(),
    )
    turn.update_audible_assistant_content("Let me check...")

    turn.discard_staged_tool_call()

    assert context.snapshot() == (
        ModelUserMessage(content="What is the weather in London?"),
        ModelAssistantMessage(content="Let me check..."),
    )


def test_multiple_tool_exchanges_preserve_exact_chronological_positions() -> None:
    context = PrivateModelContext()
    turn = _weather_turn()
    context.commit_turn(turn)
    first_bridge = "Let me check London."
    second_bridge = "I have London; now Berlin."
    audible_content = f"{first_bridge} {second_bridge} London is cooler, so take the heavier coat."
    first_call = _assistant_call()
    second_call = ModelAssistantMessage(
        content=second_bridge,
        tool_calls=(
            ToolCall(
                id="qwen-42-tool-1",
                function=SearchToolCallFunction(
                    arguments=SearchArguments(query="current weather in Berlin"),
                ),
            ),
        ),
    )
    turn.stage_tool_call(
        invocation_id=41,
        audible_text_start=0,
        audible_text_end=len(first_bridge),
        message=ModelAssistantMessage(
            content=first_bridge,
            tool_calls=first_call.tool_calls,
        ),
    )
    first_exchange = turn.commit_tool_result(_weather_outcome())
    second_start = len(first_bridge) + 1
    turn.stage_tool_call(
        invocation_id=42,
        audible_text_start=second_start,
        audible_text_end=second_start + len(second_bridge),
        message=second_call,
    )
    second_outcome = ToolSuccess(
        call_id="qwen-42-tool-1",
        tool_name=ToolName.SEARCH,
        result="Berlin is 18 degrees and clear.",
    )
    second_exchange = turn.commit_tool_result(second_outcome)
    turn.update_audible_assistant_content(audible_content)

    assert context.snapshot() == (
        turn.user_message,
        first_exchange.assistant_message,
        first_exchange.tool_message,
        second_call,
        second_exchange.tool_message,
        ModelAssistantMessage(content="London is cooler, so take the heavier coat."),
    )


def _weather_turn() -> ModelConversationTurn:
    return ModelConversationTurn(
        identity=ModelTurnIdentity(
            assistant_generation_id=11,
            user_turn_id=1,
            transcript_revision_id=7,
        ),
        user_message=ModelUserMessage(content="What is the weather in London?"),
    )


def _assistant_call() -> ModelAssistantMessage:
    return ModelAssistantMessage(
        content="Let me check that.",
        tool_calls=(
            ToolCall(
                id="qwen-41-tool-1",
                function=SearchToolCallFunction(
                    arguments=SearchArguments(query="current weather in London"),
                ),
            ),
        ),
    )


def _weather_outcome() -> ToolSuccess:
    return ToolSuccess(
        call_id="qwen-41-tool-1",
        tool_name=ToolName.SEARCH,
        result="London is 12 degrees and lightly cloudy.",
    )

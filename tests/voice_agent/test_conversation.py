from __future__ import annotations

from app.compute.voice.conversation import (
    ModelAssistantMessage,
    ModelConversationTurn,
    ModelTurnIdentity,
    ModelUserMessage,
    PrivateModelContext,
)
from app.compute.voice.tools import (
    GetWeatherArguments,
    ToolCall,
    ToolCallFunction,
    ToolName,
    ToolSuccess,
    WeatherResult,
)


def test_private_context_stages_call_without_exposing_dangling_message() -> None:
    context = PrivateModelContext()
    turn = _weather_turn()
    context.commit_turn(turn)

    stage = turn.stage_tool_call(
        invocation_id=41,
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
        message=_assistant_call(),
    )
    turn.update_audible_assistant_content("Let me check...")

    turn.discard_staged_tool_call()

    assert context.snapshot() == (
        ModelUserMessage(content="What is the weather in London?"),
        ModelAssistantMessage(content="Let me check..."),
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
                function=ToolCallFunction(
                    name=ToolName.GET_WEATHER,
                    arguments=GetWeatherArguments(location="London"),
                ),
            ),
        ),
    )


def _weather_outcome() -> ToolSuccess:
    return ToolSuccess(
        call_id="qwen-41-tool-1",
        tool_name=ToolName.GET_WEATHER,
        result=WeatherResult(
            location="London",
            temperature_celsius=12,
            conditions="lightly cloudy",
        ),
    )

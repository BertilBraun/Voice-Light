from __future__ import annotations

import asyncio

from app.compute.voice.tools import (
    GetWeatherArguments,
    SerializedToolCall,
    ToolCall,
    ToolCallFailure,
    ToolCallFailureReason,
    ToolExecutionFailure,
    ToolExecutionFailureReason,
    WeatherResult,
    WeatherToolRegistry,
)


async def immediate_weather(arguments: GetWeatherArguments) -> WeatherResult:
    return WeatherResult(
        location=arguments.location,
        temperature_celsius=12,
        conditions="lightly cloudy",
    )


def test_weather_registry_validates_london_call_and_returns_typed_result() -> None:
    async def execute() -> None:
        registry = WeatherToolRegistry(immediate_weather)
        assert "never make the function call the first assistant output" in (
            registry.specifications[0].function.description
        )
        validated = registry.validate(
            SerializedToolCall(
                id="call-1",
                name="get_weather",
                arguments_json='{"location":"London"}',
            )
        )

        assert isinstance(validated, ToolCall)
        assert validated.function.arguments == GetWeatherArguments(location="London")
        outcome = await registry.execute(validated)
        assert outcome.result == WeatherResult(
            location="London",
            temperature_celsius=12,
            conditions="lightly cloudy",
        )

    asyncio.run(execute())


def test_weather_registry_rejects_unknown_tool() -> None:
    registry = WeatherToolRegistry(immediate_weather)

    validated = registry.validate(
        SerializedToolCall(
            id="call-1",
            name="run_python",
            arguments_json='{"code":"pass"}',
        )
    )

    assert isinstance(validated, ToolCallFailure)
    assert validated.reason is ToolCallFailureReason.UNKNOWN_TOOL


def test_weather_registry_rejects_invalid_arguments() -> None:
    registry = WeatherToolRegistry(immediate_weather)

    validated = registry.validate(
        SerializedToolCall(
            id="call-1",
            name="get_weather",
            arguments_json='{"location":7}',
        )
    )

    assert isinstance(validated, ToolCallFailure)
    assert validated.reason is ToolCallFailureReason.INVALID_ARGUMENTS


def test_weather_registry_converts_handler_failure_to_typed_outcome() -> None:
    async def failing_weather(arguments: GetWeatherArguments) -> WeatherResult:
        del arguments
        raise RuntimeError("synthetic handler failure")

    async def execute() -> None:
        registry = WeatherToolRegistry(failing_weather)
        validated = registry.validate(
            SerializedToolCall(
                id="call-1",
                name="get_weather",
                arguments_json='{"location":"London"}',
            )
        )
        assert isinstance(validated, ToolCall)

        outcome = await registry.execute(validated)

        assert isinstance(outcome, ToolExecutionFailure)
        assert outcome.reason is ToolExecutionFailureReason.HANDLER_FAILURE

    asyncio.run(execute())

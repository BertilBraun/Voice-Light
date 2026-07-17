from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import ConfigDict, Field, TypeAdapter

from app.shared.base_model import FrozenBaseModel


class ToolName(StrEnum):
    GET_WEATHER = "get_weather"


class ToolCallFailureReason(StrEnum):
    MALFORMED_JSON = "malformed_json"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    DUPLICATE_CALL = "duplicate_call"
    MULTIPLE_CALLS = "multiple_calls"
    INCOMPLETE_CALL = "incomplete_call"
    UNEXPECTED_CONTENT = "unexpected_content"


class ToolExecutionFailureReason(StrEnum):
    HANDLER_FAILURE = "handler_failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


class ToolLifecycle(StrEnum):
    NOT_REQUESTED = "not_requested"
    CALL_BUFFERING = "call_buffering"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


class ToolResultCommitStatus(StrEnum):
    NOT_STAGED = "not_staged"
    STAGED = "staged"
    GENERATION_LOCAL_COMMITTED = "generation_local_committed"
    SESSION_COMMITTED = "session_committed"
    DISCARDED = "discarded"


class ToolInvalidationReason(StrEnum):
    USER_ACTIVITY = "user_activity"
    RESPONSE_REQUIRING_OVERLAP = "response_requiring_overlap"
    SPECULATIVE_INVALIDATION = "speculative_invalidation"
    GENERATION_FAILURE = "generation_failure"
    SESSION_STOPPED = "session_stopped"
    PLAYBACK_RESUME_REJECTED = "playback_resume_rejected"


class GetWeatherArguments(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    location: str = Field(min_length=1)


class WeatherResult(FrozenBaseModel):
    location: str
    temperature_celsius: int
    conditions: str


class ToolStringParameter(FrozenBaseModel):
    type: Literal["string"] = "string"
    description: str


class GetWeatherParameterProperties(FrozenBaseModel):
    location: ToolStringParameter


class GetWeatherParameters(FrozenBaseModel):
    type: Literal["object"] = "object"
    properties: GetWeatherParameterProperties
    required: tuple[Literal["location"], ...] = ("location",)
    additionalProperties: Literal[False] = False


class ToolFunctionSpecification(FrozenBaseModel):
    name: ToolName
    description: str
    parameters: GetWeatherParameters


class ToolSpecification(FrozenBaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionSpecification


class ToolCallFunction(FrozenBaseModel):
    name: ToolName
    arguments: GetWeatherArguments


class ToolCall(FrozenBaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class SerializedToolCall(FrozenBaseModel):
    id: str
    name: str
    arguments_json: str


class ToolCallFailure(FrozenBaseModel):
    call_id: str
    reason: ToolCallFailureReason
    message: str
    attempted_tool_name: str | None


class ToolSuccess(FrozenBaseModel):
    outcome: Literal["success"] = "success"
    call_id: str
    tool_name: ToolName
    result: WeatherResult


class ToolExecutionFailure(FrozenBaseModel):
    outcome: Literal["failure"] = "failure"
    call_id: str
    tool_name: ToolName | None
    reason: ToolCallFailureReason | ToolExecutionFailureReason
    message: str


ToolOutcome = Annotated[
    ToolSuccess | ToolExecutionFailure,
    Field(discriminator="outcome"),
]
tool_outcome_adapter: TypeAdapter[ToolOutcome] = TypeAdapter(ToolOutcome)


WeatherToolHandler = Callable[[GetWeatherArguments], Awaitable[WeatherResult]]


class ToolExecutor(Protocol):
    @property
    def specifications(self) -> tuple[ToolSpecification, ...]: ...

    def validate(self, request: SerializedToolCall) -> ToolCall | ToolCallFailure: ...

    async def execute(self, call: ToolCall) -> ToolSuccess | ToolExecutionFailure: ...


class WeatherToolRegistry:
    def __init__(self, weather_handler: WeatherToolHandler) -> None:
        self.weather_handler = weather_handler
        self._specifications = (
            ToolSpecification(
                function=ToolFunctionSpecification(
                    name=ToolName.GET_WEATHER,
                    description="Get deterministic current weather for a location.",
                    parameters=GetWeatherParameters(
                        properties=GetWeatherParameterProperties(
                            location=ToolStringParameter(
                                description="City or location whose current weather is requested."
                            )
                        )
                    ),
                )
            ),
        )

    @property
    def specifications(self) -> tuple[ToolSpecification, ...]:
        return self._specifications

    def validate(self, request: SerializedToolCall) -> ToolCall | ToolCallFailure:
        if request.name != ToolName.GET_WEATHER:
            return ToolCallFailure(
                call_id=request.id,
                reason=ToolCallFailureReason.UNKNOWN_TOOL,
                message=f"Unknown tool: {request.name}",
                attempted_tool_name=request.name,
            )
        try:
            arguments = GetWeatherArguments.model_validate_json(request.arguments_json)
        except ValueError as error:
            return ToolCallFailure(
                call_id=request.id,
                reason=ToolCallFailureReason.INVALID_ARGUMENTS,
                message=f"Invalid get_weather arguments: {error}",
                attempted_tool_name=request.name,
            )
        return ToolCall(
            id=request.id,
            function=ToolCallFunction(
                name=ToolName.GET_WEATHER,
                arguments=arguments,
            ),
        )

    async def execute(self, call: ToolCall) -> ToolSuccess | ToolExecutionFailure:
        try:
            result = await self.weather_handler(call.function.arguments)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return ToolExecutionFailure(
                call_id=call.id,
                tool_name=call.function.name,
                reason=ToolExecutionFailureReason.HANDLER_FAILURE,
                message=f"Weather lookup failed: {error}",
            )
        return ToolSuccess(
            call_id=call.id,
            tool_name=call.function.name,
            result=result,
        )


class DeterministicWeatherHandler:
    async def __call__(self, arguments: GetWeatherArguments) -> WeatherResult:
        await asyncio.sleep(1.0)
        normalized_location = arguments.location.strip().casefold()
        match normalized_location:
            case "london" | "london, uk" | "london, united kingdom":
                return WeatherResult(
                    location="London",
                    temperature_celsius=12,
                    conditions="lightly cloudy",
                )
            case "berlin" | "berlin, germany":
                return WeatherResult(
                    location="Berlin",
                    temperature_celsius=18,
                    conditions="clear",
                )
            case "san francisco" | "san francisco, california":
                return WeatherResult(
                    location="San Francisco",
                    temperature_celsius=16,
                    conditions="foggy",
                )
            case _:
                return WeatherResult(
                    location=arguments.location.strip(),
                    temperature_celsius=20,
                    conditions="clear (deterministic fallback)",
                )


def create_demo_tool_registry() -> WeatherToolRegistry:
    return WeatherToolRegistry(DeterministicWeatherHandler())

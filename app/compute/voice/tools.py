from __future__ import annotations

import ast
import asyncio
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import ConfigDict, Field, TypeAdapter

from app.shared.base_model import FrozenBaseModel

MAXIMUM_ABSOLUTE_INTEGER_RESULT = 10**100
MAXIMUM_ABSOLUTE_FLOAT_RESULT = 1e100
MAXIMUM_ABSOLUTE_EXPONENT = 100


class ToolName(StrEnum):
    SEARCH = "search"
    CALCULATE = "calculate"
    GET_TIME = "get_time"


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


class SearchArguments(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=240)


class CalculateArguments(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expression: str = Field(min_length=1, max_length=160)


class GetTimeArguments(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolStringParameter(FrozenBaseModel):
    type: Literal["string"] = "string"
    description: str


class SearchParameterProperties(FrozenBaseModel):
    query: ToolStringParameter


class SearchParameters(FrozenBaseModel):
    type: Literal["object"] = "object"
    properties: SearchParameterProperties
    required: tuple[Literal["query"], ...] = ("query",)
    additionalProperties: Literal[False] = False


class CalculateParameterProperties(FrozenBaseModel):
    expression: ToolStringParameter


class CalculateParameters(FrozenBaseModel):
    type: Literal["object"] = "object"
    properties: CalculateParameterProperties
    required: tuple[Literal["expression"], ...] = ("expression",)
    additionalProperties: Literal[False] = False


class GetTimeParameterProperties(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GetTimeParameters(FrozenBaseModel):
    type: Literal["object"] = "object"
    properties: GetTimeParameterProperties
    required: tuple[()] = ()
    additionalProperties: Literal[False] = False


class SearchToolFunctionSpecification(FrozenBaseModel):
    name: Literal[ToolName.SEARCH] = ToolName.SEARCH
    description: str
    parameters: SearchParameters


class CalculateToolFunctionSpecification(FrozenBaseModel):
    name: Literal[ToolName.CALCULATE] = ToolName.CALCULATE
    description: str
    parameters: CalculateParameters


class GetTimeToolFunctionSpecification(FrozenBaseModel):
    name: Literal[ToolName.GET_TIME] = ToolName.GET_TIME
    description: str
    parameters: GetTimeParameters


ToolFunctionSpecification = Annotated[
    SearchToolFunctionSpecification
    | CalculateToolFunctionSpecification
    | GetTimeToolFunctionSpecification,
    Field(discriminator="name"),
]


class ToolSpecification(FrozenBaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionSpecification


class SearchToolCallFunction(FrozenBaseModel):
    name: Literal[ToolName.SEARCH] = ToolName.SEARCH
    arguments: SearchArguments


class CalculateToolCallFunction(FrozenBaseModel):
    name: Literal[ToolName.CALCULATE] = ToolName.CALCULATE
    arguments: CalculateArguments


class GetTimeToolCallFunction(FrozenBaseModel):
    name: Literal[ToolName.GET_TIME] = ToolName.GET_TIME
    arguments: GetTimeArguments


ToolCallFunction = Annotated[
    SearchToolCallFunction | CalculateToolCallFunction | GetTimeToolCallFunction,
    Field(discriminator="name"),
]


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
    result: str


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

SearchToolHandler = Callable[[SearchArguments], Awaitable[str]]
CalculateToolHandler = Callable[[CalculateArguments], Awaitable[str]]
CurrentTimeProvider = Callable[[], datetime]


class SearchAnswerer(Protocol):
    async def answer(self, query: str) -> str: ...


class GetTimeToolHandler(Protocol):
    def set_local_time_zone(self, time_zone_name: str) -> None: ...

    async def __call__(self) -> str: ...


class ToolExecutor(Protocol):
    @property
    def specifications(self) -> tuple[ToolSpecification, ...]: ...

    def validate(self, request: SerializedToolCall) -> ToolCall | ToolCallFailure: ...

    def configure_session(self, local_time_zone: str) -> None: ...

    async def execute(self, call: ToolCall) -> ToolSuccess | ToolExecutionFailure: ...


class RuntimeToolRegistry:
    def __init__(
        self,
        search_handler: SearchToolHandler,
        calculate_handler: CalculateToolHandler,
        get_time_handler: GetTimeToolHandler,
    ) -> None:
        self.search_handler = search_handler
        self.calculate_handler = calculate_handler
        self.get_time_handler = get_time_handler
        self._specifications = _tool_specifications()

    @property
    def specifications(self) -> tuple[ToolSpecification, ...]:
        return self._specifications

    def validate(self, request: SerializedToolCall) -> ToolCall | ToolCallFailure:
        try:
            match request.name:
                case ToolName.SEARCH:
                    function: ToolCallFunction = SearchToolCallFunction(
                        arguments=SearchArguments.model_validate_json(request.arguments_json)
                    )
                case ToolName.CALCULATE:
                    function = CalculateToolCallFunction(
                        arguments=CalculateArguments.model_validate_json(request.arguments_json)
                    )
                case ToolName.GET_TIME:
                    function = GetTimeToolCallFunction(
                        arguments=GetTimeArguments.model_validate_json(request.arguments_json)
                    )
                case _:
                    return ToolCallFailure(
                        call_id=request.id,
                        reason=ToolCallFailureReason.UNKNOWN_TOOL,
                        message=f"Unknown tool: {request.name}",
                        attempted_tool_name=request.name,
                    )
        except ValueError as error:
            return ToolCallFailure(
                call_id=request.id,
                reason=ToolCallFailureReason.INVALID_ARGUMENTS,
                message=f"Invalid {request.name} arguments: {error}",
                attempted_tool_name=request.name,
            )
        return ToolCall(id=request.id, function=function)

    def configure_session(self, local_time_zone: str) -> None:
        self.get_time_handler.set_local_time_zone(local_time_zone)

    async def execute(self, call: ToolCall) -> ToolSuccess | ToolExecutionFailure:
        try:
            match call.function:
                case SearchToolCallFunction(arguments=arguments):
                    result = await self.search_handler(arguments)
                case CalculateToolCallFunction(arguments=arguments):
                    result = await self.calculate_handler(arguments)
                case GetTimeToolCallFunction():
                    result = await self.get_time_handler()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return ToolExecutionFailure(
                call_id=call.id,
                tool_name=call.function.name,
                reason=ToolExecutionFailureReason.HANDLER_FAILURE,
                message=f"{call.function.name} failed: {error}",
            )
        return ToolSuccess(
            call_id=call.id,
            tool_name=call.function.name,
            result=result,
        )


class StandardSearchHandler:
    def __init__(self, answerer: SearchAnswerer) -> None:
        self.answerer = answerer

    async def __call__(self, arguments: SearchArguments) -> str:
        return await self.answerer.answer(arguments.query)


class PythonArithmeticHandler:
    async def __call__(self, arguments: CalculateArguments) -> str:
        expression = ast.parse(arguments.expression, mode="eval")
        return str(_evaluate_arithmetic(expression.body))


class CurrentLocalTimeHandler:
    def __init__(self, current_time: CurrentTimeProvider) -> None:
        self.current_time = current_time
        self.local_time_zone = ZoneInfo("Etc/UTC")

    def set_local_time_zone(self, time_zone_name: str) -> None:
        self.local_time_zone = ZoneInfo(time_zone_name)

    async def __call__(self) -> str:
        current_time = self.current_time()
        if current_time.utcoffset() is None:
            raise ValueError("Current time must include a UTC offset.")
        local_time = current_time.astimezone(self.local_time_zone)
        return (
            f"{local_time.isoformat(timespec='seconds')} "
            f"(IANA time zone: {self.local_time_zone.key})"
        )


def create_runtime_tool_registry(search_handler: SearchToolHandler) -> RuntimeToolRegistry:
    return RuntimeToolRegistry(
        search_handler=search_handler,
        calculate_handler=PythonArithmeticHandler(),
        get_time_handler=CurrentLocalTimeHandler(_current_local_time),
    )


def runtime_tool_specifications() -> tuple[ToolSpecification, ...]:
    return _tool_specifications()


def _tool_specifications() -> tuple[ToolSpecification, ...]:
    return (
        ToolSpecification(
            function=SearchToolFunctionSpecification(
                description="Search for current or externally verifiable information.",
                parameters=SearchParameters(
                    properties=SearchParameterProperties(
                        query=ToolStringParameter(description="A concise search query.")
                    )
                ),
            )
        ),
        ToolSpecification(
            function=CalculateToolFunctionSpecification(
                description="Evaluate a basic numeric arithmetic expression.",
                parameters=CalculateParameters(
                    properties=CalculateParameterProperties(
                        expression=ToolStringParameter(
                            description="Numeric literals and basic arithmetic operators."
                        )
                    )
                ),
            )
        ),
        ToolSpecification(
            function=GetTimeToolFunctionSpecification(
                description=(
                    "Get the current date and time in the user's local time zone, including its "
                    "UTC offset and IANA time-zone name."
                ),
                parameters=GetTimeParameters(properties=GetTimeParameterProperties()),
            )
        ),
    )


def _current_local_time() -> datetime:
    return datetime.now(UTC)


def _evaluate_arithmetic(expression: ast.expr) -> int | float:
    match expression:
        case ast.Constant(value=bool()):
            raise ValueError("Boolean values are not arithmetic operands.")
        case ast.Constant(value=int() as value):
            return _bounded_integer(value)
        case ast.Constant(value=float() as value):
            return _bounded_float(value)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _evaluate_arithmetic(operand)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            value = _evaluate_arithmetic(operand)
            return _bounded_number(-value)
        case ast.BinOp(left=left, op=operator, right=right):
            left_value = _evaluate_arithmetic(left)
            right_value = _evaluate_arithmetic(right)
            result: int | float
            match operator:
                case ast.Add():
                    result = left_value + right_value
                case ast.Sub():
                    result = left_value - right_value
                case ast.Mult():
                    result = left_value * right_value
                case ast.Div():
                    result = left_value / right_value
                case ast.FloorDiv():
                    result = left_value // right_value
                case ast.Mod():
                    result = left_value % right_value
                case ast.Pow():
                    if abs(right_value) > MAXIMUM_ABSOLUTE_EXPONENT:
                        raise ValueError("The exponent is too large.")
                    result = left_value**right_value
                case _:
                    raise ValueError("The expression contains an unsupported operator.")
            return _bounded_number(result)
        case _:
            raise ValueError("Only numeric literals and basic arithmetic operators are allowed.")


def _bounded_number(value: int | float) -> int | float:
    match value:
        case bool():
            raise ValueError("Boolean results are not supported.")
        case int() as integer:
            return _bounded_integer(integer)
        case float() as decimal:
            return _bounded_float(decimal)


def _bounded_integer(value: int) -> int:
    if abs(value) > MAXIMUM_ABSOLUTE_INTEGER_RESULT:
        raise ValueError("The integer result is too large.")
    return value


def _bounded_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("The floating-point result must be finite.")
    if abs(value) > MAXIMUM_ABSOLUTE_FLOAT_RESULT:
        raise ValueError("The floating-point result is too large.")
    return value

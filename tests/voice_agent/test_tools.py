from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from app.compute.voice.tools import (
    CalculateArguments,
    CurrentLocalTimeHandler,
    GetTimeArguments,
    PythonArithmeticHandler,
    RuntimeToolRegistry,
    SearchArguments,
    SearchToolAvailability,
    SerializedToolCall,
    StandardSearchHandler,
    ToolCall,
    ToolCallAdmission,
    ToolCallFailure,
    ToolCallFailureReason,
    ToolCircuitBreaker,
    ToolExecutionFailure,
    ToolExecutionFailureReason,
    ToolName,
    ToolSuccess,
    create_runtime_tool_registry,
    runtime_tool_specifications,
)


class StaticSearchAnswerer:
    def __init__(self, answer: str) -> None:
        self.answer_text = answer
        self.queries: list[str] = []

    async def answer(self, query: str) -> str:
        self.queries.append(query)
        return self.answer_text


def create_test_tool_registry() -> RuntimeToolRegistry:
    return create_runtime_tool_registry(
        search_handler=StandardSearchHandler(StaticSearchAnswerer("test search answer"))
    )


def test_runtime_registry_exposes_search_calculate_and_get_time() -> None:
    specifications = runtime_tool_specifications()

    assert tuple(specification.function.name for specification in specifications) == (
        ToolName.SEARCH,
        ToolName.CALCULATE,
        ToolName.GET_TIME,
    )
    search_description = specifications[0].function.description
    assert "required for current weather or news" in search_description
    assert "instead of promising a future search" in search_description


def test_unconfigured_registry_does_not_expose_or_validate_search() -> None:
    registry = create_runtime_tool_registry(search_handler=None)

    assert registry.search_availability is SearchToolAvailability.UNAVAILABLE
    assert tuple(specification.function.name for specification in registry.specifications) == (
        ToolName.CALCULATE,
        ToolName.GET_TIME,
    )
    outcome = registry.validate(
        SerializedToolCall(
            id="call-1",
            name="search",
            arguments_json='{"query":"current weather in London"}',
        )
    )
    assert isinstance(outcome, ToolCallFailure)
    assert outcome.reason is ToolCallFailureReason.UNKNOWN_TOOL


def test_search_returns_pipeline_answer() -> None:
    async def execute() -> None:
        answerer = StaticSearchAnswerer("Berlin is currently sunny.")
        registry = create_runtime_tool_registry(search_handler=StandardSearchHandler(answerer))
        validated = registry.validate(
            SerializedToolCall(
                id="call-1",
                name="search",
                arguments_json='{"query":"current weather in Berlin"}',
            )
        )

        assert isinstance(validated, ToolCall)
        assert validated.function.arguments == SearchArguments(query="current weather in Berlin")
        outcome = await registry.execute(validated)
        assert outcome.result == "Berlin is currently sunny."
        assert answerer.queries == ["current weather in Berlin"]

    asyncio.run(execute())


def test_tool_circuit_breaker_rejects_only_identical_successful_calls() -> None:
    first_call = create_test_tool_registry().validate(
        SerializedToolCall(
            id="call-1",
            name="search",
            arguments_json='{"query":"current weather in London"}',
        )
    )
    repeated_call = create_test_tool_registry().validate(
        SerializedToolCall(
            id="call-2",
            name="search",
            arguments_json='{"query":"current weather in London"}',
        )
    )
    distinct_call = create_test_tool_registry().validate(
        SerializedToolCall(
            id="call-3",
            name="search",
            arguments_json='{"query":"current weather in New York"}',
        )
    )
    assert isinstance(first_call, ToolCall)
    assert isinstance(repeated_call, ToolCall)
    assert isinstance(distinct_call, ToolCall)
    circuit_breaker = ToolCircuitBreaker()

    assert circuit_breaker.admit(first_call) is ToolCallAdmission.ALLOWED
    circuit_breaker.record(
        first_call,
        ToolSuccess(
            call_id=first_call.id,
            tool_name=ToolName.SEARCH,
            result="London is cloudy.",
        ),
    )

    assert circuit_breaker.admit(repeated_call) is ToolCallAdmission.DUPLICATE
    assert circuit_breaker.admit(distinct_call) is ToolCallAdmission.ALLOWED


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("2 + 3 * 4", "14"),
        ("(10 - 4) / 3", "2.0"),
        ("2**8", "256"),
        ("17 // 5", "3"),
        ("17 % 5", "2"),
        ("-3 + +5", "2"),
    ),
)
def test_calculate_evaluates_basic_python_arithmetic(expression: str, expected: str) -> None:
    async def execute() -> None:
        registry = create_test_tool_registry()
        validated = registry.validate(
            SerializedToolCall(
                id="call-1",
                name="calculate",
                arguments_json=CalculateArguments(expression=expression).model_dump_json(),
            )
        )

        assert isinstance(validated, ToolCall)
        outcome = await registry.execute(validated)
        assert outcome.result == expected

    asyncio.run(execute())


@pytest.mark.parametrize(
    "expression",
    (
        "__import__('os').getcwd()",
        "open('secret.txt').read()",
        "[1, 2, 3][0]",
        "2**101",
    ),
)
def test_calculate_rejects_non_arithmetic_or_excessive_expressions(expression: str) -> None:
    async def execute() -> None:
        registry = create_test_tool_registry()
        validated = registry.validate(
            SerializedToolCall(
                id="call-1",
                name="calculate",
                arguments_json=CalculateArguments(expression=expression).model_dump_json(),
            )
        )

        assert isinstance(validated, ToolCall)
        outcome = await registry.execute(validated)
        assert isinstance(outcome, ToolExecutionFailure)
        assert outcome.reason is ToolExecutionFailureReason.HANDLER_FAILURE

    asyncio.run(execute())


@pytest.mark.parametrize(
    ("current_time", "expected"),
    (
        (
            datetime(2026, 7, 18, 12, 5, 9, tzinfo=UTC),
            "2026-07-18T14:05:09+02:00 (IANA time zone: Europe/Berlin)",
        ),
        (
            datetime(2026, 1, 18, 12, 5, 9, tzinfo=UTC),
            "2026-01-18T13:05:09+01:00 (IANA time zone: Europe/Berlin)",
        ),
    ),
)
def test_get_time_returns_browser_local_timestamp(current_time: datetime, expected: str) -> None:
    async def fixed_time() -> str:
        handler = CurrentLocalTimeHandler(lambda: current_time)
        handler.set_local_time_zone("Europe/Berlin")
        return await handler()

    assert asyncio.run(fixed_time()) == expected


def test_registry_applies_session_time_zone_to_get_time() -> None:
    async def execute() -> str:
        registry = RuntimeToolRegistry(
            search_handler=StandardSearchHandler(StaticSearchAnswerer("unused")),
            calculate_handler=PythonArithmeticHandler(),
            get_time_handler=CurrentLocalTimeHandler(
                lambda: datetime(2026, 7, 18, 12, 5, 9, tzinfo=UTC)
            ),
        )
        registry.configure_session("Europe/Berlin")
        validated = registry.validate(
            SerializedToolCall(id="call-1", name="get_time", arguments_json="{}")
        )

        assert isinstance(validated, ToolCall)
        outcome = await registry.execute(validated)
        assert isinstance(outcome, ToolSuccess)
        return outcome.result

    assert asyncio.run(execute()) == ("2026-07-18T14:05:09+02:00 (IANA time zone: Europe/Berlin)")


def test_registry_rejects_unknown_tool() -> None:
    registry = create_test_tool_registry()

    validated = registry.validate(
        SerializedToolCall(
            id="call-1",
            name="run_python",
            arguments_json='{"code":"pass"}',
        )
    )

    assert isinstance(validated, ToolCallFailure)
    assert validated.reason is ToolCallFailureReason.UNKNOWN_TOOL


@pytest.mark.parametrize(
    ("name", "arguments_json"),
    (
        ("search", '{"query":7}'),
        ("calculate", '{"expression":7}'),
        ("get_time", '{"timezone":"UTC"}'),
    ),
)
def test_registry_rejects_invalid_arguments(name: str, arguments_json: str) -> None:
    registry = create_test_tool_registry()

    validated = registry.validate(
        SerializedToolCall(
            id="call-1",
            name=name,
            arguments_json=arguments_json,
        )
    )

    assert isinstance(validated, ToolCallFailure)
    assert validated.reason is ToolCallFailureReason.INVALID_ARGUMENTS


def test_registry_converts_handler_failure_to_typed_outcome() -> None:
    async def failing_search(arguments: SearchArguments) -> str:
        raise RuntimeError(f"synthetic failure for {arguments.query}")

    async def execute() -> None:
        registry = RuntimeToolRegistry(
            search_handler=failing_search,
            calculate_handler=PythonArithmeticHandler(),
            get_time_handler=CurrentLocalTimeHandler(lambda: datetime(2026, 7, 18, tzinfo=UTC)),
        )
        validated = registry.validate(
            SerializedToolCall(
                id="call-1",
                name="search",
                arguments_json='{"query":"current weather"}',
            )
        )
        assert isinstance(validated, ToolCall)

        outcome = await registry.execute(validated)

        assert isinstance(outcome, ToolExecutionFailure)
        assert outcome.reason is ToolExecutionFailureReason.HANDLER_FAILURE

    asyncio.run(execute())


def test_argument_models_are_tool_specific() -> None:
    assert SearchArguments(query="news").query == "news"
    assert CalculateArguments(expression="1 + 1").expression == "1 + 1"
    assert GetTimeArguments() == GetTimeArguments()

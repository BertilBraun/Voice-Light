from __future__ import annotations

import pytest

from app.compute.voice.conversation import (
    ModelAssistantMessage,
    ModelMessage,
    ModelToolMessage,
    ModelUserMessage,
)
from app.compute.voice.tool_routing import (
    CalculationRoutingReason,
    SearchRoutingReason,
    route_required_calculation_call,
    route_required_search_call,
    temperature_conversion_spoken_result,
)
from app.compute.voice.tools import (
    CalculateArguments,
    SearchArguments,
    ToolName,
    ToolSuccess,
    runtime_tool_specifications,
)


@pytest.mark.parametrize(
    "requested_text",
    (
        "And tell me the weather in London.",
        "Can you make a tool call to check the weather for me?",
        "Search for the latest election result.",
        "Look up the current train status.",
    ),
)
def test_direct_current_information_request_routes_search(requested_text: str) -> None:
    routed = route_required_search_call(
        (ModelUserMessage(content=requested_text),),
        runtime_tool_specifications(),
        invocation_id=7,
    )

    assert routed is not None
    assert routed.reason is SearchRoutingReason.DIRECT_REQUEST
    assert routed.request.name == ToolName.SEARCH
    assert (
        SearchArguments.model_validate_json(routed.request.arguments_json).query == requested_text
    )


def test_routed_search_can_reuse_the_models_started_call_identity() -> None:
    routed = route_required_search_call(
        (ModelUserMessage(content="Search for the current weather in London."),),
        runtime_tool_specifications(),
        invocation_id=3,
        call_id="qwen-3-tool-1",
    )

    assert routed is not None
    assert routed.request.id == "qwen-3-tool-1"


def test_confirmation_routes_the_original_external_information_request() -> None:
    routed = route_required_search_call(
        (
            ModelUserMessage(content="Tell me the current weather in London."),
            ModelAssistantMessage(content="I can look that up for you."),
            ModelUserMessage(content="Please do so."),
        ),
        runtime_tool_specifications(),
        invocation_id=8,
    )

    assert routed is not None
    assert routed.reason is SearchRoutingReason.CONFIRMED_REQUEST
    assert SearchArguments.model_validate_json(routed.request.arguments_json) == SearchArguments(
        query="Tell me the current weather in London."
    )


@pytest.mark.parametrize(
    "messages",
    (
        (ModelUserMessage(content="Tell me a story."),),
        (
            ModelUserMessage(content="Tell me a story."),
            ModelAssistantMessage(content="Would you like another one?"),
            ModelUserMessage(content="Yes."),
        ),
        (ModelUserMessage(content="Calculate 49 times 49 with the tool."),),
    ),
)
def test_non_search_request_is_not_routed(messages: tuple[ModelMessage, ...]) -> None:
    assert route_required_search_call(messages, runtime_tool_specifications(), 9) is None


def test_post_search_continuation_is_not_routed_again() -> None:
    outcome = ToolSuccess(
        call_id="qwen-10-tool-1",
        tool_name=ToolName.SEARCH,
        result="London is mild and cloudy.",
    )
    messages: tuple[ModelMessage, ...] = (
        ModelUserMessage(content="What is the weather in London?"),
        ModelAssistantMessage(content="I'll check that."),
        ModelToolMessage(tool_call_id=outcome.call_id, outcome=outcome),
    )

    assert route_required_search_call(messages, runtime_tool_specifications(), 10) is None


def test_search_is_not_routed_when_the_schema_is_unavailable() -> None:
    tools = tuple(
        specification
        for specification in runtime_tool_specifications()
        if specification.function.name is not ToolName.SEARCH
    )

    assert (
        route_required_search_call(
            (ModelUserMessage(content="What is the weather?"),),
            tools,
            11,
        )
        is None
    )


@pytest.mark.parametrize(
    ("requested_text", "expected_expression"),
    (
        ("What's twenty eight Celsius in Kelvin?", "28 + 273.15"),
        ("Convert 32 Fahrenheit to Celsius.", "(32 - 32) * 5 / 9"),
        ("What is 301.15 Kelvin in Celsius?", "301.15 - 273.15"),
    ),
)
def test_explicit_temperature_conversion_routes_calculate(
    requested_text: str,
    expected_expression: str,
) -> None:
    routed = route_required_calculation_call(
        (ModelUserMessage(content=requested_text),),
        runtime_tool_specifications(),
        invocation_id=12,
    )

    assert routed is not None
    assert routed.reason is CalculationRoutingReason.EXPLICIT_TEMPERATURE
    assert routed.request.name == ToolName.CALCULATE
    assert CalculateArguments.model_validate_json(routed.request.arguments_json) == (
        CalculateArguments(expression=expected_expression)
    )


def test_contextual_temperature_range_routes_one_calculator_expression_per_value() -> None:
    routed = route_required_calculation_call(
        (
            ModelUserMessage(content="What's the current temperature in New York?"),
            ModelAssistantMessage(content="The high is 32°F and the low is 28°F."),
            ModelUserMessage(content="What's that in Celsius?"),
        ),
        runtime_tool_specifications(),
        invocation_id=13,
        call_id="qwen-13-tool-1",
    )

    assert routed is not None
    assert routed.reason is CalculationRoutingReason.CONTEXTUAL_TEMPERATURE
    assert routed.request.id == "qwen-13-tool-1"
    arguments = CalculateArguments.model_validate_json(routed.request.arguments_json)
    assert arguments.expression == "(32 - 32) * 5 / 9, (28 - 32) * 5 / 9"
    assert (
        temperature_conversion_spoken_result(routed.conversion, "0.0, -2.2222222222222223")
        == "32 degrees fahrenheit is 0 degrees celsius, and 28 degrees fahrenheit is -2.22 "
        "degrees celsius."
    )


@pytest.mark.parametrize(
    "requested_text",
    (
        "Tell me a story.",
        "It is 28 Celsius today.",
        "Is 32 Fahrenheit cold?",
    ),
)
def test_non_conversion_request_is_not_routed_to_calculate(requested_text: str) -> None:
    assert (
        route_required_calculation_call(
            (ModelUserMessage(content=requested_text),),
            runtime_tool_specifications(),
            14,
        )
        is None
    )

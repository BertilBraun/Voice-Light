from __future__ import annotations

import pytest

from app.compute.voice.conversation import (
    ModelAssistantMessage,
    ModelMessage,
    ModelToolMessage,
    ModelUserMessage,
)
from app.compute.voice.tool_routing import SearchRoutingReason, route_required_search_call
from app.compute.voice.tools import (
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

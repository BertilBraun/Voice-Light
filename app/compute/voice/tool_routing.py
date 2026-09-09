from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.compute.voice.conversation import (
    ModelAssistantMessage,
    ModelMessage,
    ModelToolMessage,
    ModelUserMessage,
)
from app.compute.voice.tools import (
    SearchArguments,
    SerializedToolCall,
    ToolName,
    ToolSpecification,
)

NORMALIZED_WORD_PATTERN = re.compile(r"[^a-z0-9]+")
SEARCH_ACTION_PATTERN = re.compile(r"\b(search|browse|look(?:\s+[a-z0-9]+){0,2}\s+up|lookup)\b")
WEATHER_PATTERN = re.compile(r"\b(weather|forecast)\b")
CURRENT_NEWS_PATTERN = re.compile(
    r"\b(current|latest|today(?:'s)?)\b.*\bnews\b|\bnews\b.*\b(current|latest|today(?:'s)?)\b"
)
EXPLICIT_EXTERNAL_TOOL_PATTERN = re.compile(
    r"\btool\s+call\b.*\b(check|find|current|latest|weather|forecast|news)\b"
)
CONFIRMATIONS = frozenset(
    {
        "do so",
        "do that",
        "go ahead",
        "please",
        "please do so",
        "yes",
        "yes please",
        "yeah",
    }
)


class SearchRoutingReason(StrEnum):
    DIRECT_REQUEST = "direct_request"
    CONFIRMED_REQUEST = "confirmed_request"


@dataclass(frozen=True)
class RoutedSearchCall:
    request: SerializedToolCall
    reason: SearchRoutingReason


def route_required_search_call(
    messages: tuple[ModelMessage, ...],
    tools: tuple[ToolSpecification, ...],
    invocation_id: int,
) -> RoutedSearchCall | None:
    if invocation_id <= 0:
        raise ValueError("The model invocation ID must be positive.")
    if not any(specification.function.name is ToolName.SEARCH for specification in tools):
        return None
    latest_user_index = _latest_user_index(messages)
    if latest_user_index is None:
        return None
    if any(isinstance(message, ModelToolMessage) for message in messages[latest_user_index + 1 :]):
        return None
    latest_user = messages[latest_user_index]
    assert isinstance(latest_user, ModelUserMessage)
    if _requires_search(latest_user.content):
        return _routed_call(invocation_id, latest_user.content, SearchRoutingReason.DIRECT_REQUEST)
    if _normalized(latest_user.content) not in CONFIRMATIONS:
        return None
    previous_assistant_index = _previous_assistant_index(messages, latest_user_index)
    if previous_assistant_index is None:
        return None
    previous_assistant = messages[previous_assistant_index]
    assert isinstance(previous_assistant, ModelAssistantMessage)
    if not _offers_search(previous_assistant.content):
        return None
    for message in reversed(messages[:previous_assistant_index]):
        if isinstance(message, ModelUserMessage) and _requires_search(message.content):
            return _routed_call(
                invocation_id,
                message.content,
                SearchRoutingReason.CONFIRMED_REQUEST,
            )
    return None


def _routed_call(
    invocation_id: int,
    query: str,
    reason: SearchRoutingReason,
) -> RoutedSearchCall:
    arguments = SearchArguments(query=query.strip())
    return RoutedSearchCall(
        request=SerializedToolCall(
            id=f"qwen-{invocation_id}-routed-search-1",
            name=ToolName.SEARCH,
            arguments_json=arguments.model_dump_json(),
        ),
        reason=reason,
    )


def _latest_user_index(messages: tuple[ModelMessage, ...]) -> int | None:
    return next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], ModelUserMessage)
        ),
        None,
    )


def _previous_assistant_index(
    messages: tuple[ModelMessage, ...],
    before_index: int,
) -> int | None:
    return next(
        (
            index
            for index in range(before_index - 1, -1, -1)
            if isinstance(messages[index], ModelAssistantMessage)
        ),
        None,
    )


def _requires_search(text: str) -> bool:
    normalized = _normalized(text)
    return any(
        pattern.search(normalized) is not None
        for pattern in (
            SEARCH_ACTION_PATTERN,
            WEATHER_PATTERN,
            CURRENT_NEWS_PATTERN,
            EXPLICIT_EXTERNAL_TOOL_PATTERN,
        )
    )


def _offers_search(text: str) -> bool:
    normalized = _normalized(text)
    return SEARCH_ACTION_PATTERN.search(normalized) is not None or (
        "check" in normalized.split() and _requires_search(normalized)
    )


def _normalized(text: str) -> str:
    return NORMALIZED_WORD_PATTERN.sub(" ", text.lower()).strip()

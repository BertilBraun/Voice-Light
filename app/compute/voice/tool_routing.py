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
    CalculateArguments,
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
TEMPERATURE_TARGET_PATTERN = re.compile(
    r"\b(?:in|into|to)\s+(?:degrees?\s+)?(?P<unit>celsius|fahrenheit|kelvin)\b",
    re.IGNORECASE,
)
TEMPERATURE_UNIT_PATTERN = re.compile(
    r"\b(?P<unit>celsius|fahrenheit|kelvin)\b|°\s*(?P<symbol>[cf])\b",
    re.IGNORECASE,
)
DIGIT_VALUE_PATTERN = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:degrees?)?\s*$",
    re.IGNORECASE,
)
NUMBER_WORDS = dict(
    zip(
        (
            "zero one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen"
        ).split(),
        range(20),
        strict=True,
    )
)
TENS_WORDS = dict(
    zip(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(),
        range(20, 100, 10),
        strict=True,
    )
)


class SearchRoutingReason(StrEnum):
    DIRECT_REQUEST = "direct_request"
    CONFIRMED_REQUEST = "confirmed_request"


class CalculationRoutingReason(StrEnum):
    EXPLICIT_TEMPERATURE = "explicit_temperature"
    CONTEXTUAL_TEMPERATURE = "contextual_temperature"


class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"


@dataclass(frozen=True)
class RoutedSearchCall:
    request: SerializedToolCall
    reason: SearchRoutingReason


@dataclass(frozen=True)
class RoutedCalculationCall:
    request: SerializedToolCall
    reason: CalculationRoutingReason
    conversion: TemperatureConversion


@dataclass(frozen=True)
class TemperatureObservation:
    value: str
    unit: TemperatureUnit


@dataclass(frozen=True)
class TemperatureConversion:
    sources: tuple[TemperatureObservation, ...]
    target: TemperatureUnit


def required_search_reason(messages: tuple[ModelMessage, ...]) -> SearchRoutingReason | None:
    latest_user_index = _latest_user_index(messages)
    if latest_user_index is None:
        return None
    if any(isinstance(message, ModelToolMessage) for message in messages[latest_user_index + 1 :]):
        return None
    latest_user = messages[latest_user_index]
    assert isinstance(latest_user, ModelUserMessage)
    if _requires_search(latest_user.content):
        return SearchRoutingReason.DIRECT_REQUEST
    if _normalized(latest_user.content) not in CONFIRMATIONS:
        return None
    previous_assistant_index = _previous_assistant_index(messages, latest_user_index)
    if previous_assistant_index is None:
        return None
    previous_assistant = messages[previous_assistant_index]
    assert isinstance(previous_assistant, ModelAssistantMessage)
    if not _offers_search(previous_assistant.content):
        return None
    if any(
        isinstance(message, ModelUserMessage) and _requires_search(message.content)
        for message in messages[:previous_assistant_index]
    ):
        return SearchRoutingReason.CONFIRMED_REQUEST
    return None


def route_required_search_call(
    messages: tuple[ModelMessage, ...],
    tools: tuple[ToolSpecification, ...],
    invocation_id: int,
    call_id: str | None = None,
) -> RoutedSearchCall | None:
    if invocation_id <= 0:
        raise ValueError("The model invocation ID must be positive.")
    if not any(specification.function.name is ToolName.SEARCH for specification in tools):
        return None
    reason = required_search_reason(messages)
    if reason is None:
        return None
    latest_user_index = _latest_user_index(messages)
    assert latest_user_index is not None
    latest_user = messages[latest_user_index]
    assert isinstance(latest_user, ModelUserMessage)
    if reason is SearchRoutingReason.DIRECT_REQUEST:
        query = latest_user.content
    else:
        previous_assistant_index = _previous_assistant_index(messages, latest_user_index)
        assert previous_assistant_index is not None
        query = next(
            message.content
            for message in reversed(messages[:previous_assistant_index])
            if isinstance(message, ModelUserMessage) and _requires_search(message.content)
        )
    return _routed_call(invocation_id, query, reason, call_id)


def route_required_calculation_call(
    messages: tuple[ModelMessage, ...],
    tools: tuple[ToolSpecification, ...],
    invocation_id: int,
    call_id: str | None = None,
) -> RoutedCalculationCall | None:
    if invocation_id <= 0:
        raise ValueError("The model invocation ID must be positive.")
    if not any(specification.function.name is ToolName.CALCULATE for specification in tools):
        return None
    latest_user_index = _latest_user_index(messages)
    if latest_user_index is None:
        return None
    if any(isinstance(message, ModelToolMessage) for message in messages[latest_user_index + 1 :]):
        return None
    latest_user = messages[latest_user_index]
    assert isinstance(latest_user, ModelUserMessage)
    target_match = TEMPERATURE_TARGET_PATTERN.search(latest_user.content)
    if target_match is None:
        return None
    target_unit = TemperatureUnit(target_match.group("unit").lower())
    sources = _temperatures_before(latest_user.content, target_match.start())
    reason = CalculationRoutingReason.EXPLICIT_TEMPERATURE
    if not sources:
        sources = _temperatures_from_context(messages[:latest_user_index])
        reason = CalculationRoutingReason.CONTEXTUAL_TEMPERATURE
    sources = tuple(source for source in sources if source.unit is not target_unit)
    if not sources:
        return None
    conversion = TemperatureConversion(sources=sources, target=target_unit)
    arguments = CalculateArguments(expression=_temperature_expression(conversion))
    return RoutedCalculationCall(
        request=SerializedToolCall(
            id=call_id or f"qwen-{invocation_id}-routed-calculate-1",
            name=ToolName.CALCULATE,
            arguments_json=arguments.model_dump_json(),
        ),
        reason=reason,
        conversion=conversion,
    )


def temperature_conversion_spoken_result(
    conversion: TemperatureConversion,
    calculated_result: str,
) -> str:
    results = tuple(_spoken_decimal(result) for result in calculated_result.split(","))
    if len(results) != len(conversion.sources):
        raise ValueError("The calculation result does not match the temperature inputs.")
    statements = tuple(
        f"{source.value} degrees {source.unit.value} is {result} degrees {conversion.target.value}"
        for source, result in zip(conversion.sources, results, strict=True)
    )
    return ", and ".join(statements) + "."


def _spoken_decimal(value: str) -> str:
    formatted = f"{float(value.strip()):.2f}".rstrip("0").rstrip(".")
    return "0" if formatted == "-0" else formatted


def _routed_call(
    invocation_id: int,
    query: str,
    reason: SearchRoutingReason,
    call_id: str | None,
) -> RoutedSearchCall:
    arguments = SearchArguments(query=query.strip())
    return RoutedSearchCall(
        request=SerializedToolCall(
            id=call_id or f"qwen-{invocation_id}-routed-search-1",
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


def _temperatures_from_context(
    messages: tuple[ModelMessage, ...],
) -> tuple[TemperatureObservation, ...]:
    for message in reversed(messages):
        if isinstance(message, ModelAssistantMessage):
            observations = _temperatures_before(message.content, len(message.content))
            if observations:
                return observations
    return ()


def _temperatures_before(text: str, before: int) -> tuple[TemperatureObservation, ...]:
    observations: list[TemperatureObservation] = []
    for unit_match in TEMPERATURE_UNIT_PATTERN.finditer(text, 0, before):
        value = _temperature_value_before(text, unit_match.start())
        if value is None:
            continue
        unit_text = unit_match.group("unit") or unit_match.group("symbol")
        assert unit_text is not None
        unit = {
            "c": TemperatureUnit.CELSIUS,
            "celsius": TemperatureUnit.CELSIUS,
            "f": TemperatureUnit.FAHRENHEIT,
            "fahrenheit": TemperatureUnit.FAHRENHEIT,
            "kelvin": TemperatureUnit.KELVIN,
        }[unit_text.lower()]
        observations.append(TemperatureObservation(value=value, unit=unit))
    return tuple(observations)


def _temperature_value_before(text: str, before: int) -> str | None:
    prefix = text[:before]
    digit_match = DIGIT_VALUE_PATTERN.search(prefix)
    if digit_match is not None:
        return digit_match.group("value")
    normalized_words = _normalized(prefix).split()
    if normalized_words and normalized_words[-1] in {"degree", "degrees"}:
        normalized_words.pop()
    for word_count in range(min(3, len(normalized_words)), 0, -1):
        parsed = _parse_spoken_number(normalized_words[-word_count:])
        if parsed is not None:
            return parsed
    return None


def _parse_spoken_number(words: list[str]) -> str | None:
    sign = 1
    if words and words[0] in {"minus", "negative"}:
        sign = -1
        words = words[1:]
    if not words:
        return None
    if len(words) == 1 and words[0] in NUMBER_WORDS:
        value = NUMBER_WORDS[words[0]]
    elif len(words) == 1 and words[0] in TENS_WORDS:
        value = TENS_WORDS[words[0]]
    elif len(words) == 2 and words[0] in TENS_WORDS and words[1] in NUMBER_WORDS:
        value = TENS_WORDS[words[0]] + NUMBER_WORDS[words[1]]
    else:
        return None
    return str(sign * value)


def _temperature_expression(conversion: TemperatureConversion) -> str:
    target = conversion.target
    return ", ".join(
        _single_temperature_expression(source, target) for source in conversion.sources
    )


def _single_temperature_expression(
    source: TemperatureObservation,
    target: TemperatureUnit,
) -> str:
    value = source.value
    match source.unit, target:
        case TemperatureUnit.CELSIUS, TemperatureUnit.FAHRENHEIT:
            return f"({value} * 9 / 5) + 32"
        case TemperatureUnit.CELSIUS, TemperatureUnit.KELVIN:
            return f"{value} + 273.15"
        case TemperatureUnit.FAHRENHEIT, TemperatureUnit.CELSIUS:
            return f"({value} - 32) * 5 / 9"
        case TemperatureUnit.FAHRENHEIT, TemperatureUnit.KELVIN:
            return f"({value} - 32) * 5 / 9 + 273.15"
        case TemperatureUnit.KELVIN, TemperatureUnit.CELSIUS:
            return f"{value} - 273.15"
        case TemperatureUnit.KELVIN, TemperatureUnit.FAHRENHEIT:
            return f"({value} - 273.15) * 9 / 5 + 32"
        case _:
            raise AssertionError("Identical temperature units do not require calculation.")

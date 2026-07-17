from __future__ import annotations

import pytest

from app.compute.voice.hermes_tool_parser import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    HermesParserEvent,
    HermesSpokenText,
    HermesToolCallCompleted,
    HermesToolCallFailed,
    HermesToolCallParser,
    HermesToolCallStarted,
)
from app.compute.voice.tools import ToolCallFailureReason

RAW_WEATHER_RESPONSE = (
    "Let me check that."
    '<tool_call>\n{"name":"get_weather","arguments":{"location":"London"}}\n</tool_call>'
)


@pytest.mark.parametrize("split_index", range(len(RAW_WEATHER_RESPONSE) + 1))
def test_hermes_parser_is_stable_at_every_two_chunk_split(split_index: int) -> None:
    events = parse_chunks(
        RAW_WEATHER_RESPONSE[:split_index],
        RAW_WEATHER_RESPONSE[split_index:],
    )

    assert_weather_events(events)


def test_hermes_parser_is_stable_at_every_character_boundary() -> None:
    events = parse_chunks(*RAW_WEATHER_RESPONSE)

    assert_weather_events(events)


@pytest.mark.parametrize("prefix_length", range(1, len(TOOL_CALL_START)))
def test_partial_opening_delimiter_never_escapes_to_spoken_text(prefix_length: int) -> None:
    parser = HermesToolCallParser(invocation_id=3)

    events = parser.add_text(f"Bridge.{TOOL_CALL_START[:prefix_length]}")

    assert spoken_text(events) == "Bridge."
    assert TOOL_CALL_START[:prefix_length] not in spoken_text(events)
    final_events = parser.finish()
    assert isinstance(final_events[-1], HermesToolCallFailed)
    assert final_events[-1].failure.reason is ToolCallFailureReason.INCOMPLETE_CALL


@pytest.mark.parametrize("prefix_length", range(1, len(TOOL_CALL_END)))
def test_partial_closing_delimiter_never_escapes_to_spoken_text(prefix_length: int) -> None:
    parser = HermesToolCallParser(invocation_id=3)

    events = parser.add_text(
        "Bridge."
        f"{TOOL_CALL_START}"
        '{"name":"get_weather","arguments":{"location":"London"}}'
        f"{TOOL_CALL_END[:prefix_length]}"
    )

    assert spoken_text(events) == "Bridge."
    final_events = parser.finish()
    assert isinstance(final_events[-1], HermesToolCallFailed)
    assert final_events[-1].failure.reason is ToolCallFailureReason.INCOMPLETE_CALL


@pytest.mark.parametrize(
    "payload",
    (
        "{",
        "[]",
        '{"name":"get_weather"}',
        '{"arguments":{"location":"London"}}',
        '{"name":"get_weather","arguments":{"location":"London"},"extra":true}',
    ),
)
def test_malformed_hermes_payload_is_a_typed_failure(payload: str) -> None:
    events = parse_chunks(f"Bridge.{TOOL_CALL_START}{payload}{TOOL_CALL_END}")

    assert isinstance(events[-1], HermesToolCallFailed)
    assert events[-1].failure.reason is ToolCallFailureReason.MALFORMED_JSON
    assert spoken_text(events) == "Bridge."


def test_duplicate_call_is_rejected_after_first_call_is_buffered() -> None:
    envelope = (
        f"{TOOL_CALL_START}"
        '{"name":"get_weather","arguments":{"location":"London"}}'
        f"{TOOL_CALL_END}"
    )

    events = parse_chunks(f"Bridge.{envelope}{envelope}")

    assert isinstance(events[-1], HermesToolCallFailed)
    assert events[-1].failure.reason is ToolCallFailureReason.DUPLICATE_CALL


def test_multiple_calls_are_rejected_after_first_call_is_buffered() -> None:
    first = (
        f"{TOOL_CALL_START}"
        '{"name":"get_weather","arguments":{"location":"London"}}'
        f"{TOOL_CALL_END}"
    )
    second = (
        f"{TOOL_CALL_START}"
        '{"name":"get_weather","arguments":{"location":"Berlin"}}'
        f"{TOOL_CALL_END}"
    )

    events = parse_chunks(f"Bridge.{first}{second}")

    assert isinstance(events[-1], HermesToolCallFailed)
    assert events[-1].failure.reason is ToolCallFailureReason.MULTIPLE_CALLS


def test_text_only_response_is_unchanged() -> None:
    raw_text = "Ordinary response with <toolish text."

    events = parse_chunks(*raw_text)

    assert spoken_text(events) == raw_text
    assert all(isinstance(event, HermesSpokenText) for event in events)


def parse_chunks(*chunks: str) -> tuple[HermesParserEvent, ...]:
    parser = HermesToolCallParser(invocation_id=3)
    events: list[HermesParserEvent] = []
    for chunk in chunks:
        events.extend(parser.add_text(chunk))
    events.extend(parser.finish())
    return tuple(events)


def assert_weather_events(events: tuple[HermesParserEvent, ...]) -> None:
    assert spoken_text(events) == "Let me check that."
    starts = [event for event in events if isinstance(event, HermesToolCallStarted)]
    calls = [event for event in events if isinstance(event, HermesToolCallCompleted)]
    failures = [event for event in events if isinstance(event, HermesToolCallFailed)]
    assert starts == [HermesToolCallStarted(call_id="qwen-3-tool-1")]
    assert len(calls) == 1
    assert calls[0].request.name == "get_weather"
    assert calls[0].request.arguments_json == '{"location":"London"}'
    assert failures == []
    assert "<tool_call>" not in spoken_text(events)
    assert "location" not in spoken_text(events)


def spoken_text(events: tuple[HermesParserEvent, ...]) -> str:
    return "".join(event.text for event in events if isinstance(event, HermesSpokenText))

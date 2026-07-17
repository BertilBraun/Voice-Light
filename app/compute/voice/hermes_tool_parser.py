from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import JsonValue, ValidationError

from app.compute.voice.tools import (
    SerializedToolCall,
    ToolCallFailure,
    ToolCallFailureReason,
)
from app.shared.base_model import FrozenBaseModel

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


class HermesParserState(StrEnum):
    SPOKEN_TEXT = "spoken_text"
    TOOL_CALL = "tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    SECOND_TOOL_CALL = "second_tool_call"
    FAILED = "failed"


@dataclass(frozen=True)
class HermesSpokenText:
    text: str


@dataclass(frozen=True)
class HermesToolCallStarted:
    call_id: str


@dataclass(frozen=True)
class HermesToolCallCompleted:
    request: SerializedToolCall


@dataclass(frozen=True)
class HermesToolCallFailed:
    failure: ToolCallFailure


HermesParserEvent = (
    HermesSpokenText | HermesToolCallStarted | HermesToolCallCompleted | HermesToolCallFailed
)


class _RawHermesToolCall(FrozenBaseModel):
    name: str
    arguments: JsonValue


class HermesToolCallParser:
    def __init__(self, invocation_id: int) -> None:
        if invocation_id <= 0:
            raise ValueError("The invocation ID must be positive.")
        self.invocation_id = invocation_id
        self.state = HermesParserState.SPOKEN_TEXT
        self.buffer = ""
        self.first_call: SerializedToolCall | None = None

    def add_text(self, text: str) -> tuple[HermesParserEvent, ...]:
        if self.state is HermesParserState.FAILED or not text:
            return ()
        self.buffer += text
        events: list[HermesParserEvent] = []
        while True:
            match self.state:
                case HermesParserState.SPOKEN_TEXT:
                    start_index = self.buffer.find(TOOL_CALL_START)
                    if start_index >= 0:
                        spoken_text = self.buffer[:start_index]
                        if spoken_text:
                            events.append(HermesSpokenText(spoken_text))
                        self.buffer = self.buffer[start_index + len(TOOL_CALL_START) :]
                        self.state = HermesParserState.TOOL_CALL
                        events.append(HermesToolCallStarted(self._call_id(1)))
                        continue
                    retained_length = _ambiguous_suffix_length(self.buffer, TOOL_CALL_START)
                    released_length = len(self.buffer) - retained_length
                    if released_length:
                        events.append(HermesSpokenText(self.buffer[:released_length]))
                        self.buffer = self.buffer[released_length:]
                    break
                case HermesParserState.TOOL_CALL:
                    end_index = self.buffer.find(TOOL_CALL_END)
                    if end_index < 0:
                        break
                    payload = self.buffer[:end_index]
                    self.buffer = self.buffer[end_index + len(TOOL_CALL_END) :]
                    parsed_event = self._parse_call(payload, call_index=1)
                    events.append(parsed_event)
                    if isinstance(parsed_event, HermesToolCallFailed):
                        self.state = HermesParserState.FAILED
                        self.buffer = ""
                        break
                    self.first_call = parsed_event.request
                    self.state = HermesParserState.AFTER_TOOL_CALL
                    continue
                case HermesParserState.AFTER_TOOL_CALL:
                    if not self.buffer:
                        break
                    stripped_buffer = self.buffer.lstrip()
                    if not stripped_buffer:
                        self.buffer = ""
                        break
                    if stripped_buffer.startswith(TOOL_CALL_START):
                        self.buffer = stripped_buffer[len(TOOL_CALL_START) :]
                        self.state = HermesParserState.SECOND_TOOL_CALL
                        events.append(HermesToolCallStarted(self._call_id(2)))
                        continue
                    retained_length = _ambiguous_suffix_length(
                        stripped_buffer,
                        TOOL_CALL_START,
                    )
                    if retained_length == len(stripped_buffer):
                        self.buffer = stripped_buffer
                        break
                    events.append(
                        self._failure(
                            call_index=2,
                            reason=ToolCallFailureReason.UNEXPECTED_CONTENT,
                            message="The model emitted content after its tool call.",
                            attempted_tool_name=None,
                        )
                    )
                    self.state = HermesParserState.FAILED
                    self.buffer = ""
                    break
                case HermesParserState.SECOND_TOOL_CALL:
                    end_index = self.buffer.find(TOOL_CALL_END)
                    if end_index < 0:
                        break
                    payload = self.buffer[:end_index]
                    second_event = self._parse_call(payload, call_index=2)
                    reason = ToolCallFailureReason.MULTIPLE_CALLS
                    attempted_tool_name: str | None = None
                    if isinstance(second_event, HermesToolCallCompleted):
                        attempted_tool_name = second_event.request.name
                        if (
                            second_event.request.name == self._require_first_call().name
                            and second_event.request.arguments_json
                            == self._require_first_call().arguments_json
                        ):
                            reason = ToolCallFailureReason.DUPLICATE_CALL
                    elif second_event.failure.attempted_tool_name is not None:
                        attempted_tool_name = second_event.failure.attempted_tool_name
                    events.append(
                        self._failure(
                            call_index=2,
                            reason=reason,
                            message="Only one tool call is allowed in this milestone.",
                            attempted_tool_name=attempted_tool_name,
                        )
                    )
                    self.state = HermesParserState.FAILED
                    self.buffer = ""
                    break
                case HermesParserState.FAILED:
                    break
        return tuple(events)

    def finish(self) -> tuple[HermesParserEvent, ...]:
        match self.state:
            case HermesParserState.SPOKEN_TEXT:
                if not self.buffer:
                    return ()
                if TOOL_CALL_START.startswith(self.buffer):
                    self.state = HermesParserState.FAILED
                    self.buffer = ""
                    return (
                        self._failure(
                            call_index=1,
                            reason=ToolCallFailureReason.INCOMPLETE_CALL,
                            message="The model ended inside a tool-call delimiter.",
                            attempted_tool_name=None,
                        ),
                    )
                event = HermesSpokenText(self.buffer)
                self.buffer = ""
                return (event,)
            case HermesParserState.TOOL_CALL | HermesParserState.SECOND_TOOL_CALL:
                call_index = 1 if self.state is HermesParserState.TOOL_CALL else 2
                self.state = HermesParserState.FAILED
                self.buffer = ""
                return (
                    self._failure(
                        call_index=call_index,
                        reason=ToolCallFailureReason.INCOMPLETE_CALL,
                        message="The model did not close its tool call.",
                        attempted_tool_name=None,
                    ),
                )
            case HermesParserState.AFTER_TOOL_CALL:
                if self.buffer and TOOL_CALL_START.startswith(self.buffer.lstrip()):
                    self.state = HermesParserState.FAILED
                    self.buffer = ""
                    return (
                        self._failure(
                            call_index=2,
                            reason=ToolCallFailureReason.INCOMPLETE_CALL,
                            message="The model ended inside a second tool-call delimiter.",
                            attempted_tool_name=None,
                        ),
                    )
                self.buffer = ""
                return ()
            case HermesParserState.FAILED:
                self.buffer = ""
                return ()

    def _parse_call(self, payload: str, call_index: int) -> HermesParserEvent:
        try:
            decoded_payload = json.loads(payload)
            raw_call = _RawHermesToolCall.model_validate(decoded_payload)
        except (json.JSONDecodeError, ValidationError) as error:
            return self._failure(
                call_index=call_index,
                reason=ToolCallFailureReason.MALFORMED_JSON,
                message=f"The tool call was not valid Hermes JSON: {error}",
                attempted_tool_name=None,
            )
        return HermesToolCallCompleted(
            SerializedToolCall(
                id=self._call_id(call_index),
                name=raw_call.name,
                arguments_json=json.dumps(
                    raw_call.arguments,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
        )

    def _failure(
        self,
        call_index: int,
        reason: ToolCallFailureReason,
        message: str,
        attempted_tool_name: str | None,
    ) -> HermesToolCallFailed:
        return HermesToolCallFailed(
            ToolCallFailure(
                call_id=self._call_id(call_index),
                reason=reason,
                message=message,
                attempted_tool_name=attempted_tool_name,
            )
        )

    def _call_id(self, call_index: int) -> str:
        return f"qwen-{self.invocation_id}-tool-{call_index}"

    def _require_first_call(self) -> SerializedToolCall:
        if self.first_call is None:
            raise AssertionError("The parser has no completed first tool call.")
        return self.first_call


def _ambiguous_suffix_length(text: str, delimiter: str) -> int:
    maximum_length = min(len(text), len(delimiter) - 1)
    for suffix_length in range(maximum_length, 0, -1):
        if delimiter.startswith(text[-suffix_length:]):
            return suffix_length
    return 0

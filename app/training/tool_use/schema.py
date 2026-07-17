from __future__ import annotations

import json
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolUseBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class ToolLatency(StrEnum):
    IMMEDIATE = "immediate"
    LATENCY_BEARING = "latency_bearing"


class SpeechStyle(StrEnum):
    CLEAN = "clean"
    ASR_FRAGMENT = "asr_fragment"
    REPAIR = "repair"
    CASUAL = "casual"
    FORMAL = "formal"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class LengthBand(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class StringParameterSchema(ToolUseBaseModel):
    type: Literal["string"] = "string"
    description: str
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=1)
    enum: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_lengths(self) -> StringParameterSchema:
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("String parameter min_length cannot exceed max_length.")
        return self


class NumberParameterSchema(ToolUseBaseModel):
    type: Literal["number"] = "number"
    description: str
    minimum: float | None = None
    maximum: float | None = None


class IntegerParameterSchema(ToolUseBaseModel):
    type: Literal["integer"] = "integer"
    description: str
    minimum: int | None = None
    maximum: int | None = None


class BooleanParameterSchema(ToolUseBaseModel):
    type: Literal["boolean"] = "boolean"
    description: str


class NullParameterSchema(ToolUseBaseModel):
    type: Literal["null"] = "null"
    description: str


class ArrayParameterSchema(ToolUseBaseModel):
    type: Literal["array"] = "array"
    description: str
    items: ParameterSchema
    min_items: int | None = Field(default=None, ge=0)
    max_items: int | None = Field(default=None, ge=0)


class NamedParameterSchema(ToolUseBaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    required: bool
    value_schema: ParameterSchema


class ObjectParameterSchema(ToolUseBaseModel):
    type: Literal["object"] = "object"
    description: str
    properties: tuple[NamedParameterSchema, ...]
    additional_properties: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_properties(self) -> ObjectParameterSchema:
        names = tuple(property_schema.name for property_schema in self.properties)
        if len(names) != len(set(names)):
            raise ValueError("Tool parameter names must be unique.")
        return self


ParameterSchema: TypeAlias = Annotated[
    StringParameterSchema
    | NumberParameterSchema
    | IntegerParameterSchema
    | BooleanParameterSchema
    | NullParameterSchema
    | ArrayParameterSchema
    | ObjectParameterSchema,
    Field(discriminator="type"),
]

ArrayParameterSchema.model_rebuild(_types_namespace={"ParameterSchema": ParameterSchema})
NamedParameterSchema.model_rebuild(_types_namespace={"ParameterSchema": ParameterSchema})
ObjectParameterSchema.model_rebuild(_types_namespace={"ParameterSchema": ParameterSchema})


class JsonStringValue(ToolUseBaseModel):
    kind: Literal["string"] = "string"
    value: str


class JsonNumberValue(ToolUseBaseModel):
    kind: Literal["number"] = "number"
    value: float


class JsonIntegerValue(ToolUseBaseModel):
    kind: Literal["integer"] = "integer"
    value: int


class JsonBooleanValue(ToolUseBaseModel):
    kind: Literal["boolean"] = "boolean"
    value: bool


class JsonNullValue(ToolUseBaseModel):
    kind: Literal["null"] = "null"


class JsonArrayValue(ToolUseBaseModel):
    kind: Literal["array"] = "array"
    items: tuple[JsonValue, ...]


class JsonObjectField(ToolUseBaseModel):
    name: str
    value: JsonValue


class JsonObjectValue(ToolUseBaseModel):
    kind: Literal["object"] = "object"
    fields: tuple[JsonObjectField, ...]

    @model_validator(mode="after")
    def validate_unique_fields(self) -> JsonObjectValue:
        names = tuple(field.name for field in self.fields)
        if len(names) != len(set(names)):
            raise ValueError("JSON object field names must be unique.")
        return self


JsonValue: TypeAlias = Annotated[
    JsonStringValue
    | JsonNumberValue
    | JsonIntegerValue
    | JsonBooleanValue
    | JsonNullValue
    | JsonArrayValue
    | JsonObjectValue,
    Field(discriminator="kind"),
]

JsonArrayValue.model_rebuild(_types_namespace={"JsonValue": JsonValue})
JsonObjectField.model_rebuild(_types_namespace={"JsonValue": JsonValue})
JsonObjectValue.model_rebuild(_types_namespace={"JsonValue": JsonValue})


class ToolDefinition(ToolUseBaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    latency: ToolLatency
    parameters: ObjectParameterSchema
    result_description: str = Field(min_length=1)


class ToolCall(ToolUseBaseModel):
    call_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    tool_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    arguments: JsonObjectValue


class Citation(ToolUseBaseModel):
    citation_id: str
    url: str
    title: str
    publisher: str | None
    published_at: str | None
    retrieved_at: str | None


class SuccessOutcome(ToolUseBaseModel):
    status: Literal["success"] = "success"
    content: str


class EmptyOutcome(ToolUseBaseModel):
    status: Literal["empty"] = "empty"
    message: str


class FailureOutcome(ToolUseBaseModel):
    status: Literal["failure"] = "failure"
    message: str


class TimeoutOutcome(ToolUseBaseModel):
    status: Literal["timeout"] = "timeout"
    message: str


class CancelledOutcome(ToolUseBaseModel):
    status: Literal["cancelled"] = "cancelled"
    message: str


ToolOutcome: TypeAlias = Annotated[
    SuccessOutcome | EmptyOutcome | FailureOutcome | TimeoutOutcome | CancelledOutcome,
    Field(discriminator="status"),
]


class SystemMessage(ToolUseBaseModel):
    kind: Literal["system"] = "system"
    message_id: str
    text: str


class UserMessage(ToolUseBaseModel):
    kind: Literal["user"] = "user"
    message_id: str
    text: str
    speech_style: SpeechStyle


class AssistantMessage(ToolUseBaseModel):
    kind: Literal["assistant"] = "assistant"
    message_id: str
    audible_text: str
    tool_calls: tuple[ToolCall, ...] = ()


class ToolResultMessage(ToolUseBaseModel):
    kind: Literal["tool_result"] = "tool_result"
    message_id: str
    call_id: str
    outcome: ToolOutcome
    citations: tuple[Citation, ...] = ()


ConversationMessage: TypeAlias = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="kind"),
]


class GenerationProvenance(ToolUseBaseModel):
    method: Literal["authored", "synthetic"]
    model_identifier: str | None
    model_revision: str | None
    quantization: str | None
    prompt_revision: str
    random_seed: int | None


class ScenarioMetadata(ToolUseBaseModel):
    family: str
    tool_need: Literal["not_needed", "required", "multiple_required"]
    expected_tool_names: tuple[str, ...]
    outcome: str
    language: Literal["en"] = "en"
    length_band: LengthBand
    speech_style: SpeechStyle
    user_turn_count: int = Field(ge=1)


class SplitMetadata(ToolUseBaseModel):
    name: DatasetSplit
    policy_revision: str
    leakage_group_id: str


class AuditMetadata(ToolUseBaseModel):
    source_license: str
    intended_license: str
    human_review: Literal["pending", "approved", "rejected"]
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RecordMetadata(ToolUseBaseModel):
    generation: GenerationProvenance
    scenario: ScenarioMetadata
    split: SplitMetadata
    audit: AuditMetadata


class ToolDialogueRecord(ToolUseBaseModel):
    schema_version: Literal["voice-light.tool-dialogue/v1"] = "voice-light.tool-dialogue/v1"
    record_id: str
    tools: tuple[ToolDefinition, ...]
    messages: tuple[ConversationMessage, ...]
    metadata: RecordMetadata

    @model_validator(mode="after")
    def validate_conversation(self) -> ToolDialogueRecord:
        if not self.messages:
            raise ValueError("A tool dialogue must begin with one system message.")
        match self.messages[0]:
            case SystemMessage():
                pass
            case _:
                raise ValueError("A tool dialogue must begin with one system message.")
        for message in self.messages[1:]:
            match message:
                case SystemMessage():
                    raise ValueError("A tool dialogue may contain only its initial system message.")
                case _:
                    pass
        tool_names = tuple(tool.name for tool in self.tools)
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Runtime tool names must be unique.")
        message_ids = tuple(message.message_id for message in self.messages)
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("Conversation message IDs must be unique.")

        seen_call_ids: set[str] = set()
        pending_call_ids: set[str] = set()
        resolved_call_ids: set[str] = set()
        for message in self.messages:
            match message:
                case AssistantMessage(tool_calls=tool_calls):
                    if pending_call_ids:
                        raise ValueError(
                            "An assistant message cannot precede pending tool results."
                        )
                    for tool_call in tool_calls:
                        if tool_call.call_id in seen_call_ids:
                            raise ValueError(f"Duplicate tool call ID: {tool_call.call_id}")
                        if tool_call.tool_name not in tool_names:
                            raise ValueError(
                                f"Unknown tool in canonical record: {tool_call.tool_name}"
                            )
                        seen_call_ids.add(tool_call.call_id)
                        pending_call_ids.add(tool_call.call_id)
                case ToolResultMessage(call_id=call_id):
                    if call_id not in pending_call_ids:
                        raise ValueError(f"Tool result does not match a pending call: {call_id}")
                    if call_id in resolved_call_ids:
                        raise ValueError(f"Duplicate tool result for call: {call_id}")
                    pending_call_ids.remove(call_id)
                    resolved_call_ids.add(call_id)
                case UserMessage():
                    if pending_call_ids:
                        raise ValueError("A user message cannot precede pending tool results.")
                case SystemMessage():
                    pass
        if pending_call_ids:
            raise ValueError("Every tool call must have one subsequent result.")
        return self


PythonJsonValue: TypeAlias = (
    str | int | float | bool | None | list["PythonJsonValue"] | dict[str, "PythonJsonValue"]
)


def json_value_to_python(value: JsonValue) -> PythonJsonValue:
    match value:
        case JsonStringValue(value=string_value):
            return string_value
        case JsonNumberValue(value=number_value):
            return number_value
        case JsonIntegerValue(value=integer_value):
            return integer_value
        case JsonBooleanValue(value=boolean_value):
            return boolean_value
        case JsonNullValue():
            return None
        case JsonArrayValue(items=items):
            return [json_value_to_python(item) for item in items]
        case JsonObjectValue(fields=fields):
            return {field.name: json_value_to_python(field.value) for field in fields}


def read_records(path: Path) -> tuple[ToolDialogueRecord, ...]:
    records: list[ToolDialogueRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(ToolDialogueRecord.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"Invalid canonical record on line {line_number}: {error}") from error
    return tuple(records)


def append_record(path: Path, record: ToolDialogueRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(record.model_dump_json())
        output_file.write("\n")


def completed_record_ids(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(record.record_id for record in read_records(path))


def validate_records(records: Iterable[ToolDialogueRecord]) -> None:
    for record in records:
        ToolDialogueRecord.model_validate(record.model_dump(mode="json"))


def contains_protocol_markup(text: str) -> bool:
    return bool(re.search(r"<\|[^>]+?\|>|</?tool_call>|</?function(?:=[^>]+)?>", text))


def canonical_json(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

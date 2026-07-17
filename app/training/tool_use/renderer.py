from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypedDict, cast

from app.training.tool_use.schema import (
    ArrayParameterSchema,
    AssistantMessage,
    BooleanParameterSchema,
    CancelledOutcome,
    ConversationMessage,
    EmptyOutcome,
    FailureOutcome,
    IntegerParameterSchema,
    NullParameterSchema,
    NumberParameterSchema,
    ObjectParameterSchema,
    ParameterSchema,
    PythonJsonValue,
    StringParameterSchema,
    SuccessOutcome,
    SystemMessage,
    TimeoutOutcome,
    ToolDefinition,
    ToolDialogueRecord,
    ToolOutcome,
    ToolResultMessage,
    UserMessage,
    json_value_to_python,
)

IGNORE_INDEX = -100


class QwenSystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class QwenUserMessage(TypedDict):
    role: Literal["user"]
    content: str


class QwenToolCallFunction(TypedDict):
    name: str
    arguments: str


class QwenToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: QwenToolCallFunction


class QwenAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str
    tool_calls: NotRequired[list[QwenToolCall]]


class QwenToolMessage(TypedDict):
    role: Literal["tool"]
    content: str
    tool_call_id: str


QwenChatMessage = QwenSystemMessage | QwenUserMessage | QwenAssistantMessage | QwenToolMessage


class QwenFunctionSpecification(TypedDict):
    name: str
    description: str
    parameters: dict[str, PythonJsonValue]


class QwenToolSpecification(TypedDict):
    type: Literal["function"]
    function: QwenFunctionSpecification


class ChatTemplateEncoding(TypedDict):
    input_ids: list[int]
    assistant_masks: NotRequired[list[int]]


class ChatTemplateTokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self,
        conversation: Sequence[QwenChatMessage],
        *,
        tools: Sequence[QwenToolSpecification],
        tokenize: Literal[True],
        add_generation_prompt: Literal[False],
        enable_thinking: Literal[False],
        chat_template: str | None = None,
        return_dict: Literal[True],
        return_assistant_tokens_mask: bool = False,
    ) -> ChatTemplateEncoding: ...


@dataclass(frozen=True)
class QwenTrainingExample:
    record_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_loss_mask: tuple[bool, ...]


def render_qwen_training_example(
    tokenizer: ChatTemplateTokenizer,
    record: ToolDialogueRecord,
) -> QwenTrainingExample:
    qwen_tools = tuple(qwen_tool_definition(tool) for tool in record.tools)
    qwen_messages = tuple(qwen_message(message) for message in record.messages)
    native_template = tokenizer.chat_template
    if native_template is None:
        raise ValueError("The Qwen tokenizer has no native chat template.")
    assistant_mask_template = add_assistant_generation_markers(native_template)
    native_encoding = cast(
        ChatTemplateEncoding,
        tokenizer.apply_chat_template(
            qwen_messages,
            tools=qwen_tools,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_dict=True,
        ),
    )
    marked_encoding = cast(
        ChatTemplateEncoding,
        tokenizer.apply_chat_template(
            qwen_messages,
            tools=qwen_tools,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            chat_template=assistant_mask_template,
            return_dict=True,
            return_assistant_tokens_mask=True,
        ),
    )
    native_ids = native_encoding["input_ids"]
    marked_ids = marked_encoding["input_ids"]
    if "assistant_masks" not in marked_encoding:
        raise ValueError("Qwen tokenizer did not return an assistant token mask.")
    assistant_loss_mask = tuple(bool(value) for value in marked_encoding["assistant_masks"])
    if native_ids != marked_ids:
        raise ValueError("Assistant mask markers changed Qwen's native rendered token IDs.")
    if len(native_ids) != len(assistant_loss_mask):
        raise ValueError("Qwen assistant mask length does not match rendered token count.")

    labels = tuple(
        token_id if include_in_loss else IGNORE_INDEX
        for token_id, include_in_loss in zip(
            native_ids,
            assistant_loss_mask,
            strict=True,
        )
    )
    if not any(assistant_loss_mask):
        raise ValueError("Rendered training record has no assistant target tokens.")
    return QwenTrainingExample(
        record_id=record.record_id,
        input_ids=tuple(native_ids),
        labels=labels,
        assistant_loss_mask=assistant_loss_mask,
    )


def render_qwen_training_records(
    tokenizer: ChatTemplateTokenizer,
    records: Sequence[ToolDialogueRecord],
) -> tuple[QwenTrainingExample, ...]:
    return tuple(render_qwen_training_example(tokenizer, record) for record in records)


def qwen_message(message: ConversationMessage) -> QwenChatMessage:
    match message:
        case SystemMessage(text=text):
            return {"role": "system", "content": text}
        case UserMessage(text=text):
            return {"role": "user", "content": text}
        case AssistantMessage(audible_text=audible_text, tool_calls=tool_calls):
            assistant_message: QwenAssistantMessage = {
                "role": "assistant",
                "content": audible_text,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(
                                json_value_to_python(tool_call.arguments),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                    for tool_call in tool_calls
                ]
            return assistant_message
        case ToolResultMessage(call_id=call_id, outcome=outcome):
            return {
                "role": "tool",
                "content": _tool_result_content(outcome),
                "tool_call_id": call_id,
            }


def qwen_tool_definition(tool: ToolDefinition) -> QwenToolSpecification:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameter_schema_to_json(tool.parameters),
        },
    }


def parameter_schema_to_json(schema: ParameterSchema) -> dict[str, PythonJsonValue]:
    match schema:
        case StringParameterSchema(
            description=description,
            min_length=min_length,
            max_length=max_length,
            enum=enum,
        ):
            result: dict[str, PythonJsonValue] = {
                "type": "string",
                "description": description,
            }
            if min_length is not None:
                result["minLength"] = min_length
            if max_length is not None:
                result["maxLength"] = max_length
            if enum is not None:
                result["enum"] = list(enum)
            return result
        case NumberParameterSchema(
            description=description,
            minimum=minimum,
            maximum=maximum,
        ):
            result = {"type": "number", "description": description}
            if minimum is not None:
                result["minimum"] = minimum
            if maximum is not None:
                result["maximum"] = maximum
            return result
        case IntegerParameterSchema(
            description=description,
            minimum=minimum,
            maximum=maximum,
        ):
            result = {"type": "integer", "description": description}
            if minimum is not None:
                result["minimum"] = minimum
            if maximum is not None:
                result["maximum"] = maximum
            return result
        case BooleanParameterSchema(description=description):
            return {"type": "boolean", "description": description}
        case NullParameterSchema(description=description):
            return {"type": "null", "description": description}
        case ArrayParameterSchema(
            description=description,
            items=items,
            min_items=min_items,
            max_items=max_items,
        ):
            result = {
                "type": "array",
                "description": description,
                "items": parameter_schema_to_json(items),
            }
            if min_items is not None:
                result["minItems"] = min_items
            if max_items is not None:
                result["maxItems"] = max_items
            return result
        case ObjectParameterSchema(
            description=description,
            properties=properties,
            additional_properties=additional_properties,
        ):
            property_values: dict[str, PythonJsonValue] = {
                property_schema.name: parameter_schema_to_json(property_schema.value_schema)
                for property_schema in properties
            }
            return {
                "type": "object",
                "description": description,
                "properties": property_values,
                "required": [
                    property_schema.name
                    for property_schema in properties
                    if property_schema.required
                ],
                "additionalProperties": additional_properties,
            }


def add_assistant_generation_markers(native_template: str) -> str:
    assistant_branch = '{%- elif message.role == "assistant" %}'
    tool_branch = "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
    if native_template.count(assistant_branch) != 1:
        raise ValueError("Expected exactly one assistant branch in Qwen's native chat template.")
    if native_template.count(tool_branch) != 1:
        raise ValueError("Expected exactly one assistant-to-tool boundary in Qwen's chat template.")
    marked_template = native_template.replace(
        assistant_branch,
        f"{assistant_branch}{{% generation %}}",
        1,
    )
    return marked_template.replace(
        tool_branch,
        (
            "        {{- '<|im_end|>\\n' }}{% endgeneration %}\n"
            '    {%- elif message.role == "tool" %}'
        ),
        1,
    )


def _tool_result_content(outcome: ToolOutcome) -> str:
    match outcome:
        case SuccessOutcome(content=content):
            return content
        case EmptyOutcome(message=message):
            return f"No result: {message}"
        case FailureOutcome(message=message):
            return f"Tool failure: {message}"
        case TimeoutOutcome(message=message):
            return f"Tool timeout: {message}"
        case CancelledOutcome(message=message):
            return f"Tool cancelled: {message}"

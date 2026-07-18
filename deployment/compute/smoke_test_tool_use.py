from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum

import torch

from app.compute.voice.hermes_tool_parser import (
    HermesSpokenText,
    HermesToolCallCompleted,
    HermesToolCallFailed,
    HermesToolCallParser,
)
from app.compute.voice.llm_worker_protocol import (
    LlmAssistantMessage,
    LlmToolMessage,
    LlmUserMessage,
    StartLlmCommand,
)
from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_ADAPTER_NAME,
    LANGUAGE_MODEL_ADAPTER_REVISION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
)
from app.compute.voice.qwen_config import (
    QwenAdapterConfiguration,
    QwenModelConfiguration,
)
from app.compute.voice.qwen_worker import QwenRuntime
from app.compute.voice.tools import (
    CalculateArguments,
    CalculateToolCallFunction,
    SearchArguments,
    ToolCall,
    ToolCallFailure,
    ToolName,
    ToolSuccess,
    create_runtime_tool_registry,
    runtime_tool_specifications,
)
from app.shared.base_model import FrozenBaseModel


class SmokeCase(StrEnum):
    ORDINARY = "ordinary_no_tool"
    CALCULATE = "calculate"
    SEARCH = "search"
    POST_TOOL_CONTINUATION = "post_tool_continuation"


@dataclass(frozen=True)
class SmokeRequest:
    case: SmokeCase
    command: StartLlmCommand
    expected_tool: ToolName | None


class SmokeObservation(FrozenBaseModel):
    case: SmokeCase
    generated_text: str
    spoken_text: str
    generated_tools: tuple[str, ...]
    parser_failures: tuple[str, ...]
    passed: bool


class ToolUseSmokeReport(FrozenBaseModel):
    base_model: str
    base_revision: str
    adapter: str
    adapter_revision: str
    observations: tuple[SmokeObservation, ...]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = QwenRuntime(
        QwenModelConfiguration(
            model_name=LANGUAGE_MODEL_NAME,
            model_revision=LANGUAGE_MODEL_REVISION,
            adapter=QwenAdapterConfiguration(
                repository_id=LANGUAGE_MODEL_ADAPTER_NAME,
                revision=LANGUAGE_MODEL_ADAPTER_REVISION,
            ),
        )
    )
    observations = tuple(
        _run_smoke_request(runtime, smoke_request, random_seed=17 + request_index)
        for request_index, smoke_request in enumerate(_smoke_requests())
    )
    report = ToolUseSmokeReport(
        base_model=LANGUAGE_MODEL_NAME,
        base_revision=LANGUAGE_MODEL_REVISION,
        adapter=LANGUAGE_MODEL_ADAPTER_NAME,
        adapter_revision=LANGUAGE_MODEL_ADAPTER_REVISION,
        observations=observations,
    )
    print(report.model_dump_json(indent=2))
    failed_cases = tuple(observation.case for observation in observations if not observation.passed)
    if failed_cases:
        failed_case_names = ", ".join(failed_cases)
        raise RuntimeError(f"Tool-use smoke test failed: {failed_case_names}")


def _run_smoke_request(
    runtime: QwenRuntime,
    smoke_request: SmokeRequest,
    random_seed: int,
) -> SmokeObservation:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    generated_text = "".join(runtime.stream_text(smoke_request.command, threading.Event()))
    parser = HermesToolCallParser(smoke_request.command.invocation_id)
    parser_events = (*parser.add_text(generated_text), *parser.finish())
    spoken_text = "".join(
        event.text for event in parser_events if isinstance(event, HermesSpokenText)
    ).strip()
    generated_tools = tuple(
        event.request.name for event in parser_events if isinstance(event, HermesToolCallCompleted)
    )
    parser_failures = tuple(
        event.failure.message for event in parser_events if isinstance(event, HermesToolCallFailed)
    )
    registry = create_runtime_tool_registry(_unused_search)
    validated_calls = tuple(
        registry.validate(event.request)
        for event in parser_events
        if isinstance(event, HermesToolCallCompleted)
    )
    expected_tools = (
        () if smoke_request.expected_tool is None else (smoke_request.expected_tool.value,)
    )
    validation_failures = tuple(
        call.message for call in validated_calls if isinstance(call, ToolCallFailure)
    )
    return SmokeObservation(
        case=smoke_request.case,
        generated_text=generated_text,
        spoken_text=spoken_text,
        generated_tools=generated_tools,
        parser_failures=(*parser_failures, *validation_failures),
        passed=(
            bool(spoken_text)
            and generated_tools == expected_tools
            and not parser_failures
            and not validation_failures
        ),
    )


async def _unused_search(arguments: SearchArguments) -> str:
    del arguments
    raise AssertionError("The smoke test validates calls without executing tools.")


def _smoke_requests() -> tuple[SmokeRequest, ...]:
    tools = runtime_tool_specifications()
    calculate_call = ToolCall(
        id="smoke-calculate",
        function=CalculateToolCallFunction(arguments=CalculateArguments(expression="37 * 14")),
    )
    calculate_result = ToolSuccess(
        call_id=calculate_call.id,
        tool_name=ToolName.CALCULATE,
        result="518",
    )
    return (
        SmokeRequest(
            case=SmokeCase.ORDINARY,
            command=StartLlmCommand(
                invocation_id=1,
                assistant_generation_id=1,
                messages=(
                    LlmUserMessage(content="In one short sentence, wish me a calm afternoon."),
                ),
                tools=tools,
            ),
            expected_tool=None,
        ),
        SmokeRequest(
            case=SmokeCase.CALCULATE,
            command=StartLlmCommand(
                invocation_id=2,
                assistant_generation_id=2,
                messages=(LlmUserMessage(content="What is 37 multiplied by 14?"),),
                tools=tools,
            ),
            expected_tool=ToolName.CALCULATE,
        ),
        SmokeRequest(
            case=SmokeCase.SEARCH,
            command=StartLlmCommand(
                invocation_id=3,
                assistant_generation_id=3,
                messages=(LlmUserMessage(content="Search for the current weather in Berlin."),),
                tools=tools,
            ),
            expected_tool=ToolName.SEARCH,
        ),
        SmokeRequest(
            case=SmokeCase.POST_TOOL_CONTINUATION,
            command=StartLlmCommand(
                invocation_id=4,
                assistant_generation_id=4,
                messages=(
                    LlmUserMessage(content="What is 37 multiplied by 14?"),
                    LlmAssistantMessage(
                        content="I'll calculate that.",
                        tool_calls=(calculate_call,),
                    ),
                    LlmToolMessage(
                        tool_call_id=calculate_call.id,
                        content=calculate_result,
                    ),
                ),
                tools=tools,
            ),
            expected_tool=None,
        ),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from app.compute.voice.hermes_tool_parser import (
    HermesSpokenText,
    HermesToolCallCompleted,
    HermesToolCallFailed,
    HermesToolCallParser,
)
from app.compute.voice.llm_worker_protocol import (
    LlmAssistantMessage,
    LlmEndEvent,
    LlmSpokenTextDeltaEvent,
    LlmToolCallEvent,
    LlmToolCallFailureEvent,
    LlmToolCallStartedEvent,
    LlmToolMessage,
    LlmUserMessage,
    LlmWorkerErrorEvent,
    StartLlmCommand,
)
from app.compute.voice.models import (
    QWEN_PYTHON_PATH,
    QwenWorkerConfiguration,
    QwenWorkerProcess,
)
from app.compute.voice.qwen_config import language_model_configuration_from_environment
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
    adapter: str | None
    adapter_revision: str | None
    observations: tuple[SmokeObservation, ...]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configuration = tool_use_worker_configuration(os.environ)
    worker = QwenWorkerProcess(QWEN_PYTHON_PATH, configuration)
    try:
        observations = tuple(
            _run_worker_smoke_request(worker, smoke_request) for smoke_request in _smoke_requests()
        )
    finally:
        worker.close()
    adapter = configuration.model.adapter
    report = ToolUseSmokeReport(
        base_model=configuration.model.model_name,
        base_revision=configuration.model.model_revision,
        adapter=None if adapter is None else adapter.repository_id,
        adapter_revision=None if adapter is None else adapter.revision,
        observations=observations,
    )
    print(report.model_dump_json(indent=2))
    failed_cases = tuple(observation.case for observation in observations if not observation.passed)
    if failed_cases:
        failed_case_names = ", ".join(failed_cases)
        raise RuntimeError(f"Tool-use smoke test failed: {failed_case_names}")


def tool_use_worker_configuration(
    environment: Mapping[str, str],
) -> QwenWorkerConfiguration:
    return QwenWorkerConfiguration(
        model=language_model_configuration_from_environment(environment),
        component_name="Qwen language model",
    )


def _run_smoke_request(
    generated_text: str,
    smoke_request: SmokeRequest,
) -> SmokeObservation:
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


def _run_worker_smoke_request(
    worker: QwenWorkerProcess,
    smoke_request: SmokeRequest,
) -> SmokeObservation:
    spoken_parts: list[str] = []
    generated_tools: list[str] = []
    failures: list[str] = []
    registry = create_runtime_tool_registry(_unused_search)
    worker.send(smoke_request.command)
    while True:
        event = worker.read_event()
        match event:
            case LlmSpokenTextDeltaEvent(text=text):
                spoken_parts.append(text)
            case LlmToolCallStartedEvent():
                pass
            case LlmToolCallEvent(request=request):
                generated_tools.append(request.name)
                validated_call = registry.validate(request)
                if isinstance(validated_call, ToolCallFailure):
                    failures.append(validated_call.message)
            case LlmToolCallFailureEvent(failure=failure):
                failures.append(failure.message)
            case LlmEndEvent():
                break
            case LlmWorkerErrorEvent(message=message):
                raise RuntimeError(f"Tool-use smoke generation failed: {message}")
            case _:
                raise RuntimeError("Tool-use smoke generation returned an unexpected event.")
    spoken_text = "".join(spoken_parts).strip()
    expected_tools = (
        () if smoke_request.expected_tool is None else (smoke_request.expected_tool.value,)
    )
    return SmokeObservation(
        case=smoke_request.case,
        generated_text=spoken_text,
        spoken_text=spoken_text,
        generated_tools=tuple(generated_tools),
        parser_failures=tuple(failures),
        passed=(bool(spoken_text) and tuple(generated_tools) == expected_tools and not failures),
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

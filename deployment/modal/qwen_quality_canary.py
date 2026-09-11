from __future__ import annotations

import os
from dataclasses import dataclass

from huggingface_hub import snapshot_download

import modal
from app.compute.voice.llm_worker_protocol import (
    LlmAssistantMessage,
    LlmEndEvent,
    LlmSpokenTextDeltaEvent,
    LlmToolCallEvent,
    LlmToolCallFailureEvent,
    LlmToolCallStartedEvent,
    LlmUserMessage,
    LlmWorkerErrorEvent,
    StartLlmCommand,
)
from app.compute.voice.models import (
    QwenWorkerConfiguration,
    QwenWorkerProcess,
    qwen_python_path,
)
from app.compute.voice.qwen_config import language_model_configuration_from_environment
from app.compute.voice.tools import runtime_tool_specifications
from app.shared.base_model import FrozenBaseModel
from deployment.modal.voice_light import (
    MODEL_CACHE_MOUNT,
    image,
    model_cache,
)

APPLICATION_NAME = "VoiceLightQwenQualityCanary"

app = modal.App(APPLICATION_NAME)


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    messages: tuple[LlmUserMessage | LlmAssistantMessage, ...]


class DiagnosticObservation(FrozenBaseModel):
    name: str
    spoken_text: str
    tool_names: tuple[str, ...]
    tool_arguments: tuple[str, ...]
    failures: tuple[str, ...]


@app.function(
    image=image,
    gpu="A10",
    timeout=1_800,
    volumes={str(MODEL_CACHE_MOUNT): model_cache},
)
def evaluate(model_name: str, model_revision: str) -> None:
    snapshot_download(model_name, revision=model_revision)
    model_cache.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VOICE_LIGHT_MERGED_LANGUAGE_MODEL_NAME"] = model_name
    os.environ["VOICE_LIGHT_MERGED_LANGUAGE_MODEL_REVISION"] = model_revision
    os.environ["VOICE_LIGHT_QWEN_BACKEND"] = "transformers"

    cases = (
        DiagnosticCase(
            name="funny_fact",
            messages=(LlmUserMessage(content="Can you tell me a funny fact, please?"),),
        ),
        DiagnosticCase(
            name="correct_bad_fact",
            messages=(
                LlmUserMessage(content="Can you tell me a funny fact, please?"),
                LlmAssistantMessage(
                    content=(
                        "The average person eats about 400 pounds of food in their lifetime. "
                        "That's roughly the weight of a 200-pound bear."
                    )
                ),
                LlmUserMessage(
                    content=(
                        "That makes no sense. Four hundred pounds is two two-hundred-pound "
                        "bears, and a person eats much more than that in a lifetime."
                    )
                ),
            ),
        ),
        DiagnosticCase(
            name="moose_comparison",
            messages=(LlmUserMessage(content="How does a 400-pound total compare with a moose?"),),
        ),
        DiagnosticCase(
            name="two_city_times",
            messages=(
                LlmUserMessage(content="What are the current times in London and New York?"),
            ),
        ),
        DiagnosticCase(
            name="current_weather",
            messages=(LlmUserMessage(content="What is the current weather in London?"),),
        ),
        DiagnosticCase(
            name="arithmetic",
            messages=(LlmUserMessage(content="What is 49 times 49?"),),
        ),
        DiagnosticCase(
            name="short_story",
            messages=(
                LlmUserMessage(
                    content="Tell me a coherent, imaginative story in about five sentences."
                ),
            ),
        ),
    )
    configuration = QwenWorkerConfiguration(
        model=language_model_configuration_from_environment(os.environ),
        component_name="Qwen quality canary",
    )
    worker = QwenWorkerProcess(qwen_python_path(configuration.model.backend), configuration)
    observations: list[DiagnosticObservation] = []
    try:
        for invocation_id, case in enumerate(cases, start=1):
            command = StartLlmCommand(
                invocation_id=invocation_id,
                assistant_generation_id=invocation_id,
                messages=case.messages,
                tools=runtime_tool_specifications(),
            )
            spoken_parts: list[str] = []
            tool_names: list[str] = []
            tool_arguments: list[str] = []
            failures: list[str] = []
            worker.send(command)
            while True:
                event = worker.read_event()
                match event:
                    case LlmSpokenTextDeltaEvent(text=text):
                        spoken_parts.append(text)
                    case LlmToolCallStartedEvent():
                        pass
                    case LlmToolCallEvent(request=request):
                        tool_names.append(request.name)
                        tool_arguments.append(request.arguments_json)
                    case LlmToolCallFailureEvent(failure=failure):
                        failures.append(failure.message)
                    case LlmEndEvent():
                        break
                    case LlmWorkerErrorEvent(message=message):
                        raise RuntimeError(message)
                    case _:
                        raise RuntimeError("Qwen quality canary returned an unexpected event.")
            observations.append(
                DiagnosticObservation(
                    name=case.name,
                    spoken_text="".join(spoken_parts).strip(),
                    tool_names=tuple(tool_names),
                    tool_arguments=tuple(tool_arguments),
                    failures=tuple(failures),
                )
            )
    finally:
        worker.close()
    for observation in observations:
        print(observation.model_dump_json())

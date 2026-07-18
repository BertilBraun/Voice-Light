from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypedDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from app.compute.voice.hermes_tool_parser import (
    HermesParserEvent,
    HermesSpokenText,
    HermesToolCallCompleted,
    HermesToolCallFailed,
    HermesToolCallParser,
    HermesToolCallStarted,
)
from app.compute.voice.llm_worker_protocol import (
    CancelLlmCommand,
    GenerateTextLlmCommand,
    LlmAssistantMessage,
    LlmCancelledEvent,
    LlmEndEvent,
    LlmSpokenTextDeltaEvent,
    LlmToolCallEvent,
    LlmToolCallFailureEvent,
    LlmToolCallStartedEvent,
    LlmToolMessage,
    LlmUserMessage,
    LlmWorkerCommand,
    LlmWorkerErrorEvent,
    LlmWorkerEvent,
    LlmWorkerReadyEvent,
    ShutdownLlmCommand,
    StartLlmCommand,
    llm_worker_command_adapter,
)
from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


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
QwenGenerationCommand = StartLlmCommand | GenerateTextLlmCommand


class GenerationStreamer(Protocol):
    def on_finalized_text(self, text: str, stream_end: bool = False) -> None: ...


class QwenChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[QwenChatMessage],
        *,
        tokenize: Literal[False],
        add_generation_prompt: Literal[True],
        tools: list[dict[str, object]],
        enable_thinking: Literal[False],
    ) -> str: ...


class CancellationStoppingCriteria(StoppingCriteria):
    def __init__(self, cancellation_event: threading.Event) -> None:
        self.cancellation_event = cancellation_event

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **generation_arguments: torch.Tensor,
    ) -> torch.BoolTensor:
        del scores, generation_arguments
        return torch.full(
            (input_ids.shape[0],),
            self.cancellation_event.is_set(),
            device=input_ids.device,
            dtype=torch.bool,
        )


class QwenRuntime:
    def __init__(self, model_name: str, model_revision: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

    def stream_text(
        self,
        command: QwenGenerationCommand,
        cancellation_event: threading.Event,
    ) -> Iterator[str]:
        prompt = render_qwen_prompt(self.tokenizer, command)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_errors: list[Exception] = []

        def generate() -> None:
            match command:
                case StartLlmCommand():
                    self.model.generate(
                        **model_inputs,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList(
                            [CancellationStoppingCriteria(cancellation_event)]
                        ),
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                    )
                case GenerateTextLlmCommand(max_new_tokens=max_new_tokens):
                    self.model.generate(
                        **model_inputs,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList(
                            [CancellationStoppingCriteria(cancellation_event)]
                        ),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

        model_thread = threading.Thread(
            target=_run_model_generation,
            args=(generate, streamer, generation_errors),
            daemon=True,
            name=f"qwen-model-{command.invocation_id}",
        )
        model_thread.start()
        stream_completed = False
        try:
            yield from streamer
            stream_completed = True
            if generation_errors:
                raise RuntimeError(
                    f"Qwen model generation failed: {generation_errors[0]}"
                ) from generation_errors[0]
        finally:
            if not stream_completed:
                cancellation_event.set()
            model_thread.join()

    def count_response_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


@dataclass(frozen=True)
class ActiveWorkerInvocation:
    invocation_id: int
    cancellation_event: threading.Event
    thread: threading.Thread


class QwenWorkerController:
    def __init__(self, runtime: QwenRuntime) -> None:
        self.runtime = runtime
        self.output_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_invocation: ActiveWorkerInvocation | None = None

    def run(self) -> None:
        self._send_event(LlmWorkerReadyEvent())
        for line in sys.stdin:
            try:
                command = llm_worker_command_adapter.validate_json(line)
                if self._handle_command(command):
                    return
            except Exception:
                logger.exception("Qwen worker command handling failed")

    def _handle_command(self, command: LlmWorkerCommand) -> bool:
        match command:
            case StartLlmCommand() | GenerateTextLlmCommand():
                self._start(command)
                return False
            case CancelLlmCommand():
                self._cancel(command)
                return False
            case ShutdownLlmCommand():
                self._shutdown()
                return True

    def _start(self, command: QwenGenerationCommand) -> None:
        with self.state_lock:
            if self.active_invocation is not None:
                self._send_event(
                    LlmWorkerErrorEvent(
                        invocation_id=command.invocation_id,
                        message="Qwen worker already has an active invocation.",
                    )
                )
                return
            cancellation_event = threading.Event()
            invocation_thread = threading.Thread(
                target=self._generate,
                args=(command, cancellation_event),
                daemon=True,
                name=f"qwen-stream-{command.invocation_id}",
            )
            self.active_invocation = ActiveWorkerInvocation(
                invocation_id=command.invocation_id,
                cancellation_event=cancellation_event,
                thread=invocation_thread,
            )
            invocation_thread.start()

    def _cancel(self, command: CancelLlmCommand) -> None:
        with self.state_lock:
            active_invocation = self.active_invocation
            if (
                active_invocation is None
                or active_invocation.invocation_id != command.invocation_id
            ):
                logger.warning(
                    "ignoring Qwen cancellation for inactive invocation: invocation=%d",
                    command.invocation_id,
                )
                return
            active_invocation.cancellation_event.set()

    def _shutdown(self) -> None:
        with self.state_lock:
            active_invocation = self.active_invocation
            if active_invocation is not None:
                active_invocation.cancellation_event.set()
        if active_invocation is not None:
            active_invocation.thread.join()

    def _generate(
        self,
        command: QwenGenerationCommand,
        cancellation_event: threading.Event,
    ) -> None:
        terminal_event: LlmWorkerEvent | None = None
        raw_response_text = ""
        try:
            match command:
                case StartLlmCommand():
                    parser = HermesToolCallParser(command.invocation_id)
                    for text_delta in self.runtime.stream_text(command, cancellation_event):
                        raw_response_text += text_delta
                        cumulative_token_count = self.runtime.count_response_tokens(
                            raw_response_text
                        )
                        for parser_event in parser.add_text(text_delta):
                            self._send_parser_event(
                                command.invocation_id,
                                cumulative_token_count,
                                parser_event,
                            )
                    cumulative_token_count = self.runtime.count_response_tokens(raw_response_text)
                    for parser_event in parser.finish():
                        self._send_parser_event(
                            command.invocation_id,
                            cumulative_token_count,
                            parser_event,
                        )
                case GenerateTextLlmCommand():
                    for text_delta in self.runtime.stream_text(command, cancellation_event):
                        raw_response_text += text_delta
                        cumulative_token_count = self.runtime.count_response_tokens(
                            raw_response_text
                        )
                        self._send_event(
                            LlmSpokenTextDeltaEvent(
                                invocation_id=command.invocation_id,
                                text=text_delta,
                                cumulative_token_count=cumulative_token_count,
                            )
                        )
                    cumulative_token_count = self.runtime.count_response_tokens(raw_response_text)
            if cancellation_event.is_set():
                terminal_event = LlmCancelledEvent(invocation_id=command.invocation_id)
            else:
                terminal_event = LlmEndEvent(
                    invocation_id=command.invocation_id,
                    cumulative_token_count=cumulative_token_count,
                )
        except Exception as error:
            logger.exception("Qwen generation failed: invocation=%d", command.invocation_id)
            terminal_event = LlmWorkerErrorEvent(
                invocation_id=command.invocation_id,
                message=str(error),
            )
        finally:
            with self.state_lock:
                active_invocation = self.active_invocation
                if (
                    active_invocation is not None
                    and active_invocation.invocation_id == command.invocation_id
                ):
                    self.active_invocation = None
            if terminal_event is not None:
                self._send_event(terminal_event)

    def _send_parser_event(
        self,
        invocation_id: int,
        cumulative_token_count: int,
        parser_event: HermesParserEvent,
    ) -> None:
        match parser_event:
            case HermesSpokenText():
                self._send_event(
                    LlmSpokenTextDeltaEvent(
                        invocation_id=invocation_id,
                        text=parser_event.text,
                        cumulative_token_count=cumulative_token_count,
                    )
                )
            case HermesToolCallStarted():
                self._send_event(
                    LlmToolCallStartedEvent(
                        invocation_id=invocation_id,
                        call_id=parser_event.call_id,
                        cumulative_token_count=cumulative_token_count,
                    )
                )
            case HermesToolCallCompleted():
                self._send_event(
                    LlmToolCallEvent(
                        invocation_id=invocation_id,
                        request=parser_event.request,
                        cumulative_token_count=cumulative_token_count,
                    )
                )
            case HermesToolCallFailed():
                self._send_event(
                    LlmToolCallFailureEvent(
                        invocation_id=invocation_id,
                        failure=parser_event.failure,
                        cumulative_token_count=cumulative_token_count,
                    )
                )

    def _send_event(self, event: LlmWorkerEvent) -> None:
        with self.output_lock:
            sys.stdout.write(event.model_dump_json() + "\n")
            sys.stdout.flush()


def _run_model_generation(
    generate: Callable[[], None],
    streamer: GenerationStreamer,
    generation_errors: list[Exception],
) -> None:
    try:
        generate()
    except Exception as error:
        generation_errors.append(error)
        streamer.on_finalized_text("", stream_end=True)


def _qwen_chat_message(
    message: LlmUserMessage | LlmAssistantMessage | LlmToolMessage,
) -> QwenChatMessage:
    match message:
        case LlmUserMessage():
            return {"role": "user", "content": message.content}
        case LlmAssistantMessage():
            assistant_message: QwenAssistantMessage = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments.model_dump_json(),
                        },
                    }
                    for call in message.tool_calls
                ]
            return assistant_message
        case LlmToolMessage():
            return {
                "role": "tool",
                "content": message.content.model_dump_json(),
                "tool_call_id": message.tool_call_id,
            }


def qwen_chat_messages(command: StartLlmCommand) -> list[QwenChatMessage]:
    return [
        {"role": "system", "content": LANGUAGE_MODEL_SYSTEM_PROMPT},
        *(_qwen_chat_message(message) for message in command.messages),
    ]


def render_qwen_prompt(
    tokenizer: QwenChatTemplateTokenizer,
    command: QwenGenerationCommand,
) -> str:
    match command:
        case StartLlmCommand():
            messages = qwen_chat_messages(command)
            tools = [specification.model_dump(mode="json") for specification in command.tools]
        case GenerateTextLlmCommand(system_prompt=system_prompt, user_prompt=user_prompt):
            messages = [
                QwenSystemMessage(role="system", content=system_prompt),
                QwenUserMessage(role="user", content=user_prompt),
            ]
            tools = []
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        enable_thinking=False,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    options = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    QwenWorkerController(QwenRuntime(options.model, options.revision)).run()


if __name__ == "__main__":
    main()

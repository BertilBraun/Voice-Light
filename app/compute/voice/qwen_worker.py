from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypedDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from app.compute.voice.llm_worker_protocol import (
    CancelLlmCommand,
    LlmCancelledEvent,
    LlmEndEvent,
    LlmTextDeltaEvent,
    LlmWorkerCommand,
    LlmWorkerErrorEvent,
    LlmWorkerEvent,
    LlmWorkerReadyEvent,
    ShutdownLlmCommand,
    StartLlmCommand,
    llm_worker_command_adapter,
)
from app.compute.voice.model_constants import LANGUAGE_MODEL_NAME, LANGUAGE_MODEL_REVISION

LANGUAGE_MODEL_SYSTEM_PROMPT: Final = (
    "You are a conversational voice agent. Respond naturally and directly to the user's latest "
    "message. Use the complete conversation history as context and do not repeat earlier answers. "
    "Start with substantive content instead of filler acknowledgements such as 'Sure' or 'Of "
    "course.' Use plain text without Markdown or emoji."
)
logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationStreamer(Protocol):
    def on_finalized_text(self, text: str, stream_end: bool = False) -> None: ...


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
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            LANGUAGE_MODEL_NAME,
            revision=LANGUAGE_MODEL_REVISION,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            LANGUAGE_MODEL_NAME,
            revision=LANGUAGE_MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

    def stream_text(
        self,
        command: StartLlmCommand,
        cancellation_event: threading.Event,
    ) -> Iterator[str]:
        messages: list[ChatMessage] = [
            {"role": "system", "content": LANGUAGE_MODEL_SYSTEM_PROMPT},
            *(
                {"role": message.role.value, "content": message.content}
                for message in command.messages
            ),
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_errors: list[Exception] = []

        def generate() -> None:
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

        model_thread = threading.Thread(
            target=_run_model_generation,
            args=(generate, streamer, generation_errors),
            daemon=True,
            name=f"qwen-model-{command.generation_id}",
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


@dataclass(frozen=True)
class ActiveWorkerGeneration:
    generation_id: int
    cancellation_event: threading.Event
    thread: threading.Thread


class QwenWorkerController:
    def __init__(self, runtime: QwenRuntime) -> None:
        self.runtime = runtime
        self.output_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_generation: ActiveWorkerGeneration | None = None

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
            case StartLlmCommand():
                self._start(command)
                return False
            case CancelLlmCommand():
                self._cancel(command)
                return False
            case ShutdownLlmCommand():
                self._shutdown()
                return True

    def _start(self, command: StartLlmCommand) -> None:
        with self.state_lock:
            if self.active_generation is not None:
                self._send_event(
                    LlmWorkerErrorEvent(
                        generation_id=command.generation_id,
                        message="Qwen worker already has an active generation.",
                    )
                )
                return
            cancellation_event = threading.Event()
            generation_thread = threading.Thread(
                target=self._generate,
                args=(command, cancellation_event),
                daemon=True,
                name=f"qwen-stream-{command.generation_id}",
            )
            self.active_generation = ActiveWorkerGeneration(
                generation_id=command.generation_id,
                cancellation_event=cancellation_event,
                thread=generation_thread,
            )
            generation_thread.start()

    def _cancel(self, command: CancelLlmCommand) -> None:
        with self.state_lock:
            active_generation = self.active_generation
            if (
                active_generation is None
                or active_generation.generation_id != command.generation_id
            ):
                logger.warning(
                    "ignoring Qwen cancellation for inactive generation: generation=%d",
                    command.generation_id,
                )
                return
            active_generation.cancellation_event.set()

    def _shutdown(self) -> None:
        with self.state_lock:
            active_generation = self.active_generation
            if active_generation is not None:
                active_generation.cancellation_event.set()
        if active_generation is not None:
            active_generation.thread.join()

    def _generate(
        self,
        command: StartLlmCommand,
        cancellation_event: threading.Event,
    ) -> None:
        terminal_event: LlmWorkerEvent | None = None
        try:
            for text_delta in self.runtime.stream_text(command, cancellation_event):
                self._send_event(
                    LlmTextDeltaEvent(
                        generation_id=command.generation_id,
                        text=text_delta,
                    )
                )
            if cancellation_event.is_set():
                terminal_event = LlmCancelledEvent(generation_id=command.generation_id)
            else:
                terminal_event = LlmEndEvent(generation_id=command.generation_id)
        except Exception as error:
            logger.exception("Qwen generation failed: generation=%d", command.generation_id)
            terminal_event = LlmWorkerErrorEvent(
                generation_id=command.generation_id,
                message=str(error),
            )
        finally:
            with self.state_lock:
                active_generation = self.active_generation
                if (
                    active_generation is not None
                    and active_generation.generation_id == command.generation_id
                ):
                    self.active_generation = None
            if terminal_event is not None:
                self._send_event(terminal_event)

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    QwenWorkerController(QwenRuntime()).run()


if __name__ == "__main__":
    main()

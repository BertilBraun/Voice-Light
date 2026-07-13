from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Final

import numpy as np
import torch
from pocket_tts import TTSModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from app.shared.compute_api import ConversationMessage

LANGUAGE_MODEL_NAME: Final = "Qwen/Qwen3-1.7B"
LANGUAGE_MODEL_REVISION: Final = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LANGUAGE_MODEL_SYSTEM_PROMPT: Final = (
    "You are a conversational voice agent. Respond naturally and directly to the user's latest "
    "message. Use the complete conversation history as context and do not repeat earlier answers. "
    "Use plain text without Markdown or emoji."
)
POCKET_TTS_LANGUAGE: Final = "english"
POCKET_TTS_VOICE: Final = "alba"


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


class TransformersLanguageModel:
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

    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": LANGUAGE_MODEL_SYSTEM_PROMPT},
            *(message.model_dump(mode="json") for message in conversation),
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
        cancellation_event = threading.Event()
        generation_thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                **model_inputs,
                "streamer": streamer,
                "stopping_criteria": StoppingCriteriaList(
                    [CancellationStoppingCriteria(cancellation_event)]
                ),
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
            },
            daemon=True,
        )
        generation_thread.start()
        try:
            while True:
                text_delta = await asyncio.to_thread(_next_text, streamer)
                if text_delta is None:
                    break
                yield text_delta
        finally:
            cancellation_event.set()
            await asyncio.to_thread(generation_thread.join)


class PocketSpeechSynthesizer:
    def __init__(self) -> None:
        self.model = TTSModel.load_model(language=POCKET_TTS_LANGUAGE)
        self.voice_state = self.model.get_state_for_audio_prompt(POCKET_TTS_VOICE)

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        audio_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        event_loop = asyncio.get_running_loop()
        cancellation_event = threading.Event()

        def synthesize() -> None:
            stream: Iterator[torch.Tensor] = self.model.generate_audio_stream(
                self.voice_state,
                text,
            )
            try:
                for audio_chunk in stream:
                    if cancellation_event.is_set():
                        return
                    samples = audio_chunk.detach().cpu().float().numpy()
                    pcm_bytes = (np.clip(samples, -1.0, 1.0) * 32_767.0).astype("<i2").tobytes()
                    event_loop.call_soon_threadsafe(audio_queue.put_nowait, pcm_bytes)
            except Exception as error:
                event_loop.call_soon_threadsafe(audio_queue.put_nowait, error)
            finally:
                stream.close()
                event_loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        synthesis_thread = threading.Thread(target=synthesize, daemon=True)
        synthesis_thread.start()
        try:
            while (audio := await audio_queue.get()) is not None:
                if isinstance(audio, Exception):
                    raise audio
                yield audio
        finally:
            cancellation_event.set()
            await asyncio.to_thread(synthesis_thread.join)


def _next_text(streamer: TextIteratorStreamer) -> str | None:
    try:
        return next(streamer)
    except StopIteration:
        return None

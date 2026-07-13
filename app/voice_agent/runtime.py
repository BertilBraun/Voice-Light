from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

INPUT_SAMPLE_RATE: Final = 16_000
LANGUAGE_MODEL_NAME: Final = "Qwen/Qwen3-1.7B"
LANGUAGE_MODEL_SYSTEM_PROMPT: Final = (
    "You are a conversational voice agent. Respond naturally and directly to the user's latest "
    "message. Use the complete conversation history as context and do not repeat earlier answers. "
    "Use plain text without Markdown or emoji."
)
COSYVOICE_ENGLISH_LANGUAGE_TAG: Final = "<|en|>"
COSYVOICE_PROMPT_TEXT: Final = "希望你以后能够做的比我还好呦。"
COSYVOICE_SPEAKER_ID: Final = "voice-light-prompt"


class SileroSpeechDetector:
    def __init__(self) -> None:
        self.iterator = VADIterator(
            load_silero_vad(),
            sampling_rate=INPUT_SAMPLE_RATE,
            threshold=0.4,
            min_silence_duration_ms=250,
            speech_pad_ms=100,
        )
        self.pending_samples = np.empty(0, dtype=np.float32)
        self.speech_active = False

    def process_audio(self, pcm_bytes: bytes) -> bool:
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32_768.0
        self.pending_samples = np.concatenate((self.pending_samples, samples))
        while len(self.pending_samples) >= 512:
            frame = torch.from_numpy(self.pending_samples[:512])
            self.pending_samples = self.pending_samples[512:]
            event = self.iterator(frame)
            if event is not None and "start" in event:
                self.speech_active = True
            if event is not None and "end" in event:
                self.speech_active = False
        return self.speech_active


class TransformersLanguageModel:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            LANGUAGE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")

    async def stream_response(
        self,
        conversation: tuple[tuple[str, str], ...],
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": LANGUAGE_MODEL_SYSTEM_PROMPT},
            *({"role": role, "content": content} for role, content in conversation),
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
        generation_thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                **model_inputs,
                "streamer": streamer,
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
            },
            daemon=True,
        )
        generation_thread.start()
        while True:
            text_delta = await asyncio.to_thread(_next_text, streamer)
            if text_delta is None:
                break
            yield text_delta
        await asyncio.to_thread(generation_thread.join)


class CosyVoiceSpeechSynthesizer:
    def __init__(self, model_directory: Path, prompt_audio_path: Path) -> None:
        from cosyvoice.cli.cosyvoice import AutoModel

        self.model = AutoModel(model_dir=model_directory.as_posix(), fp16=True)
        self.prompt_audio_path = prompt_audio_path
        speaker_added = self.model.add_zero_shot_spk(
            COSYVOICE_PROMPT_TEXT,
            self.prompt_audio_path.as_posix(),
            COSYVOICE_SPEAKER_ID,
        )
        assert speaker_added is True

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        audio_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        event_loop = asyncio.get_running_loop()

        def synthesize() -> None:
            try:
                outputs = self.model.inference_cross_lingual(
                    f"{COSYVOICE_ENGLISH_LANGUAGE_TAG}{text}",
                    "",
                    zero_shot_spk_id=COSYVOICE_SPEAKER_ID,
                    stream=True,
                )
                produced_audio = False
                for output in outputs:
                    produced_audio = True
                    speech = output["tts_speech"].squeeze().detach().cpu().numpy()
                    pcm = (np.clip(speech, -1.0, 1.0) * 32_767.0).astype("<i2").tobytes()
                    event_loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
                if not produced_audio:
                    raise RuntimeError("CosyVoice produced no audio.")
            except Exception as error:
                event_loop.call_soon_threadsafe(audio_queue.put_nowait, error)
            finally:
                event_loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        synthesis_thread = threading.Thread(target=synthesize, daemon=True)
        synthesis_thread.start()
        while (audio := await audio_queue.get()) is not None:
            if isinstance(audio, Exception):
                raise audio
            yield audio
        await asyncio.to_thread(synthesis_thread.join)


def _next_text(streamer: TextIteratorStreamer) -> str | None:
    try:
        return next(streamer)
    except StopIteration:
        return None

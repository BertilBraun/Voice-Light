from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
import whisper
from pydantic import BaseModel
from silero_vad import VADIterator, load_silero_vad
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from whisper.model import Whisper

from app.voice_agent.interfaces import TranscriptionSession

INPUT_SAMPLE_RATE: Final = 16_000
LANGUAGE_MODEL_NAME: Final = "Qwen/Qwen3-0.6B"
LANGUAGE_MODEL_SYSTEM_PROMPT: Final = (
    "You are a voice agent. Keep responses short, factual, conversational, and easy to speak "
    "aloud. Use plain text without Markdown or emoji."
)
COSYVOICE_ENGLISH_LANGUAGE_TAG: Final = "<|en|>"


class WhisperTranscriptionOutput(BaseModel):
    text: str


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


class BufferedWhisperTranscriber:
    def __init__(self) -> None:
        download_directory = Path(os.environ["XDG_CACHE_HOME"]) / "whisper"
        self.model = whisper.load_model(
            "small.en",
            device="cuda",
            download_root=download_directory.as_posix(),
        )

    def start_session(self) -> TranscriptionSession:
        return BufferedWhisperSession(self.model)


class BufferedWhisperSession:
    def __init__(self, model: Whisper) -> None:
        self.model = model
        self.audio_bytes = bytearray()
        self.last_text = ""

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        self.audio_bytes.extend(pcm_bytes)
        return None

    async def finish(self) -> str:
        if not self.audio_bytes:
            return ""
        self.last_text = await asyncio.to_thread(self._transcribe)
        return self.last_text

    async def close(self) -> None:
        self.audio_bytes.clear()

    def _transcribe(self) -> str:
        samples = np.frombuffer(self.audio_bytes, dtype="<i2").astype(np.float32) / 32_768.0
        raw_output = self.model.transcribe(
            samples,
            language="en",
            fp16=True,
            temperature=0.0,
        )
        output = WhisperTranscriptionOutput.model_validate(cast(dict[str, object], raw_output))
        return output.text


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
                    self.prompt_audio_path.as_posix(),
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

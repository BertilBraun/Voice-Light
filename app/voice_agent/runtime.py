from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel
from silero_vad import VADIterator, load_silero_vad
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
from transformers.pipelines import AutomaticSpeechRecognitionPipeline

from app.voice_agent.interfaces import TranscriptionSession

INPUT_SAMPLE_RATE: Final = 16_000
PARTIAL_TRANSCRIPTION_INTERVAL_SAMPLES: Final = 12_800


class AsrPipelineOutput(BaseModel):
    text: str


class SileroSpeechDetector:
    def __init__(self) -> None:
        self.iterator = VADIterator(
            load_silero_vad(),
            sampling_rate=INPUT_SAMPLE_RATE,
            threshold=0.5,
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


class BufferedNemotronTranscriber:
    def __init__(self) -> None:
        self.pipeline: AutomaticSpeechRecognitionPipeline = pipeline(
            task="automatic-speech-recognition",
            model="nvidia/nemotron-speech-streaming-en-0.6b",
            device=0,
            dtype=torch.bfloat16,
        )

    def start_session(self) -> TranscriptionSession:
        return BufferedNemotronSession(self.pipeline)


class BufferedNemotronSession:
    def __init__(self, asr_pipeline: AutomaticSpeechRecognitionPipeline) -> None:
        self.pipeline = asr_pipeline
        self.audio_bytes = bytearray()
        self.samples_at_last_partial = 0
        self.last_text = ""

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        self.audio_bytes.extend(pcm_bytes)
        sample_count = len(self.audio_bytes) // 2
        if sample_count - self.samples_at_last_partial < PARTIAL_TRANSCRIPTION_INTERVAL_SAMPLES:
            return None
        self.samples_at_last_partial = sample_count
        self.last_text = await asyncio.to_thread(self._transcribe)
        return self.last_text

    async def finish(self) -> str:
        if not self.audio_bytes:
            return ""
        self.last_text = await asyncio.to_thread(self._transcribe)
        return self.last_text

    async def close(self) -> None:
        self.audio_bytes.clear()

    def _transcribe(self) -> str:
        samples = np.frombuffer(self.audio_bytes, dtype="<i2").astype(np.float32) / 32_768.0
        raw_output = self.pipeline({"array": samples, "sampling_rate": INPUT_SAMPLE_RATE})
        output = AsrPipelineOutput.model_validate(raw_output)
        return output.text


class TransformersLanguageModel:
    def __init__(self) -> None:
        model_name = "Qwen/Qwen3-1.7B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        ).to("cuda")

    async def stream_response(
        self,
        conversation: tuple[tuple[str, str], ...],
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": "You are a concise, friendly voice assistant."},
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
                "max_new_tokens": 160,
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

    async def stream_audio(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        text_queue: queue.Queue[str | None] = queue.Queue()
        audio_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        event_loop = asyncio.get_running_loop()

        def text_generator() -> Iterator[str]:
            while (text := text_queue.get()) is not None:
                yield text

        def synthesize() -> None:
            try:
                outputs = self.model.inference_cross_lingual(
                    text_generator(),
                    self.prompt_audio_path.as_posix(),
                    stream=True,
                )
                for output in outputs:
                    speech = output["tts_speech"].squeeze().detach().cpu().numpy()
                    pcm = (np.clip(speech, -1.0, 1.0) * 32_767.0).astype("<i2").tobytes()
                    event_loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
            except Exception as error:
                event_loop.call_soon_threadsafe(audio_queue.put_nowait, error)
            finally:
                event_loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        synthesis_thread = threading.Thread(target=synthesize, daemon=True)
        synthesis_thread.start()

        async def supply_text() -> None:
            async for text in text_chunks:
                text_queue.put(text)
            text_queue.put(None)

        supply_task = asyncio.create_task(supply_text())
        while (audio := await audio_queue.get()) is not None:
            if isinstance(audio, Exception):
                raise audio
            yield audio
        await supply_task
        await asyncio.to_thread(synthesis_thread.join)


def _next_text(streamer: TextIteratorStreamer) -> str | None:
    try:
        return next(streamer)
    except StopIteration:
        return None

from __future__ import annotations

from typing import Final, Protocol

import numpy as np
import torch
from silero_vad import load_silero_vad

from app.compute.voice.interfaces import SpeechDetectionObservation

INPUT_SAMPLE_RATE: Final = 16_000
SILERO_FRAME_SAMPLES: Final = 512
SPEECH_THRESHOLD: Final = 0.4
SILENCE_THRESHOLD: Final = SPEECH_THRESHOLD - 0.15
MINIMUM_SILENCE_SAMPLES: Final = INPUT_SAMPLE_RATE * 250 // 1_000


class SileroVadModel(Protocol):
    def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor: ...

    def reset_states(self) -> None: ...


class SileroSpeechDetector:
    def __init__(self, model: SileroVadModel) -> None:
        self.model = model
        self.model.reset_states()
        self.pending_samples = np.empty(0, dtype=np.float32)
        self.speech_active = False
        self.current_sample = 0
        self.pending_silence_started_at_sample: int | None = None
        self.latest_speech_probability: float | None = None

    def process_audio(self, pcm_bytes: bytes) -> SpeechDetectionObservation:
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32_768.0
        self.pending_samples = np.concatenate((self.pending_samples, samples))
        while len(self.pending_samples) >= SILERO_FRAME_SAMPLES:
            frame = torch.from_numpy(self.pending_samples[:SILERO_FRAME_SAMPLES])
            self.pending_samples = self.pending_samples[SILERO_FRAME_SAMPLES:]
            self._observe_frame(frame)
        return SpeechDetectionObservation(
            is_speech=self.speech_active,
            speech_probability=self.latest_speech_probability,
            pending_silence_samples=self._pending_silence_samples(),
        )

    @torch.no_grad()
    def _observe_frame(self, frame: torch.Tensor) -> None:
        self.current_sample += len(frame)
        speech_probability = float(self.model(frame, INPUT_SAMPLE_RATE).item())
        self.latest_speech_probability = speech_probability
        if speech_probability >= SPEECH_THRESHOLD:
            self.pending_silence_started_at_sample = None
            self.speech_active = True
            return
        if speech_probability >= SILENCE_THRESHOLD or not self.speech_active:
            return
        if self.pending_silence_started_at_sample is None:
            self.pending_silence_started_at_sample = self.current_sample
            return
        if self._pending_silence_samples() < MINIMUM_SILENCE_SAMPLES:
            return
        self.speech_active = False
        self.pending_silence_started_at_sample = None

    def _pending_silence_samples(self) -> int:
        if self.pending_silence_started_at_sample is None:
            return 0
        return self.current_sample - self.pending_silence_started_at_sample


class SileroSpeechDetectorFactory:
    def __init__(self) -> None:
        self.model: SileroVadModel = load_silero_vad()
        self.create()

    def create(self) -> SileroSpeechDetector:
        return SileroSpeechDetector(self.model)

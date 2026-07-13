from __future__ import annotations

from typing import Final

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

INPUT_SAMPLE_RATE: Final = 16_000


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

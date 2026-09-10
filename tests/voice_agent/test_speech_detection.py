from __future__ import annotations

from collections import deque

import pytest
import torch

from app.compute.voice.speech_detection import SILERO_FRAME_SAMPLES, SileroSpeechDetector


class ScriptedSileroModel:
    def __init__(self, probabilities: tuple[float, ...]) -> None:
        self.probabilities = deque(probabilities)
        self.reset_count = 0

    def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert len(audio) == SILERO_FRAME_SAMPLES
        assert sample_rate == 16_000
        return torch.tensor(self.probabilities.popleft())

    def reset_states(self) -> None:
        self.reset_count += 1


def test_detector_exposes_pending_silence_without_advancing_authoritative_endpoint() -> None:
    detector = SileroSpeechDetector(ScriptedSileroModel((0.9, 0.1, *(0.1 for _ in range(7)), 0.1)))
    frame = b"\x00\x00" * SILERO_FRAME_SAMPLES

    speech = detector.process_audio(frame)
    first_silence = detector.process_audio(frame)
    pending_observations = tuple(detector.process_audio(frame) for _ in range(7))
    endpoint = detector.process_audio(frame)

    assert speech.is_speech is True
    assert first_silence.is_speech is True
    assert first_silence.pending_silence_samples == 0
    assert [observation.pending_silence_samples for observation in pending_observations] == [
        512,
        1_024,
        1_536,
        2_048,
        2_560,
        3_072,
        3_584,
    ]
    assert all(observation.is_speech for observation in pending_observations)
    assert endpoint.is_speech is False
    assert endpoint.pending_silence_samples == 0


def test_detector_clears_pending_silence_when_speech_probability_recovers() -> None:
    detector = SileroSpeechDetector(ScriptedSileroModel((0.9, 0.1, 0.1, 0.1, 0.9)))
    frame = b"\x00\x00" * SILERO_FRAME_SAMPLES

    detector.process_audio(frame)
    detector.process_audio(frame)
    detector.process_audio(frame)
    pending = detector.process_audio(frame)
    recovered = detector.process_audio(frame)

    assert pending.is_speech is True
    assert pending.pending_silence_samples == 1_024
    assert recovered.is_speech is True
    assert recovered.speech_probability == pytest.approx(0.9)
    assert recovered.pending_silence_samples == 0

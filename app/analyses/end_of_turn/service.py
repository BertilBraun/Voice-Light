from __future__ import annotations

import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.audio.wav import mono_samples


@dataclass(frozen=True)
class WaveformEnvelope:
    sample_rate: int
    duration_seconds: float
    samples_per_bin: int
    minimums: list[float]
    maximums: list[float]


@dataclass(frozen=True)
class SpeechSegment:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class EndOfTurnEvent:
    time_seconds: float
    speech_start_seconds: float
    speech_end_seconds: float
    silence_seconds: float


@dataclass(frozen=True)
class BaselineResult:
    name: str
    description: str
    frame_seconds: float
    min_silence_seconds: float
    threshold: float
    speech_segments: list[SpeechSegment]
    end_of_turn_events: list[EndOfTurnEvent]


@dataclass(frozen=True)
class AudioAnalysis:
    speaker1_waveform: WaveformEnvelope
    speaker2_waveform: WaveformEnvelope
    baseline_results: list[BaselineResult]


def waveform_to_json(waveform: WaveformEnvelope) -> dict[str, object]:
    return asdict(waveform)


def baseline_to_json(baseline_result: BaselineResult) -> dict[str, object]:
    return asdict(baseline_result)


def analysis_to_json(audio_analysis: AudioAnalysis) -> dict[str, object]:
    return {
        "speaker1_waveform": waveform_to_json(audio_analysis.speaker1_waveform),
        "speaker2_waveform": waveform_to_json(audio_analysis.speaker2_waveform),
        "baseline_results": [
            baseline_to_json(baseline_result) for baseline_result in audio_analysis.baseline_results
        ],
    }


def build_waveform_envelope(wave_path: Path, target_bins: int = 3600) -> WaveformEnvelope:
    with wave.open(str(wave_path), "rb") as wave_reader:
        frame_count = wave_reader.getnframes()
        sample_rate = wave_reader.getframerate()
        sample_width = wave_reader.getsampwidth()
        channel_count = wave_reader.getnchannels()
        samples_per_bin = max(1, math.ceil(frame_count / target_bins))
        maximum_amplitude = float((1 << (sample_width * 8 - 1)) - 1)
        minimums: list[float] = []
        maximums: list[float] = []

        while wave_reader.tell() < frame_count:
            frames_to_read = min(samples_per_bin, frame_count - wave_reader.tell())
            fragment = wave_reader.readframes(frames_to_read)
            samples = mono_samples(
                fragment=fragment,
                sample_width=sample_width,
                channel_count=channel_count,
            )
            minimum = float(np.min(samples))
            maximum = float(np.max(samples))
            minimums.append(max(-1.0, minimum / maximum_amplitude))
            maximums.append(min(1.0, maximum / maximum_amplitude))

    duration_seconds = frame_count / sample_rate
    return WaveformEnvelope(
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        samples_per_bin=samples_per_bin,
        minimums=minimums,
        maximums=maximums,
    )


def analyze_session_audio(
    speaker1_path: Path,
    speaker2_path: Path,
    baseline_results: list[BaselineResult],
) -> AudioAnalysis:
    speaker1_waveform = build_waveform_envelope(wave_path=speaker1_path)
    speaker2_waveform = build_waveform_envelope(wave_path=speaker2_path)
    return AudioAnalysis(
        speaker1_waveform=speaker1_waveform,
        speaker2_waveform=speaker2_waveform,
        baseline_results=baseline_results,
    )

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import read_mono_wave_audio

TARGET_SAMPLE_RATE = 16000
FRAME_SECONDS = 512 / TARGET_SAMPLE_RATE


class SileroSpeechTimestamp(TypedDict):
    start: float
    end: float


class SileroVadModel(Protocol):
    def reset_states(self) -> None:
        raise NotImplementedError


class GetSpeechTimestamps(Protocol):
    def __call__(
        self,
        audio: Tensor,
        model: SileroVadModel,
        threshold: float,
        sampling_rate: int,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        return_seconds: bool,
        time_resolution: int,
    ) -> list[SileroSpeechTimestamp]:
        raise NotImplementedError


class LoadSileroVad(Protocol):
    def __call__(self, onnx: bool) -> SileroVadModel:
        raise NotImplementedError


@dataclass(frozen=True)
class SileroVadRuntime:
    model: SileroVadModel
    get_speech_timestamps: GetSpeechTimestamps


@dataclass(frozen=True)
class SileroVadDetector:
    info: EndOfTurnDetectorInfo
    threshold: float
    frame_seconds: float
    min_speech_seconds: float
    min_silence_seconds: float
    silero_min_silence_seconds: float
    speech_pad_seconds: float
    use_onnx: bool

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        runtime = _load_silero_runtime(use_onnx=self.use_onnx)
        audio = _read_mono_float_audio(
            wave_path=speaker1_path,
            target_sample_rate=TARGET_SAMPLE_RATE,
        )
        timestamps = runtime.get_speech_timestamps(
            torch.from_numpy(audio),
            runtime.model,
            threshold=self.threshold,
            sampling_rate=TARGET_SAMPLE_RATE,
            min_speech_duration_ms=_seconds_to_milliseconds(self.min_speech_seconds),
            min_silence_duration_ms=_seconds_to_milliseconds(self.silero_min_silence_seconds),
            speech_pad_ms=_seconds_to_milliseconds(self.speech_pad_seconds),
            return_seconds=True,
            time_resolution=6,
        )
        speech_segments = _speech_segments_from_silero_timestamps(timestamps=timestamps)
        end_of_turn_events = _end_of_turn_events(
            speech_segments=speech_segments,
            audio_duration_seconds=len(audio) / TARGET_SAMPLE_RATE,
            min_silence_seconds=self.min_silence_seconds,
        )
        return BaselineResult(
            name="silero_vad",
            description=self.info.description,
            frame_seconds=self.frame_seconds,
            min_silence_seconds=self.min_silence_seconds,
            threshold=self.threshold,
            speech_segments=speech_segments,
            pause_spans=[],
            backchannel_spans=[],
            end_of_turn_events=end_of_turn_events,
        )


def silero_vad_detector() -> SileroVadDetector:
    return SileroVadDetector(
        info=EndOfTurnDetectorInfo(
            mode=_silero_vad_mode(),
            label="Silero VAD",
            description=(
                "Silero VAD speech timestamps on 16 kHz mono audio with fixed 400 ms silence "
                "hysteresis for end-of-turn events."
            ),
        ),
        threshold=0.5,
        frame_seconds=FRAME_SECONDS,
        min_speech_seconds=0.1,
        min_silence_seconds=0.4,
        silero_min_silence_seconds=0.1,
        speech_pad_seconds=0.0,
        use_onnx=False,
    )


def _silero_vad_mode() -> EndOfTurnDetectorMode:
    try:
        return EndOfTurnDetectorMode.SILERO_VAD
    except AttributeError as error:
        raise ValueError(
            "Add EndOfTurnDetectorMode.SILERO_VAD before wiring silero_vad_detector()."
        ) from error


@lru_cache(maxsize=2)
def _load_silero_runtime(use_onnx: bool) -> SileroVadRuntime:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as error:
        raise ImportError(
            "Silero VAD runtime is not installed. Install `silero-vad` before using this detector."
        ) from error

    load_model = cast(LoadSileroVad, load_silero_vad)
    return SileroVadRuntime(
        model=load_model(onnx=use_onnx),
        get_speech_timestamps=cast(GetSpeechTimestamps, get_speech_timestamps),
    )


def _read_mono_float_audio(
    wave_path: Path,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    audio = read_mono_wave_audio(wave_path=wave_path)
    if audio.frame_count == 0:
        raise ValueError("Cannot run Silero VAD on an audio file with no frames.")

    normalized_samples = _normalize_pcm_samples(
        samples=audio.samples,
        sample_width=audio.sample_width,
    )
    return _resample_audio(
        audio=normalized_samples,
        source_sample_rate=audio.sample_rate,
        target_sample_rate=target_sample_rate,
    )


def _normalize_pcm_samples(
    samples: NDArray[np.float64],
    sample_width: int,
) -> NDArray[np.float32]:
    maximum_amplitude = float((1 << (sample_width * 8 - 1)) - 1)
    clipped_samples = np.clip(samples / maximum_amplitude, -1.0, 1.0)
    return clipped_samples.astype(np.float32)


def _resample_audio(
    audio: NDArray[np.float32],
    source_sample_rate: int,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    if source_sample_rate == target_sample_rate:
        return audio

    duration_seconds = len(audio) / source_sample_rate
    target_sample_count = max(1, round(duration_seconds * target_sample_rate))
    source_positions = np.linspace(0.0, duration_seconds, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, duration_seconds, num=target_sample_count, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _speech_segments_from_silero_timestamps(
    timestamps: list[SileroSpeechTimestamp],
) -> list[SpeechSegment]:
    return [
        SpeechSegment(
            start_seconds=_rounded_seconds(timestamp["start"]),
            end_seconds=_rounded_seconds(timestamp["end"]),
        )
        for timestamp in timestamps
    ]


def _end_of_turn_events(
    speech_segments: list[SpeechSegment],
    audio_duration_seconds: float,
    min_silence_seconds: float,
) -> list[EndOfTurnEvent]:
    end_of_turn_events: list[EndOfTurnEvent] = []
    for segment_index, speech_segment in enumerate(speech_segments):
        next_segment_start_seconds = _next_segment_start_seconds(
            speech_segments=speech_segments,
            segment_index=segment_index,
            audio_duration_seconds=audio_duration_seconds,
        )
        silence_seconds = _rounded_seconds(next_segment_start_seconds - speech_segment.end_seconds)
        if silence_seconds >= min_silence_seconds:
            end_of_turn_events.append(
                EndOfTurnEvent(
                    time_seconds=_rounded_seconds(speech_segment.end_seconds + min_silence_seconds),
                    speech_start_seconds=speech_segment.start_seconds,
                    speech_end_seconds=speech_segment.end_seconds,
                    silence_seconds=silence_seconds,
                )
            )
    return end_of_turn_events


def _next_segment_start_seconds(
    speech_segments: list[SpeechSegment],
    segment_index: int,
    audio_duration_seconds: float,
) -> float:
    next_segment_index = segment_index + 1
    if next_segment_index < len(speech_segments):
        return speech_segments[next_segment_index].start_seconds
    return audio_duration_seconds


def _seconds_to_milliseconds(seconds: float) -> int:
    return round(seconds * 1000)


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, 6)

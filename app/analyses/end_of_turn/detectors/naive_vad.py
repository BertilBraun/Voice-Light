from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import mono_samples


@dataclass(frozen=True)
class NaiveVadFloorDetector:
    info: EndOfTurnDetectorInfo = EndOfTurnDetectorInfo(
        mode=EndOfTurnDetectorMode.NAIVE_VAD_FLOOR,
        label="Naive VAD floor",
        description="RMS VAD on speaker 1 with fixed silence hysteresis.",
    )

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        return run_naive_vad_floor(wave_path=speaker1_path)


def run_naive_vad_floor(
    wave_path: Path,
    frame_seconds: float = 0.03,
    min_speech_seconds: float = 0.12,
    min_silence_seconds: float = 0.7,
) -> BaselineResult:
    frame_energies = _read_frame_energies(wave_path=wave_path, frame_seconds=frame_seconds)
    threshold = _adaptive_threshold(frame_energies=frame_energies)
    speech_flags = [frame_energy >= threshold for frame_energy in frame_energies]
    speech_segments = _speech_segments_from_flags(
        speech_flags=speech_flags,
        frame_seconds=frame_seconds,
        min_speech_seconds=min_speech_seconds,
    )
    end_of_turn_events = _end_of_turn_events(
        speech_segments=speech_segments,
        min_silence_seconds=min_silence_seconds,
    )
    return BaselineResult(
        name="naive_vad_floor",
        description="RMS VAD on speaker 1 with fixed silence hysteresis.",
        frame_seconds=frame_seconds,
        min_silence_seconds=min_silence_seconds,
        threshold=threshold,
        speech_segments=speech_segments,
        end_of_turn_events=end_of_turn_events,
    )


def _read_frame_energies(wave_path: Path, frame_seconds: float) -> list[float]:
    with wave.open(str(wave_path), "rb") as wave_reader:
        sample_rate = wave_reader.getframerate()
        sample_width = wave_reader.getsampwidth()
        channel_count = wave_reader.getnchannels()
        frames_per_window = max(1, round(sample_rate * frame_seconds))
        frame_energies: list[float] = []

        while True:
            fragment = wave_reader.readframes(frames_per_window)
            if not fragment:
                break
            samples = mono_samples(
                fragment=fragment,
                sample_width=sample_width,
                channel_count=channel_count,
            )
            frame_energies.append(float(np.sqrt(np.mean(samples * samples))))

    return frame_energies


def _adaptive_threshold(frame_energies: list[float]) -> float:
    if not frame_energies:
        raise ValueError("Cannot run VAD on an audio file with no frames.")

    sorted_energies = sorted(frame_energies)
    noise_floor = _percentile(sorted_values=sorted_energies, percentile=0.20)
    speech_level = _percentile(sorted_values=sorted_energies, percentile=0.90)
    return max(noise_floor * 3.0, speech_level * 0.08, 1.0)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile for an empty list.")
    bounded_percentile = min(1.0, max(0.0, percentile))
    index = round((len(sorted_values) - 1) * bounded_percentile)
    return sorted_values[index]


def _speech_segments_from_flags(
    speech_flags: list[bool],
    frame_seconds: float,
    min_speech_seconds: float,
) -> list[SpeechSegment]:
    speech_segments: list[SpeechSegment] = []
    active_start_index: int | None = None

    for frame_index, is_speech in enumerate(speech_flags):
        if is_speech and active_start_index is None:
            active_start_index = frame_index
        if not is_speech and active_start_index is not None:
            _append_segment_if_long_enough(
                speech_segments=speech_segments,
                start_index=active_start_index,
                end_index=frame_index,
                frame_seconds=frame_seconds,
                min_speech_seconds=min_speech_seconds,
            )
            active_start_index = None

    if active_start_index is not None:
        _append_segment_if_long_enough(
            speech_segments=speech_segments,
            start_index=active_start_index,
            end_index=len(speech_flags),
            frame_seconds=frame_seconds,
            min_speech_seconds=min_speech_seconds,
        )

    return speech_segments


def _append_segment_if_long_enough(
    speech_segments: list[SpeechSegment],
    start_index: int,
    end_index: int,
    frame_seconds: float,
    min_speech_seconds: float,
) -> None:
    start_seconds = _rounded_seconds(start_index * frame_seconds)
    end_seconds = _rounded_seconds(end_index * frame_seconds)
    if _rounded_seconds(end_seconds - start_seconds) >= min_speech_seconds:
        speech_segments.append(SpeechSegment(start_seconds=start_seconds, end_seconds=end_seconds))


def _end_of_turn_events(
    speech_segments: list[SpeechSegment],
    min_silence_seconds: float,
) -> list[EndOfTurnEvent]:
    end_of_turn_events: list[EndOfTurnEvent] = []
    for current_segment, next_segment in zip(
        speech_segments,
        speech_segments[1:],
        strict=False,
    ):
        silence_seconds = next_segment.start_seconds - current_segment.end_seconds
        if silence_seconds >= min_silence_seconds:
            end_of_turn_events.append(
                EndOfTurnEvent(
                    time_seconds=_rounded_seconds(
                        current_segment.end_seconds + min_silence_seconds
                    ),
                    speech_start_seconds=current_segment.start_seconds,
                    speech_end_seconds=current_segment.end_seconds,
                    silence_seconds=silence_seconds,
                )
            )
    return end_of_turn_events


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, 6)

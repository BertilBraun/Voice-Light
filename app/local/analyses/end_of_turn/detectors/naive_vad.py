from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.local.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.local.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.shared.audio.wav import iter_mono_wave_audio_chunks, read_mono_wave_audio

FULL_RECORDING_READ_CHUNK_SECONDS = 30.0


@dataclass(frozen=True)
class NaiveVadFloorDetector:
    info: EndOfTurnDetectorInfo
    frame_seconds: float
    min_speech_seconds: float
    min_silence_seconds: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        return run_naive_vad_floor(
            wave_path=speaker1_path,
            result_name=self.info.mode.value,
            description=self.info.description,
            frame_seconds=self.frame_seconds,
            min_speech_seconds=self.min_speech_seconds,
            min_silence_seconds=self.min_silence_seconds,
        )


def run_naive_vad_floor(
    wave_path: Path,
    result_name: str,
    description: str,
    frame_seconds: float = 0.03,
    min_speech_seconds: float = 0.12,
    min_silence_seconds: float = 0.7,
) -> BaselineResult:
    frame_energies = _read_frame_energies(wave_path=wave_path, frame_seconds=frame_seconds)
    return _vad_result_from_frame_energies(
        frame_energies=frame_energies,
        result_name=result_name,
        description=description,
        frame_seconds=frame_seconds,
        min_speech_seconds=min_speech_seconds,
        min_silence_seconds=min_silence_seconds,
    )


def run_naive_vad_floor_full_recording(
    wave_path: Path,
    result_name: str,
    description: str,
    frame_seconds: float = 0.03,
    min_speech_seconds: float = 0.12,
    min_silence_seconds: float = 0.7,
) -> BaselineResult:
    frame_energies = _read_full_recording_frame_energies(
        wave_path=wave_path,
        frame_seconds=frame_seconds,
    )
    return _vad_result_from_frame_energies(
        frame_energies=frame_energies,
        result_name=result_name,
        description=description,
        frame_seconds=frame_seconds,
        min_speech_seconds=min_speech_seconds,
        min_silence_seconds=min_silence_seconds,
    )


def _vad_result_from_frame_energies(
    frame_energies: list[float],
    result_name: str,
    description: str,
    frame_seconds: float,
    min_speech_seconds: float,
    min_silence_seconds: float,
) -> BaselineResult:
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
        name=result_name,
        description=description,
        frame_seconds=frame_seconds,
        min_silence_seconds=min_silence_seconds,
        threshold=threshold,
        speech_segments=speech_segments,
        pause_spans=[],
        backchannel_spans=[],
        end_of_turn_events=end_of_turn_events,
    )


def _read_full_recording_frame_energies(
    wave_path: Path,
    frame_seconds: float,
) -> list[float]:
    frame_energies: list[float] = []
    pending_samples = np.empty(0, dtype=np.float64)
    sample_rate: int | None = None
    frames_per_window: int | None = None
    for audio in iter_mono_wave_audio_chunks(
        wave_path=wave_path,
        chunk_duration_seconds=FULL_RECORDING_READ_CHUNK_SECONDS,
    ):
        if sample_rate is None:
            sample_rate = audio.sample_rate
            frames_per_window = max(1, round(sample_rate * frame_seconds))
        assert audio.sample_rate == sample_rate
        assert frames_per_window is not None
        samples = (
            np.concatenate((pending_samples, audio.samples))
            if len(pending_samples)
            else audio.samples
        )
        complete_sample_count = len(samples) - len(samples) % frames_per_window
        complete_samples = samples[:complete_sample_count]
        if complete_sample_count:
            windows = complete_samples.reshape(-1, frames_per_window)
            energies = np.sqrt(np.mean(np.square(windows), axis=1))
            frame_energies.extend(float(energy) for energy in energies)
        pending_samples = samples[complete_sample_count:].copy()
    if len(pending_samples):
        frame_energies.append(float(np.sqrt(np.mean(np.square(pending_samples)))))
    return frame_energies


def _read_frame_energies(
    wave_path: Path,
    frame_seconds: float,
) -> list[float]:
    audio = read_mono_wave_audio(wave_path=wave_path)
    frames_per_window = max(1, round(audio.sample_rate * frame_seconds))
    frame_energies: list[float] = []

    for start_index in range(0, audio.frame_count, frames_per_window):
        samples = audio.samples[start_index : start_index + frames_per_window]
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


def naive_vad_floor_detector() -> NaiveVadFloorDetector:
    return NaiveVadFloorDetector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.NAIVE_VAD_FLOOR,
            label="Naive VAD floor",
            description="RMS VAD on speaker 1 with fixed 700 ms silence hysteresis.",
        ),
        frame_seconds=0.03,
        min_speech_seconds=0.12,
        min_silence_seconds=0.7,
    )


def naive_vad_fast_detector() -> NaiveVadFloorDetector:
    return NaiveVadFloorDetector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.NAIVE_VAD_FAST,
            label="Naive VAD fast",
            description="RMS VAD on speaker 1 with shorter 400 ms silence hysteresis.",
        ),
        frame_seconds=0.03,
        min_speech_seconds=0.09,
        min_silence_seconds=0.4,
    )

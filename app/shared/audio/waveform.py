from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS, mono_samples


@dataclass(frozen=True)
class WaveformPoint:
    minimum_amplitude: float
    maximum_amplitude: float


@dataclass(frozen=True)
class WaveformEnvelope:
    duration_seconds: float
    sample_rate: int
    points: tuple[WaveformPoint, ...]


def full_waveform_envelope(wave_path: Path, point_count: int) -> WaveformEnvelope:
    return _waveform_envelope(
        wave_path=wave_path,
        point_count=point_count,
        maximum_duration_seconds=None,
    )


def capped_waveform_envelope(wave_path: Path, point_count: int) -> WaveformEnvelope:
    return _waveform_envelope(
        wave_path=wave_path,
        point_count=point_count,
        maximum_duration_seconds=ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
    )


def _waveform_envelope(
    wave_path: Path,
    point_count: int,
    maximum_duration_seconds: float | None,
) -> WaveformEnvelope:
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    resolved_path = wave_path.resolve()
    if not resolved_path.is_file():
        raise ValueError(f"Audio file does not exist: {resolved_path}")
    if resolved_path.suffix.lower() != ".wav":
        return _ffmpeg_waveform_envelope(
            audio_path=resolved_path,
            point_count=point_count,
            maximum_duration_seconds=maximum_duration_seconds,
        )
    return _cached_waveform_envelope(
        wave_path=resolved_path,
        modified_nanoseconds=resolved_path.stat().st_mtime_ns,
        point_count=point_count,
        maximum_duration_seconds=maximum_duration_seconds,
    )


def _ffmpeg_waveform_envelope(
    audio_path: Path,
    point_count: int,
    maximum_duration_seconds: float | None,
) -> WaveformEnvelope:
    waveform_sample_rate = 100
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    command.extend(
        (
            "-i",
            str(audio_path),
        )
    )
    if maximum_duration_seconds is not None:
        command.extend(("-t", str(maximum_duration_seconds)))
    command.extend(
        (
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(waveform_sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        )
    )
    completed = subprocess.run(tuple(command), check=True, capture_output=True)
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    if len(samples) == 0:
        return WaveformEnvelope(
            duration_seconds=0.0,
            sample_rate=waveform_sample_rate,
            points=(),
        )
    samples_per_point = max(1, int(np.ceil(len(samples) / point_count)))
    points = tuple(
        WaveformPoint(
            minimum_amplitude=max(-1.0, float(np.min(fragment))),
            maximum_amplitude=min(1.0, float(np.max(fragment))),
        )
        for start_index in range(0, len(samples), samples_per_point)
        if len(fragment := samples[start_index : start_index + samples_per_point])
    )
    return WaveformEnvelope(
        duration_seconds=len(samples) / waveform_sample_rate,
        sample_rate=waveform_sample_rate,
        points=points,
    )


@lru_cache(maxsize=64)
def _cached_waveform_envelope(
    wave_path: Path,
    modified_nanoseconds: int,
    point_count: int,
    maximum_duration_seconds: float | None,
) -> WaveformEnvelope:
    del modified_nanoseconds
    try:
        with wave.open(str(wave_path), "rb") as wave_reader:
            sample_rate = wave_reader.getframerate()
            sample_width = wave_reader.getsampwidth()
            channel_count = wave_reader.getnchannels()
            source_frame_count = wave_reader.getnframes()
            frame_count = (
                source_frame_count
                if maximum_duration_seconds is None
                else min(source_frame_count, round(sample_rate * maximum_duration_seconds))
            )
            points = _read_waveform_points(
                wave_reader=wave_reader,
                frame_count=frame_count,
                sample_width=sample_width,
                channel_count=channel_count,
                point_count=point_count,
            )
    except (OSError, wave.Error) as error:
        raise ValueError(f"Could not read WAV file {wave_path}: {error}") from error

    return WaveformEnvelope(
        duration_seconds=frame_count / sample_rate,
        sample_rate=sample_rate,
        points=points,
    )


def _read_waveform_points(
    wave_reader: wave.Wave_read,
    frame_count: int,
    sample_width: int,
    channel_count: int,
    point_count: int,
) -> tuple[WaveformPoint, ...]:
    frames_per_point = max(1, int(np.ceil(frame_count / point_count)))
    maximum_amplitude = float(1 << (sample_width * 8 - 1))
    points: list[WaveformPoint] = []
    while len(points) < point_count:
        fragment = wave_reader.readframes(frames_per_point)
        if not fragment:
            break
        samples = mono_samples(
            fragment=fragment,
            sample_width=sample_width,
            channel_count=channel_count,
        )
        points.append(
            WaveformPoint(
                minimum_amplitude=max(-1.0, float(np.min(samples)) / maximum_amplitude),
                maximum_amplitude=min(1.0, float(np.max(samples)) / maximum_amplitude),
            )
        )
    return tuple(points)

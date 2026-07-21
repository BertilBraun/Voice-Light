from __future__ import annotations

import io
import subprocess
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ANALYSIS_AUDIO_MAX_DURATION_SECONDS = 180.0
PLAYBACK_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class MonoWaveAudio:
    samples: NDArray[np.float64]
    sample_rate: int
    sample_width: int
    channel_count: int
    frame_count: int
    duration_seconds: float


def mono_samples(
    fragment: bytes,
    sample_width: int,
    channel_count: int,
) -> NDArray[np.float64]:
    if sample_width == 1:
        unsigned_samples = np.frombuffer(fragment, dtype=np.uint8).reshape(-1, channel_count)
        return (unsigned_samples[:, 0].astype(np.float64) - 128.0).copy()
    if sample_width == 2:
        return (
            np.frombuffer(fragment, dtype="<i2").reshape(-1, channel_count)[:, 0].astype(np.float64)
        )
    if sample_width == 3:
        sample_bytes = np.frombuffer(fragment, dtype=np.uint8).reshape(-1, channel_count, 3)
        first_channel = sample_bytes[:, 0, :].astype(np.uint32)
        unsigned_values = (
            first_channel[:, 0] | (first_channel[:, 1] << 8) | (first_channel[:, 2] << 16)
        )
        signed_values = unsigned_values.astype(np.int32)
        signed_values[signed_values >= 0x800000] -= 0x1000000
        return signed_values.astype(np.float64)
    if sample_width == 4:
        return (
            np.frombuffer(fragment, dtype="<i4").reshape(-1, channel_count)[:, 0].astype(np.float64)
        )
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def read_mono_wave_audio(wave_path: Path) -> MonoWaveAudio:
    return read_mono_wave_audio_window(
        wave_path=wave_path,
        start_seconds=0.0,
        maximum_duration_seconds=ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
    )


def read_mono_wave_audio_window(
    wave_path: Path,
    start_seconds: float,
    maximum_duration_seconds: float,
) -> MonoWaveAudio:
    if start_seconds < 0.0:
        raise ValueError("start_seconds must be non-negative")
    if maximum_duration_seconds <= 0.0:
        raise ValueError("maximum_duration_seconds must be positive")
    with wave.open(str(wave_path), "rb") as wave_reader:
        sample_rate = wave_reader.getframerate()
        sample_width = wave_reader.getsampwidth()
        channel_count = wave_reader.getnchannels()
        source_frame_count = wave_reader.getnframes()
        start_frame = min(source_frame_count, round(start_seconds * sample_rate))
        frame_count = min(
            source_frame_count - start_frame,
            round(maximum_duration_seconds * sample_rate),
        )
        wave_reader.setpos(start_frame)
        fragment = wave_reader.readframes(frame_count)

    samples = mono_samples(
        fragment=fragment,
        sample_width=sample_width,
        channel_count=channel_count,
    )
    return MonoWaveAudio(
        samples=samples,
        sample_rate=sample_rate,
        sample_width=sample_width,
        channel_count=channel_count,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
    )


def iter_mono_wave_audio_chunks(
    wave_path: Path,
    chunk_duration_seconds: float,
) -> Iterator[MonoWaveAudio]:
    if chunk_duration_seconds <= 0.0:
        raise ValueError("chunk_duration_seconds must be positive")
    with wave.open(str(wave_path), "rb") as wave_reader:
        sample_rate = wave_reader.getframerate()
        sample_width = wave_reader.getsampwidth()
        channel_count = wave_reader.getnchannels()
        source_frame_count = wave_reader.getnframes()
        chunk_frame_count = max(1, round(chunk_duration_seconds * sample_rate))
        remaining_frame_count = source_frame_count
        while remaining_frame_count > 0:
            frame_count = min(remaining_frame_count, chunk_frame_count)
            fragment = wave_reader.readframes(frame_count)
            samples = mono_samples(
                fragment=fragment,
                sample_width=sample_width,
                channel_count=channel_count,
            )
            actual_frame_count = len(samples)
            if actual_frame_count == 0:
                break
            yield MonoWaveAudio(
                samples=samples,
                sample_rate=sample_rate,
                sample_width=sample_width,
                channel_count=channel_count,
                frame_count=actual_frame_count,
                duration_seconds=actual_frame_count / sample_rate,
            )
            remaining_frame_count -= actual_frame_count


def capped_wave_bytes(wave_path: Path) -> bytes:
    audio = read_mono_wave_audio(wave_path=wave_path)
    return playback_wave_bytes(audio=audio)


def capped_audio_wave_bytes(audio_path: Path) -> bytes:
    if audio_path.suffix.lower() == ".wav":
        return capped_wave_bytes(audio_path)
    completed = subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-t",
            str(ANALYSIS_AUDIO_MAX_DURATION_SECONDS),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ),
        check=True,
        capture_output=True,
    )
    output_buffer = io.BytesIO()
    with wave.open(output_buffer, "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(PLAYBACK_SAMPLE_WIDTH_BYTES)
        wave_writer.setframerate(16_000)
        wave_writer.writeframes(completed.stdout)
    return output_buffer.getvalue()


def wave_window_bytes(
    wave_path: Path,
    start_seconds: float,
    maximum_duration_seconds: float,
) -> bytes:
    audio = read_mono_wave_audio_window(
        wave_path=wave_path,
        start_seconds=start_seconds,
        maximum_duration_seconds=maximum_duration_seconds,
    )
    return playback_wave_bytes(audio=audio)


def resampled_wave_window_bytes(
    wave_path: Path,
    start_seconds: float,
    maximum_duration_seconds: float,
    sample_rate: int,
) -> bytes:
    if start_seconds < 0.0:
        raise ValueError("start_seconds must be non-negative")
    if maximum_duration_seconds <= 0.0:
        raise ValueError("maximum_duration_seconds must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    output_buffer = io.BytesIO()
    with wave.open(str(wave_path), "rb") as wave_reader:
        source_sample_rate = wave_reader.getframerate()
        source_sample_width = wave_reader.getsampwidth()
        channel_count = wave_reader.getnchannels()
        source_frame_count = wave_reader.getnframes()
        start_frame = min(source_frame_count, round(start_seconds * source_sample_rate))
        remaining_frames = min(
            source_frame_count - start_frame,
            round(maximum_duration_seconds * source_sample_rate),
        )
        wave_reader.setpos(start_frame)
        source_maximum_amplitude = float((1 << (source_sample_width * 8 - 1)) - 1)
        target_maximum_amplitude = float((1 << (PLAYBACK_SAMPLE_WIDTH_BYTES * 8 - 1)) - 1)
        with wave.open(output_buffer, "wb") as wave_writer:
            wave_writer.setnchannels(1)
            wave_writer.setsampwidth(PLAYBACK_SAMPLE_WIDTH_BYTES)
            wave_writer.setframerate(sample_rate)
            while remaining_frames > 0:
                chunk_frame_count = min(source_sample_rate, remaining_frames)
                samples = mono_samples(
                    fragment=wave_reader.readframes(chunk_frame_count),
                    sample_width=source_sample_width,
                    channel_count=channel_count,
                )
                target_frame_count = max(
                    1,
                    round(len(samples) * sample_rate / source_sample_rate),
                )
                target_positions = (
                    np.arange(target_frame_count, dtype=np.float64)
                    * source_sample_rate
                    / sample_rate
                )
                resampled_samples = np.interp(
                    target_positions,
                    np.arange(len(samples), dtype=np.float64),
                    samples,
                )
                scaled_samples = np.round(
                    resampled_samples * (target_maximum_amplitude / source_maximum_amplitude)
                )
                clipped_samples = np.clip(
                    scaled_samples,
                    -target_maximum_amplitude - 1,
                    target_maximum_amplitude,
                )
                wave_writer.writeframesraw(clipped_samples.astype("<i2").tobytes())
                remaining_frames -= len(samples)
    return output_buffer.getvalue()


def playback_wave_bytes(audio: MonoWaveAudio) -> bytes:
    fragment = playback_pcm16_fragment(audio=audio)

    output_buffer = io.BytesIO()
    with wave.open(output_buffer, "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(PLAYBACK_SAMPLE_WIDTH_BYTES)
        wave_writer.setframerate(audio.sample_rate)
        wave_writer.setnframes(audio.frame_count)
        wave_writer.writeframes(fragment)
    return output_buffer.getvalue()


def playback_pcm16_fragment(audio: MonoWaveAudio) -> bytes:
    source_maximum_amplitude = float((1 << (audio.sample_width * 8 - 1)) - 1)
    target_maximum_amplitude = float((1 << (PLAYBACK_SAMPLE_WIDTH_BYTES * 8 - 1)) - 1)
    scaled_samples = np.round(audio.samples * (target_maximum_amplitude / source_maximum_amplitude))
    clipped_samples = np.clip(
        scaled_samples,
        -target_maximum_amplitude - 1,
        target_maximum_amplitude,
    )
    return clipped_samples.astype("<i2").tobytes()

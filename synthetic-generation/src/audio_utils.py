from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class SpeechBounds:
    onset_seconds: float
    offset_seconds: float
    duration_seconds: float
    original_duration_seconds: float


def to_mono_float32(audio_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio_samples, dtype=np.float32)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return np.mean(samples, axis=1, dtype=np.float32)
    raise ValueError(f"Expected one- or two-dimensional audio, got shape {samples.shape}")


def write_mono_wav(audio_path: Path, audio_samples: np.ndarray, sample_rate: int) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    mono_samples = to_mono_float32(audio_samples)
    sf.write(audio_path, mono_samples, sample_rate, format="WAV", subtype="PCM_16")


def write_audio_bytes_as_mono_wav(audio_path: Path, audio_bytes: bytes) -> tuple[float, int]:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    samples, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
    mono_samples = to_mono_float32(samples)
    sf.write(audio_path, mono_samples, sample_rate, format="WAV", subtype="PCM_16")
    return audio_duration_seconds(audio_path), audio_sample_rate(audio_path)


def audio_duration_seconds(audio_path: Path) -> float:
    audio_info = sf.info(audio_path)
    return float(audio_info.frames) / float(audio_info.samplerate)


def audio_sample_rate(audio_path: Path) -> int:
    return int(sf.info(audio_path).samplerate)


def trim_to_speech(audio_path: Path, padding_seconds: float = 0.03) -> SpeechBounds:
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    mono_samples = to_mono_float32(samples)
    if mono_samples.size == 0:
        raise ValueError(f"Audio file has no samples: {audio_path}")
    original_duration_seconds = float(mono_samples.size) / float(sample_rate)
    onset_seconds, offset_seconds = detect_major_speech_bounds(mono_samples, sample_rate)
    padded_onset_seconds = max(0.0, onset_seconds - padding_seconds)
    padded_offset_seconds = min(original_duration_seconds, offset_seconds + padding_seconds)
    start_sample = int(round(padded_onset_seconds * sample_rate))
    end_sample = max(start_sample + 1, int(round(padded_offset_seconds * sample_rate)))
    trimmed_samples = mono_samples[start_sample:end_sample]
    sf.write(audio_path, trimmed_samples, sample_rate, format="WAV", subtype="PCM_16")
    return SpeechBounds(
        onset_seconds=padded_onset_seconds,
        offset_seconds=padded_offset_seconds,
        duration_seconds=float(trimmed_samples.size) / float(sample_rate),
        original_duration_seconds=original_duration_seconds,
    )


def detect_major_audio_onset_seconds(audio_path: Path) -> float:
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    mono_samples = to_mono_float32(samples)
    if mono_samples.size == 0:
        raise ValueError(f"Audio file has no samples: {audio_path}")
    onset_seconds, _offset_seconds = detect_major_speech_bounds(mono_samples, sample_rate)
    return onset_seconds


def detect_major_speech_bounds(mono_samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    frame_size = max(1, int(sample_rate * 0.02))
    hop_size = max(1, int(sample_rate * 0.005))
    rms_values: list[float] = []
    frame_starts: list[int] = []
    for frame_start in range(0, max(1, mono_samples.size - frame_size + 1), hop_size):
        frame = mono_samples[frame_start : frame_start + frame_size]
        rms_values.append(float(np.sqrt(np.mean(np.square(frame)))))
        frame_starts.append(frame_start)
    if len(rms_values) == 0:
        return 0.0, 0.0
    rms_array = np.asarray(rms_values, dtype=np.float32)
    peak_rms = float(np.max(rms_array))
    if peak_rms <= 0:
        duration_seconds = float(mono_samples.size) / float(sample_rate)
        return 0.0, duration_seconds
    noise_floor = float(np.percentile(rms_array, 20))
    onset_threshold = max(noise_floor * 8.0, peak_rms * 0.18, 0.003)
    offset_threshold = max(noise_floor * 3.0, peak_rms * 0.025, 0.0006)
    required_frames = 3
    for index in range(0, len(rms_array) - required_frames + 1):
        window = rms_array[index : index + required_frames]
        if bool(np.all(window >= onset_threshold)):
            onset_index = index
            offset_index = find_speech_offset_index(
                rms_array,
                offset_threshold,
                required_frames,
            )
            return (
                float(frame_starts[onset_index]) / float(sample_rate),
                min(
                    float(frame_starts[offset_index] + frame_size) / float(sample_rate),
                    float(mono_samples.size) / float(sample_rate),
                ),
            )
    first_major_index = int(np.argmax(rms_array >= onset_threshold))
    if rms_array[first_major_index] >= onset_threshold:
        offset_index = find_speech_offset_index(rms_array, offset_threshold, 1)
        return (
            float(frame_starts[first_major_index]) / float(sample_rate),
            min(
                float(frame_starts[offset_index] + frame_size) / float(sample_rate),
                float(mono_samples.size) / float(sample_rate),
            ),
        )
    duration_seconds = float(mono_samples.size) / float(sample_rate)
    return 0.0, duration_seconds


def find_speech_offset_index(
    rms_array: np.ndarray,
    threshold: float,
    required_frames: int,
) -> int:
    for index in range(len(rms_array) - required_frames, -1, -1):
        window = rms_array[index : index + required_frames]
        if bool(np.all(window >= threshold)):
            return index + required_frames - 1
    return len(rms_array) - 1


def validate_decodable_audio(audio_path: Path) -> None:
    with sf.SoundFile(audio_path) as audio_file:
        if audio_file.frames <= 0:
            raise ValueError(f"Audio file has no frames: {audio_path}")
        if audio_file.samplerate <= 0:
            raise ValueError(f"Audio file has invalid sample rate: {audio_path}")


def write_test_tone(
    audio_path: Path, frequency_hz: float, duration_seconds: float, sample_rate: int
) -> None:
    frame_count = int(duration_seconds * sample_rate)
    times = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
    samples = 0.1 * np.sin(2.0 * np.pi * frequency_hz * times)
    write_mono_wav(audio_path, samples, sample_rate)

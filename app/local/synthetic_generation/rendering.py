from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from app.local.synthetic_generation.models import (
    PauseEvent,
    SpeakerRole,
    SyntheticConversationPlan,
    SyntheticModel,
)


class TtsProvenance(SyntheticModel):
    backend_id: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    model_license: str = Field(min_length=1)
    seed: int = Field(ge=0)
    generation_seconds: float | None = Field(default=None, ge=0.0)
    real_time_factor: float | None = Field(default=None, ge=0.0)


class RenderedEventSource(SyntheticModel):
    event_id: str
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: TtsProvenance


class AcousticAugmentation(SyntheticModel):
    user_gain_db: float = 0.0
    assistant_gain_db: float = 0.0
    signal_to_noise_db: float | None = Field(default=None, gt=0.0)
    reverb_decay_seconds: float | None = Field(default=None, gt=0.0)
    codec: Literal["none", "mulaw_8bit"] = "none"
    microphone_low_hz: float | None = Field(default=None, ge=0.0)
    microphone_high_hz: float | None = Field(default=None, gt=0.0)
    user_to_assistant_crosstalk_db: float | None = Field(default=None, le=0.0)
    assistant_to_user_crosstalk_db: float | None = Field(default=None, le=0.0)

    @model_validator(mode="after")
    def validate_microphone_band(self) -> AcousticAugmentation:
        if (
            self.microphone_low_hz is not None
            and self.microphone_high_hz is not None
            and self.microphone_low_hz >= self.microphone_high_hz
        ):
            raise ValueError("Microphone low cutoff must be below its high cutoff.")
        return self


class SyntheticRenderRequest(SyntheticModel):
    plan: SyntheticConversationPlan
    sample_rate_hz: int = Field(default=16_000, gt=0)
    sources: tuple[RenderedEventSource, ...]
    augmentation: AcousticAugmentation = AcousticAugmentation()

    @model_validator(mode="after")
    def validate_sources(self) -> SyntheticRenderRequest:
        expected_ids = {
            event.event_id for event in self.plan.events if not isinstance(event, PauseEvent)
        }
        source_ids = {source.event_id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("Rendered source event IDs must be unique.")
        if source_ids != expected_ids:
            missing = sorted(expected_ids - source_ids)
            unexpected = sorted(source_ids - expected_ids)
            raise ValueError(
                f"Rendered sources do not match voiced events; missing={missing}, "
                f"unexpected={unexpected}."
            )
        return self


class SyntheticRenderManifest(SyntheticModel):
    schema_version: Literal["voice-light-synthetic-render-v1"] = "voice-light-synthetic-render-v1"
    plan: SyntheticConversationPlan
    sample_rate_hz: int
    sources: tuple[RenderedEventSource, ...]
    augmentation: AcousticAugmentation
    user_audio_path: Path
    assistant_audio_path: Path
    user_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assistant_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_normalization_gain: float = Field(gt=0.0, le=1.0)
    assistant_normalization_gain: float = Field(gt=0.0, le=1.0)


def render_conversation(
    request: SyntheticRenderRequest,
    output_directory: Path,
) -> SyntheticRenderManifest:
    output_directory.mkdir(parents=True, exist_ok=True)
    source_by_event_id = {source.event_id: source for source in request.sources}
    speakers_by_id = {speaker.speaker_id: speaker for speaker in request.plan.speakers}
    sample_count = round(request.plan.duration_seconds * request.sample_rate_hz)
    user_track = np.zeros(sample_count, dtype=np.float32)
    assistant_track = np.zeros(sample_count, dtype=np.float32)
    for event in request.plan.events:
        if isinstance(event, PauseEvent):
            continue
        source = source_by_event_id[event.event_id]
        if _file_sha256(source.audio_path) != source.audio_sha256:
            raise ValueError(f"Source audio hash changed for event {event.event_id}.")
        source_samples, source_rate_hz = _read_pcm16_wave(source.audio_path)
        fitted = _fit_event_duration(
            source_samples=source_samples,
            source_rate_hz=source_rate_hz,
            output_rate_hz=request.sample_rate_hz,
            duration_seconds=event.end_seconds - event.start_seconds,
        )
        output_start = round(event.start_seconds * request.sample_rate_hz)
        output_end = output_start + fitted.size
        role = speakers_by_id[event.speaker_id].role
        track = user_track if role is SpeakerRole.USER else assistant_track
        track[output_start:output_end] += fitted
    generator = np.random.default_rng(request.plan.seed)
    user_track = _augment_track(
        user_track,
        request.sample_rate_hz,
        request.augmentation,
        request.augmentation.user_gain_db,
        generator,
    )
    assistant_track = _augment_track(
        assistant_track,
        request.sample_rate_hz,
        request.augmentation,
        request.augmentation.assistant_gain_db,
        generator,
    )
    clean_user = user_track.copy()
    clean_assistant = assistant_track.copy()
    if request.augmentation.assistant_to_user_crosstalk_db is not None:
        user_track += clean_assistant * _db_to_amplitude(
            request.augmentation.assistant_to_user_crosstalk_db
        )
    if request.augmentation.user_to_assistant_crosstalk_db is not None:
        assistant_track += clean_user * _db_to_amplitude(
            request.augmentation.user_to_assistant_crosstalk_db
        )
    user_track, user_normalization_gain = _normalize(user_track)
    assistant_track, assistant_normalization_gain = _normalize(assistant_track)
    user_audio_path = output_directory / "speaker1-user.wav"
    assistant_audio_path = output_directory / "speaker2-assistant.wav"
    _write_pcm16_wave(user_audio_path, user_track, request.sample_rate_hz)
    _write_pcm16_wave(assistant_audio_path, assistant_track, request.sample_rate_hz)
    manifest = SyntheticRenderManifest(
        plan=request.plan,
        sample_rate_hz=request.sample_rate_hz,
        sources=request.sources,
        augmentation=request.augmentation,
        user_audio_path=user_audio_path,
        assistant_audio_path=assistant_audio_path,
        user_audio_sha256=_file_sha256(user_audio_path),
        assistant_audio_sha256=_file_sha256(assistant_audio_path),
        user_normalization_gain=user_normalization_gain,
        assistant_normalization_gain=assistant_normalization_gain,
    )
    (output_directory / "render.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def _augment_track(
    samples: np.ndarray,
    sample_rate_hz: int,
    augmentation: AcousticAugmentation,
    gain_db: float,
    generator: np.random.Generator,
) -> np.ndarray:
    augmented = samples * _db_to_amplitude(gain_db)
    if augmentation.reverb_decay_seconds is not None:
        impulse_size = max(1, round(augmentation.reverb_decay_seconds * sample_rate_hz))
        impulse_times = np.arange(impulse_size, dtype=np.float32) / sample_rate_hz
        impulse = np.exp(-6.0 * impulse_times / augmentation.reverb_decay_seconds)
        impulse[0] = 1.0
        impulse[1:] *= 0.15
        augmented = np.convolve(augmented, impulse, mode="full")[: samples.size]
    if augmentation.signal_to_noise_db is not None:
        signal_power = max(float(np.mean(np.square(augmented))), 1e-8)
        noise_power = signal_power / (10.0 ** (augmentation.signal_to_noise_db / 10.0))
        augmented += generator.normal(0.0, np.sqrt(noise_power), augmented.size)
    if augmentation.microphone_low_hz is not None or augmentation.microphone_high_hz is not None:
        augmented = _limit_bandwidth(
            augmented,
            sample_rate_hz,
            augmentation.microphone_low_hz,
            augmentation.microphone_high_hz,
        )
    if augmentation.codec == "mulaw_8bit":
        augmented = _mulaw_round_trip(augmented)
    return augmented.astype(np.float32)


def _fit_event_duration(
    source_samples: np.ndarray,
    source_rate_hz: int,
    output_rate_hz: int,
    duration_seconds: float,
) -> np.ndarray:
    output_count = round(duration_seconds * output_rate_hz)
    if source_samples.size == 0:
        raise ValueError("Rendered event source contains no samples.")
    source_positions = np.linspace(0.0, 1.0, source_samples.size, endpoint=True)
    output_positions = np.linspace(0.0, 1.0, output_count, endpoint=True)
    fitted = np.interp(output_positions, source_positions, source_samples).astype(np.float32)
    fade_count = min(round(0.01 * output_rate_hz), output_count // 2)
    if fade_count > 0:
        fade = np.linspace(0.0, 1.0, fade_count, dtype=np.float32)
        fitted[:fade_count] *= fade
        fitted[-fade_count:] *= fade[::-1]
    return fitted


def _limit_bandwidth(
    samples: np.ndarray,
    sample_rate_hz: int,
    low_hz: float | None,
    high_hz: float | None,
) -> np.ndarray:
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate_hz)
    if low_hz is not None:
        spectrum[frequencies < low_hz] = 0.0
    if high_hz is not None:
        spectrum[frequencies > high_hz] = 0.0
    return np.fft.irfft(spectrum, n=samples.size).astype(np.float32)


def _mulaw_round_trip(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    mu = 255.0
    encoded = np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / np.log1p(mu)
    quantized = np.round((encoded + 1.0) * 127.5) / 127.5 - 1.0
    return np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(mu)) / mu


def _read_pcm16_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio_file:
        if audio_file.getsampwidth() != 2:
            raise ValueError(f"Source WAV must use 16-bit PCM: {path}")
        channel_count = audio_file.getnchannels()
        sample_rate_hz = audio_file.getframerate()
        frames = audio_file.readframes(audio_file.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channel_count > 1:
        samples = samples.reshape(-1, channel_count).mean(axis=1)
    return samples, sample_rate_hz


def _write_pcm16_wave(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    encoded = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(encoded.tobytes())


def _normalize(samples: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    gain = min(1.0, 0.98 / peak) if peak > 0.0 else 1.0
    return (samples * gain).astype(np.float32), gain


def _db_to_amplitude(decibels: float) -> float:
    return 10.0 ** (decibels / 20.0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

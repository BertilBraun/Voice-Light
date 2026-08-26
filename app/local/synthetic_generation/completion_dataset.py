from __future__ import annotations

import hashlib
import random
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from app.local.synthetic_generation.models import SyntheticModel


class CompletionBoundaryKind(StrEnum):
    HOLD = "hold"
    END_OF_TURN = "end_of_turn"


class TtsProvider(StrEnum):
    QWEN3_VOICE_DESIGN = "qwen3_voice_design"
    CHATTERBOX_MULTILINGUAL = "chatterbox_multilingual"
    VOXTREAM2 = "voxtream2"


class SyntheticEnglishSpeechPrompt(SyntheticModel):
    prompt_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    text: str = Field(min_length=1)
    voice_instruction: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    seed: int = Field(ge=0)

    @field_validator("voice_instruction")
    @classmethod
    def validate_clean_voice_instruction(cls, value: str) -> str:
        prohibited_phrases = (
            "ambient sound",
            "background noise",
            "breathy",
            "hushed",
            "room tone",
            "whisper",
        )
        normalized_value = value.casefold()
        matched_phrases = tuple(
            phrase for phrase in prohibited_phrases if phrase in normalized_value
        )
        if matched_phrases:
            raise ValueError(
                "English voice instructions require clean voiced speech; prohibited phrases: "
                f"{', '.join(matched_phrases)}."
            )
        return value

    @model_validator(mode="after")
    def validate_long_form_text(self) -> SyntheticEnglishSpeechPrompt:
        if len(self.text.split()) < 45:
            raise ValueError("Synthetic speech prompts require at least 45 words.")
        return self


class TtsGenerationProvenanceBase(SyntheticModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    model_license: str = Field(min_length=1)
    generation_seconds: float = Field(gt=0.0)
    real_time_factor: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)


class QwenVoiceDesignProvenance(TtsGenerationProvenanceBase):
    provider: Literal[TtsProvider.QWEN3_VOICE_DESIGN] = TtsProvider.QWEN3_VOICE_DESIGN


class ChatterboxMultilingualProvenance(TtsGenerationProvenanceBase):
    provider: Literal[TtsProvider.CHATTERBOX_MULTILINGUAL] = TtsProvider.CHATTERBOX_MULTILINGUAL
    runtime_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    reference_audio_path: Path
    reference_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exaggeration: float = Field(ge=0.0, le=1.0)
    classifier_free_guidance_weight: float = Field(ge=0.0)
    temperature: float = Field(gt=0.0)


class Voxtream2Provenance(TtsGenerationProvenanceBase):
    provider: Literal[TtsProvider.VOXTREAM2] = TtsProvider.VOXTREAM2
    runtime_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_audio_path: Path
    reference_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaking_rate_syllables_per_second: float = Field(gt=0.0)


TtsGenerationProvenance = Annotated[
    QwenVoiceDesignProvenance | ChatterboxMultilingualProvenance | Voxtream2Provenance,
    Field(discriminator="provider"),
]


class SilenceDetectionConfiguration(SyntheticModel):
    frame_milliseconds: int = Field(default=20, gt=0)
    minimum_silence_milliseconds: int = Field(default=500, ge=500)
    absolute_rms_threshold: float = Field(default=0.001, gt=0.0)
    peak_rms_ratio: float = Field(default=0.01, gt=0.0, lt=1.0)


DEFAULT_SILENCE_DETECTION = SilenceDetectionConfiguration()
FINAL_FADE_MILLISECONDS = 10


class CompletionBoundaryAnnotation(SyntheticModel):
    kind: CompletionBoundaryKind
    time_seconds: float = Field(ge=0.0)
    silence_duration_seconds: float = Field(ge=0.0)


class GeneratedUtteranceAnnotation(SyntheticModel):
    schema_version: str = "voice-light-synthetic-completion-v1"
    utterance_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    prompt: SyntheticEnglishSpeechPrompt
    audio_path: Path
    sample_rate_hz: int = Field(gt=0)
    original_duration_seconds: float = Field(gt=0.0)
    trimmed_duration_seconds: float = Field(gt=0.0)
    trimmed_trailing_seconds: float = Field(ge=0.0)
    detection: SilenceDetectionConfiguration
    boundaries: tuple[CompletionBoundaryAnnotation, ...] = Field(min_length=1)
    provenance: TtsGenerationProvenance

    @model_validator(mode="after")
    def validate_final_boundary(self) -> GeneratedUtteranceAnnotation:
        end_boundaries = tuple(
            boundary
            for boundary in self.boundaries
            if boundary.kind is CompletionBoundaryKind.END_OF_TURN
        )
        if len(end_boundaries) != 1:
            raise ValueError("An utterance requires exactly one end-of-turn boundary.")
        if end_boundaries[0].time_seconds != self.trimmed_duration_seconds:
            raise ValueError("The end-of-turn boundary must equal the trimmed duration.")
        return self


class WindowBoundaryAnnotation(SyntheticModel):
    kind: CompletionBoundaryKind
    time_seconds: float = Field(ge=0.0, lt=20.0)
    silence_duration_seconds: float = Field(ge=0.0)


class CompletionWindowPlan(SyntheticModel):
    schema_version: str = "voice-light-synthetic-completion-window-v1"
    window_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    utterance_id: str
    prompt_id: str
    anchor_kind: CompletionBoundaryKind
    source_audio_path: Path
    source_start_seconds: float = Field(ge=0.0)
    source_end_seconds: float = Field(gt=0.0)
    left_padding_seconds: float = Field(ge=0.0)
    right_padding_seconds: float = Field(ge=0.0)
    duration_seconds: float = Field(default=20.0)
    boundaries: tuple[WindowBoundaryAnnotation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_duration(self) -> CompletionWindowPlan:
        represented = (
            self.left_padding_seconds
            + self.source_end_seconds
            - self.source_start_seconds
            + self.right_padding_seconds
        )
        if not np.isclose(represented, self.duration_seconds, atol=1e-6):
            raise ValueError("Window source and padding must represent exactly 20 seconds.")
        return self


def analyze_generated_samples(
    samples: np.ndarray,
    sample_rate_hz: int,
    utterance_id: str,
    prompt: SyntheticEnglishSpeechPrompt,
    audio_path: Path,
    provenance: TtsGenerationProvenance,
    configuration: SilenceDetectionConfiguration = DEFAULT_SILENCE_DETECTION,
) -> tuple[np.ndarray, GeneratedUtteranceAnnotation]:
    if samples.ndim != 1:
        raise ValueError("Generated speech analysis requires mono audio.")
    if samples.size == 0 or sample_rate_hz <= 0:
        raise ValueError("Generated speech analysis requires non-empty audio and a sample rate.")
    frame_size = round(sample_rate_hz * configuration.frame_milliseconds / 1000.0)
    frame_count = (samples.size + frame_size - 1) // frame_size
    rms_values = np.asarray(
        [
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            samples[
                                frame_index * frame_size : min(
                                    samples.size, (frame_index + 1) * frame_size
                                )
                            ]
                        )
                    )
                )
            )
            for frame_index in range(frame_count)
        ],
        dtype=np.float32,
    )
    peak_rms = float(np.max(rms_values))
    threshold = max(
        configuration.absolute_rms_threshold,
        peak_rms * configuration.peak_rms_ratio,
    )
    active_indices = np.flatnonzero(rms_values >= threshold)
    if active_indices.size == 0:
        raise ValueError("Generated audio contains no speech-like energy.")
    first_active_frame = int(active_indices[0])
    last_active_frame = int(active_indices[-1])
    frame_seconds = configuration.frame_milliseconds / 1000.0
    end_of_turn_seconds = min(
        samples.size / sample_rate_hz,
        (last_active_frame + 1) * frame_seconds,
    )
    trimmed_sample_count = min(samples.size, round(end_of_turn_seconds * sample_rate_hz))
    boundaries = list(
        _internal_silence_boundaries(
            active=rms_values >= threshold,
            first_active_frame=first_active_frame,
            last_active_frame=last_active_frame,
            frame_seconds=frame_seconds,
            minimum_silence_seconds=configuration.minimum_silence_milliseconds / 1000.0,
        )
    )
    boundaries.append(
        CompletionBoundaryAnnotation(
            kind=CompletionBoundaryKind.END_OF_TURN,
            time_seconds=end_of_turn_seconds,
            silence_duration_seconds=(samples.size / sample_rate_hz) - end_of_turn_seconds,
        )
    )
    annotation = GeneratedUtteranceAnnotation(
        utterance_id=utterance_id,
        prompt=prompt,
        audio_path=audio_path,
        sample_rate_hz=sample_rate_hz,
        original_duration_seconds=samples.size / sample_rate_hz,
        trimmed_duration_seconds=end_of_turn_seconds,
        trimmed_trailing_seconds=(samples.size - trimmed_sample_count) / sample_rate_hz,
        detection=configuration,
        boundaries=tuple(boundaries),
        provenance=provenance,
    )
    trimmed_samples = samples[:trimmed_sample_count].copy()
    _apply_final_fade(trimmed_samples, sample_rate_hz)
    return trimmed_samples, annotation


def completion_window_plans(
    annotation: GeneratedUtteranceAnnotation,
    seed: int,
    minimum_boundary_position_seconds: float = 4.0,
    maximum_boundary_position_seconds: float = 19.2,
) -> tuple[CompletionWindowPlan, ...]:
    if not 0.0 <= minimum_boundary_position_seconds < maximum_boundary_position_seconds < 20.0:
        raise ValueError("Boundary positions must define an increasing range inside 20 seconds.")
    plans = []
    for boundary_index, anchor in enumerate(annotation.boundaries):
        generator = random.Random(_window_seed(seed, annotation.utterance_id, boundary_index))
        requested_position = generator.uniform(
            minimum_boundary_position_seconds,
            maximum_boundary_position_seconds,
        )
        source_start = max(0.0, anchor.time_seconds - requested_position)
        left_padding = max(0.0, requested_position - anchor.time_seconds)
        available_source_seconds = 20.0 - left_padding
        source_end = min(
            annotation.trimmed_duration_seconds,
            source_start + available_source_seconds,
        )
        right_padding = 20.0 - left_padding - (source_end - source_start)
        window_boundaries = tuple(
            WindowBoundaryAnnotation(
                kind=boundary.kind,
                time_seconds=left_padding + boundary.time_seconds - source_start,
                silence_duration_seconds=boundary.silence_duration_seconds,
            )
            for boundary in annotation.boundaries
            if source_start <= boundary.time_seconds < source_end + right_padding - 1e-9
            and 0.0 <= left_padding + boundary.time_seconds - source_start < 20.0
        )
        digest = hashlib.sha256(
            f"{annotation.utterance_id}:{boundary_index}:{source_start:.6f}".encode()
        ).hexdigest()
        plans.append(
            CompletionWindowPlan(
                window_id=digest,
                utterance_id=annotation.utterance_id,
                prompt_id=annotation.prompt.prompt_id,
                anchor_kind=anchor.kind,
                source_audio_path=annotation.audio_path,
                source_start_seconds=source_start,
                source_end_seconds=source_end,
                left_padding_seconds=left_padding,
                right_padding_seconds=right_padding,
                boundaries=window_boundaries,
            )
        )
    return tuple(plans)


def _internal_silence_boundaries(
    active: np.ndarray,
    first_active_frame: int,
    last_active_frame: int,
    frame_seconds: float,
    minimum_silence_seconds: float,
) -> tuple[CompletionBoundaryAnnotation, ...]:
    boundaries = []
    silence_start: int | None = None
    for frame_index in range(first_active_frame, last_active_frame + 1):
        if not bool(active[frame_index]) and silence_start is None:
            silence_start = frame_index
        if bool(active[frame_index]) and silence_start is not None:
            silence_seconds = (frame_index - silence_start) * frame_seconds
            if silence_seconds >= minimum_silence_seconds:
                boundaries.append(
                    CompletionBoundaryAnnotation(
                        kind=CompletionBoundaryKind.HOLD,
                        time_seconds=silence_start * frame_seconds,
                        silence_duration_seconds=silence_seconds,
                    )
                )
            silence_start = None
    return tuple(boundaries)


def _window_seed(seed: int, prompt_id: str, boundary_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{prompt_id}:{boundary_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _apply_final_fade(samples: np.ndarray, sample_rate_hz: int) -> None:
    fade_sample_count = min(
        samples.size,
        round(sample_rate_hz * FINAL_FADE_MILLISECONDS / 1000.0),
    )
    if fade_sample_count == 0:
        return
    samples[-fade_sample_count:] *= np.linspace(
        1.0,
        0.0,
        fade_sample_count,
        dtype=np.float32,
    )

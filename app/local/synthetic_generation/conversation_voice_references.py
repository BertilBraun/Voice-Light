from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import Field

from app.local.synthetic_generation.audio_files import write_mono_pcm16_audio
from app.local.synthetic_generation.completion_dataset import (
    DEFAULT_SILENCE_DETECTION,
    SilenceDetectionConfiguration,
)
from app.local.synthetic_generation.conversation_prompts import (
    Affect,
    ConversationPromptSetId,
    Energy,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    SegmentDelivery,
    SpeakingPace,
    qwen_voice_instruction,
)
from app.local.synthetic_generation.models import SyntheticModel


class TtsBackendIdentity(SyntheticModel):
    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    model_license: str = Field(min_length=1)


@dataclass(frozen=True)
class ReferenceSynthesisRequest:
    plan_id: str
    text: str
    voice_instruction: str
    seed: int


@dataclass(frozen=True)
class ReferenceSynthesisResult:
    plan_id: str
    samples: np.ndarray
    sample_rate_hz: int
    generation_seconds: float
    batch_seed: int


class ReferenceSpeechSynthesizer(Protocol):
    @property
    def identity(self) -> TtsBackendIdentity: ...

    def generate_batch(
        self,
        requests: tuple[ReferenceSynthesisRequest, ...],
    ) -> tuple[ReferenceSynthesisResult, ...]: ...


class ConversationVoiceReference(SyntheticModel):
    schema_version: Literal["voice-light-conversation-voice-reference-v1"] = (
        "voice-light-conversation-voice-reference-v1"
    )
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    reference_text: str = Field(min_length=1)
    reference_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    voice_instruction: str = Field(min_length=1)
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0, le=12.0)
    trimmed_leading_seconds: float = Field(ge=0.0)
    trimmed_trailing_seconds: float = Field(ge=0.0)
    request_seed: int = Field(ge=0)
    batch_seed: int = Field(ge=0)
    generation_seconds: float = Field(ge=0.0)
    real_time_factor: float = Field(ge=0.0)
    backend: TtsBackendIdentity


class ConversationVoiceReferenceManifest(SyntheticModel):
    schema_version: Literal["voice-light-conversation-voice-references-v1"] = (
        "voice-light-conversation-voice-references-v1"
    )
    prompt_set_id: ConversationPromptSetId
    prompt_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: TtsBackendIdentity
    detection: SilenceDetectionConfiguration
    references: tuple[ConversationVoiceReference, ...]


@dataclass(frozen=True)
class TrimmedSpeech:
    samples: np.ndarray
    sample_rate_hz: int
    original_duration_seconds: float
    leading_seconds: float
    trailing_seconds: float
    active_frames: np.ndarray
    frame_seconds: float


def generate_conversation_voice_references(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    output_directory: Path,
    synthesizer: ReferenceSpeechSynthesizer,
    batch_size: int,
    detection: SilenceDetectionConfiguration = DEFAULT_SILENCE_DETECTION,
) -> ConversationVoiceReferenceManifest:
    if batch_size <= 0:
        raise ValueError("Voice-reference batch size must be positive.")
    output_directory.mkdir(parents=True, exist_ok=True)
    records_path = output_directory / "voice-references.jsonl"
    manifest_path = output_directory / "voice-references.json"
    _validate_existing_manifest(
        manifest_path,
        prompt_set,
        prompt_set_path,
        synthesizer.identity,
    )
    requests = tuple(_reference_request(plan) for plan in prompt_set.plans)
    existing = _read_references(records_path)
    _validate_checkpoint(existing, requests, output_directory, synthesizer.identity)
    references_by_plan = {reference.plan_id: reference for reference in existing}
    pending = tuple(request for request in requests if request.plan_id not in references_by_plan)
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        results = synthesizer.generate_batch(batch)
        results_by_plan = _validated_results(batch, results)
        for request in batch:
            reference = _materialize_reference(
                request,
                results_by_plan[request.plan_id],
                output_directory,
                synthesizer.identity,
                detection,
            )
            _append_reference(records_path, reference)
            references_by_plan[reference.plan_id] = reference
        _write_manifest_atomically(
            manifest_path,
            _manifest(
                prompt_set,
                prompt_set_path,
                synthesizer.identity,
                detection,
                references_by_plan,
            ),
        )
        print(
            f"references={len(references_by_plan)}/{len(requests)}",
            flush=True,
        )
    manifest = _manifest(
        prompt_set,
        prompt_set_path,
        synthesizer.identity,
        detection,
        references_by_plan,
    )
    _write_manifest_atomically(manifest_path, manifest)
    return manifest


def load_voice_reference_manifest(path: Path) -> ConversationVoiceReferenceManifest:
    manifest = ConversationVoiceReferenceManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return manifest.model_copy(
        update={
            "references": tuple(
                reference.model_copy(update={"audio_path": path.parent / reference.audio_path})
                for reference in manifest.references
            )
        }
    )


def trim_generated_speech(
    samples: np.ndarray,
    sample_rate_hz: int,
    item_id: str,
    detection: SilenceDetectionConfiguration = DEFAULT_SILENCE_DETECTION,
) -> TrimmedSpeech:
    mono_samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono_samples.size == 0 or sample_rate_hz <= 0:
        raise ValueError(f"TTS returned empty audio for {item_id}.")
    frame_size = round(sample_rate_hz * detection.frame_milliseconds / 1000.0)
    root_mean_squares = np.asarray(
        [
            float(
                np.sqrt(
                    np.mean(
                        np.square(mono_samples[start : min(start + frame_size, mono_samples.size)])
                    )
                )
            )
            for start in range(0, mono_samples.size, frame_size)
        ],
        dtype=np.float32,
    )
    threshold = max(
        detection.absolute_rms_threshold,
        float(np.max(root_mean_squares)) * detection.peak_rms_ratio,
    )
    active_indices = np.flatnonzero(root_mean_squares >= threshold)
    if active_indices.size == 0:
        raise ValueError(f"TTS returned no speech-like energy for {item_id}.")
    first_frame = int(active_indices[0])
    last_frame = int(active_indices[-1])
    start_sample = first_frame * frame_size
    end_sample = min(mono_samples.size, (last_frame + 1) * frame_size)
    trimmed = mono_samples[start_sample:end_sample].copy()
    _fade_edges(trimmed, sample_rate_hz)
    return TrimmedSpeech(
        samples=trimmed,
        sample_rate_hz=sample_rate_hz,
        original_duration_seconds=mono_samples.size / sample_rate_hz,
        leading_seconds=start_sample / sample_rate_hz,
        trailing_seconds=(mono_samples.size - end_sample) / sample_rate_hz,
        active_frames=root_mean_squares[first_frame : last_frame + 1] >= threshold,
        frame_seconds=detection.frame_milliseconds / 1000.0,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_pcm16_wave(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Wave output path must end in .wav: {path}")
    write_mono_pcm16_audio(path, samples, sample_rate_hz)


def _reference_request(plan: EnglishConversationPromptPlan) -> ReferenceSynthesisRequest:
    delivery = SegmentDelivery(
        pace=SpeakingPace.MODERATE,
        energy=Energy.CONVERSATIONAL,
        affect=Affect.NEUTRAL,
    )
    return ReferenceSynthesisRequest(
        plan_id=plan.plan_id,
        text=plan.voice_reference_text,
        voice_instruction=qwen_voice_instruction(plan.base_user_voice, delivery),
        seed=_reference_seed(plan.seed, plan.plan_id),
    )


def _materialize_reference(
    request: ReferenceSynthesisRequest,
    result: ReferenceSynthesisResult,
    output_directory: Path,
    backend: TtsBackendIdentity,
    detection: SilenceDetectionConfiguration,
) -> ConversationVoiceReference:
    trimmed = trim_generated_speech(
        result.samples,
        result.sample_rate_hz,
        request.plan_id,
        detection,
    )
    relative_audio_path = Path("audio") / f"{request.plan_id}.flac"
    audio_path = output_directory / relative_audio_path
    write_mono_pcm16_audio(audio_path, trimmed.samples, trimmed.sample_rate_hz)
    duration_seconds = trimmed.samples.size / trimmed.sample_rate_hz
    return ConversationVoiceReference(
        plan_id=request.plan_id,
        reference_text=request.text,
        reference_text_sha256=hashlib.sha256(request.text.encode()).hexdigest(),
        voice_instruction=request.voice_instruction,
        audio_path=relative_audio_path,
        audio_sha256=file_sha256(audio_path),
        sample_rate_hz=trimmed.sample_rate_hz,
        duration_seconds=duration_seconds,
        trimmed_leading_seconds=trimmed.leading_seconds,
        trimmed_trailing_seconds=trimmed.trailing_seconds,
        request_seed=request.seed,
        batch_seed=result.batch_seed,
        generation_seconds=result.generation_seconds,
        real_time_factor=result.generation_seconds / duration_seconds,
        backend=backend,
    )


def _manifest(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    backend: TtsBackendIdentity,
    detection: SilenceDetectionConfiguration,
    references_by_plan: dict[str, ConversationVoiceReference],
) -> ConversationVoiceReferenceManifest:
    references = tuple(
        references_by_plan[plan.plan_id]
        for plan in prompt_set.plans
        if plan.plan_id in references_by_plan
    )
    return ConversationVoiceReferenceManifest(
        prompt_set_id=prompt_set.set_id,
        prompt_set_sha256=file_sha256(prompt_set_path),
        backend=backend,
        detection=detection,
        references=references,
    )


def _validate_checkpoint(
    existing: tuple[ConversationVoiceReference, ...],
    requests: tuple[ReferenceSynthesisRequest, ...],
    output_directory: Path,
    backend: TtsBackendIdentity,
) -> None:
    requests_by_plan = {request.plan_id: request for request in requests}
    if len(requests_by_plan) != len(requests):
        raise ValueError("Conversation plan IDs must be unique.")
    seen: set[str] = set()
    for reference in existing:
        if reference.plan_id in seen:
            raise ValueError(f"Checkpoint repeats reference {reference.plan_id}.")
        seen.add(reference.plan_id)
        request = requests_by_plan.get(reference.plan_id)
        if request is None:
            raise ValueError(f"Checkpoint contains unknown reference {reference.plan_id}.")
        if reference.reference_text != request.text:
            raise ValueError(f"Checkpoint reference text changed for {reference.plan_id}.")
        if reference.voice_instruction != request.voice_instruction:
            raise ValueError(f"Checkpoint voice instruction changed for {reference.plan_id}.")
        if reference.backend != backend:
            raise ValueError(f"Checkpoint backend changed for {reference.plan_id}.")
        audio_path = output_directory / reference.audio_path
        if not audio_path.exists() or file_sha256(audio_path) != reference.audio_sha256:
            raise ValueError(f"Checkpoint audio is missing or changed for {reference.plan_id}.")


def _validate_existing_manifest(
    manifest_path: Path,
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    backend: TtsBackendIdentity,
) -> None:
    if not manifest_path.exists():
        return
    manifest = ConversationVoiceReferenceManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.prompt_set_id != prompt_set.set_id:
        raise ValueError("Existing references use a different prompt set ID.")
    if manifest.prompt_set_sha256 != file_sha256(prompt_set_path):
        raise ValueError("Existing references use different prompt-set content.")
    if manifest.backend != backend:
        raise ValueError("Existing references use a different VoiceDesign backend.")


def _validated_results(
    requests: tuple[ReferenceSynthesisRequest, ...],
    results: tuple[ReferenceSynthesisResult, ...],
) -> dict[str, ReferenceSynthesisResult]:
    requested_ids = tuple(request.plan_id for request in requests)
    result_ids = tuple(result.plan_id for result in results)
    if result_ids != requested_ids:
        raise ValueError("VoiceDesign results must match request order and plan IDs.")
    return {result.plan_id: result for result in results}


def _reference_seed(plan_seed: int, plan_id: str) -> int:
    digest = hashlib.sha256(f"{plan_seed}:{plan_id}:reference".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _append_reference(path: Path, reference: ConversationVoiceReference) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(f"{reference.model_dump_json()}\n")


def _read_references(path: Path) -> tuple[ConversationVoiceReference, ...]:
    if not path.exists():
        return ()
    return tuple(
        ConversationVoiceReference.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_manifest_atomically(
    path: Path,
    manifest: ConversationVoiceReferenceManifest,
) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.partial")
    temporary_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _fade_edges(samples: np.ndarray, sample_rate_hz: int) -> None:
    fade_count = min(samples.size // 2, round(0.005 * sample_rate_hz))
    if fade_count == 0:
        return
    fade = np.linspace(0.0, 1.0, fade_count, dtype=np.float32)
    samples[:fade_count] *= fade
    samples[-fade_count:] *= fade[::-1]

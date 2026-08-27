from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import Field, model_validator

from app.local.synthetic_generation.completion_dataset import (
    DEFAULT_SILENCE_DETECTION,
    SilenceDetectionConfiguration,
)
from app.local.synthetic_generation.conversation_compiler import (
    MeasuredSilence,
    RenderedUserClip,
)
from app.local.synthetic_generation.conversation_prompts import (
    CompletionUserPrompt,
    ConversationPromptSetId,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    HoldUserPrompt,
    InterruptionFloorClaimUserPrompt,
    NonFloorFeedbackUserPrompt,
    ResponseFloorClaimUserPrompt,
    SpeakingPace,
    UserPrompt,
)
from app.local.synthetic_generation.conversation_voice_references import (
    ConversationVoiceReference,
    ConversationVoiceReferenceManifest,
    TtsBackendIdentity,
    file_sha256,
    trim_generated_speech,
    write_pcm16_wave,
)
from app.local.synthetic_generation.models import SyntheticModel


@dataclass(frozen=True)
class SpeechSynthesisRequest:
    clause_id: str
    text: str
    delivery_instruction: str
    speed: float
    seed: int


@dataclass(frozen=True)
class SpeechSynthesisResult:
    clause_id: str
    samples: np.ndarray
    sample_rate_hz: int
    generation_seconds: float
    batch_seed: int
    clone_prompt: VoiceClonePromptProvenance


def cosyvoice3_reference_prompt(reference_text: str) -> str:
    if not reference_text.strip():
        raise ValueError("CosyVoice reference transcript must not be empty.")
    return f"You are a helpful assistant.<|endofprompt|>{reference_text}"


class VoiceClonePromptProvenance(SyntheticModel):
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    x_vector_only_mode: Literal[False] = False
    backend: TtsBackendIdentity


class BatchSpeechSynthesizer(Protocol):
    @property
    def identity(self) -> TtsBackendIdentity: ...

    def generate_batch(
        self,
        reference: ConversationVoiceReference,
        requests: tuple[SpeechSynthesisRequest, ...],
    ) -> tuple[SpeechSynthesisResult, ...]: ...


class RenderedClauseProvenance(SyntheticModel):
    clause_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_seed: int = Field(ge=0)
    batch_seed: int = Field(ge=0)
    sample_rate_hz: int = Field(gt=0)
    original_duration_seconds: float = Field(gt=0.0)
    trimmed_duration_seconds: float = Field(gt=0.0)
    trimmed_leading_seconds: float = Field(ge=0.0)
    trimmed_trailing_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    real_time_factor: float = Field(ge=0.0)


class RenderedConversationUserUnit(SyntheticModel):
    schema_version: Literal["voice-light-rendered-conversation-user-unit-v1"] = (
        "voice-light-rendered-conversation-user-unit-v1"
    )
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    prompt: UserPrompt
    backend: TtsBackendIdentity
    reference: ConversationVoiceReference
    clone_prompt: VoiceClonePromptProvenance
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip: RenderedUserClip
    clauses: tuple[RenderedClauseProvenance, ...] = Field(min_length=1, max_length=2)


class ConversationTtsManifest(SyntheticModel):
    schema_version: Literal["voice-light-conversation-user-tts-v1"] = (
        "voice-light-conversation-user-tts-v1"
    )
    prompt_set_id: ConversationPromptSetId
    prompt_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: TtsBackendIdentity
    detection: SilenceDetectionConfiguration
    clone_prompts: tuple[VoiceClonePromptProvenance, ...]
    rendered_units: tuple[RenderedConversationUserUnit, ...]

    @model_validator(mode="after")
    def validate_clone_bindings(self) -> ConversationTtsManifest:
        clone_prompts_by_plan = {prompt.plan_id: prompt for prompt in self.clone_prompts}
        if len(clone_prompts_by_plan) != len(self.clone_prompts):
            raise ValueError("Clone-prompt plan IDs must be unique.")
        for unit in self.rendered_units:
            if clone_prompts_by_plan.get(unit.plan_id) != unit.clone_prompt:
                raise ValueError(
                    f"Rendered unit {unit.prompt.unit_id} has no matching clone prompt."
                )
        return self


@dataclass(frozen=True)
class _SpeechActivity:
    samples: np.ndarray
    original_duration_seconds: float
    leading_seconds: float
    trailing_seconds: float
    internal_silences: tuple[MeasuredSilence, ...]


@dataclass(frozen=True)
class _PreparedUnit:
    plan: EnglishConversationPromptPlan
    prompt: UserPrompt
    requests: tuple[SpeechSynthesisRequest, ...]


def render_conversation_user_audio(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    reference_manifest: ConversationVoiceReferenceManifest,
    reference_manifest_path: Path,
    output_directory: Path,
    synthesizer: BatchSpeechSynthesizer,
    batch_size: int,
    detection: SilenceDetectionConfiguration = DEFAULT_SILENCE_DETECTION,
) -> ConversationTtsManifest:
    if batch_size <= 0:
        raise ValueError("Conversation TTS batch size must be positive.")
    output_directory.mkdir(parents=True, exist_ok=True)
    records_path = output_directory / "rendered-units.jsonl"
    manifest_path = output_directory / "render.json"
    references_by_plan = _validated_references(
        prompt_set,
        prompt_set_path,
        reference_manifest,
        reference_manifest_path,
    )
    _validate_existing_manifest(
        manifest_path,
        prompt_set,
        prompt_set_path,
        reference_manifest_path,
        synthesizer.identity,
    )
    existing = _read_records(records_path)
    expected = _prepared_units(prompt_set)
    _validate_checkpoint(
        existing,
        expected,
        references_by_plan,
        output_directory,
        synthesizer.identity,
    )
    rendered_by_unit_id = {
        _record_key(record.plan_id, record.prompt.unit_id): record for record in existing
    }
    for plan in prompt_set.plans:
        pending = tuple(
            prepared
            for prepared in expected
            if prepared.plan.plan_id == plan.plan_id
            and _record_key(prepared.plan.plan_id, prepared.prompt.unit_id)
            not in rendered_by_unit_id
        )
        reference = references_by_plan[plan.plan_id]
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            requests = tuple(request for prepared in batch for request in prepared.requests)
            results = synthesizer.generate_batch(reference, requests)
            results_by_clause = _validated_results(requests, results)
            for prepared in batch:
                record = _materialize_unit(
                    prepared,
                    reference,
                    results_by_clause,
                    output_directory,
                    detection,
                    synthesizer.identity,
                )
                _append_record(records_path, record)
                rendered_by_unit_id[_record_key(record.plan_id, record.prompt.unit_id)] = record
            manifest = _manifest(
                prompt_set,
                prompt_set_path,
                reference_manifest_path,
                synthesizer.identity,
                detection,
                expected,
                rendered_by_unit_id,
            )
            _write_manifest_atomically(manifest_path, manifest)
            print(
                f"rendered={len(rendered_by_unit_id)}/{len(expected)} user units",
                flush=True,
            )
    manifest = _manifest(
        prompt_set,
        prompt_set_path,
        reference_manifest_path,
        synthesizer.identity,
        detection,
        expected,
        rendered_by_unit_id,
    )
    _write_manifest_atomically(manifest_path, manifest)
    return manifest


def load_rendered_user_clips(manifest_path: Path) -> tuple[RenderedUserClip, ...]:
    manifest = ConversationTtsManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    return tuple(
        unit.clip.model_copy(update={"audio_path": manifest_path.parent / unit.clip.audio_path})
        for unit in manifest.rendered_units
    )


def _prepared_units(prompt_set: EnglishConversationPromptSet) -> tuple[_PreparedUnit, ...]:
    prepared = []
    for plan in prompt_set.plans:
        for prompt in sorted(plan.user_prompts, key=lambda item: item.sequence_index):
            texts = _prompt_clauses(prompt)
            requests = tuple(
                SpeechSynthesisRequest(
                    clause_id=_clause_id(plan.plan_id, prompt.unit_id, clause_index),
                    text=text,
                    delivery_instruction=_delivery_instruction(prompt),
                    speed=_speech_speed(prompt.delivery.pace),
                    seed=_clause_seed(plan.seed, plan.plan_id, prompt.unit_id, clause_index),
                )
                for clause_index, text in enumerate(texts)
            )
            prepared.append(
                _PreparedUnit(
                    plan=plan,
                    prompt=prompt,
                    requests=requests,
                )
            )
    return tuple(prepared)


def _delivery_instruction(prompt: UserPrompt) -> str:
    delivery = prompt.delivery
    return (
        f"Speak in English at a {delivery.pace.value} pace with {delivery.energy.value} energy "
        f"and a {delivery.affect.value} conversational affect."
    )


def _speech_speed(pace: SpeakingPace) -> float:
    match pace:
        case SpeakingPace.SLOW:
            return 0.96
        case SpeakingPace.MODERATE:
            return 1.06
        case SpeakingPace.FAST:
            return 1.16


def _materialize_unit(
    prepared: _PreparedUnit,
    reference: ConversationVoiceReference,
    results_by_clause: dict[str, SpeechSynthesisResult],
    output_directory: Path,
    detection: SilenceDetectionConfiguration,
    backend: TtsBackendIdentity,
) -> RenderedConversationUserUnit:
    activities = tuple(
        _trim_to_speech(results_by_clause[request.clause_id], detection)
        for request in prepared.requests
    )
    sample_rate_hz = activities[0][1]
    if any(activity_sample_rate != sample_rate_hz for _, activity_sample_rate in activities):
        raise ValueError(f"TTS clauses for {prepared.prompt.unit_id} use different sample rates.")
    activity = activities[0][0]
    samples = activity.samples
    continuation_silences = activity.internal_silences
    relative_audio_path = Path("audio") / f"{prepared.plan.plan_id}_{prepared.prompt.unit_id}.wav"
    audio_path = output_directory / relative_audio_path
    write_pcm16_wave(audio_path, samples, sample_rate_hz)
    duration_seconds = samples.size / sample_rate_hz
    clip = RenderedUserClip(
        clip_id=f"{prepared.plan.plan_id}_{prepared.prompt.unit_id}",
        audio_path=relative_audio_path,
        audio_sha256=file_sha256(audio_path),
        duration_seconds=duration_seconds,
        active_start_seconds=0.0,
        active_end_seconds=duration_seconds,
        continuation_silences=continuation_silences,
    )
    clause_provenance = tuple(
        _clause_provenance(request, results_by_clause[request.clause_id], activity)
        for request, (activity, _) in zip(prepared.requests, activities, strict=True)
    )
    clone_prompts = {
        results_by_clause[request.clause_id].clone_prompt for request in prepared.requests
    }
    if len(clone_prompts) != 1:
        raise ValueError(f"TTS changed clone prompt within user unit {prepared.prompt.unit_id}.")
    clone_prompt = next(iter(clone_prompts))
    if clone_prompt.plan_id != prepared.plan.plan_id:
        raise ValueError(f"TTS returned the wrong clone prompt for {prepared.prompt.unit_id}.")
    if clone_prompt.reference_audio_sha256 != reference.audio_sha256:
        raise ValueError(f"TTS used the wrong reference audio for {prepared.prompt.unit_id}.")
    if clone_prompt.reference_text_sha256 != reference.reference_text_sha256:
        raise ValueError(f"TTS used the wrong reference text for {prepared.prompt.unit_id}.")
    return RenderedConversationUserUnit(
        plan_id=prepared.plan.plan_id,
        prompt=prepared.prompt,
        backend=backend,
        reference=reference,
        clone_prompt=clone_prompt,
        prompt_sha256=_prompt_sha256(prepared),
        clip=clip,
        clauses=clause_provenance,
    )


def _trim_to_speech(
    result: SpeechSynthesisResult,
    detection: SilenceDetectionConfiguration,
) -> tuple[_SpeechActivity, int]:
    trimmed = trim_generated_speech(
        result.samples,
        result.sample_rate_hz,
        result.clause_id,
        detection,
    )
    internal_silences = _internal_silences(
        trimmed.active_frames,
        trimmed.frame_seconds,
        detection.minimum_silence_milliseconds / 1000.0,
        trimmed.samples.size / result.sample_rate_hz,
    )
    return (
        _SpeechActivity(
            samples=trimmed.samples,
            original_duration_seconds=trimmed.original_duration_seconds,
            leading_seconds=trimmed.leading_seconds,
            trailing_seconds=trimmed.trailing_seconds,
            internal_silences=internal_silences,
        ),
        result.sample_rate_hz,
    )


def _internal_silences(
    active: np.ndarray,
    frame_seconds: float,
    minimum_silence_seconds: float,
    duration_seconds: float,
) -> tuple[MeasuredSilence, ...]:
    silences = []
    silence_start: int | None = None
    for frame_index, is_active in enumerate(active):
        if not bool(is_active) and silence_start is None:
            silence_start = frame_index
        if bool(is_active) and silence_start is not None:
            silence_duration = (frame_index - silence_start) * frame_seconds
            if silence_duration >= minimum_silence_seconds:
                silences.append(
                    MeasuredSilence(
                        start_seconds=silence_start * frame_seconds,
                        end_seconds=min(frame_index * frame_seconds, duration_seconds),
                    )
                )
            silence_start = None
    return tuple(silences)


def _prompt_clauses(prompt: UserPrompt) -> tuple[str, ...]:
    match prompt:
        case NonFloorFeedbackUserPrompt(text=text):
            return (text.value,)
        case (
            CompletionUserPrompt(text=text)
            | HoldUserPrompt(text=text)
            | ResponseFloorClaimUserPrompt(text=text)
            | InterruptionFloorClaimUserPrompt(text=text)
        ):
            return (text,)


def _manifest(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    reference_manifest_path: Path,
    backend: TtsBackendIdentity,
    detection: SilenceDetectionConfiguration,
    expected: tuple[_PreparedUnit, ...],
    rendered_by_unit_id: dict[str, RenderedConversationUserUnit],
) -> ConversationTtsManifest:
    rendered = tuple(
        rendered_by_unit_id[_record_key(prepared.plan.plan_id, prepared.prompt.unit_id)]
        for prepared in expected
        if _record_key(prepared.plan.plan_id, prepared.prompt.unit_id) in rendered_by_unit_id
    )
    clone_prompts_by_plan = {unit.clone_prompt.plan_id: unit.clone_prompt for unit in rendered}
    return ConversationTtsManifest(
        prompt_set_id=prompt_set.set_id,
        prompt_set_sha256=file_sha256(prompt_set_path),
        reference_manifest_sha256=file_sha256(reference_manifest_path),
        backend=backend,
        detection=detection,
        clone_prompts=tuple(
            clone_prompts_by_plan[plan.plan_id]
            for plan in prompt_set.plans
            if plan.plan_id in clone_prompts_by_plan
        ),
        rendered_units=rendered,
    )


def _validate_checkpoint(
    existing: tuple[RenderedConversationUserUnit, ...],
    expected: tuple[_PreparedUnit, ...],
    references_by_plan: dict[str, ConversationVoiceReference],
    output_directory: Path,
    backend: TtsBackendIdentity,
) -> None:
    expected_by_unit_id = {
        _record_key(prepared.plan.plan_id, prepared.prompt.unit_id): prepared
        for prepared in expected
    }
    if len(expected_by_unit_id) != len(expected):
        raise ValueError("User unit IDs must be unique across the prompt set.")
    seen: set[str] = set()
    for record in existing:
        record_key = _record_key(record.plan_id, record.prompt.unit_id)
        if record_key in seen:
            raise ValueError(f"Checkpoint repeats user unit {record_key}.")
        seen.add(record_key)
        if record_key not in expected_by_unit_id:
            raise ValueError(f"Checkpoint contains unknown user unit {record_key}.")
        prepared = expected_by_unit_id[record_key]
        if record.prompt_sha256 != _prompt_sha256(prepared):
            raise ValueError(f"Checkpoint prompt changed for user unit {record_key}.")
        if record.backend != backend:
            raise ValueError(f"Checkpoint backend changed for user unit {record_key}.")
        if record.reference != references_by_plan[record.plan_id]:
            raise ValueError(f"Checkpoint reference changed for user unit {record_key}.")
        if record.clone_prompt.reference_audio_sha256 != record.reference.audio_sha256:
            raise ValueError(f"Checkpoint clone prompt changed for user unit {record_key}.")
        if record.clone_prompt.reference_text_sha256 != record.reference.reference_text_sha256:
            raise ValueError(f"Checkpoint clone prompt text changed for user unit {record_key}.")
        audio_path = output_directory / record.clip.audio_path
        if not audio_path.exists() or file_sha256(audio_path) != record.clip.audio_sha256:
            raise ValueError(f"Checkpoint audio is missing or changed for user unit {record_key}.")


def _validate_existing_manifest(
    manifest_path: Path,
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    reference_manifest_path: Path,
    backend: TtsBackendIdentity,
) -> None:
    if not manifest_path.exists():
        return
    manifest = ConversationTtsManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.prompt_set_id != prompt_set.set_id:
        raise ValueError("Existing conversation render uses a different prompt set ID.")
    if manifest.prompt_set_sha256 != file_sha256(prompt_set_path):
        raise ValueError("Existing conversation render uses different prompt-set content.")
    if manifest.reference_manifest_sha256 != file_sha256(reference_manifest_path):
        raise ValueError("Existing conversation render uses different voice references.")
    if manifest.backend != backend:
        raise ValueError("Existing conversation render uses a different TTS backend.")


def _validated_references(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    manifest: ConversationVoiceReferenceManifest,
    manifest_path: Path,
) -> dict[str, ConversationVoiceReference]:
    if not manifest_path.exists():
        raise ValueError("Voice-reference manifest file does not exist.")
    if manifest.prompt_set_id != prompt_set.set_id:
        raise ValueError("Voice references do not match the conversation prompt set ID.")
    if manifest.prompt_set_sha256 != file_sha256(prompt_set_path):
        raise ValueError("Voice references do not match the conversation prompt-set content.")
    references_by_plan = {reference.plan_id: reference for reference in manifest.references}
    if len(references_by_plan) != len(manifest.references):
        raise ValueError("Voice-reference plan IDs must be unique.")
    expected_plan_ids = {plan.plan_id for plan in prompt_set.plans}
    if set(references_by_plan) != expected_plan_ids:
        missing = sorted(expected_plan_ids - set(references_by_plan))
        unexpected = sorted(set(references_by_plan) - expected_plan_ids)
        raise ValueError(
            "Voice references must exactly match plans; "
            f"missing={missing}, unexpected={unexpected}."
        )
    for plan in prompt_set.plans:
        reference = references_by_plan[plan.plan_id]
        if reference.reference_text != plan.voice_reference_text:
            raise ValueError(f"Voice reference text changed for plan {plan.plan_id}.")
        if (
            reference.reference_text_sha256
            != hashlib.sha256(plan.voice_reference_text.encode()).hexdigest()
        ):
            raise ValueError(f"Voice reference text hash changed for plan {plan.plan_id}.")
        if not reference.audio_path.exists():
            raise ValueError(f"Voice reference audio is missing for plan {plan.plan_id}.")
        if file_sha256(reference.audio_path) != reference.audio_sha256:
            raise ValueError(f"Voice reference audio changed for plan {plan.plan_id}.")
    return references_by_plan


def _validated_results(
    requests: tuple[SpeechSynthesisRequest, ...],
    results: tuple[SpeechSynthesisResult, ...],
) -> dict[str, SpeechSynthesisResult]:
    requested_ids = tuple(request.clause_id for request in requests)
    result_ids = tuple(result.clause_id for result in results)
    if result_ids != requested_ids:
        raise ValueError("TTS results must exactly match request order and clause IDs.")
    return {result.clause_id: result for result in results}


def _clause_provenance(
    request: SpeechSynthesisRequest,
    result: SpeechSynthesisResult,
    activity: _SpeechActivity,
) -> RenderedClauseProvenance:
    return RenderedClauseProvenance(
        clause_id=request.clause_id,
        text_sha256=hashlib.sha256(request.text.encode()).hexdigest(),
        request_seed=request.seed,
        batch_seed=result.batch_seed,
        sample_rate_hz=result.sample_rate_hz,
        original_duration_seconds=activity.original_duration_seconds,
        trimmed_duration_seconds=activity.samples.size / result.sample_rate_hz,
        trimmed_leading_seconds=activity.leading_seconds,
        trimmed_trailing_seconds=activity.trailing_seconds,
        generation_seconds=result.generation_seconds,
        real_time_factor=result.generation_seconds
        / (activity.samples.size / result.sample_rate_hz),
    )


def _prompt_sha256(prepared: _PreparedUnit) -> str:
    content = f"{prepared.plan.plan_id}\n{prepared.plan.seed}\n{prepared.prompt.model_dump_json()}"
    return hashlib.sha256(content.encode()).hexdigest()


def _clause_id(plan_id: str, unit_id: str, clause_index: int) -> str:
    suffix = "before" if clause_index == 0 else "after"
    return f"{plan_id}_{unit_id}_{suffix}"


def _clause_seed(plan_seed: int, plan_id: str, unit_id: str, clause_index: int) -> int:
    digest = hashlib.sha256(f"{plan_seed}:{plan_id}:{unit_id}:{clause_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _record_key(plan_id: str, unit_id: str) -> str:
    return f"{plan_id}:{unit_id}"


def _append_record(path: Path, record: RenderedConversationUserUnit) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(f"{record.model_dump_json()}\n")


def _read_records(path: Path) -> tuple[RenderedConversationUserUnit, ...]:
    if not path.exists():
        return ()
    return tuple(
        RenderedConversationUserUnit.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_manifest_atomically(path: Path, manifest: ConversationTtsManifest) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.partial")
    temporary_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)

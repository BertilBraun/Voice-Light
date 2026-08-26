from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import Field

from app.local.synthetic_generation.completion_dataset import (
    DEFAULT_SILENCE_DETECTION,
    SilenceDetectionConfiguration,
)
from app.local.synthetic_generation.conversation_compiler import (
    MeasuredSilence,
    RenderedUserClip,
)
from app.local.synthetic_generation.conversation_prompts import (
    BaseUserVoice,
    CompletionUserPrompt,
    ConversationPromptSetId,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    HoldUserPrompt,
    InterruptionFloorClaimUserPrompt,
    NonFloorFeedbackUserPrompt,
    ResponseFloorClaimUserPrompt,
    UserPrompt,
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
class SpeechSynthesisRequest:
    clause_id: str
    text: str
    voice_instruction: str
    seed: int


@dataclass(frozen=True)
class SpeechSynthesisResult:
    clause_id: str
    samples: np.ndarray
    sample_rate_hz: int
    generation_seconds: float
    batch_seed: int


class BatchSpeechSynthesizer(Protocol):
    @property
    def identity(self) -> TtsBackendIdentity: ...

    def generate_batch(
        self,
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
    base_user_voice: BaseUserVoice
    backend: TtsBackendIdentity
    voice_instruction: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip: RenderedUserClip
    clauses: tuple[RenderedClauseProvenance, ...] = Field(min_length=1, max_length=2)


class ConversationTtsManifest(SyntheticModel):
    schema_version: Literal["voice-light-conversation-user-tts-v1"] = (
        "voice-light-conversation-user-tts-v1"
    )
    prompt_set_id: ConversationPromptSetId
    prompt_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: TtsBackendIdentity
    detection: SilenceDetectionConfiguration
    rendered_units: tuple[RenderedConversationUserUnit, ...]


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
    voice_instruction: str
    requests: tuple[SpeechSynthesisRequest, ...]


def render_conversation_user_audio(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
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
    _validate_existing_manifest(
        manifest_path,
        prompt_set,
        prompt_set_path,
        synthesizer.identity,
    )
    existing = _read_records(records_path)
    expected = _prepared_units(prompt_set)
    _validate_checkpoint(existing, expected, output_directory, synthesizer.identity)
    rendered_by_unit_id = {
        _record_key(record.plan_id, record.prompt.unit_id): record for record in existing
    }
    pending = tuple(
        prepared
        for prepared in expected
        if _record_key(prepared.plan.plan_id, prepared.prompt.unit_id) not in rendered_by_unit_id
    )
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        requests = tuple(request for prepared in batch for request in prepared.requests)
        results = synthesizer.generate_batch(requests)
        results_by_clause = _validated_results(requests, results)
        for prepared in batch:
            record = _materialize_unit(
                prepared,
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
            instruction = qwen_voice_instruction(plan.base_user_voice, prompt.delivery)
            texts = _prompt_clauses(prompt)
            requests = tuple(
                SpeechSynthesisRequest(
                    clause_id=_clause_id(plan.plan_id, prompt.unit_id, clause_index),
                    text=text,
                    voice_instruction=instruction,
                    seed=_clause_seed(plan.seed, plan.plan_id, prompt.unit_id, clause_index),
                )
                for clause_index, text in enumerate(texts)
            )
            prepared.append(
                _PreparedUnit(
                    plan=plan,
                    prompt=prompt,
                    voice_instruction=instruction,
                    requests=requests,
                )
            )
    return tuple(prepared)


def _materialize_unit(
    prepared: _PreparedUnit,
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
        raise ValueError(f"HOLD clauses for {prepared.prompt.unit_id} use different sample rates.")
    if isinstance(prepared.prompt, HoldUserPrompt):
        samples, continuation_silences = _compose_hold(
            tuple(activity for activity, _ in activities),
            sample_rate_hz,
            prepared.prompt.pause_duration_seconds,
        )
    else:
        activity = activities[0][0]
        samples = activity.samples
        continuation_silences = activity.internal_silences
    relative_audio_path = Path("audio") / f"{prepared.plan.plan_id}_{prepared.prompt.unit_id}.wav"
    audio_path = output_directory / relative_audio_path
    _write_pcm16_wave(audio_path, samples, sample_rate_hz)
    duration_seconds = samples.size / sample_rate_hz
    clip = RenderedUserClip(
        clip_id=f"{prepared.plan.plan_id}_{prepared.prompt.unit_id}",
        audio_path=relative_audio_path,
        audio_sha256=_file_sha256(audio_path),
        duration_seconds=duration_seconds,
        active_start_seconds=0.0,
        active_end_seconds=duration_seconds,
        continuation_silences=continuation_silences,
    )
    clause_provenance = tuple(
        _clause_provenance(request, results_by_clause[request.clause_id], activity)
        for request, (activity, _) in zip(prepared.requests, activities, strict=True)
    )
    return RenderedConversationUserUnit(
        plan_id=prepared.plan.plan_id,
        prompt=prepared.prompt,
        base_user_voice=prepared.plan.base_user_voice,
        backend=backend,
        voice_instruction=prepared.voice_instruction,
        prompt_sha256=_prompt_sha256(prepared),
        clip=clip,
        clauses=clause_provenance,
    )


def _trim_to_speech(
    result: SpeechSynthesisResult,
    detection: SilenceDetectionConfiguration,
) -> tuple[_SpeechActivity, int]:
    samples = np.asarray(result.samples, dtype=np.float32).reshape(-1)
    if samples.size == 0 or result.sample_rate_hz <= 0:
        raise ValueError(f"TTS returned empty audio for {result.clause_id}.")
    frame_size = round(result.sample_rate_hz * detection.frame_milliseconds / 1000.0)
    root_mean_squares = np.asarray(
        [
            float(
                np.sqrt(np.mean(np.square(samples[start : min(start + frame_size, samples.size)])))
            )
            for start in range(0, samples.size, frame_size)
        ],
        dtype=np.float32,
    )
    threshold = max(
        detection.absolute_rms_threshold,
        float(np.max(root_mean_squares)) * detection.peak_rms_ratio,
    )
    active_indices = np.flatnonzero(root_mean_squares >= threshold)
    if active_indices.size == 0:
        raise ValueError(f"TTS returned no speech-like energy for {result.clause_id}.")
    first_frame = int(active_indices[0])
    last_frame = int(active_indices[-1])
    start_sample = first_frame * frame_size
    end_sample = min(samples.size, (last_frame + 1) * frame_size)
    trimmed = samples[start_sample:end_sample].copy()
    active = root_mean_squares[first_frame : last_frame + 1] >= threshold
    frame_seconds = detection.frame_milliseconds / 1000.0
    internal_silences = _internal_silences(
        active,
        frame_seconds,
        detection.minimum_silence_milliseconds / 1000.0,
        trimmed.size / result.sample_rate_hz,
    )
    _fade_edges(trimmed, result.sample_rate_hz)
    original_duration = samples.size / result.sample_rate_hz
    return (
        _SpeechActivity(
            samples=trimmed,
            original_duration_seconds=original_duration,
            leading_seconds=start_sample / result.sample_rate_hz,
            trailing_seconds=(samples.size - end_sample) / result.sample_rate_hz,
            internal_silences=internal_silences,
        ),
        result.sample_rate_hz,
    )


def _compose_hold(
    activities: tuple[_SpeechActivity, ...],
    sample_rate_hz: int,
    pause_duration_seconds: float,
) -> tuple[np.ndarray, tuple[MeasuredSilence, ...]]:
    if len(activities) != 2:
        raise ValueError("A HOLD prompt requires exactly two rendered clauses.")
    first, second = activities
    pause_sample_count = round(pause_duration_seconds * sample_rate_hz)
    pause_start_seconds = first.samples.size / sample_rate_hz
    exact_pause_seconds = pause_sample_count / sample_rate_hz
    samples = np.concatenate(
        (first.samples, np.zeros(pause_sample_count, dtype=np.float32), second.samples)
    )
    second_offset = pause_start_seconds + exact_pause_seconds
    silences = (
        *first.internal_silences,
        MeasuredSilence(
            start_seconds=pause_start_seconds,
            end_seconds=second_offset,
        ),
        *tuple(
            MeasuredSilence(
                start_seconds=second_offset + silence.start_seconds,
                end_seconds=second_offset + silence.end_seconds,
            )
            for silence in second.internal_silences
        ),
    )
    return samples, silences


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
        case HoldUserPrompt(text_before_pause=before, text_after_pause=after):
            return (before, after)
        case (
            CompletionUserPrompt(text=text)
            | NonFloorFeedbackUserPrompt(text=text)
            | ResponseFloorClaimUserPrompt(text=text)
            | InterruptionFloorClaimUserPrompt(text=text)
        ):
            return (text,)


def _manifest(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
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
    return ConversationTtsManifest(
        prompt_set_id=prompt_set.set_id,
        prompt_set_sha256=_file_sha256(prompt_set_path),
        backend=backend,
        detection=detection,
        rendered_units=rendered,
    )


def _validate_checkpoint(
    existing: tuple[RenderedConversationUserUnit, ...],
    expected: tuple[_PreparedUnit, ...],
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
        audio_path = output_directory / record.clip.audio_path
        if not audio_path.exists() or _file_sha256(audio_path) != record.clip.audio_sha256:
            raise ValueError(f"Checkpoint audio is missing or changed for user unit {record_key}.")


def _validate_existing_manifest(
    manifest_path: Path,
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    backend: TtsBackendIdentity,
) -> None:
    if not manifest_path.exists():
        return
    manifest = ConversationTtsManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.prompt_set_id != prompt_set.set_id:
        raise ValueError("Existing conversation render uses a different prompt set ID.")
    if manifest.prompt_set_sha256 != _file_sha256(prompt_set_path):
        raise ValueError("Existing conversation render uses different prompt-set content.")
    if manifest.backend != backend:
        raise ValueError("Existing conversation render uses a different TTS backend.")


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
    content = (
        f"{prepared.plan.plan_id}\n{prepared.plan.seed}\n"
        f"{prepared.prompt.model_dump_json()}\n{prepared.voice_instruction}"
    )
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


def _write_pcm16_wave(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(encoded.tobytes())


def _fade_edges(samples: np.ndarray, sample_rate_hz: int) -> None:
    fade_count = min(samples.size // 2, round(0.005 * sample_rate_hz))
    if fade_count == 0:
        return
    fade = np.linspace(0.0, 1.0, fade_count, dtype=np.float32)
    samples[:fade_count] *= fade
    samples[-fade_count:] *= fade[::-1]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

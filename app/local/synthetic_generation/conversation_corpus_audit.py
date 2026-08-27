from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.local.synthetic_generation.conversation_pipeline import (
    SyntheticConversationCorpusManifest,
)
from app.local.synthetic_generation.conversation_prompts import (
    EnglishConversationPromptSet,
    floor_user_turn_length,
)
from app.local.synthetic_generation.conversation_tts import ConversationTtsManifest
from app.local.synthetic_generation.models import SyntheticModel


class CorpusDimensionCount(SyntheticModel):
    value: str
    count: int = Field(ge=0)


class CorpusQualityFlag(SyntheticModel):
    plan_id: str
    code: Literal[
        "duplicate_topic",
        "similar_topic",
        "missing_audio",
        "audio_hash_mismatch",
        "extended_duration_outlier",
    ]
    detail: str


class ExtendedTurnDurationSummary(SyntheticModel):
    rendered_count: int = Field(ge=0)
    within_twenty_to_thirty_seconds_count: int = Field(ge=0)
    minimum_seconds: float | None = Field(default=None, ge=0.0)
    maximum_seconds: float | None = Field(default=None, ge=0.0)


class ConversationCorpusAudit(SyntheticModel):
    schema_version: Literal["voice-light-conversation-corpus-audit-v1"] = (
        "voice-light-conversation-corpus-audit-v1"
    )
    prompt_set_id: str
    plan_count: int = Field(ge=1)
    planned_conversation_hours: float = Field(gt=0.0)
    measured_conversation_hours: float | None = Field(default=None, gt=0.0)
    rendered_user_audio_hours: float | None = Field(default=None, gt=0.0)
    generation_hours: float | None = Field(default=None, ge=0.0)
    real_time_factor: float | None = Field(default=None, ge=0.0)
    domains: tuple[CorpusDimensionCount, ...]
    paces: tuple[CorpusDimensionCount, ...]
    affects: tuple[CorpusDimensionCount, ...]
    accents: tuple[CorpusDimensionCount, ...]
    conditions: tuple[CorpusDimensionCount, ...]
    turn_lengths: tuple[CorpusDimensionCount, ...]
    extended_turn_durations: ExtendedTurnDurationSummary
    quality_flags: tuple[CorpusQualityFlag, ...]


def audit_conversation_corpus(
    prompt_set_path: Path,
    tts_manifest_path: Path | None = None,
    corpus_manifest_path: Path | None = None,
) -> ConversationCorpusAudit:
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        prompt_set_path.read_text(encoding="utf-8")
    )
    tts_manifest = _load_tts_manifest(prompt_set, prompt_set_path, tts_manifest_path)
    corpus_manifest = _load_corpus_manifest(prompt_set, corpus_manifest_path)
    flags = list(_topic_flags(prompt_set))
    extended_durations: list[float] = []
    rendered_user_seconds: float | None = None
    generation_seconds: float | None = None
    if tts_manifest is not None:
        rendered_user_seconds = 0.0
        generation_seconds = 0.0
        for unit in tts_manifest.rendered_units:
            audio_path = tts_manifest_path.parent / unit.clip.audio_path
            if not audio_path.is_file():
                flags.append(
                    CorpusQualityFlag(
                        plan_id=unit.plan_id,
                        code="missing_audio",
                        detail=f"Missing rendered unit {unit.prompt.unit_id}: {audio_path}",
                    )
                )
                continue
            if _file_sha256(audio_path) != unit.clip.audio_sha256:
                flags.append(
                    CorpusQualityFlag(
                        plan_id=unit.plan_id,
                        code="audio_hash_mismatch",
                        detail=f"Hash mismatch for rendered unit {unit.prompt.unit_id}.",
                    )
                )
            rendered_user_seconds += unit.clip.duration_seconds
            generation_seconds += sum(clause.generation_seconds for clause in unit.clauses)
            length = floor_user_turn_length(unit.prompt)
            if length is not None and length.value == "extended":
                extended_durations.append(unit.clip.duration_seconds)
                if not 15.0 <= unit.clip.duration_seconds <= 35.0:
                    flags.append(
                        CorpusQualityFlag(
                            plan_id=unit.plan_id,
                            code="extended_duration_outlier",
                            detail=(
                                f"Extended unit {unit.prompt.unit_id} rendered as "
                                f"{unit.clip.duration_seconds:.3f} seconds."
                            ),
                        )
                    )
    real_time_factor = (
        generation_seconds / rendered_user_seconds
        if generation_seconds is not None and rendered_user_seconds
        else None
    )
    return ConversationCorpusAudit(
        prompt_set_id=prompt_set.set_id,
        plan_count=len(prompt_set.plans),
        planned_conversation_hours=(
            sum(plan.target_duration_seconds for plan in prompt_set.plans) / 3600.0
        ),
        measured_conversation_hours=(
            sum(item.source_duration_seconds for item in corpus_manifest.conversations) / 3600.0
            if corpus_manifest is not None
            else None
        ),
        rendered_user_audio_hours=(
            rendered_user_seconds / 3600.0 if rendered_user_seconds is not None else None
        ),
        generation_hours=(generation_seconds / 3600.0 if generation_seconds is not None else None),
        real_time_factor=real_time_factor,
        domains=_counts(plan.domain.value for plan in prompt_set.plans),
        paces=_counts(
            prompt.delivery.pace.value for plan in prompt_set.plans for prompt in plan.user_prompts
        ),
        affects=_counts(
            prompt.delivery.affect.value
            for plan in prompt_set.plans
            for prompt in plan.user_prompts
        ),
        accents=_counts(plan.base_user_voice.accent.value for plan in prompt_set.plans),
        conditions=_counts(
            prompt.condition for plan in prompt_set.plans for prompt in plan.user_prompts
        ),
        turn_lengths=_counts(
            length.value
            for plan in prompt_set.plans
            for prompt in plan.user_prompts
            if (length := floor_user_turn_length(prompt)) is not None
        ),
        extended_turn_durations=ExtendedTurnDurationSummary(
            rendered_count=len(extended_durations),
            within_twenty_to_thirty_seconds_count=sum(
                20.0 <= duration <= 30.0 for duration in extended_durations
            ),
            minimum_seconds=min(extended_durations, default=None),
            maximum_seconds=max(extended_durations, default=None),
        ),
        quality_flags=tuple(flags),
    )


def _load_tts_manifest(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    path: Path | None,
) -> ConversationTtsManifest | None:
    if path is None:
        return None
    manifest = ConversationTtsManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.prompt_set_id != prompt_set.set_id:
        raise ValueError("TTS manifest belongs to a different prompt set.")
    if manifest.prompt_set_sha256 != _file_sha256(prompt_set_path):
        raise ValueError("TTS manifest prompt-set hash does not match the audited prompt set.")
    return manifest


def _load_corpus_manifest(
    prompt_set: EnglishConversationPromptSet,
    path: Path | None,
) -> SyntheticConversationCorpusManifest | None:
    if path is None:
        return None
    manifest = SyntheticConversationCorpusManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if manifest.prompt_set_id != prompt_set.set_id:
        raise ValueError("Compiled corpus belongs to a different prompt set.")
    return manifest


def _topic_flags(prompt_set: EnglishConversationPromptSet) -> tuple[CorpusQualityFlag, ...]:
    flags: list[CorpusQualityFlag] = []
    normalized_topics: list[tuple[str, str, frozenset[str]]] = []
    for plan in prompt_set.plans:
        normalized = " ".join(plan.topic.casefold().split())
        tokens = frozenset(token.strip(".,!?;:'\"") for token in normalized.split())
        for previous_plan_id, previous_topic, previous_tokens in normalized_topics:
            if normalized == previous_topic:
                flags.append(
                    CorpusQualityFlag(
                        plan_id=plan.plan_id,
                        code="duplicate_topic",
                        detail=f"Topic duplicates {previous_plan_id}: {plan.topic}",
                    )
                )
                continue
            union = tokens | previous_tokens
            similarity = len(tokens & previous_tokens) / len(union) if union else 0.0
            if len(tokens) >= 4 and len(previous_tokens) >= 4 and similarity >= 0.75:
                flags.append(
                    CorpusQualityFlag(
                        plan_id=plan.plan_id,
                        code="similar_topic",
                        detail=(
                            f"Topic has Jaccard similarity {similarity:.3f} to "
                            f"{previous_plan_id}: {plan.topic}"
                        ),
                    )
                )
        normalized_topics.append((plan.plan_id, normalized, tokens))
    return tuple(flags)


def _counts(values: Iterable[str]) -> tuple[CorpusDimensionCount, ...]:
    return tuple(
        CorpusDimensionCount(value=value, count=count)
        for value, count in sorted(Counter(values).items())
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit a synthetic conversation corpus.")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--tts-manifest", type=Path)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parsed = parser.parse_args(arguments)
    report = audit_conversation_corpus(
        parsed.prompts,
        parsed.tts_manifest,
        parsed.corpus_manifest,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = parsed.output.with_suffix(f"{parsed.output.suffix}.partial")
    temporary_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(parsed.output)
    print(report.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()

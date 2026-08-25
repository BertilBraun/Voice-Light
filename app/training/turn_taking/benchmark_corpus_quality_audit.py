from __future__ import annotations

import hashlib
from collections import defaultdict
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_models import (
    BenchmarkModel,
    TurnCompletionCandidate,
    TurnCompletionInventory,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class CorpusQualityReviewLabel(StrEnum):
    USABLE = "usable"
    UNUSABLE_SOURCE = "unusable_source"
    WRONG_ALIGNMENT_OR_CHANNEL = "wrong_alignment_or_channel"
    UNSURE = "unsure"


class CorpusQualityAuditConfiguration(BenchmarkModel):
    seed: str = "corpus-quality-control-v1"
    items_per_dataset: int = Field(default=5, gt=0)
    context_before_seconds: float = Field(default=4.0, gt=0.0)
    context_after_seconds: float = Field(default=3.0, gt=0.0)


class CorpusQualityAuditItem(BenchmarkModel):
    order: int = Field(gt=0)
    audit_id: str = Field(pattern=SHA256_PATTERN)
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    dataset_name: str
    conversation_id: str
    external_id: str
    user_side: str
    categories: tuple[str, ...] = Field(min_length=1)
    anchor_seconds: float = Field(ge=0.0)
    boundary_kind: str
    clip_path: str
    clip_start_seconds: float = Field(ge=0.0)
    clip_end_seconds: float = Field(gt=0.0)
    boundary_offset_seconds: float = Field(ge=0.0)
    source_window_ids: tuple[str, ...] = Field(min_length=1)


class CorpusQualityAuditManifest(BenchmarkModel):
    schema_version: Literal["voice-light-corpus-quality-audit-v1"] = (
        "voice-light-corpus-quality-audit-v1"
    )
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    split: Literal[TrainingCorpusSplit.VALIDATION] = TrainingCorpusSplit.VALIDATION
    corpus_repository: str
    corpus_revision: str
    configuration: CorpusQualityAuditConfiguration
    dataset_names: tuple[str, ...] = Field(min_length=1)
    item_count: int = Field(gt=0)
    items_sha256: str = Field(pattern=SHA256_PATTERN)
    items: tuple[CorpusQualityAuditItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> CorpusQualityAuditManifest:
        if len(self.items) != self.item_count:
            raise ValueError("Corpus quality audit item count does not match its manifest.")
        if corpus_quality_audit_rows_sha256(self.items) != self.items_sha256:
            raise ValueError("Corpus quality audit items do not match their manifest hash.")
        if tuple(item.order for item in self.items) != tuple(range(1, len(self.items) + 1)):
            raise ValueError("Corpus quality audit item order must be contiguous and one-based.")
        if tuple(sorted({item.dataset_name for item in self.items})) != self.dataset_names:
            raise ValueError("Corpus quality audit datasets do not match its manifest.")
        return self


class CorpusQualityAuditReview(BenchmarkModel):
    audit_id: str = Field(pattern=SHA256_PATTERN)
    review_label: CorpusQualityReviewLabel
    notes: str = ""


class CorpusQualityAuditReviewArtifact(BenchmarkModel):
    schema_version: Literal["voice-light-corpus-quality-reviews-v1"] = (
        "voice-light-corpus-quality-reviews-v1"
    )
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    reviewer: str = Field(min_length=1)
    reviews: tuple[CorpusQualityAuditReview, ...]

    @model_validator(mode="after")
    def validate_reviews(self) -> CorpusQualityAuditReviewArtifact:
        audit_ids = tuple(review.audit_id for review in self.reviews)
        if len(set(audit_ids)) != len(audit_ids):
            raise ValueError("Corpus quality review artifact contains duplicate audit IDs.")
        return self


def build_corpus_quality_audit_manifest(
    inventory: TurnCompletionInventory,
    configuration: CorpusQualityAuditConfiguration,
) -> CorpusQualityAuditManifest:
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Corpus quality audits are restricted to validation.")
    candidates = _select_candidates(inventory.candidates, configuration)
    ordered_items = tuple(
        _audit_item(candidate, order, configuration)
        for order, candidate in enumerate(
            sorted(
                candidates,
                key=lambda candidate: _stable_digest(
                    configuration.seed,
                    "order",
                    candidate.candidate_id,
                ),
            ),
            start=1,
        )
    )
    return CorpusQualityAuditManifest(
        inventory_sha256=inventory.manifest.candidate_sha256,
        corpus_repository=inventory.manifest.corpus_repository,
        corpus_revision=inventory.manifest.corpus_revision,
        configuration=configuration,
        dataset_names=tuple(sorted({item.dataset_name for item in ordered_items})),
        item_count=len(ordered_items),
        items_sha256=corpus_quality_audit_rows_sha256(ordered_items),
        items=ordered_items,
    )


def corpus_quality_audit_rows_sha256(
    items: tuple[CorpusQualityAuditItem, ...],
) -> str:
    payload = "\n".join(item.model_dump_json() for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_candidates(
    candidates: tuple[TurnCompletionCandidate, ...],
    configuration: CorpusQualityAuditConfiguration,
) -> tuple[TurnCompletionCandidate, ...]:
    by_dataset: defaultdict[str, list[TurnCompletionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_dataset[candidate.dataset_name].append(candidate)
    selected: list[TurnCompletionCandidate] = []
    for dataset_name in sorted(by_dataset):
        dataset_candidates = by_dataset[dataset_name]
        if len(dataset_candidates) < configuration.items_per_dataset:
            raise ValueError(
                f"Corpus quality audit requested {configuration.items_per_dataset} cases "
                f"from {dataset_name}, but only {len(dataset_candidates)} are available."
            )
        selected.extend(
            sorted(
                dataset_candidates,
                key=lambda candidate: _stable_digest(
                    configuration.seed,
                    dataset_name,
                    candidate.candidate_id,
                ),
            )[: configuration.items_per_dataset]
        )
    return tuple(selected)


def _audit_item(
    candidate: TurnCompletionCandidate,
    order: int,
    configuration: CorpusQualityAuditConfiguration,
) -> CorpusQualityAuditItem:
    clip_start = max(0.0, candidate.anchor_seconds - configuration.context_before_seconds)
    clip_end = candidate.anchor_seconds + configuration.context_after_seconds
    audit_id = _stable_digest(configuration.seed, "audit-item", candidate.candidate_id)
    return CorpusQualityAuditItem(
        order=order,
        audit_id=audit_id,
        candidate_id=candidate.candidate_id,
        dataset_name=candidate.dataset_name,
        conversation_id=candidate.conversation_id,
        external_id=candidate.external_id,
        user_side=candidate.user_side,
        categories=candidate.categories,
        anchor_seconds=candidate.anchor_seconds,
        boundary_kind=candidate.boundary_kind.value,
        clip_path=f"clips/{order:04d}-{audit_id[:12]}.wav",
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        boundary_offset_seconds=candidate.anchor_seconds - clip_start,
        source_window_ids=candidate.source_window_ids,
    )


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

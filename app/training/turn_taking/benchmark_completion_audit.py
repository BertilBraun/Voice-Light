from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_completion_metrics import (
    validate_completion_predictions,
)
from app.training.turn_taking.benchmark_models import (
    BenchmarkModel,
    CompletionDetectorKind,
    CompletionDetectorProvenance,
    CompletionPredictionArtifact,
    TurnCompletionCandidate,
    TurnCompletionInventory,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class CompletionAuditGroup(StrEnum):
    AMBIGUOUS = "ambiguous"
    CONFIDENT_HOLD = "confident_hold"
    CONFIDENT_EOT = "confident_eot"


class CompletionAuditSelectionReason(StrEnum):
    LABEL_MIDPOINT = "label_midpoint"
    COMPLETION_CONTINUATION_TENSION = "completion_continuation_tension"
    VOICE_LIGHT_DISAGREEMENT = "voice_light_disagreement"
    CROSS_MODEL_DISAGREEMENT = "cross_model_disagreement"
    REPRESENTATIVE_CONTROL = "representative_control"


class CompletionAuditReviewLabel(StrEnum):
    SAFE_TO_TAKE = "safe_to_take"
    HOLD = "hold"
    AMBIGUOUS_UNRATABLE = "ambiguous_unratable"


class CompletionAuditErrorTag(StrEnum):
    CONTINUATION = "continuation"
    BACKCHANNEL = "backchannel"
    OVERLAP = "overlap"
    TRANSCRIPT_ERROR = "transcript_error"
    TIMING_ERROR = "timing_error"
    CENSORED_CONTEXT = "censored_context"
    AUDIO_QUALITY = "audio_quality"


class CompletionAuditConfiguration(BenchmarkModel):
    seed: str = "completion-label-audit-v1"
    ambiguous_count: int = Field(default=80, ge=0)
    confident_hold_count: int = Field(default=120, ge=0)
    confident_eot_count: int = Field(default=120, ge=0)
    confident_challenge_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    double_review_count: int = Field(default=100, ge=0)
    hold_completion_maximum: float = Field(default=0.2, ge=0.0, le=1.0)
    hold_continuation_minimum: float = Field(default=0.8, ge=0.0, le=1.0)
    eot_completion_minimum: float = Field(default=0.8, ge=0.0, le=1.0)
    conflicting_continuation_minimum: float = Field(default=0.8, ge=0.0, le=1.0)
    context_before_seconds: float = Field(default=4.0, gt=0.0)
    context_after_seconds: float = Field(default=3.0, gt=0.0)


class CompletionAuditDetectorScore(BenchmarkModel):
    probability: float = Field(ge=0.0, le=1.0)
    native_gate_seconds: float = Field(gt=0.0)


class CompletionAuditPredictionSource(BenchmarkModel):
    role: Literal["voice_light", "smart_turn", "livekit"]
    predictions_sha256: str = Field(pattern=SHA256_PATTERN)
    detector: CompletionDetectorProvenance


class CompletionAuditItem(BenchmarkModel):
    order: int = Field(gt=0)
    audit_id: str = Field(pattern=SHA256_PATTERN)
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    group: CompletionAuditGroup
    selection_reasons: tuple[CompletionAuditSelectionReason, ...] = Field(min_length=1)
    challenge_score: float = Field(ge=0.0, le=1.0)
    double_review: bool
    dataset_name: str
    conversation_id: str
    external_id: str
    user_side: str
    categories: tuple[str, ...] = Field(min_length=1)
    anchor_seconds: float = Field(ge=0.0)
    boundary_kind: str
    completion_probability: float = Field(ge=0.0, le=1.0)
    continuation_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    voice_light: CompletionAuditDetectorScore
    smart_turn: CompletionAuditDetectorScore
    livekit: CompletionAuditDetectorScore
    clip_path: str
    clip_start_seconds: float = Field(ge=0.0)
    clip_end_seconds: float = Field(gt=0.0)
    boundary_offset_seconds: float = Field(ge=0.0)
    source_window_ids: tuple[str, ...] = Field(min_length=1)


class CompletionAuditManifest(BenchmarkModel):
    schema_version: Literal["voice-light-completion-label-audit-v1"] = (
        "voice-light-completion-label-audit-v1"
    )
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    split: Literal[TrainingCorpusSplit.VALIDATION] = TrainingCorpusSplit.VALIDATION
    corpus_repository: str
    corpus_revision: str
    configuration: CompletionAuditConfiguration
    prediction_sources: tuple[CompletionAuditPredictionSource, ...]
    item_count: int = Field(ge=0)
    items_sha256: str = Field(pattern=SHA256_PATTERN)
    items: tuple[CompletionAuditItem, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> CompletionAuditManifest:
        if len(self.items) != self.item_count:
            raise ValueError("Completion audit item count does not match its manifest.")
        if completion_audit_rows_sha256(self.items) != self.items_sha256:
            raise ValueError("Completion audit items do not match their manifest hash.")
        if tuple(item.order for item in self.items) != tuple(range(1, len(self.items) + 1)):
            raise ValueError("Completion audit item order must be contiguous and one-based.")
        return self


class CompletionAuditReview(BenchmarkModel):
    audit_id: str = Field(pattern=SHA256_PATTERN)
    review_label: CompletionAuditReviewLabel
    error_tags: tuple[CompletionAuditErrorTag, ...] = ()
    notes: str = ""


class CompletionAuditReviewArtifact(BenchmarkModel):
    schema_version: Literal["voice-light-completion-label-reviews-v1"] = (
        "voice-light-completion-label-reviews-v1"
    )
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    reviewer: str = Field(min_length=1)
    reviews: tuple[CompletionAuditReview, ...]

    @model_validator(mode="after")
    def validate_reviews(self) -> CompletionAuditReviewArtifact:
        audit_ids = tuple(review.audit_id for review in self.reviews)
        if len(set(audit_ids)) != len(audit_ids):
            raise ValueError("Completion audit review artifact contains duplicate audit IDs.")
        return self


class CompletionAuditReviewerSummary(BenchmarkModel):
    reviewer: str
    reviewed_support: int = Field(ge=0)


class CompletionAuditGroupReviewSummary(BenchmarkModel):
    group: CompletionAuditGroup
    selected_support: int = Field(ge=0)
    reviewed_support: int = Field(ge=0)
    safe_to_take_count: int = Field(ge=0)
    hold_count: int = Field(ge=0)
    ambiguous_unratable_count: int = Field(ge=0)


class CompletionAuditTagCount(BenchmarkModel):
    tag: CompletionAuditErrorTag
    count: int = Field(ge=0)


class CompletionAuditAgreementSummary(BenchmarkModel):
    confident_consensus_support: int = Field(ge=0)
    automatic_label_agreement_count: int = Field(ge=0)
    automatic_label_agreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    confident_eot_consensus_support: int = Field(ge=0)
    unsafe_eot_error_count: int = Field(ge=0)
    unsafe_eot_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    double_review_support: int = Field(ge=0)
    exact_double_review_agreement_count: int = Field(ge=0)
    exact_double_review_agreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    cohen_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)


class CompletionAuditAnalysisReport(BenchmarkModel):
    schema_version: Literal["voice-light-completion-label-audit-analysis-v1"] = (
        "voice-light-completion-label-audit-analysis-v1"
    )
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    reviewer_summaries: tuple[CompletionAuditReviewerSummary, ...]
    group_summaries: tuple[CompletionAuditGroupReviewSummary, ...]
    agreement: CompletionAuditAgreementSummary
    error_tag_counts: tuple[CompletionAuditTagCount, ...]


def analyze_completion_audit_reviews(
    manifest: CompletionAuditManifest,
    review_artifacts: Iterable[CompletionAuditReviewArtifact],
) -> CompletionAuditAnalysisReport:
    artifacts = tuple(review_artifacts)
    if not artifacts:
        raise ValueError("Completion audit analysis requires at least one review artifact.")
    if len({artifact.reviewer for artifact in artifacts}) != len(artifacts):
        raise ValueError("Completion audit review artifacts must use unique reviewer names.")
    item_by_id = {item.audit_id: item for item in manifest.items}
    reviews_by_id: defaultdict[str, list[tuple[str, CompletionAuditReview]]] = defaultdict(list)
    for artifact in artifacts:
        if artifact.manifest_sha256 != manifest.items_sha256:
            raise ValueError("Completion audit review artifact uses a different manifest hash.")
        for review in artifact.reviews:
            if review.audit_id not in item_by_id:
                raise ValueError(f"Review references unknown audit item {review.audit_id}.")
            reviews_by_id[review.audit_id].append((artifact.reviewer, review))

    consensus = {
        audit_id: label
        for audit_id, reviews in reviews_by_id.items()
        if (label := _consensus_label(tuple(review.review_label for _, review in reviews)))
        is not None
    }
    group_summaries = tuple(
        _group_review_summary(group, manifest, consensus) for group in CompletionAuditGroup
    )
    agreement = _agreement_summary(manifest, reviews_by_id, consensus)
    tag_counts = Counter(
        tag
        for reviews in reviews_by_id.values()
        for _, review in reviews
        for tag in review.error_tags
    )
    return CompletionAuditAnalysisReport(
        manifest_sha256=manifest.items_sha256,
        reviewer_summaries=tuple(
            CompletionAuditReviewerSummary(
                reviewer=artifact.reviewer,
                reviewed_support=len(artifact.reviews),
            )
            for artifact in sorted(artifacts, key=lambda artifact: artifact.reviewer)
        ),
        group_summaries=group_summaries,
        agreement=agreement,
        error_tag_counts=tuple(
            CompletionAuditTagCount(tag=tag, count=tag_counts[tag])
            for tag in CompletionAuditErrorTag
        ),
    )


class _AuditCandidate(BenchmarkModel):
    candidate: TurnCompletionCandidate
    group: CompletionAuditGroup
    challenge_score: float = Field(ge=0.0, le=1.0)
    challenge_reasons: tuple[CompletionAuditSelectionReason, ...]
    voice_light: CompletionAuditDetectorScore
    smart_turn: CompletionAuditDetectorScore
    livekit: CompletionAuditDetectorScore


def build_completion_audit_manifest(
    inventory: TurnCompletionInventory,
    voice_light_predictions: CompletionPredictionArtifact,
    smart_turn_predictions: CompletionPredictionArtifact,
    livekit_predictions: CompletionPredictionArtifact,
    configuration: CompletionAuditConfiguration,
) -> CompletionAuditManifest:
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Completion label audits are restricted to validation.")
    artifacts = (
        ("voice_light", voice_light_predictions),
        ("smart_turn", smart_turn_predictions),
        ("livekit", livekit_predictions),
    )
    for _, artifact in artifacts:
        if artifact.manifest.split is not TrainingCorpusSplit.VALIDATION:
            raise ValueError("Completion audit predictions must use validation.")
        if artifact.manifest.inventory_sha256 != inventory.manifest.candidate_sha256:
            raise ValueError("Completion audit prediction inventory hash does not match.")
        coverage = validate_completion_predictions(inventory.candidates, artifact.predictions)
        if coverage.missing_candidate_support or coverage.prediction_support != len(
            inventory.candidates
        ):
            raise ValueError("Completion audit requires one score for every candidate.")
    for role, artifact in artifacts:
        _validate_source_role(role, artifact)

    score_maps = tuple(_first_score_by_candidate(artifact) for _, artifact in artifacts)
    rows = tuple(
        _audit_candidate(
            candidate=candidate,
            voice_light=score_maps[0][candidate.candidate_id],
            smart_turn=score_maps[1][candidate.candidate_id],
            livekit=score_maps[2][candidate.candidate_id],
            configuration=configuration,
        )
        for candidate in inventory.candidates
    )
    selected = _select_audit_candidates(rows, configuration)
    ordered = tuple(
        sorted(
            selected,
            key=lambda row: _stable_digest(
                configuration.seed,
                "presentation-order",
                row.candidate.candidate_id,
            ),
        )
    )
    double_review_ids = {
        row.candidate.candidate_id
        for row in _balanced_select(
            ordered,
            min(configuration.double_review_count, len(ordered)),
            lambda row: _stable_digest(
                configuration.seed,
                "double-review",
                row.candidate.candidate_id,
            ),
        )
    }
    items = tuple(
        _audit_item(
            row=row,
            order=index,
            double_review=row.candidate.candidate_id in double_review_ids,
            configuration=configuration,
        )
        for index, row in enumerate(ordered, start=1)
    )
    sources = tuple(
        CompletionAuditPredictionSource(
            role=role,
            predictions_sha256=artifact.manifest.predictions_sha256,
            detector=artifact.manifest.detector,
        )
        for role, artifact in artifacts
    )
    return CompletionAuditManifest(
        inventory_sha256=inventory.manifest.candidate_sha256,
        corpus_repository=inventory.manifest.corpus_repository,
        corpus_revision=inventory.manifest.corpus_revision,
        configuration=configuration,
        prediction_sources=sources,
        item_count=len(items),
        items_sha256=completion_audit_rows_sha256(items),
        items=items,
    )


def completion_audit_rows_sha256(items: tuple[CompletionAuditItem, ...]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _consensus_label(
    labels: tuple[CompletionAuditReviewLabel, ...],
) -> CompletionAuditReviewLabel | None:
    counts = Counter(labels)
    if not counts:
        return None
    ordered = counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _group_review_summary(
    group: CompletionAuditGroup,
    manifest: CompletionAuditManifest,
    consensus: dict[str, CompletionAuditReviewLabel],
) -> CompletionAuditGroupReviewSummary:
    items = tuple(item for item in manifest.items if item.group is group)
    labels = tuple(consensus[item.audit_id] for item in items if item.audit_id in consensus)
    return CompletionAuditGroupReviewSummary(
        group=group,
        selected_support=len(items),
        reviewed_support=len(labels),
        safe_to_take_count=labels.count(CompletionAuditReviewLabel.SAFE_TO_TAKE),
        hold_count=labels.count(CompletionAuditReviewLabel.HOLD),
        ambiguous_unratable_count=labels.count(CompletionAuditReviewLabel.AMBIGUOUS_UNRATABLE),
    )


def _agreement_summary(
    manifest: CompletionAuditManifest,
    reviews_by_id: dict[str, list[tuple[str, CompletionAuditReview]]],
    consensus: dict[str, CompletionAuditReviewLabel],
) -> CompletionAuditAgreementSummary:
    confident_items = tuple(
        item
        for item in manifest.items
        if item.group is not CompletionAuditGroup.AMBIGUOUS and item.audit_id in consensus
    )
    expected_labels = {
        CompletionAuditGroup.CONFIDENT_HOLD: CompletionAuditReviewLabel.HOLD,
        CompletionAuditGroup.CONFIDENT_EOT: CompletionAuditReviewLabel.SAFE_TO_TAKE,
    }
    agreement_count = sum(
        consensus[item.audit_id] is expected_labels[item.group] for item in confident_items
    )
    confident_eot_items = tuple(
        item for item in confident_items if item.group is CompletionAuditGroup.CONFIDENT_EOT
    )
    unsafe_eot_count = sum(
        consensus[item.audit_id] is CompletionAuditReviewLabel.HOLD for item in confident_eot_items
    )
    double_pairs = tuple(
        tuple(
            review.review_label
            for _, review in sorted(reviews_by_id[item.audit_id], key=lambda pair: pair[0])[:2]
        )
        for item in manifest.items
        if item.double_review and len(reviews_by_id[item.audit_id]) >= 2
    )
    exact_count = sum(first is second for first, second in double_pairs)
    return CompletionAuditAgreementSummary(
        confident_consensus_support=len(confident_items),
        automatic_label_agreement_count=agreement_count,
        automatic_label_agreement_rate=_optional_rate(agreement_count, len(confident_items)),
        confident_eot_consensus_support=len(confident_eot_items),
        unsafe_eot_error_count=unsafe_eot_count,
        unsafe_eot_error_rate=_optional_rate(unsafe_eot_count, len(confident_eot_items)),
        double_review_support=len(double_pairs),
        exact_double_review_agreement_count=exact_count,
        exact_double_review_agreement_rate=_optional_rate(exact_count, len(double_pairs)),
        cohen_kappa=_cohen_kappa(double_pairs),
    )


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _cohen_kappa(
    pairs: tuple[tuple[CompletionAuditReviewLabel, CompletionAuditReviewLabel], ...],
) -> float | None:
    if not pairs:
        return None
    observed = sum(first is second for first, second in pairs) / len(pairs)
    first_counts = Counter(first for first, _ in pairs)
    second_counts = Counter(second for _, second in pairs)
    expected = sum(
        first_counts[label] / len(pairs) * second_counts[label] / len(pairs)
        for label in CompletionAuditReviewLabel
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _first_score_by_candidate(
    artifact: CompletionPredictionArtifact,
) -> dict[str, CompletionAuditDetectorScore]:
    scores: dict[str, CompletionAuditDetectorScore] = {}
    for prediction in artifact.predictions:
        if prediction.candidate_id in scores:
            continue
        scores[prediction.candidate_id] = CompletionAuditDetectorScore(
            probability=prediction.completion_probability,
            native_gate_seconds=prediction.elapsed_seconds,
        )
    return scores


def _audit_candidate(
    candidate: TurnCompletionCandidate,
    voice_light: CompletionAuditDetectorScore,
    smart_turn: CompletionAuditDetectorScore,
    livekit: CompletionAuditDetectorScore,
    configuration: CompletionAuditConfiguration,
) -> _AuditCandidate:
    completion = candidate.target_points[0].completion_probability
    continuation = candidate.continuation_probability
    conflicting = (
        completion >= configuration.eot_completion_minimum
        and continuation is not None
        and continuation >= configuration.conflicting_continuation_minimum
    )
    if (
        completion <= configuration.hold_completion_maximum
        and continuation is not None
        and continuation >= configuration.hold_continuation_minimum
    ):
        group = CompletionAuditGroup.CONFIDENT_HOLD
        target = 0.0
    elif completion >= configuration.eot_completion_minimum and not conflicting:
        group = CompletionAuditGroup.CONFIDENT_EOT
        target = 1.0
    else:
        group = CompletionAuditGroup.AMBIGUOUS
        target = completion

    midpoint_uncertainty = 1.0 - 2.0 * abs(completion - 0.5)
    continuation_tension = (
        1.0 if continuation is None else min(1.0, abs(completion + continuation - 1.0))
    )
    voice_disagreement = abs(voice_light.probability - target)
    model_probabilities = (
        voice_light.probability,
        smart_turn.probability,
        livekit.probability,
    )
    model_spread = max(model_probabilities) - min(model_probabilities)
    if group is CompletionAuditGroup.AMBIGUOUS:
        challenge_score = (
            0.35 * midpoint_uncertainty
            + 0.30 * continuation_tension
            + 0.25 * voice_disagreement
            + 0.10 * model_spread
        )
    else:
        external_error = (
            abs(smart_turn.probability - target) + abs(livekit.probability - target)
        ) / 2.0
        label_margin = min(abs(completion - 0.2), abs(completion - 0.8)) / 0.2
        challenge_score = (
            0.55 * voice_disagreement
            + 0.25 * external_error
            + 0.10 * model_spread
            + 0.10 * (1.0 - min(1.0, label_margin))
        )
    reasons = _challenge_reasons(
        midpoint_uncertainty=midpoint_uncertainty,
        continuation_tension=continuation_tension,
        voice_disagreement=voice_disagreement,
        model_spread=model_spread,
    )
    return _AuditCandidate(
        candidate=candidate,
        group=group,
        challenge_score=min(1.0, max(0.0, challenge_score)),
        challenge_reasons=reasons,
        voice_light=voice_light,
        smart_turn=smart_turn,
        livekit=livekit,
    )


def _challenge_reasons(
    midpoint_uncertainty: float,
    continuation_tension: float,
    voice_disagreement: float,
    model_spread: float,
) -> tuple[CompletionAuditSelectionReason, ...]:
    scored = (
        (midpoint_uncertainty, CompletionAuditSelectionReason.LABEL_MIDPOINT),
        (
            continuation_tension,
            CompletionAuditSelectionReason.COMPLETION_CONTINUATION_TENSION,
        ),
        (voice_disagreement, CompletionAuditSelectionReason.VOICE_LIGHT_DISAGREEMENT),
        (model_spread, CompletionAuditSelectionReason.CROSS_MODEL_DISAGREEMENT),
    )
    return tuple(reason for _, reason in sorted(scored, key=lambda item: (-item[0], item[1]))[:2])


def _validate_source_role(role: str, artifact: CompletionPredictionArtifact) -> None:
    detector_kind = artifact.manifest.detector.detector_kind
    match role:
        case "voice_light":
            expected = CompletionDetectorKind.VOICE_LIGHT_COMPLETION
        case "smart_turn":
            expected = CompletionDetectorKind.PIPECAT_SMART_TURN_V3_2
        case "livekit":
            expected = CompletionDetectorKind.LIVEKIT_V1_MINI
        case _:
            raise AssertionError(f"Unhandled completion audit source role {role!r}.")
    if detector_kind is not expected:
        raise ValueError(f"Completion audit {role} predictions use the wrong detector kind.")


def _select_audit_candidates(
    rows: tuple[_AuditCandidate, ...],
    configuration: CompletionAuditConfiguration,
) -> tuple[_AuditCandidate, ...]:
    by_group = {
        group: tuple(row for row in rows if row.group is group) for group in CompletionAuditGroup
    }
    requested = {
        CompletionAuditGroup.AMBIGUOUS: configuration.ambiguous_count,
        CompletionAuditGroup.CONFIDENT_HOLD: configuration.confident_hold_count,
        CompletionAuditGroup.CONFIDENT_EOT: configuration.confident_eot_count,
    }
    for group, count in requested.items():
        if count > len(by_group[group]):
            raise ValueError(
                f"Completion audit requested {count} {group.value} cases, "
                f"but only {len(by_group[group])} are available."
            )

    ambiguous = _balanced_select(
        by_group[CompletionAuditGroup.AMBIGUOUS],
        configuration.ambiguous_count,
        lambda row: (-row.challenge_score, row.candidate.candidate_id),
    )
    selected = list(ambiguous)
    for group, count in (
        (CompletionAuditGroup.CONFIDENT_HOLD, configuration.confident_hold_count),
        (CompletionAuditGroup.CONFIDENT_EOT, configuration.confident_eot_count),
    ):
        challenge_count = round(count * configuration.confident_challenge_fraction)
        challenge = _balanced_select(
            by_group[group],
            challenge_count,
            lambda row: (-row.challenge_score, row.candidate.candidate_id),
        )
        challenge_ids = {row.candidate.candidate_id for row in challenge}
        controls = _balanced_select(
            tuple(
                row for row in by_group[group] if row.candidate.candidate_id not in challenge_ids
            ),
            count - len(challenge),
            lambda row: _stable_digest(
                configuration.seed,
                "representative-control",
                row.candidate.candidate_id,
            ),
        )
        selected.extend(challenge)
        selected.extend(
            row.model_copy(
                update={
                    "challenge_reasons": (CompletionAuditSelectionReason.REPRESENTATIVE_CONTROL,)
                }
            )
            for row in controls
        )
    return tuple(selected)


def _balanced_select(
    rows: Iterable[_AuditCandidate],
    count: int,
    rank: Callable[[_AuditCandidate], object],
) -> tuple[_AuditCandidate, ...]:
    groups: defaultdict[tuple[str, str], list[_AuditCandidate]] = defaultdict(list)
    for row in rows:
        groups[(row.candidate.dataset_name, row.candidate.conversation_id)].append(row)
    for values in groups.values():
        values.sort(key=rank)
    group_keys = sorted(groups)
    selected: list[_AuditCandidate] = []
    while len(selected) < count:
        progressed = False
        for key in group_keys:
            values = groups[key]
            if values:
                selected.append(values.pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError("Completion audit selection could not satisfy its requested count.")
    return tuple(selected)


def _audit_item(
    row: _AuditCandidate,
    order: int,
    double_review: bool,
    configuration: CompletionAuditConfiguration,
) -> CompletionAuditItem:
    candidate = row.candidate
    clip_start = max(0.0, candidate.anchor_seconds - configuration.context_before_seconds)
    clip_end = candidate.anchor_seconds + configuration.context_after_seconds
    audit_id = _stable_digest(configuration.seed, "audit-item", candidate.candidate_id)
    return CompletionAuditItem(
        order=order,
        audit_id=audit_id,
        candidate_id=candidate.candidate_id,
        group=row.group,
        selection_reasons=row.challenge_reasons,
        challenge_score=row.challenge_score,
        double_review=double_review,
        dataset_name=candidate.dataset_name,
        conversation_id=candidate.conversation_id,
        external_id=candidate.external_id,
        user_side=candidate.user_side,
        categories=candidate.categories,
        anchor_seconds=candidate.anchor_seconds,
        boundary_kind=candidate.boundary_kind.value,
        completion_probability=candidate.target_points[0].completion_probability,
        continuation_probability=candidate.continuation_probability,
        voice_light=row.voice_light,
        smart_turn=row.smart_turn,
        livekit=row.livekit,
        clip_path=f"clips/{order:04d}-{audit_id[:12]}.wav",
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        boundary_offset_seconds=candidate.anchor_seconds - clip_start,
        source_window_ids=candidate.source_window_ids,
    )


def _stable_digest(seed: str, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{purpose}|{value}".encode()).hexdigest()

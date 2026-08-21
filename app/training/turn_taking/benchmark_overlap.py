from __future__ import annotations

import re
from collections.abc import Iterable

from app.training.turn_taking.benchmark_models import (
    AudioProvenanceRecord,
    ExactOverlap,
    OverlapAuditReport,
)

MUNDO_SOURCE_TOKEN = "mundo"
MUNDO_SOURCE_ALIASES = frozenset(("dataset3",))


def audit_smart_turn_overlap(
    local_records: Iterable[AudioProvenanceRecord],
    smart_turn_records: Iterable[AudioProvenanceRecord],
    external_repository: str,
    external_revision: str,
) -> OverlapAuditReport:
    local = tuple(local_records)
    external = tuple(smart_turn_records)
    exact_overlaps = _exact_overlaps(local=local, external=external)
    local_source_tokens = {_normalized_source(record.source_name) for record in local}
    external_source_tokens = {_normalized_source(record.source_name) for record in external}
    provenance_risk_sources = tuple(
        sorted(
            source
            for source in local_source_tokens
            if _is_mundo_source(source)
            and any(_is_mundo_source(other) for other in external_source_tokens)
        )
    )
    unverified_count = sum(
        record.audio_sha256 is None and record.pcm_sha256 is None for record in local
    )
    return OverlapAuditReport(
        external_repository=external_repository,
        external_revision=external_revision,
        local_record_count=len(local),
        external_record_count=len(external),
        exact_overlaps=exact_overlaps,
        provenance_risk_sources=provenance_risk_sources,
        unverified_local_record_count=unverified_count,
        exact_hash_comparison_performed=bool(external)
        and all(
            record.audio_sha256 is not None or record.pcm_sha256 is not None
            for record in (*local, *external)
        ),
        clean_comparative_claim_permitted=(
            not exact_overlaps and not provenance_risk_sources and unverified_count == 0
        ),
    )


def _exact_overlaps(
    local: tuple[AudioProvenanceRecord, ...],
    external: tuple[AudioProvenanceRecord, ...],
) -> tuple[ExactOverlap, ...]:
    overlaps: list[ExactOverlap] = []
    for local_record in local:
        for external_record in external:
            match = _matched_hash(local_record=local_record, external_record=external_record)
            if match is None:
                continue
            hash_kind, matched_sha256 = match
            overlaps.append(
                ExactOverlap(
                    local_source_name=local_record.source_name,
                    local_external_id=local_record.external_id,
                    external_source_name=external_record.source_name,
                    external_external_id=external_record.external_id,
                    matched_hash_kind=hash_kind,
                    matched_sha256=matched_sha256,
                )
            )
    return tuple(
        sorted(
            overlaps,
            key=lambda overlap: (
                overlap.matched_sha256,
                overlap.local_source_name,
                overlap.external_source_name,
            ),
        )
    )


def _matched_hash(
    local_record: AudioProvenanceRecord,
    external_record: AudioProvenanceRecord,
) -> tuple[str, str] | None:
    if (
        local_record.audio_sha256 is not None
        and local_record.audio_sha256 == external_record.audio_sha256
    ):
        return "audio_sha256", local_record.audio_sha256
    if (
        local_record.pcm_sha256 is not None
        and local_record.pcm_sha256 == external_record.pcm_sha256
    ):
        return "pcm_sha256", local_record.pcm_sha256
    return None


def _normalized_source(source_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", source_name.lower())


def _is_mundo_source(normalized_source: str) -> bool:
    return MUNDO_SOURCE_TOKEN in normalized_source or normalized_source in MUNDO_SOURCE_ALIASES

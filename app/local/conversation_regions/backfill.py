from __future__ import annotations

import argparse
from uuid import UUID

from app.local.config import DATABASE_URL
from app.local.conversation_regions.models import ConversationRegionConfig
from app.local.conversation_regions.repository import ConversationRegionRepository
from app.local.conversation_regions.service import analyze_conversation_regions
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.shared.quality import METRIC_VERSION


def backfill_conversation_regions(
    repository: ConversationRegionRepository,
    dataset_id: UUID | None,
    config: ConversationRegionConfig,
) -> int:
    evidence_records = repository.list_evidence(
        dataset_id=dataset_id,
        metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
    )
    for evidence in evidence_records:
        repository.upsert(
            sample_id=evidence.sample_id,
            analysis=analyze_conversation_regions(
                speaker1_vad=evidence.speaker1_vad,
                speaker2_vad=evidence.speaker2_vad,
                annotation=evidence.annotation,
                config=config,
            ),
        )
    return len(evidence_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=UUID)
    options = parser.parse_args()
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required.")
    processed_count = backfill_conversation_regions(
        repository=ConversationRegionRepository(DATABASE_URL),
        dataset_id=options.dataset_id,
        config=ConversationRegionConfig(),
    )
    print(f"Backfilled conversational regions for {processed_count} samples.")


if __name__ == "__main__":
    main()

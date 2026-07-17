from __future__ import annotations

from app.local.config import DATABASE_URL
from app.local.synchronization_review.acoustic_evaluation import evaluate_acoustic_offsets
from app.local.synchronization_review.evaluation import reviewed_offset_labels
from app.local.synchronization_review.repository import SynchronizationReviewRepository


def main() -> None:
    repository = SynchronizationReviewRepository(database_url=DATABASE_URL)
    labels = reviewed_offset_labels(stored_alignments=repository.load_reviewed_alignments())
    report = evaluate_acoustic_offsets(labels=labels)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

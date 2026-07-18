from __future__ import annotations

import argparse

from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.config import DATABASE_URL
from app.local.synchronization_review.activity_evaluation import (
    evaluate_activity_offset_estimator,
    reviewed_activity_curves,
)
from app.local.synchronization_review.evaluation import (
    EvidenceScope,
    ReviewProvenance,
    evaluate_offset_estimator,
    reviewed_examples,
    reviewed_offset_labels,
)
from app.local.synchronization_review.full_recording_evidence import (
    SUPPORTED_FULL_EVIDENCE_MODELS,
    full_recording_evidence_records,
)
from app.local.synchronization_review.initial_recording_evidence import (
    initial_recording_evidence_records,
)
from app.local.synchronization_review.repository import SynchronizationReviewRepository
from app.shared.asr import AsrModelId
from app.shared.quality import METRIC_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synchronization offset estimation.")
    parser.add_argument(
        "--evidence-scope",
        choices=tuple(scope.value for scope in EvidenceScope),
        default=EvidenceScope.INITIAL_180_SECONDS.value,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=AsrModelId,
        choices=SUPPORTED_FULL_EVIDENCE_MODELS,
        default=list(SUPPORTED_FULL_EVIDENCE_MODELS),
        help="Full-recording ASR models to evaluate.",
    )
    parser.add_argument(
        "--activity-optimizer",
        action="store_true",
        help="Cross-validate the full-recording audio-activity basin optimizer.",
    )
    arguments = parser.parse_args()
    evidence_scope = EvidenceScope(arguments.evidence_scope)
    model_ids = tuple(arguments.models)
    repository = SynchronizationReviewRepository(database_url=DATABASE_URL)
    labels = reviewed_offset_labels(stored_alignments=repository.load_reviewed_alignments())
    if arguments.activity_optimizer:
        database_labels = tuple(
            label for label in labels if label.provenance is ReviewProvenance.DATABASE
        )
        activity_report = evaluate_activity_offset_estimator(
            examples=reviewed_activity_curves(labels=database_labels)
        )
        print(activity_report.model_dump_json(indent=2))
        return
    match evidence_scope:
        case EvidenceScope.INITIAL_180_SECONDS:
            evidence_records = initial_recording_evidence_records(
                transcript_pairs=repository.load_transcript_pairs(),
                annotations=repository.load_annotations(metric_version=METRIC_VERSION),
            )
        case EvidenceScope.FULL_RECORDING:
            full_repository = FullRecordingAsrRepository(database_url=DATABASE_URL)
            transcripts = full_repository.list_latest_transcripts(model_ids=model_ids)
            annotations = repository.load_annotations(metric_version=METRIC_VERSION)
            evidence_records = full_recording_evidence_records(
                transcripts=transcripts,
                model_ids=model_ids,
                annotations=annotations,
            )
    examples = reviewed_examples(labels=labels, evidence_records=evidence_records)
    report = evaluate_offset_estimator(examples=examples)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

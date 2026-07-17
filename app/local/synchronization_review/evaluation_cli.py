from __future__ import annotations

import argparse

from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.config import DATABASE_URL
from app.local.synchronization_review.evaluation import (
    EvidenceScope,
    evaluate_offset_estimator,
    evidence_records_from_candidates,
    reviewed_examples,
    reviewed_offset_labels,
)
from app.local.synchronization_review.full_recording_evidence import (
    SUPPORTED_FULL_EVIDENCE_MODELS,
    full_recording_evidence_records,
)
from app.local.synchronization_review.repository import SynchronizationReviewRepository
from app.local.synchronization_review.service import synchronization_candidates
from app.shared.asr import AsrModelId


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
        default=[AsrModelId.PARAKEET_TDT],
        help="Full-recording ASR models to evaluate.",
    )
    arguments = parser.parse_args()
    evidence_scope = EvidenceScope(arguments.evidence_scope)
    model_ids = tuple(arguments.models)
    repository = SynchronizationReviewRepository(database_url=DATABASE_URL)
    labels = reviewed_offset_labels(stored_alignments=repository.load_reviewed_alignments())
    match evidence_scope:
        case EvidenceScope.INITIAL_180_SECONDS:
            candidate_response = synchronization_candidates(repository=repository)
            evidence_records = evidence_records_from_candidates(
                candidates=candidate_response.candidates
            )
        case EvidenceScope.FULL_RECORDING:
            full_repository = FullRecordingAsrRepository(database_url=DATABASE_URL)
            transcripts = full_repository.list_latest_transcripts(model_ids=model_ids)
            evidence_records = full_recording_evidence_records(
                transcripts=transcripts,
                model_ids=model_ids,
            )
    examples = reviewed_examples(labels=labels, evidence_records=evidence_records)
    report = evaluate_offset_estimator(examples=examples)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

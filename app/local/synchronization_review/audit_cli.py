from __future__ import annotations

import argparse
from pathlib import Path

from app.local.config import DATABASE_URL
from app.local.data.sessions import SpeakerName, session_audio_path
from app.local.synchronization_review.audit import (
    SYNCHRONIZATION_AUDIT_PATH,
    SYNCHRONIZATION_AUDIT_STATIC_PATH,
    build_synchronization_audit_report,
)
from app.local.synchronization_review.repository import SynchronizationReviewRepository
from app.local.synchronization_review.service import synchronization_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a read-only overlapping-window synchronization audit."
    )
    parser.add_argument(
        "--external-id",
        help="Analyze one session for a quick verification run before the full audit.",
    )
    arguments = parser.parse_args()
    response = synchronization_candidates(
        repository=SynchronizationReviewRepository(database_url=DATABASE_URL)
    )
    candidates = tuple(
        candidate
        for candidate in response.candidates
        if arguments.external_id is None or candidate.external_id == arguments.external_id
    )
    if not candidates:
        raise ValueError(f"No synchronization candidate found for {arguments.external_id}.")
    track_paths = tuple(
        (
            session_audio_path(
                identifier=candidate.external_id,
                speaker_name=SpeakerName.SPEAKER1,
            ),
            session_audio_path(
                identifier=candidate.external_id,
                speaker_name=SpeakerName.SPEAKER2,
            ),
        )
        for candidate in candidates
    )
    report = build_synchronization_audit_report(
        candidates=candidates,
        track_paths=track_paths,
    )
    report_path = (
        SYNCHRONIZATION_AUDIT_PATH
        if arguments.external_id is None
        else SYNCHRONIZATION_AUDIT_PATH.with_name(
            f"synchronization-audit-{arguments.external_id}.json"
        )
    )
    _write_report(report_json=report.model_dump_json(indent=2), report_path=report_path)
    if arguments.external_id is None:
        _write_report(
            report_json=report.model_dump_json(indent=2),
            report_path=SYNCHRONIZATION_AUDIT_STATIC_PATH,
        )
    print(
        f"Wrote {report.analyzed_session_count} read-only synchronization audit results "
        f"to {report_path}."
    )


def _write_report(report_json: str, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(report_json, encoding="utf-8")
    temporary_path.replace(report_path)


if __name__ == "__main__":
    main()

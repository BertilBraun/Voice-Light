from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from app.local.training_corpus.hub_publish import verify_upload_entries
from app.local.training_corpus.hub_upload_plan import (
    CorpusUploadPlan,
    CorpusUploadPlanEntry,
    CorpusUploadPlanSummary,
    UploadPlanAction,
    UploadPlanReason,
)

REVISION = "1" * 40


def test_verify_upload_entries_accepts_unchanged_planned_files(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_bytes(b"corpus")
    entry = upload_entry(path)

    assert verify_upload_entries(plan(entry)) == (entry,)


@pytest.mark.parametrize("change", ("missing", "size", "hash"))
def test_verify_upload_entries_rejects_files_changed_after_planning(
    tmp_path: Path,
    change: str,
) -> None:
    path = tmp_path / "corpus.json"
    path.write_bytes(b"corpus")
    entry = upload_entry(path)
    if change == "missing":
        path.unlink()
    elif change == "size":
        path.write_bytes(b"different-size")
    else:
        path.write_bytes(b"change")

    with pytest.raises(ValueError, match="missing|size changed|hash changed"):
        verify_upload_entries(plan(entry))


def upload_entry(path: Path) -> CorpusUploadPlanEntry:
    content = path.read_bytes()
    return CorpusUploadPlanEntry(
        path=PurePosixPath("corpus.json"),
        local_path=path,
        action=UploadPlanAction.ADD,
        reason=UploadPlanReason.MISSING_REMOTELY,
        local_size_bytes=len(content),
        local_sha256=hashlib.sha256(content).hexdigest(),
    )


def plan(entry: CorpusUploadPlanEntry) -> CorpusUploadPlan:
    return CorpusUploadPlan(
        generated_at=datetime(2026, 8, 21, tzinfo=UTC),
        repository_id="BertilBraun/voice-light-audio",
        base_revision=REVISION,
        verified_head_revision=REVISION,
        summary=CorpusUploadPlanSummary(
            add_file_count=1,
            update_file_count=0,
            unchanged_file_count=0,
            excluded_file_count=0,
            upload_file_count=1,
            upload_bytes=entry.local_size_bytes,
        ),
        entries=(entry,),
    )

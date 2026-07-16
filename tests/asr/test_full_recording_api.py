from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.local.asr import router as full_recording_router
from app.local.asr.full_recording_models import (
    FullRecordingAsrBatchRecord,
    FullRecordingAsrBatchRequest,
    FullRecordingAsrBatchStatus,
    FullRecordingAsrSampleScope,
)
from app.local.main import app
from app.shared.asr import AsrModelId


class StubFullRecordingRepository:
    def __init__(self, record: FullRecordingAsrBatchRecord) -> None:
        self.record = record
        self.requests: list[FullRecordingAsrBatchRequest] = []

    def create_batch(self, request: FullRecordingAsrBatchRequest) -> FullRecordingAsrBatchRecord:
        self.requests.append(request)
        return self.record

    def get_batch(self, batch_id: UUID) -> FullRecordingAsrBatchRecord:
        if batch_id != self.record.id:
            raise ValueError(f"Full-recording ASR batch not found: {batch_id}")
        return self.record


def test_full_recording_api_queues_work_and_reports_compact_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = batch_record()
    repository = StubFullRecordingRepository(record)
    monkeypatch.setattr(
        full_recording_router,
        "full_recording_asr_repository",
        lambda: repository,
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/api/asr/full-recording/batches",
            json={
                "idempotency_key": record.idempotency_key,
                "dataset_id": str(record.dataset_id),
                "sample_scope": FullRecordingAsrSampleScope.QUALITY_ANALYZED.value,
                "models": [AsrModelId.PARAKEET_TDT.value],
            },
        )
        status_response = client.get(f"/api/asr/full-recording/batches/{record.id}")

    assert create_response.status_code == 200
    assert status_response.status_code == 200
    assert create_response.json()["total_track_count"] == 12
    assert status_response.json()["recent_errors"] == ["one failed track"]
    assert repository.requests[0].models == (AsrModelId.PARAKEET_TDT,)


def batch_record() -> FullRecordingAsrBatchRecord:
    now = datetime.now(UTC)
    return FullRecordingAsrBatchRecord(
        id=uuid4(),
        idempotency_key="luel-parakeet-full-test",
        dataset_id=uuid4(),
        sample_scope=FullRecordingAsrSampleScope.QUALITY_ANALYZED,
        models=(AsrModelId.PARAKEET_TDT,),
        status=FullRecordingAsrBatchStatus.RUNNING,
        total_track_count=12,
        pending_track_count=8,
        running_track_count=1,
        completed_track_count=2,
        failed_track_count=1,
        recent_errors=("one failed track",),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=None,
    )

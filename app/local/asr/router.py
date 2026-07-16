from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.local.asr.client import HttpRemoteAsrClient
from app.local.asr.full_recording_models import (
    FullRecordingAsrBatchRecord,
    FullRecordingAsrBatchRequest,
)
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.repository import AsrTranscriptRepository
from app.local.asr.service import cached_asr_transcripts
from app.local.config import COMPUTE_TOKEN, DATABASE_URL, REMOTE_ASR_ENDPOINT_URL
from app.shared.asr import CachedAsrRequest, CachedAsrResponse

router = APIRouter(prefix="/api/asr", tags=["asr"])


@router.post("/full-recording/batches")
def create_full_recording_asr_batch(
    request: FullRecordingAsrBatchRequest,
) -> FullRecordingAsrBatchRecord:
    try:
        return full_recording_asr_repository().create_batch(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/full-recording/batches/{batch_id}")
def full_recording_asr_batch_status(batch_id: UUID) -> FullRecordingAsrBatchRecord:
    try:
        return full_recording_asr_repository().get_batch(batch_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/transcriptions")
def cached_asr_transcriptions_api(request: CachedAsrRequest) -> CachedAsrResponse:
    try:
        return cached_asr_transcripts(
            audio_path=Path(request.audio_path),
            requested_models=request.models,
            cache=asr_transcript_repository(),
            remote_client_factory=remote_asr_client,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def asr_transcript_repository() -> AsrTranscriptRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for cached ASR.")
    return AsrTranscriptRepository(database_url=DATABASE_URL)


def full_recording_asr_repository() -> FullRecordingAsrRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for full-recording ASR.")
    return FullRecordingAsrRepository(database_url=DATABASE_URL)


def remote_asr_client() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )

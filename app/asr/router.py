from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.asr.client import HttpRemoteAsrClient
from app.asr.repository import AsrTranscriptRepository
from app.asr.schemas import CachedAsrRequest, CachedAsrResponse
from app.asr.service import cached_asr_transcripts
from app.config import COMPUTE_TOKEN, DATABASE_URL, REMOTE_ASR_ENDPOINT_URL

router = APIRouter(prefix="/api/asr", tags=["asr"])


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


def remote_asr_client() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )

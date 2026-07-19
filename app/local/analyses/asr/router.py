from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.local.analyses.asr.catalog import available_asr_models
from app.local.analyses.asr.models import AsrAnalysisRequest, AsrAnalysisResponse, AsrModelInfo
from app.local.analyses.asr.service import analyze_asr
from app.local.asr.client import HttpRemoteAsrClient
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.repository import AsrTranscriptRepository
from app.local.config import COMPUTE_TOKEN, DATABASE_URL, REMOTE_ASR_ENDPOINT_URL

router = APIRouter(prefix="/api/asr", tags=["asr"])


@router.get("/models")
def list_asr_models() -> dict[str, tuple[AsrModelInfo, ...]]:
    return {"models": available_asr_models()}


@router.post("/analyze")
def analyze_asr_api(request: AsrAnalysisRequest) -> AsrAnalysisResponse:
    try:
        return analyze_asr(
            session_id=request.session_id,
            speaker_track=request.speaker_track,
            selected_models=request.models,
            reference_words=request.reference_words,
            cache=asr_transcript_repository(),
            ingested_transcript_source=full_recording_asr_repository(),
            remote_client_factory=remote_asr_client,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def asr_transcript_repository() -> AsrTranscriptRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for ASR analysis.")
    return AsrTranscriptRepository(database_url=DATABASE_URL)


def full_recording_asr_repository() -> FullRecordingAsrRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for ASR analysis.")
    return FullRecordingAsrRepository(database_url=DATABASE_URL)


def remote_asr_client() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )

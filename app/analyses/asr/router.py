from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.analyses.asr.catalog import available_asr_models
from app.analyses.asr.models import AsrAnalysisRequest, AsrAnalysisResponse, AsrModelInfo
from app.analyses.asr.service import analyze_asr
from app.asr.client import HttpRemoteAsrClient
from app.asr.repository import AsrTranscriptRepository
from app.config import COMPUTE_TOKEN, DATABASE_URL, REMOTE_ASR_ENDPOINT_URL

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
            remote_client_factory=remote_asr_client,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def asr_transcript_repository() -> AsrTranscriptRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for ASR analysis.")
    return AsrTranscriptRepository(database_url=DATABASE_URL)


def remote_asr_client() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )

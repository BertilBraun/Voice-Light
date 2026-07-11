from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.analyses.asr.models import AsrAnalysisRequest, AsrAnalysisResponse, AsrModelInfo
from app.analyses.asr.service import analyze_asr, available_asr_models

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
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

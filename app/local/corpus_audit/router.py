from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.local.config import DATABASE_URL
from app.local.conversation_regions.models import CONVERSATION_REGION_ANALYSIS_VERSION
from app.local.corpus_audit.models import CorpusAuditReport, CorpusAuditRequest
from app.local.corpus_audit.repository import CorpusAuditRepository
from app.local.corpus_audit.service import generate_corpus_audit
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.shared.quality import METRIC_VERSION

router = APIRouter(prefix="/api/corpus-audit", tags=["corpus-audit"])


def repository() -> CorpusAuditRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for corpus audit APIs.")
    return CorpusAuditRepository(DATABASE_URL)


@router.post("")
def corpus_audit(request: CorpusAuditRequest, response: Response) -> CorpusAuditReport:
    try:
        response.headers["Cache-Control"] = "no-store"
        evidence = repository().load_evidence(
            dataset_ids=request.dataset_ids,
            minimum_quality=request.minimum_quality,
            metric_version=METRIC_VERSION,
            annotation_version=ANNOTATION_VERSION,
            region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        )
        return generate_corpus_audit(evidence=evidence, request=request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

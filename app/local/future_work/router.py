from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.local.config import FUTURE_WORK_ROOT
from app.local.future_work.service import list_future_work_documents, load_future_work_document
from app.local.future_work.views import render_future_work_document, render_future_work_index

router = APIRouter(prefix="/future-work", tags=["future-work"])


@router.get("", response_class=HTMLResponse)
def future_work_index() -> HTMLResponse:
    documents = list_future_work_documents(documents_root=FUTURE_WORK_ROOT)
    return HTMLResponse(render_future_work_index(documents), headers={"Cache-Control": "no-store"})


@router.get("/{slug}", response_class=HTMLResponse)
def future_work_document(slug: str) -> HTMLResponse:
    try:
        document = load_future_work_document(documents_root=FUTURE_WORK_ROOT, slug=slug)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Future-work document not found") from error
    return HTMLResponse(
        render_future_work_document(document), headers={"Cache-Control": "no-store"}
    )

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.local.future_work.service import (
    list_future_work_documents,
    load_future_work_document,
)
from app.local.main import app


def test_future_work_page_lists_and_renders_repository_documents() -> None:
    with TestClient(app) as client:
        overview_response = client.get("/")
        index_response = client.get("/future-work")
        document_response = client.get("/future-work/speculative-speech-tool-pipelining")

    assert overview_response.status_code == 200
    assert 'href="/future-work"' in overview_response.text
    assert index_response.status_code == 200
    assert "Speculative Speech During Tool Use" in index_response.text
    assert "Speaker Separation for Conversational Voice-Agent Data" in index_response.text
    assert document_response.status_code == 200
    assert "Central Constraint: Prefix Commitment" in document_response.text
    assert '<pre><code class="language-text">' in document_response.text


def test_future_work_document_renders_tables() -> None:
    with TestClient(app) as client:
        response = client.get("/future-work/speaker-separation-for-conversational-data")

    assert response.status_code == 200
    assert "<table>" in response.text
    assert "Very quiet backchannel" in response.text


def test_future_work_document_returns_not_found_for_unknown_slug() -> None:
    with TestClient(app) as client:
        response = client.get("/future-work/not-a-real-idea")

    assert response.status_code == 404


def test_future_work_service_discovers_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "later-idea.md").write_text(
        "# Later Idea\n\nA concise description for the index.\n\n## Notes\n\nMore detail.",
        encoding="utf-8",
    )
    (tmp_path / "first-idea.md").write_text(
        "# First Idea\n\nAnother description.\n", encoding="utf-8"
    )

    documents = list_future_work_documents(documents_root=tmp_path)

    assert tuple(document.slug for document in documents) == ("first-idea", "later-idea")
    assert documents[1].summary == "A concise description for the index."


@pytest.mark.parametrize("slug", ("../README", "UPPERCASE", "contains spaces"))
def test_future_work_service_rejects_unsafe_slugs(tmp_path: Path, slug: str) -> None:
    with pytest.raises(ValueError, match="Invalid future-work document slug"):
        load_future_work_document(documents_root=tmp_path, slug=slug)


def test_future_work_service_requires_title_and_summary(tmp_path: Path) -> None:
    (tmp_path / "missing-metadata.md").write_text("## Notes only\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must start with an H1 title"):
        load_future_work_document(documents_root=tmp_path, slug="missing-metadata")

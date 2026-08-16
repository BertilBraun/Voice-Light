from __future__ import annotations

from pathlib import Path

WEB_ROOT = Path(__file__).parents[2] / "app" / "local" / "web" / "pages"


def test_corpus_review_page_exposes_readiness_gate() -> None:
    html = (WEB_ROOT / "corpus-review" / "index.html").read_text(encoding="utf-8")
    script = (WEB_ROOT / "corpus-review" / "app.js").read_text(encoding="utf-8")

    assert 'id="gate-issues"' in html
    assert "/readiness" in script
    assert "READY TO PUBLISH" in script
    assert "BLOCKED" in script


def test_corpus_review_links_pin_exact_training_crop() -> None:
    script = (WEB_ROOT / "corpus-review" / "app.js").read_text(encoding="utf-8")

    for parameter in (
        "review_item_id",
        "dataset_id",
        "sample_id",
        "user_side",
        "start_seconds",
    ):
        assert parameter in script

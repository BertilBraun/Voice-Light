from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class FutureWorkDocument:
    slug: str
    title: str
    summary: str
    rendered_content: str


def list_future_work_documents(documents_root: Path) -> tuple[FutureWorkDocument, ...]:
    documents = tuple(
        load_future_work_document(documents_root=documents_root, slug=path.stem)
        for path in sorted(documents_root.glob("*.md"))
    )
    return tuple(sorted(documents, key=lambda document: document.title.casefold()))


def load_future_work_document(documents_root: Path, slug: str) -> FutureWorkDocument:
    if SLUG_PATTERN.fullmatch(slug) is None:
        raise ValueError(f"Invalid future-work document slug: {slug}")

    document_path = documents_root / f"{slug}.md"
    if not document_path.is_file():
        raise FileNotFoundError(document_path)

    markdown_text = document_path.read_text(encoding="utf-8")
    title, summary = parse_document_metadata(markdown_text=markdown_text, path=document_path)
    renderer = MarkdownIt("commonmark", {"html": False, "typographer": True}).enable("table")
    return FutureWorkDocument(
        slug=slug,
        title=title,
        summary=summary,
        rendered_content=renderer.render(markdown_text),
    )


def parse_document_metadata(markdown_text: str, path: Path) -> tuple[str, str]:
    lines = markdown_text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"Future-work document must start with an H1 title: {path}")

    title = lines[0].removeprefix("# ").strip()
    if not title:
        raise ValueError(f"Future-work document title must not be empty: {path}")

    summary_lines: list[str] = []
    for line in lines[1:]:
        stripped_line = line.strip()
        if not stripped_line:
            if summary_lines:
                break
            continue
        if stripped_line.startswith(("#", "- ", "* ", ">", "```")):
            if summary_lines:
                break
            continue
        summary_lines.append(stripped_line)

    if not summary_lines:
        raise ValueError(f"Future-work document must have an introductory paragraph: {path}")
    return title, " ".join(summary_lines)

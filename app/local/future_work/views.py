from __future__ import annotations

from html import escape

from app.local.future_work.service import FutureWorkDocument


def render_future_work_index(documents: tuple[FutureWorkDocument, ...]) -> str:
    document_links = "\n".join(_render_document_link(document) for document in documents)
    return _page_shell(
        title="Future Work · Voice Light",
        body=f"""
        <header class="page-header">
          <a class="back-link" href="/">← Voice Light</a>
          <p class="eyebrow">Research notebook</p>
          <h1>Future Work</h1>
          <p class="lede">Ideas worth keeping visible, even before they become active projects.</p>
        </header>
        <main class="document-grid">
          {document_links}
        </main>
        <footer>
          Add an idea by creating a Markdown file in <code>docs/future-work</code>.
        </footer>
        """,
    )


def render_future_work_document(document: FutureWorkDocument) -> str:
    return _page_shell(
        title=f"{escape(document.title)} · Future Work",
        body=f"""
        <header class="article-header">
          <a class="back-link" href="/future-work">← All future work</a>
        </header>
        <main class="document-shell">
          <article class="markdown-body">{document.rendered_content}</article>
        </main>
        """,
    )


def _render_document_link(document: FutureWorkDocument) -> str:
    return f"""
    <a class="document-card" href="/future-work/{escape(document.slug)}">
      <span>
        <strong>{escape(document.title)}</strong>
        <span class="summary">{escape(document.summary)}</span>
      </span>
      <span class="open-label">Read <span aria-hidden="true">→</span></span>
    </a>
    """


def _page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <link rel="stylesheet" href="/pages/future-work/styles.css" />
  </head>
  <body>
    <div class="site-shell">
      {body}
    </div>
  </body>
</html>
"""

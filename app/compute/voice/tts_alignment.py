from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from app.compute.voice.interfaces import SynthesizedWordBoundary


@dataclass
class _PendingTranscriptBoundary:
    text_offset: int | None


class TranscriptBoundaryTracker:
    def __init__(self) -> None:
        self.pending: deque[_PendingTranscriptBoundary] = deque()
        self.last_emitted: SynthesizedWordBoundary | None = None
        self.last_source_text_offset = 0

    def add_source_word(
        self,
        token_entry_flags: Sequence[bool],
        text_offset: int,
    ) -> tuple[SynthesizedWordBoundary, ...]:
        if text_offset <= 0:
            raise ValueError("Source word text offsets must be positive.")
        if text_offset <= self.last_source_text_offset:
            raise ValueError("Source word text offsets must increase monotonically.")
        self.last_source_text_offset = text_offset
        if not any(token_entry_flags):
            return self._coalesce_removed_word(text_offset)
        first_token_entry = True
        for has_tokens in token_entry_flags:
            if not has_tokens:
                continue
            mapping_offset = text_offset if first_token_entry else None
            self.pending.append(_PendingTranscriptBoundary(mapping_offset))
            first_token_entry = False
        return ()

    def consume_transcript_word(self, start_sample: int) -> SynthesizedWordBoundary | None:
        if start_sample < 0:
            raise ValueError("Transcript boundary sample offsets cannot be negative.")
        if not self.pending:
            raise AssertionError("Kyutai emitted a transcript word without an input mapping.")
        mapping = self.pending.popleft()
        if mapping.text_offset is None:
            return None
        boundary = SynthesizedWordBoundary(
            text_offset=mapping.text_offset,
            start_sample=start_sample,
        )
        self.last_emitted = boundary
        return boundary

    def _coalesce_removed_word(
        self,
        text_offset: int,
    ) -> tuple[SynthesizedWordBoundary, ...]:
        for mapping in reversed(self.pending):
            if mapping.text_offset is not None:
                mapping.text_offset = max(mapping.text_offset, text_offset)
                return ()
        if self.last_emitted is None:
            return ()
        if text_offset <= self.last_emitted.text_offset:
            return ()
        boundary = SynthesizedWordBoundary(
            text_offset=text_offset,
            start_sample=self.last_emitted.start_sample,
        )
        self.last_emitted = boundary
        return (boundary,)

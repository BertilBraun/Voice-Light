from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewedAlignment:
    external_id: str
    speaker2_shift_seconds: float


REVIEWED_ALIGNMENTS = (
    ReviewedAlignment(external_id="pmt_095", speaker2_shift_seconds=-2.9),
    ReviewedAlignment(external_id="pmt_192", speaker2_shift_seconds=-4.0),
    ReviewedAlignment(external_id="pmt_205", speaker2_shift_seconds=-10.0),
    ReviewedAlignment(external_id="pmt_226", speaker2_shift_seconds=-2.5),
    ReviewedAlignment(external_id="pmt_236", speaker2_shift_seconds=-4.8),
    ReviewedAlignment(external_id="pmt_284", speaker2_shift_seconds=-6.8),
)


def reviewed_alignment(external_id: str) -> ReviewedAlignment | None:
    return next(
        (alignment for alignment in REVIEWED_ALIGNMENTS if alignment.external_id == external_id),
        None,
    )

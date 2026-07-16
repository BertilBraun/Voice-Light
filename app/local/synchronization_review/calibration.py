from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewedAlignment:
    external_id: str
    speaker2_shift_seconds: float


REVIEWED_ALIGNMENTS = (
    ReviewedAlignment(external_id="pmt_284", speaker2_shift_seconds=-6.8),
    ReviewedAlignment(external_id="pmt_205", speaker2_shift_seconds=-10.0),
    ReviewedAlignment(external_id="pmt_236", speaker2_shift_seconds=-4.8),
    ReviewedAlignment(external_id="pmt_008", speaker2_shift_seconds=-4.2),
    ReviewedAlignment(external_id="pmt_132", speaker2_shift_seconds=-3.2),
    ReviewedAlignment(external_id="pmt_161", speaker2_shift_seconds=-3.2),
    ReviewedAlignment(external_id="pmt_226", speaker2_shift_seconds=-2.5),
    ReviewedAlignment(external_id="pmt_180", speaker2_shift_seconds=-3.6),
    ReviewedAlignment(external_id="pmt_315", speaker2_shift_seconds=-2.6),
    ReviewedAlignment(external_id="pmt_095", speaker2_shift_seconds=-2.9),
    ReviewedAlignment(external_id="pmt_192", speaker2_shift_seconds=-4.0),
    ReviewedAlignment(external_id="pmt_256", speaker2_shift_seconds=-3.6),
    ReviewedAlignment(external_id="pmt_217", speaker2_shift_seconds=-4.2),
    ReviewedAlignment(external_id="pmt_237", speaker2_shift_seconds=-2.4),
    ReviewedAlignment(external_id="pmt_087", speaker2_shift_seconds=-1.6),
    ReviewedAlignment(external_id="pmt_225", speaker2_shift_seconds=-2.8),
    ReviewedAlignment(external_id="pmt_219", speaker2_shift_seconds=-1.4),
    ReviewedAlignment(external_id="pmt_318", speaker2_shift_seconds=-0.4),
    ReviewedAlignment(external_id="pmt_144", speaker2_shift_seconds=-2.0),
)


def reviewed_alignment(external_id: str) -> ReviewedAlignment | None:
    return next(
        (alignment for alignment in REVIEWED_ALIGNMENTS if alignment.external_id == external_id),
        None,
    )

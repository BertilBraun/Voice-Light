from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewedAlignment:
    external_id: str
    speaker2_shift_seconds: float


REVIEWED_ALIGNMENTS = (
    ReviewedAlignment(external_id="sample_284", speaker2_shift_seconds=-6.8),
    ReviewedAlignment(external_id="sample_205", speaker2_shift_seconds=-10.0),
    ReviewedAlignment(external_id="sample_236", speaker2_shift_seconds=-4.8),
    ReviewedAlignment(external_id="sample_008", speaker2_shift_seconds=-4.2),
    ReviewedAlignment(external_id="sample_132", speaker2_shift_seconds=-3.2),
    ReviewedAlignment(external_id="sample_161", speaker2_shift_seconds=-3.2),
    ReviewedAlignment(external_id="sample_226", speaker2_shift_seconds=-2.5),
    ReviewedAlignment(external_id="sample_180", speaker2_shift_seconds=-3.6),
    ReviewedAlignment(external_id="sample_315", speaker2_shift_seconds=-2.6),
    ReviewedAlignment(external_id="sample_095", speaker2_shift_seconds=-2.9),
    ReviewedAlignment(external_id="sample_192", speaker2_shift_seconds=-4.0),
    ReviewedAlignment(external_id="sample_256", speaker2_shift_seconds=-3.6),
    ReviewedAlignment(external_id="sample_217", speaker2_shift_seconds=-4.2),
    ReviewedAlignment(external_id="sample_237", speaker2_shift_seconds=-2.4),
    ReviewedAlignment(external_id="sample_087", speaker2_shift_seconds=-1.6),
    ReviewedAlignment(external_id="sample_225", speaker2_shift_seconds=-2.8),
    ReviewedAlignment(external_id="sample_219", speaker2_shift_seconds=-1.4),
    ReviewedAlignment(external_id="sample_318", speaker2_shift_seconds=-0.4),
    ReviewedAlignment(external_id="sample_144", speaker2_shift_seconds=-2.0),
    ReviewedAlignment(external_id="sample_306", speaker2_shift_seconds=0.8),
    ReviewedAlignment(external_id="sample_246", speaker2_shift_seconds=-1.2),
    ReviewedAlignment(external_id="sample_214", speaker2_shift_seconds=-1.6),
    ReviewedAlignment(external_id="sample_310", speaker2_shift_seconds=-1.6),
    ReviewedAlignment(external_id="sample_245", speaker2_shift_seconds=-1.2),
    ReviewedAlignment(external_id="sample_222", speaker2_shift_seconds=-2.6),
    ReviewedAlignment(external_id="sample_324", speaker2_shift_seconds=-2.4),
    ReviewedAlignment(external_id="sample_298", speaker2_shift_seconds=-1.2),
    ReviewedAlignment(external_id="sample_007", speaker2_shift_seconds=-2.0),
)

UNRESOLVED_ALIGNMENT_IDS = ("sample_326",)


def reviewed_alignment(
    external_id: str,
    stored_alignments: tuple[ReviewedAlignment, ...] = (),
) -> ReviewedAlignment | None:
    return next(
        (
            alignment
            for alignment in (*stored_alignments, *REVIEWED_ALIGNMENTS)
            if alignment.external_id == external_id
        ),
        None,
    )


def is_unresolved_alignment(external_id: str) -> bool:
    return external_id in UNRESOLVED_ALIGNMENT_IDS

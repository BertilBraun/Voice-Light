from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.local.alignment_migration.models import AlignmentSidecar


@dataclass(frozen=True)
class TranscriptAlignment:
    speaker1_offset_seconds: float
    speaker2_offset_seconds: float
    aligned_duration_seconds: float | None

    def offset_seconds(self, speaker: str) -> float:
        match speaker:
            case "Speaker1":
                return self.speaker1_offset_seconds
            case "Speaker2":
                return self.speaker2_offset_seconds
            case _:
                raise ValueError(f"Unknown transcript speaker: {speaker}")


def transcript_alignment(metadata_path: Path) -> TranscriptAlignment:
    sidecar_path = metadata_path.with_name("alignment.json")
    if not sidecar_path.is_file():
        return TranscriptAlignment(
            speaker1_offset_seconds=0.0,
            speaker2_offset_seconds=0.0,
            aligned_duration_seconds=None,
        )
    sidecar = AlignmentSidecar.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.sample_external_id != metadata_path.parent.name:
        raise ValueError(
            f"Alignment sidecar sample identifier does not match {metadata_path.name}."
        )
    return TranscriptAlignment(
        speaker1_offset_seconds=(
            sidecar.speaker1.prepended_silence_frame_count / sidecar.sample_rate
        ),
        speaker2_offset_seconds=(
            sidecar.speaker2.prepended_silence_frame_count / sidecar.sample_rate
        ),
        aligned_duration_seconds=(
            max(
                sidecar.speaker1.aligned_frame_count,
                sidecar.speaker2.aligned_frame_count,
            )
            / sidecar.sample_rate
        ),
    )

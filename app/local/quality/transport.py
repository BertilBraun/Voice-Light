from __future__ import annotations

from pathlib import Path

from app.shared.audio.transport import prepare_analysis_audio


def prepare_quality_transport_audio(source_path: Path, output_path: Path) -> None:
    prepare_analysis_audio(source_path=source_path, output_path=output_path)

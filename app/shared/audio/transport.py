from __future__ import annotations

import subprocess
from pathlib import Path

from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS
from app.shared.quality import QUALITY_SAMPLE_RATE


def prepare_analysis_audio(source_path: Path, output_path: Path) -> None:
    command = (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-t",
        str(ANALYSIS_AUDIO_MAX_DURATION_SECONDS),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(QUALITY_SAMPLE_RATE),
        "-c:a",
        "flac",
        "-compression_level",
        "0",
        str(output_path),
    )
    subprocess.run(command, check=True)

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import File, Form, UploadFile

from app.compute.quality.service import score_two_track_sample
from app.shared.audio import load_audio
from app.shared.compute_api import QualityAnalysisResponse, QualityAnalysisUpload
from app.shared.storage.local import LocalStorageBackend


async def analyze_uploaded_quality(
    request_json: Annotated[str, Form()],
    speaker1: Annotated[UploadFile, File()],
    speaker2: Annotated[UploadFile, File()],
) -> QualityAnalysisResponse:
    request = QualityAnalysisUpload.model_validate_json(request_json)
    with tempfile.TemporaryDirectory(prefix="voice-light-quality-") as directory_name:
        directory = Path(directory_name)
        speaker1_path = directory / "speaker1.audio"
        speaker2_path = directory / "speaker2.audio"
        await write_upload(speaker1, speaker1_path)
        await write_upload(speaker2, speaker2_path)
        storage = LocalStorageBackend()
        speaker1_audio = load_audio(storage, speaker1_path.as_posix())
        speaker2_audio = load_audio(storage, speaker2_path.as_posix())
        quality_result = score_two_track_sample(
            sample_id=request.sample_id,
            speaker1_audio=speaker1_audio,
            speaker2_audio=speaker2_audio,
            speaker1_uri="speaker1",
            speaker2_uri="speaker2",
        )
        return QualityAnalysisResponse(
            speaker1_metadata=(request.speaker1_original_metadata or speaker1_audio.metadata),
            speaker2_metadata=(request.speaker2_original_metadata or speaker2_audio.metadata),
            quality_result=quality_result,
        )


async def write_upload(upload: UploadFile, path: Path) -> None:
    with path.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            output.write(chunk)

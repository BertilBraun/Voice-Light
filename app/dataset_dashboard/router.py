from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from app.audio.wav import capped_wave_bytes
from app.config import DATABASE_URL
from app.db.models import (
    DashboardSample,
    DatasetRecord,
    IngestionJobRecord,
    SampleListFilter,
    TrackSide,
)
from app.db.repository import Repository
from app.ingestion.service import IngestionService

router = APIRouter(prefix="/api/dataset-dashboard", tags=["dataset-dashboard"])


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    datasets: tuple[DatasetRecord, ...]


class SampleListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    samples: tuple[DashboardSample, ...]


class IngestionJobListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    jobs: tuple[IngestionJobRecord, ...]


class LocalIngestionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_name: str
    root_path: str


def repository() -> Repository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for dataset dashboard APIs.")
    return Repository(DATABASE_URL)


@router.get("/datasets")
def list_datasets() -> DatasetListResponse:
    try:
        return DatasetListResponse(datasets=tuple(repository().list_datasets()))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/samples")
def list_samples(
    dataset_id: UUID | None = None,
    quality_min: float | None = None,
    quality_max: float | None = None,
    duration_min: float | None = None,
    duration_max: float | None = None,
    flag: str | None = None,
    speech_ratio_min: float | None = None,
    speech_ratio_max: float | None = None,
    overlap_ratio_min: float | None = None,
    overlap_ratio_max: float | None = None,
    silence_ratio_min: float | None = None,
    silence_ratio_max: float | None = None,
    asr_model: str | None = None,
    wer_min: float | None = None,
    wer_max: float | None = None,
    timestamp_p90_max: float | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> SampleListResponse:
    try:
        sample_filter = SampleListFilter(
            dataset_id=dataset_id,
            quality_min=quality_min,
            quality_max=quality_max,
            duration_min=duration_min,
            duration_max=duration_max,
            flag=flag,
            speech_ratio_min=speech_ratio_min,
            speech_ratio_max=speech_ratio_max,
            overlap_ratio_min=overlap_ratio_min,
            overlap_ratio_max=overlap_ratio_max,
            silence_ratio_min=silence_ratio_min,
            silence_ratio_max=silence_ratio_max,
            asr_model=asr_model,
            wer_min=wer_min,
            wer_max=wer_max,
            timestamp_p90_max=timestamp_p90_max,
            limit=limit,
            offset=offset,
        )
        return SampleListResponse(samples=tuple(repository().list_dashboard_samples(sample_filter)))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/jobs")
def list_jobs(limit: int = Query(default=50, ge=1, le=200)) -> IngestionJobListResponse:
    try:
        return IngestionJobListResponse(jobs=tuple(repository().list_ingestion_jobs(limit)))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/ingest/local")
def ingest_local_dataset(
    request: LocalIngestionRequest, background_tasks: BackgroundTasks
) -> dict[str, str]:
    root_path = Path(request.root_path).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Dataset root does not exist: {request.root_path}"
        )
    ingestion_service = IngestionService(repository())
    background_tasks.add_task(
        ingestion_service.ingest_local_dataset, request.dataset_name, root_path
    )
    return {"status": "queued"}


@router.get("/audio/{sample_id}/{side}")
def sample_audio(
    sample_id: UUID, side: TrackSide, background_tasks: BackgroundTasks
) -> FileResponse:
    try:
        matched_sample = repository().get_dashboard_sample(sample_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    matched_track = next((track for track in matched_sample.tracks if track.side == side), None)
    if matched_track is None:
        raise HTTPException(status_code=404, detail=f"Track not found: {side.value}")
    source_path = Path(matched_track.access_uri)
    playback_wave_path = write_playback_wave_file(source_path)
    background_tasks.add_task(delete_file, playback_wave_path)
    return FileResponse(
        playback_wave_path,
        media_type="audio/wav",
        filename=playback_wave_path.name,
        headers={"Cache-Control": "no-store"},
    )


def write_playback_wave_file(wave_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"{wave_path.stem}_dataset_playback_",
        suffix=".wav",
        delete=False,
    ) as playback_wave_file:
        playback_wave_file.write(capped_wave_bytes(wave_path=wave_path))
        return Path(playback_wave_file.name)


def delete_file(path: Path) -> None:
    path.unlink(missing_ok=True)

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import Field

from app.local.config import DATABASE_URL
from app.local.db.models import (
    ConversationDatasetSummary,
    DashboardSample,
    DatasetRecord,
    IngestionJobRecord,
    SampleListFilter,
    TrackSide,
)
from app.local.db.repository import Repository
from app.local.ingestion.discovery import DatasetLayout
from app.local.ingestion.service import IngestionService
from app.shared.audio.wav import capped_wave_bytes
from app.shared.audio.waveform import capped_waveform_envelope, full_waveform_envelope
from app.shared.base_model import FrozenBaseModel

router = APIRouter(prefix="/api/dataset-dashboard", tags=["dataset-dashboard"])


class DatasetListResponse(FrozenBaseModel):
    datasets: tuple[DatasetRecord, ...]


class SampleListResponse(FrozenBaseModel):
    samples: tuple[DashboardSample, ...]


class IngestionJobListResponse(FrozenBaseModel):
    jobs: tuple[IngestionJobRecord, ...]


class LocalIngestionRequest(FrozenBaseModel):
    dataset_name: str
    root_path: str
    layout: DatasetLayout = DatasetLayout.TWO_AUDIO_FILES
    max_workers: int = Field(default=1, ge=1, le=4)
    max_duration_hours: float | None = Field(default=None, gt=0.0)


class IngestionQueueResponse(FrozenBaseModel):
    status: str


class WaveformPointResponse(FrozenBaseModel):
    minimum_amplitude: float
    maximum_amplitude: float


class WaveformResponse(FrozenBaseModel):
    duration_seconds: float
    sample_rate: int
    points: tuple[WaveformPointResponse, ...]


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


@router.get("/conversation-summary")
def conversation_summary(
    dataset_id: UUID | None = None,
    quality_min: float | None = None,
    overlap_ratio_max: float | None = None,
    flag: str | None = None,
) -> ConversationDatasetSummary:
    try:
        return repository().conversation_dataset_summary(
            SampleListFilter(
                dataset_id=dataset_id,
                quality_min=quality_min,
                overlap_ratio_max=overlap_ratio_max,
                flag=flag,
            )
        )
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
) -> IngestionQueueResponse:
    root_path = Path(request.root_path).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Dataset root does not exist: {request.root_path}"
        )
    ingestion_service = IngestionService(repository())
    background_tasks.add_task(
        ingestion_service.ingest_local_dataset,
        request.dataset_name,
        root_path,
        request.layout,
        request.max_workers,
        request.max_duration_hours,
    )
    return IngestionQueueResponse(status="queued")


@router.get("/audio/{sample_id}/{side}")
def sample_audio(sample_id: UUID, side: TrackSide, trimmed: bool = False) -> Response:
    source_path = sample_track_path(sample_id=sample_id, side=side)
    if trimmed:
        return Response(
            content=capped_wave_bytes(wave_path=source_path),
            media_type="audio/wav",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    return FileResponse(
        source_path,
        media_type="audio/wav",
        filename=source_path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/waveform/{sample_id}/{side}")
def sample_waveform(
    sample_id: UUID,
    side: TrackSide,
    points: int = Query(default=1200, ge=100, le=5000),
    trimmed: bool = False,
) -> WaveformResponse:
    source_path = sample_track_path(sample_id=sample_id, side=side)
    try:
        envelope = (
            capped_waveform_envelope(wave_path=source_path, point_count=points)
            if trimmed
            else full_waveform_envelope(wave_path=source_path, point_count=points)
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return WaveformResponse(
        duration_seconds=envelope.duration_seconds,
        sample_rate=envelope.sample_rate,
        points=tuple(
            WaveformPointResponse(
                minimum_amplitude=point.minimum_amplitude,
                maximum_amplitude=point.maximum_amplitude,
            )
            for point in envelope.points
        ),
    )


def sample_track_path(sample_id: UUID, side: TrackSide) -> Path:
    try:
        matched_sample = repository().get_dashboard_sample(sample_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    matched_track = next((track for track in matched_sample.tracks if track.side == side), None)
    if matched_track is None:
        raise HTTPException(status_code=404, detail=f"Track not found: {side.value}")
    source_path = Path(matched_track.access_uri)
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {source_path}")
    return source_path

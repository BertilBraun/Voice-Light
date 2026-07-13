from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from app.local.config import COMPUTE_TOKEN, REMOTE_QUALITY_ENDPOINT_URL
from app.local.db.models import DatasetCreate, DatasetStorageKind, JobStatus, TrackSide
from app.local.db.repository import (
    AudioMetadataInput,
    QualityResultInput,
    Repository,
    SampleTrackInput,
)
from app.local.ingestion.discovery import DatasetLayout, DiscoveredSample, discover_samples
from app.local.quality.client import HttpRemoteQualityClient, RemoteQualityClient
from app.local.quality.remote_models import (
    AudioSource,
    LocalAudioSource,
    RemoteQualityRequest,
    UriAudioSource,
)
from app.shared.audio import probe_local_audio_metadata
from app.shared.quality import METRIC_VERSION, AudioMetadata, QualityResult
from app.shared.storage.base import StorageBackend
from app.shared.storage.local import LocalStorageBackend


@dataclass(frozen=True)
class ProcessedSample:
    discovered: DiscoveredSample
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    quality_result: QualityResult


class IngestionService:
    def __init__(
        self,
        repository: Repository,
        remote_quality_client: RemoteQualityClient | None = None,
    ) -> None:
        self.repository = repository
        self.remote_quality_client = remote_quality_client or HttpRemoteQualityClient(
            REMOTE_QUALITY_ENDPOINT_URL,
            COMPUTE_TOKEN,
        )

    def ingest_local_dataset(
        self,
        dataset_name: str,
        root: Path,
        layout: DatasetLayout = DatasetLayout.TWO_AUDIO_FILES,
        max_workers: int = 4,
    ) -> None:
        self.ingest_dataset(
            dataset_name=dataset_name,
            root=root.resolve().as_posix(),
            storage=LocalStorageBackend(),
            storage_kind=DatasetStorageKind.LOCAL,
            layout=layout,
            max_workers=max_workers,
        )

    def ingest_dataset(
        self,
        dataset_name: str,
        root: str,
        storage: StorageBackend,
        storage_kind: DatasetStorageKind,
        layout: DatasetLayout,
        max_workers: int,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        job = self.repository.create_ingestion_job(root, f"Queued ingestion for {dataset_name}")
        try:
            dataset = self.repository.upsert_dataset(
                DatasetCreate(
                    name=dataset_name,
                    storage_kind=storage_kind,
                    root_uri=root,
                    description=f"{layout.value} dataset",
                )
            )
            samples = discover_samples(storage, root, layout)
            completed_sample_ids = self.repository.completed_quality_sample_ids(
                dataset.id,
                METRIC_VERSION,
            )
            pending_samples = pending_discovered_samples(samples, completed_sample_ids)
            skipped_samples = len(samples) - len(pending_samples)
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.RUNNING,
                f"Discovered {len(samples)} samples; skipped {skipped_samples} completed samples",
                dataset_id=dataset.id,
                total_samples=len(samples),
                processed_samples=skipped_samples,
            )
            processed_samples = skipped_samples
            failed_samples = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_sample,
                        storage,
                        discovered_sample,
                        self.remote_quality_client,
                    )
                    for discovered_sample in pending_samples
                ]
                for future in as_completed(futures):
                    try:
                        persist_processed_sample(
                            self.repository, dataset.id, storage, future.result()
                        )
                        processed_samples += 1
                    except Exception:
                        failed_samples += 1
                    self.repository.update_ingestion_job(
                        job.id,
                        JobStatus.RUNNING,
                        f"Processed {processed_samples + failed_samples} of {len(samples)} samples",
                        processed_samples=processed_samples,
                        failed_samples=failed_samples,
                    )
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.COMPLETED,
                "Ingestion completed",
                processed_samples=processed_samples,
                failed_samples=failed_samples,
            )
        except Exception as error:
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.FAILED,
                "Ingestion failed",
                error=f"{type(error).__name__}: {error}",
            )


def pending_discovered_samples(
    samples: list[DiscoveredSample],
    completed_sample_ids: set[str],
) -> list[DiscoveredSample]:
    return [sample for sample in samples if sample.external_id not in completed_sample_ids]


def process_sample(
    storage: StorageBackend,
    discovered: DiscoveredSample,
    remote_quality_client: RemoteQualityClient,
) -> ProcessedSample:
    response = remote_quality_client.analyze(
        RemoteQualityRequest(
            sample_id=discovered.external_id,
            speaker1=remote_audio_source(storage, discovered.speaker1_path),
            speaker2=remote_audio_source(storage, discovered.speaker2_path),
        )
    )
    return ProcessedSample(
        discovered=discovered,
        speaker1_metadata=response.speaker1_metadata,
        speaker2_metadata=response.speaker2_metadata,
        quality_result=response.quality_result,
    )


def remote_audio_source(storage: StorageBackend, path: str) -> AudioSource:
    access_uri = storage.access_uri(path)
    if urlparse(access_uri).scheme in {"http", "https"}:
        return UriAudioSource(uri=access_uri)
    return LocalAudioSource(
        filename=Path(path).name,
        path=path,
        original_metadata=probe_local_audio_metadata(Path(path)),
    )


def persist_processed_sample(
    repository: Repository,
    dataset_id: UUID,
    storage: StorageBackend,
    processed: ProcessedSample,
) -> None:
    sample = repository.upsert_sample(dataset_id, processed.discovered.external_id)
    repository.update_sample_duration(
        sample.id,
        min(
            processed.speaker1_metadata.duration_seconds,
            processed.speaker2_metadata.duration_seconds,
        ),
    )
    for side, speaker_index, path, metadata in (
        (TrackSide.SPEAKER1, 1, processed.discovered.speaker1_path, processed.speaker1_metadata),
        (TrackSide.SPEAKER2, 2, processed.discovered.speaker2_path, processed.speaker2_metadata),
    ):
        track = repository.upsert_sample_track(
            sample.id,
            SampleTrackInput(
                side=side,
                speaker_index=speaker_index,
                storage_uri=path,
                access_uri=storage.access_uri(path),
                duration_seconds=metadata.duration_seconds,
                sample_rate=metadata.sample_rate,
                channels=metadata.channels,
                sample_count=metadata.sample_count,
            ),
        )
        repository.upsert_audio_metadata(
            AudioMetadataInput(
                sample_track_id=track.id,
                duration_seconds=metadata.duration_seconds,
                sample_rate=metadata.sample_rate,
                channels=metadata.channels,
                sample_count=metadata.sample_count,
                payload=metadata.model_dump(),
            )
        )
    repository.insert_quality_result(quality_result_input(sample.id, processed.quality_result))


def quality_result_input(sample_id: UUID, quality_result: QualityResult) -> QualityResultInput:
    audio_quality = quality_result.audio_quality
    interaction_density = quality_result.interaction_density
    timing_reliability = quality_result.timing_reliability
    return QualityResultInput(
        sample_id=sample_id,
        metric_version=quality_result.metric_version,
        status=quality_result.status.value,
        total_quality_score=quality_result.total_quality_score,
        raw_quality_score=quality_result.raw_quality_score,
        calibrated_quality_score=quality_result.calibrated_quality_score,
        interaction_density_score=interaction_density.quality_score
        if interaction_density is not None
        else None,
        timing_reliability_score=timing_reliability.quality_score
        if timing_reliability is not None
        else None,
        audio_quality_score=audio_quality.quality_score if audio_quality is not None else None,
        speech_ratio=interaction_density.speech_ratio if interaction_density is not None else None,
        silence_ratio=interaction_density.silence_ratio
        if interaction_density is not None
        else None,
        overlap_ratio=interaction_density.overlap_ratio
        if interaction_density is not None
        else None,
        duration_mismatch_seconds=audio_quality.duration_gap_seconds
        if audio_quality is not None
        else None,
        track_correlation=audio_quality.track_correlation if audio_quality is not None else None,
        energy_envelope_correlation=audio_quality.energy_envelope_correlation
        if audio_quality is not None
        else None,
        flags=quality_result.calibration_flags,
        payload=quality_result.model_dump(mode="json"),
    )

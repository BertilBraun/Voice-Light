from __future__ import annotations

import logging
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
from app.local.ingestion.conversation import analyze_conversation
from app.local.ingestion.discovery import DatasetLayout, DiscoveredSample, discover_samples
from app.local.quality.client import HttpRemoteQualityClient, RemoteQualityClient
from app.local.quality.remote_models import (
    AudioSource,
    LocalAudioSource,
    RemoteQualityRequest,
    UriAudioSource,
)
from app.shared.audio import load_audio, probe_local_audio_metadata
from app.shared.quality import (
    METRIC_VERSION,
    AudioMetadata,
    ProcessingStatus,
    QualityResult,
    RunConfig,
)
from app.shared.quality_analysis.service import score_two_track_sample
from app.shared.storage.base import StorageBackend
from app.shared.storage.local import LocalStorageBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedSample:
    discovered: DiscoveredSample
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    quality_result: QualityResult


@dataclass(frozen=True)
class IngestionSummary:
    status: JobStatus
    message: str
    error: str | None


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
        max_workers: int = 1,
        max_duration_hours: float | None = None,
    ) -> None:
        self.ingest_dataset(
            dataset_name=dataset_name,
            root=root.resolve().as_posix(),
            storage=LocalStorageBackend(),
            storage_kind=DatasetStorageKind.LOCAL,
            layout=layout,
            max_workers=max_workers,
            max_duration_hours=max_duration_hours,
        )

    def ingest_dataset(
        self,
        dataset_name: str,
        root: str,
        storage: StorageBackend,
        storage_kind: DatasetStorageKind,
        layout: DatasetLayout,
        max_workers: int,
        max_duration_hours: float | None = None,
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
            selected_samples = select_duration_limited_samples(
                storage=storage,
                samples=pending_samples,
                max_duration_hours=max_duration_hours,
            )
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.RUNNING,
                (
                    f"Discovered {len(samples)} samples; skipped {skipped_samples} completed "
                    f"samples; selected {len(selected_samples)} samples"
                ),
                dataset_id=dataset.id,
                total_samples=len(selected_samples),
                processed_samples=0,
            )
            processed_samples = 0
            failed_samples = 0
            sample_errors: list[str] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_local_sample,
                        storage,
                        discovered_sample,
                        self.repository.database_url,
                    ): discovered_sample
                    for discovered_sample in selected_samples
                }
                for future in as_completed(futures):
                    discovered_sample = futures[future]
                    try:
                        persist_processed_sample(
                            self.repository, dataset.id, storage, future.result()
                        )
                        processed_samples += 1
                    except Exception as error:
                        failed_samples += 1
                        sample_error = (
                            f"{discovered_sample.external_id}: {type(error).__name__}: {error}"
                        )
                        sample_errors.append(sample_error)
                        logger.exception(
                            "ingestion sample failed: %s",
                            discovered_sample.external_id,
                        )
                    self.repository.update_ingestion_job(
                        job.id,
                        JobStatus.RUNNING,
                        (
                            f"Processed {processed_samples + failed_samples} of "
                            f"{len(selected_samples)} samples"
                        ),
                        processed_samples=processed_samples,
                        failed_samples=failed_samples,
                    )
            summary = ingestion_summary(
                processed_samples=processed_samples,
                failed_samples=failed_samples,
                sample_errors=tuple(sample_errors),
            )
            self.repository.update_ingestion_job(
                job.id,
                summary.status,
                summary.message,
                processed_samples=processed_samples,
                failed_samples=failed_samples,
                error=summary.error,
            )
        except Exception as error:
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.FAILED,
                "Ingestion failed",
                error=f"{type(error).__name__}: {error}",
            )


def ingestion_summary(
    processed_samples: int,
    failed_samples: int,
    sample_errors: tuple[str, ...],
) -> IngestionSummary:
    assert processed_samples >= 0
    assert failed_samples >= 0
    assert len(sample_errors) == failed_samples
    if failed_samples:
        total_samples = processed_samples + failed_samples
        return IngestionSummary(
            status=JobStatus.FAILED,
            message=f"Ingestion failed for {failed_samples} of {total_samples} samples",
            error="\n".join(sample_errors),
        )
    return IngestionSummary(
        status=JobStatus.COMPLETED,
        message="Ingestion completed",
        error=None,
    )


def pending_discovered_samples(
    samples: list[DiscoveredSample],
    completed_sample_ids: set[str],
) -> list[DiscoveredSample]:
    return [sample for sample in samples if sample.external_id not in completed_sample_ids]


def select_duration_limited_samples(
    storage: StorageBackend,
    samples: list[DiscoveredSample],
    max_duration_hours: float | None,
) -> list[DiscoveredSample]:
    if max_duration_hours is None:
        return samples
    if max_duration_hours <= 0.0:
        raise ValueError("max_duration_hours must be positive")
    maximum_duration_seconds = max_duration_hours * 3600.0
    selected_samples: list[DiscoveredSample] = []
    selected_duration_seconds = 0.0
    for sample in samples:
        speaker1_path = local_storage_path(storage, sample.speaker1_path)
        speaker2_path = local_storage_path(storage, sample.speaker2_path)
        duration_seconds = min(
            probe_local_audio_metadata(speaker1_path).duration_seconds,
            probe_local_audio_metadata(speaker2_path).duration_seconds,
        )
        if (
            selected_samples
            and selected_duration_seconds + duration_seconds > maximum_duration_seconds
        ):
            break
        selected_samples.append(sample)
        selected_duration_seconds += duration_seconds
    return selected_samples


def process_local_sample(
    storage: StorageBackend,
    discovered: DiscoveredSample,
    database_url: str,
) -> ProcessedSample:
    speaker1_path = local_storage_path(storage, discovered.speaker1_path)
    speaker2_path = local_storage_path(storage, discovered.speaker2_path)
    speaker1_metadata = probe_local_audio_metadata(speaker1_path)
    speaker2_metadata = probe_local_audio_metadata(speaker2_path)
    duration_gap_seconds = abs(
        speaker1_metadata.duration_seconds - speaker2_metadata.duration_seconds
    )
    maximum_duration_seconds = max(
        speaker1_metadata.duration_seconds, speaker2_metadata.duration_seconds
    )
    duration_gap_ratio = (
        duration_gap_seconds / maximum_duration_seconds if maximum_duration_seconds > 0.0 else 0.0
    )
    if duration_gap_ratio > 0.01:
        return ProcessedSample(
            discovered=discovered,
            speaker1_metadata=speaker1_metadata,
            speaker2_metadata=speaker2_metadata,
            quality_result=invalid_duration_quality_result(
                discovered=discovered,
                duration_seconds=min(
                    speaker1_metadata.duration_seconds,
                    speaker2_metadata.duration_seconds,
                ),
                duration_gap_seconds=duration_gap_seconds,
                duration_gap_ratio=duration_gap_ratio,
            ),
        )
    annotation = analyze_conversation(
        sample_id=discovered.external_id,
        speaker1_path=speaker1_path,
        speaker2_path=speaker2_path,
        database_url=database_url,
    )
    speaker1_audio = load_audio(storage, discovered.speaker1_path, target_sample_rate=16_000)
    speaker2_audio = load_audio(storage, discovered.speaker2_path, target_sample_rate=16_000)
    quality_result = score_two_track_sample(
        sample_id=discovered.external_id,
        speaker1_audio=speaker1_audio,
        speaker2_audio=speaker2_audio,
        speaker1_uri=storage.access_uri(discovered.speaker1_path),
        speaker2_uri=storage.access_uri(discovered.speaker2_path),
        conversation_annotation=annotation,
    )
    return ProcessedSample(
        discovered=discovered,
        speaker1_metadata=speaker1_metadata,
        speaker2_metadata=speaker2_metadata,
        quality_result=quality_result,
    )


def local_storage_path(storage: StorageBackend, path: str) -> Path:
    access_uri = storage.access_uri(path)
    source_path = Path(access_uri)
    if source_path.is_file():
        return source_path
    if urlparse(access_uri).scheme in {"http", "https", "s3", "gs"}:
        raise ValueError("Local quality analysis requires locally materialized audio files.")
    raise ValueError(f"Audio file not found: {source_path}")


def invalid_duration_quality_result(
    discovered: DiscoveredSample,
    duration_seconds: float,
    duration_gap_seconds: float,
    duration_gap_ratio: float,
) -> QualityResult:
    return QualityResult(
        metric_version=RunConfig().metric_version,
        sample_id=discovered.external_id,
        status=ProcessingStatus.INVALID,
        speaker1_uri=discovered.speaker1_path,
        speaker2_uri=discovered.speaker2_path,
        duration_seconds=duration_seconds,
        interaction_density=None,
        timing_reliability=None,
        audio_quality=None,
        conversation_annotation=None,
        event_candidates=(),
        raw_quality_score=0.0,
        calibrated_quality_score=0.0,
        calibration_flags=(
            "duration_mismatch_invalid",
            f"duration_gap_seconds:{duration_gap_seconds:.3f}",
            f"duration_gap_ratio:{duration_gap_ratio:.6f}",
        ),
        total_quality_score=0.0,
        error=None,
    )


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
    conversation_annotation = quality_result.conversation_annotation
    flags = list(quality_result.calibration_flags)
    if audio_quality is not None:
        flags.extend(flag for flag in audio_quality.flags if flag not in flags)
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
        conversation_quality_score=conversation_annotation.quality_score
        if conversation_annotation is not None
        else None,
        interaction_count=conversation_annotation.interaction_count
        if conversation_annotation is not None
        else None,
        speech_segment_count=conversation_annotation.speech_segment_count
        if conversation_annotation is not None
        else None,
        turn_count=conversation_annotation.turn_count
        if conversation_annotation is not None
        else None,
        turn_taking_count=conversation_annotation.turn_taking_count
        if conversation_annotation is not None
        else None,
        pause_count=conversation_annotation.pause_count
        if conversation_annotation is not None
        else None,
        backchannel_count=conversation_annotation.backchannel_count
        if conversation_annotation is not None
        else None,
        interruption_count=conversation_annotation.interruption_count
        if conversation_annotation is not None
        else None,
        usable_event_count=conversation_annotation.usable_event_count
        if conversation_annotation is not None
        else None,
        conversation_events_per_hour=conversation_annotation.events_per_hour
        if conversation_annotation is not None
        else None,
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
        flags=tuple(flags),
        payload=quality_result.model_dump(mode="json"),
    )

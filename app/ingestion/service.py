from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.db.models import DatasetCreate, DatasetStorageKind, JobStatus, TrackSide
from app.db.repository import AudioMetadataInput, QualityResultInput, Repository, SampleTrackInput
from app.ingestion.discovery import discover_two_track_samples
from app.quality.audio import load_audio
from app.quality.models import QualityResult
from app.quality.service import score_two_track_sample
from app.storage.local import LocalStorageBackend


class IngestionService:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def ingest_local_dataset(self, dataset_name: str, root: Path) -> None:
        job = self.repository.create_ingestion_job(
            str(root), f"Queued ingestion for {dataset_name}"
        )
        storage = LocalStorageBackend()
        try:
            dataset = self.repository.upsert_dataset(
                DatasetCreate(
                    name=dataset_name,
                    storage_kind=DatasetStorageKind.LOCAL,
                    root_uri=str(root.resolve()),
                    description="Local two-track dataset",
                )
            )
            samples = discover_two_track_samples(storage, root)
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.RUNNING,
                f"Discovered {len(samples)} samples",
                dataset_id=dataset.id,
                total_samples=len(samples),
            )
            processed_samples = 0
            failed_samples = 0
            for discovered_sample in samples:
                try:
                    sample = self.repository.upsert_sample(
                        dataset.id, discovered_sample.external_id
                    )
                    speaker1_metadata = load_audio(discovered_sample.speaker1_path).metadata
                    speaker2_metadata = load_audio(discovered_sample.speaker2_path).metadata
                    speaker1_track = self.repository.upsert_sample_track(
                        sample.id,
                        SampleTrackInput(
                            side=TrackSide.SPEAKER1,
                            speaker_index=1,
                            storage_uri=str(discovered_sample.speaker1_path),
                            access_uri=storage.access_uri(str(discovered_sample.speaker1_path)),
                            duration_seconds=speaker1_metadata.duration_seconds,
                            sample_rate=speaker1_metadata.sample_rate,
                            channels=speaker1_metadata.channels,
                            sample_count=speaker1_metadata.sample_count,
                        ),
                    )
                    speaker2_track = self.repository.upsert_sample_track(
                        sample.id,
                        SampleTrackInput(
                            side=TrackSide.SPEAKER2,
                            speaker_index=2,
                            storage_uri=str(discovered_sample.speaker2_path),
                            access_uri=storage.access_uri(str(discovered_sample.speaker2_path)),
                            duration_seconds=speaker2_metadata.duration_seconds,
                            sample_rate=speaker2_metadata.sample_rate,
                            channels=speaker2_metadata.channels,
                            sample_count=speaker2_metadata.sample_count,
                        ),
                    )
                    self.repository.upsert_audio_metadata(
                        AudioMetadataInput(
                            sample_track_id=speaker1_track.id,
                            duration_seconds=speaker1_metadata.duration_seconds,
                            sample_rate=speaker1_metadata.sample_rate,
                            channels=speaker1_metadata.channels,
                            sample_count=speaker1_metadata.sample_count,
                            payload=speaker1_metadata.model_dump(),
                        )
                    )
                    self.repository.upsert_audio_metadata(
                        AudioMetadataInput(
                            sample_track_id=speaker2_track.id,
                            duration_seconds=speaker2_metadata.duration_seconds,
                            sample_rate=speaker2_metadata.sample_rate,
                            channels=speaker2_metadata.channels,
                            sample_count=speaker2_metadata.sample_count,
                            payload=speaker2_metadata.model_dump(),
                        )
                    )
                    quality_result = score_two_track_sample(
                        discovered_sample.external_id,
                        discovered_sample.speaker1_path,
                        discovered_sample.speaker2_path,
                    )
                    self.repository.insert_quality_result(
                        quality_result_input(sample.id, quality_result)
                    )
                    processed_samples += 1
                except Exception:
                    failed_samples += 1
                self.repository.update_ingestion_job(
                    job.id,
                    JobStatus.RUNNING,
                    f"Processed {processed_samples} of {len(samples)} samples",
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

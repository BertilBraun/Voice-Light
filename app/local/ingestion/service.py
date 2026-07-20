from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse
from uuid import UUID

from app.local.asr.client import HttpRemoteAsrClient
from app.local.asr.full_recording_models import (
    FullRecordingAsrTranscriptBundle,
    FullRecordingAsrTranscriptPair,
)
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.full_recording_service import (
    FullRecordingAsrTrack,
    RemoteAsrClientFactory,
    sha256_file,
    transcribe_full_recording_track,
)
from app.local.config import (
    COMPUTE_TOKEN,
    REMOTE_ASR_ENDPOINT_URL,
    REMOTE_QUALITY_ENDPOINT_URL,
)
from app.local.db.models import (
    DatasetCreate,
    DatasetStorageKind,
    JobStatus,
    SampleTrackRecord,
    TrackSide,
)
from app.local.db.repository import (
    AudioMetadataInput,
    QualityResultInput,
    Repository,
    SampleTrackInput,
)
from app.local.ingestion.alignment import pad_audio_tracks_to_shared_timeline
from app.local.ingestion.conversation import analyze_conversation
from app.local.ingestion.discovery import (
    DatasetLayout,
    DiscoveredSample,
    discover_samples,
    track_durations_are_compatible,
)
from app.local.ingestion.language import (
    LanguagePreflightService,
    LanguageTrack,
    RemoteWhisperLanguageProbeTranscriber,
)
from app.local.ingestion.language_repository import LanguageAssessmentRepository
from app.local.quality.client import HttpRemoteQualityClient, RemoteQualityClient
from app.local.quality.remote_models import (
    AudioSource,
    LocalAudioSource,
    RemoteQualityRequest,
    UriAudioSource,
)
from app.shared.asr import AsrModelId
from app.shared.audio import load_audio, probe_local_audio_metadata
from app.shared.quality import (
    METRIC_VERSION,
    AudioMetadata,
    QualityResult,
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
class RegisteredSample:
    discovered: DiscoveredSample
    sample_id: UUID
    speaker1_track_id: UUID
    speaker2_track_id: UUID
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    speaker1_source_audio_sha256: str
    speaker2_source_audio_sha256: str


@dataclass(frozen=True)
class ReadySample:
    registered: RegisteredSample
    transcripts: FullRecordingAsrTranscriptBundle


@dataclass(frozen=True)
class ProcessedReadySample:
    ready: ReadySample
    quality_result: QualityResult


@dataclass(frozen=True)
class QualityResultProvenance:
    speaker1_parakeet_full_asr_transcript_id: UUID
    speaker2_parakeet_full_asr_transcript_id: UUID
    speaker1_canary_full_asr_transcript_id: UUID
    speaker2_canary_full_asr_transcript_id: UUID


class FullTranscriptPairRepository(Protocol):
    def get_exact_transcript_pair(
        self,
        sample_id: UUID,
        speaker1_track_id: UUID,
        speaker1_source_audio_sha256: str,
        speaker2_track_id: UUID,
        speaker2_source_audio_sha256: str,
        model_id: AsrModelId,
    ) -> FullRecordingAsrTranscriptPair | None: ...


@dataclass(frozen=True)
class IngestionSummary:
    status: JobStatus
    message: str
    error: str | None


class SampleProcessingStatus(StrEnum):
    ANALYZED = "analyzed"
    LANGUAGE_EXCLUDED = "language_excluded"


@dataclass(frozen=True)
class DurationValidatedSamples:
    accepted: list[DiscoveredSample]
    excluded: list[DiscoveredSample]


class IngestionService:
    def __init__(
        self,
        repository: Repository,
        remote_quality_client: RemoteQualityClient | None = None,
        remote_asr_client_factory: RemoteAsrClientFactory | None = None,
        language_preflight_service: LanguagePreflightService | None = None,
    ) -> None:
        self.repository = repository
        self.remote_quality_client = remote_quality_client or HttpRemoteQualityClient(
            REMOTE_QUALITY_ENDPOINT_URL,
            COMPUTE_TOKEN,
        )
        self.remote_asr_client_factory = (
            remote_asr_client_factory or default_remote_asr_client_factory
        )
        self.language_preflight_service = (
            language_preflight_service or self.default_language_preflight_service()
        )

    def default_language_preflight_service(self) -> LanguagePreflightService:
        return LanguagePreflightService(
            store=LanguageAssessmentRepository(self.repository.database_url),
            transcriber=RemoteWhisperLanguageProbeTranscriber(
                remote_client=self.remote_asr_client_factory()
            ),
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
            duration_validated = duration_validated_samples(
                storage=storage,
                samples=samples,
            )
            completed_sample_ids = self.repository.completed_quality_sample_ids(
                dataset.id,
                METRIC_VERSION,
            )
            pending_samples = pending_discovered_samples(
                duration_validated.accepted,
                completed_sample_ids,
            )
            skipped_samples = len(duration_validated.accepted) - len(pending_samples)
            selected_samples = select_duration_limited_samples(
                storage=storage,
                samples=pending_samples,
                max_duration_hours=max_duration_hours,
            )
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.RUNNING,
                (
                    f"Discovered {len(samples)} samples; excluded "
                    f"{len(duration_validated.excluded)} duration-mismatched samples; skipped "
                    f"{skipped_samples} completed samples; selected {len(selected_samples)} samples"
                ),
                dataset_id=dataset.id,
                total_samples=len(selected_samples),
                processed_samples=0,
            )
            processed_samples = 0
            analyzed_samples = 0
            language_excluded_samples = 0
            failed_samples = 0
            sample_errors: list[str] = []
            full_asr_repository = FullRecordingAsrRepository(
                database_url=self.repository.database_url
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_ingestion_sample,
                        self.repository,
                        full_asr_repository,
                        dataset.id,
                        storage,
                        discovered_sample,
                        self.remote_asr_client_factory,
                        self.language_preflight_service,
                    ): discovered_sample
                    for discovered_sample in selected_samples
                }
                for future in as_completed(futures):
                    discovered_sample = futures[future]
                    try:
                        processing_status = future.result()
                        processed_samples += 1
                        match processing_status:
                            case SampleProcessingStatus.ANALYZED:
                                analyzed_samples += 1
                            case SampleProcessingStatus.LANGUAGE_EXCLUDED:
                                language_excluded_samples += 1
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
                            f"Assessed {processed_samples + failed_samples} of "
                            f"{len(selected_samples)} samples; analyzed {analyzed_samples}; "
                            f"language-excluded {language_excluded_samples}"
                        ),
                        processed_samples=processed_samples,
                        failed_samples=failed_samples,
                    )
            summary = ingestion_summary(
                processed_samples=processed_samples,
                analyzed_samples=analyzed_samples,
                language_excluded_samples=language_excluded_samples,
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
    analyzed_samples: int,
    language_excluded_samples: int,
    failed_samples: int,
    sample_errors: tuple[str, ...],
) -> IngestionSummary:
    assert processed_samples >= 0
    assert analyzed_samples >= 0
    assert language_excluded_samples >= 0
    assert failed_samples >= 0
    assert processed_samples == analyzed_samples + language_excluded_samples
    assert len(sample_errors) == failed_samples
    if failed_samples:
        total_samples = processed_samples + failed_samples
        return IngestionSummary(
            status=JobStatus.FAILED,
            message=(
                f"Ingestion failed for {failed_samples} of {total_samples} samples; "
                f"analyzed {analyzed_samples}; language-excluded {language_excluded_samples}"
            ),
            error="\n".join(sample_errors),
        )
    return IngestionSummary(
        status=JobStatus.COMPLETED,
        message=(
            f"Ingestion completed; analyzed {analyzed_samples}; "
            f"language-excluded {language_excluded_samples}"
        ),
        error=None,
    )


def pending_discovered_samples(
    samples: list[DiscoveredSample],
    completed_sample_ids: set[str],
) -> list[DiscoveredSample]:
    return [sample for sample in samples if sample.external_id not in completed_sample_ids]


def duration_validated_samples(
    storage: StorageBackend,
    samples: list[DiscoveredSample],
) -> DurationValidatedSamples:
    accepted: list[DiscoveredSample] = []
    excluded: list[DiscoveredSample] = []
    for sample in samples:
        speaker1_duration_seconds = probe_local_audio_metadata(
            local_storage_path(storage, sample.speaker1_path)
        ).duration_seconds
        speaker2_duration_seconds = probe_local_audio_metadata(
            local_storage_path(storage, sample.speaker2_path)
        ).duration_seconds
        target = (
            accepted
            if track_durations_are_compatible(
                speaker1_duration_seconds=speaker1_duration_seconds,
                speaker2_duration_seconds=speaker2_duration_seconds,
            )
            else excluded
        )
        target.append(sample)
    return DurationValidatedSamples(accepted=accepted, excluded=excluded)


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


def register_sample(
    repository: Repository,
    dataset_id: UUID,
    storage: StorageBackend,
    discovered: DiscoveredSample,
) -> RegisteredSample:
    speaker1_path = local_storage_path(storage, discovered.speaker1_path)
    speaker2_path = local_storage_path(storage, discovered.speaker2_path)
    speaker1_metadata = probe_local_audio_metadata(speaker1_path)
    speaker2_metadata = probe_local_audio_metadata(speaker2_path)
    speaker1_source_audio_sha256 = sha256_file(speaker1_path)
    speaker2_source_audio_sha256 = sha256_file(speaker2_path)
    sample = repository.upsert_sample(dataset_id, discovered.external_id)
    repository.update_sample_duration(
        sample.id,
        max(speaker1_metadata.duration_seconds, speaker2_metadata.duration_seconds),
    )
    speaker1_track = register_track(
        repository=repository,
        sample_id=sample.id,
        storage=storage,
        side=TrackSide.SPEAKER1,
        speaker_index=1,
        path=discovered.speaker1_path,
        metadata=speaker1_metadata,
        audio_sha256=speaker1_source_audio_sha256,
    )
    speaker2_track = register_track(
        repository=repository,
        sample_id=sample.id,
        storage=storage,
        side=TrackSide.SPEAKER2,
        speaker_index=2,
        path=discovered.speaker2_path,
        metadata=speaker2_metadata,
        audio_sha256=speaker2_source_audio_sha256,
    )
    return RegisteredSample(
        discovered=discovered,
        sample_id=sample.id,
        speaker1_track_id=speaker1_track.id,
        speaker2_track_id=speaker2_track.id,
        speaker1_metadata=speaker1_metadata,
        speaker2_metadata=speaker2_metadata,
        speaker1_source_audio_sha256=speaker1_source_audio_sha256,
        speaker2_source_audio_sha256=speaker2_source_audio_sha256,
    )


def register_track(
    repository: Repository,
    sample_id: UUID,
    storage: StorageBackend,
    side: TrackSide,
    speaker_index: int,
    path: str,
    metadata: AudioMetadata,
    audio_sha256: str,
) -> SampleTrackRecord:
    track = repository.upsert_sample_track(
        sample_id,
        SampleTrackInput(
            side=side,
            speaker_index=speaker_index,
            storage_uri=path,
            access_uri=storage.access_uri(path),
            duration_seconds=metadata.duration_seconds,
            sample_rate=metadata.sample_rate,
            channels=metadata.channels,
            sample_count=metadata.sample_count,
            audio_sha256=audio_sha256,
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
    return track


def ready_sample(
    full_asr_repository: FullTranscriptPairRepository,
    registered: RegisteredSample,
) -> ReadySample:
    parakeet = full_asr_repository.get_exact_transcript_pair(
        sample_id=registered.sample_id,
        speaker1_track_id=registered.speaker1_track_id,
        speaker1_source_audio_sha256=registered.speaker1_source_audio_sha256,
        speaker2_track_id=registered.speaker2_track_id,
        speaker2_source_audio_sha256=registered.speaker2_source_audio_sha256,
        model_id=AsrModelId.PARAKEET_TDT,
    )
    canary = full_asr_repository.get_exact_transcript_pair(
        sample_id=registered.sample_id,
        speaker1_track_id=registered.speaker1_track_id,
        speaker1_source_audio_sha256=registered.speaker1_source_audio_sha256,
        speaker2_track_id=registered.speaker2_track_id,
        speaker2_source_audio_sha256=registered.speaker2_source_audio_sha256,
        model_id=AsrModelId.CANARY,
    )
    assert parakeet is not None
    assert canary is not None
    return ReadySample(
        registered=registered,
        transcripts=FullRecordingAsrTranscriptBundle(
            parakeet=parakeet,
            canary=canary,
        ),
    )


def process_ingestion_sample(
    repository: Repository,
    full_asr_repository: FullRecordingAsrRepository,
    dataset_id: UUID,
    storage: StorageBackend,
    discovered: DiscoveredSample,
    remote_asr_client_factory: RemoteAsrClientFactory,
    language_preflight_service: LanguagePreflightService,
) -> SampleProcessingStatus:
    registered = register_sample(
        repository=repository,
        dataset_id=dataset_id,
        storage=storage,
        discovered=discovered,
    )
    language_assessment = language_preflight_service.assess_sample(
        speaker1=LanguageTrack(
            sample_track_id=registered.speaker1_track_id,
            side=TrackSide.SPEAKER1,
            source_path=local_storage_path(storage, discovered.speaker1_path),
            source_audio_sha256=registered.speaker1_source_audio_sha256,
        ),
        speaker2=LanguageTrack(
            sample_track_id=registered.speaker2_track_id,
            side=TrackSide.SPEAKER2,
            source_path=local_storage_path(storage, discovered.speaker2_path),
            source_audio_sha256=registered.speaker2_source_audio_sha256,
        ),
    )
    if not language_assessment.allows_full_analysis:
        return SampleProcessingStatus.LANGUAGE_EXCLUDED
    for track in (
        FullRecordingAsrTrack(
            sample_track_id=registered.speaker1_track_id,
            access_uri=storage.access_uri(discovered.speaker1_path),
            models=(AsrModelId.PARAKEET_TDT, AsrModelId.CANARY),
            source_audio_sha256=registered.speaker1_source_audio_sha256,
        ),
        FullRecordingAsrTrack(
            sample_track_id=registered.speaker2_track_id,
            access_uri=storage.access_uri(discovered.speaker2_path),
            models=(AsrModelId.PARAKEET_TDT, AsrModelId.CANARY),
            source_audio_sha256=registered.speaker2_source_audio_sha256,
        ),
    ):
        transcribe_full_recording_track(
            track=track,
            store=full_asr_repository,
            remote_client_factory=remote_asr_client_factory,
        )
    ready = ready_sample(
        full_asr_repository=full_asr_repository,
        registered=registered,
    )
    persist_processed_ready_sample(
        repository=repository,
        processed=process_ready_sample(storage=storage, ready=ready),
    )
    return SampleProcessingStatus.ANALYZED


def default_remote_asr_client_factory() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )


def process_ready_sample(
    storage: StorageBackend,
    ready: ReadySample,
) -> ProcessedReadySample:
    discovered = ready.registered.discovered
    speaker1_audio = load_audio(storage, discovered.speaker1_path, target_sample_rate=16_000)
    speaker2_audio = load_audio(storage, discovered.speaker2_path, target_sample_rate=16_000)
    shared_timeline = pad_audio_tracks_to_shared_timeline(
        speaker1=speaker1_audio,
        speaker2=speaker2_audio,
    )
    annotation = analyze_conversation(
        speaker1_audio=shared_timeline.speaker1,
        speaker2_audio=shared_timeline.speaker2,
        transcripts=ready.transcripts,
    )
    quality_result = score_two_track_sample(
        sample_id=discovered.external_id,
        speaker1_audio=shared_timeline.speaker1,
        speaker2_audio=shared_timeline.speaker2,
        speaker1_uri=storage.access_uri(discovered.speaker1_path),
        speaker2_uri=storage.access_uri(discovered.speaker2_path),
        conversation_annotation=annotation,
    )
    return ProcessedReadySample(
        ready=ready,
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
                audio_sha256=sha256_file(
                    local_storage_path(storage, path),
                ),
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
    repository.insert_quality_result(
        quality_result_input(
            sample_id=sample.id,
            quality_result=processed.quality_result,
            provenance=None,
        )
    )


def persist_processed_ready_sample(
    repository: Repository,
    processed: ProcessedReadySample,
) -> None:
    ready = processed.ready
    repository.insert_quality_result(
        quality_result_input(
            sample_id=ready.registered.sample_id,
            quality_result=processed.quality_result,
            provenance=QualityResultProvenance(
                speaker1_parakeet_full_asr_transcript_id=(ready.transcripts.parakeet.speaker1.id),
                speaker2_parakeet_full_asr_transcript_id=(ready.transcripts.parakeet.speaker2.id),
                speaker1_canary_full_asr_transcript_id=ready.transcripts.canary.speaker1.id,
                speaker2_canary_full_asr_transcript_id=ready.transcripts.canary.speaker2.id,
            ),
        )
    )


def quality_result_input(
    sample_id: UUID,
    quality_result: QualityResult,
    provenance: QualityResultProvenance | None,
) -> QualityResultInput:
    audio_quality = quality_result.audio_quality
    interaction_density = quality_result.interaction_density
    timing_reliability = quality_result.timing_reliability
    conversation_annotation = quality_result.conversation_annotation
    conversation_count_estimate = quality_result.conversation_count_estimate
    estimated_counts = (
        conversation_count_estimate.estimated if conversation_count_estimate is not None else None
    )
    flags = list(quality_result.quality_flags)
    if audio_quality is not None:
        flags.extend(flag for flag in audio_quality.flags if flag not in flags)
    return QualityResultInput(
        sample_id=sample_id,
        metric_version=quality_result.metric_version,
        status=quality_result.status.value,
        total_quality_score=quality_result.total_quality_score,
        raw_quality_score=quality_result.raw_quality_score,
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
        annotation_duration_seconds=conversation_count_estimate.annotation_duration_seconds
        if conversation_count_estimate is not None
        else None,
        represented_duration_seconds=conversation_count_estimate.represented_duration_seconds
        if conversation_count_estimate is not None
        else None,
        estimated_speech_segment_count=estimated_counts.speech_segment_count
        if estimated_counts is not None
        else None,
        estimated_interaction_count=estimated_counts.interaction_count
        if estimated_counts is not None
        else None,
        estimated_turn_count=estimated_counts.turn_count if estimated_counts is not None else None,
        estimated_turn_taking_count=estimated_counts.turn_taking_count
        if estimated_counts is not None
        else None,
        estimated_pause_count=estimated_counts.pause_count
        if estimated_counts is not None
        else None,
        estimated_backchannel_count=estimated_counts.backchannel_count
        if estimated_counts is not None
        else None,
        estimated_interruption_count=estimated_counts.interruption_count
        if estimated_counts is not None
        else None,
        estimated_usable_event_count=estimated_counts.usable_event_count
        if estimated_counts is not None
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
        speaker1_parakeet_full_asr_transcript_id=(
            provenance.speaker1_parakeet_full_asr_transcript_id
        )
        if provenance is not None
        else None,
        speaker2_parakeet_full_asr_transcript_id=(
            provenance.speaker2_parakeet_full_asr_transcript_id
        )
        if provenance is not None
        else None,
        speaker1_canary_full_asr_transcript_id=(provenance.speaker1_canary_full_asr_transcript_id)
        if provenance is not None
        else None,
        speaker2_canary_full_asr_transcript_id=(provenance.speaker2_canary_full_asr_transcript_id)
        if provenance is not None
        else None,
        flags=tuple(flags),
        payload=quality_result.model_dump(mode="json"),
    )

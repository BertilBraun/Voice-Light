from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlparse
from uuid import UUID

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.full_recording_repository import (
    FullRecordingAsrRepository,
    transcript_result_from_record,
)
from app.local.asr.full_recording_service import (
    missing_model_ids,
    requested_results,
    validate_transcript_timestamps,
)
from app.local.config import COMPUTE_BASE_URL, COMPUTE_TOKEN
from app.local.conversation_regions.models import ConversationRegionConfig
from app.local.conversation_regions.repository import ConversationRegionRepository
from app.local.conversation_regions.service import analyze_conversation_regions
from app.local.db.models import (
    DatasetCreate,
    DatasetStorageKind,
    JobStatus,
    SampleTrackRecord,
    TrackLanguageAssessment,
    TrackLanguageStatus,
    TrackSide,
)
from app.local.db.repository import AudioMetadataInput, Repository, SampleTrackInput
from app.local.ingestion.conversation import analyze_conversation_from_remote_evidence
from app.local.ingestion.discovery import DiscoveredSample
from app.local.ingestion.language import (
    SampleLanguageAssessment,
    sample_language_status,
)
from app.local.ingestion.language_repository import LanguageAssessmentRepository
from app.local.ingestion.manifest import (
    ManifestDiscoveredSample,
    discover_manifest_samples,
    read_meetings_manifest,
)
from app.local.ingestion.remote_client import HttpDatasetAnalysisClient
from app.local.ingestion.service import (
    ProcessedReadySample,
    RegisteredSample,
    SampleProcessingStatus,
    ingestion_summary,
    persist_processed_ready_sample,
    ready_sample,
)
from app.local.ingestion.vad_repository import VadRepository
from app.shared.asr import AsrModelId
from app.shared.audio.s3 import S3AudioSource
from app.shared.compute_api import (
    DatasetAsrRequest,
    DatasetAsrResponse,
    DatasetLanguageRequest,
    DatasetLanguageResponse,
    DatasetLanguageTrackResponse,
    DatasetQualityRequest,
    DatasetQualityResponse,
    DatasetTrackTranscripts,
    MaterializedAudio,
)
from app.shared.language import LANGUAGE_ASSESSMENT_VERSION
from app.shared.quality import METRIC_VERSION, AudioMetadata
from app.shared.quality_analysis.service import quality_result_with_conversation_annotation

logger = logging.getLogger(__name__)
FULL_RECORDING_MODELS = (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY)


class DatasetAnalysisClient(Protocol):
    def assess_language(self, request: DatasetLanguageRequest) -> DatasetLanguageResponse: ...

    def transcribe(self, request: DatasetAsrRequest) -> DatasetAsrResponse: ...

    def analyze_quality(self, request: DatasetQualityRequest) -> DatasetQualityResponse: ...


@dataclass(frozen=True)
class ManifestRegisteredSample:
    discovered: ManifestDiscoveredSample
    sample_id: UUID
    speaker1_track: SampleTrackRecord
    speaker2_track: SampleTrackRecord


@dataclass(frozen=True)
class ManifestLanguageReady:
    discovered: ManifestDiscoveredSample
    registered: RegisteredSample
    assessment: SampleLanguageAssessment


class ManifestIngestionService:
    def __init__(
        self,
        repository: Repository,
        analysis_client: DatasetAnalysisClient,
    ) -> None:
        self.repository = repository
        self.analysis_client = analysis_client
        self.full_asr_repository = FullRecordingAsrRepository(repository.database_url)
        self.language_repository = LanguageAssessmentRepository(repository.database_url)
        self.vad_repository = VadRepository(repository.database_url)
        self.conversation_region_repository = ConversationRegionRepository(repository.database_url)

    @classmethod
    def create_default(cls, repository: Repository) -> ManifestIngestionService:
        return cls(
            repository=repository,
            analysis_client=HttpDatasetAnalysisClient(
                compute_base_url=COMPUTE_BASE_URL,
                api_key=COMPUTE_TOKEN,
            ),
        )

    def ingest(
        self,
        dataset_name: str,
        manifest_path: Path,
        max_workers: int,
        max_duration_hours: float | None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        resolved_manifest_path = manifest_path.resolve()
        job = self.repository.create_ingestion_job(
            resolved_manifest_path.as_posix(),
            f"Queued manifest ingestion for {dataset_name}",
        )
        try:
            manifest = read_meetings_manifest(resolved_manifest_path)
            dataset = self.repository.upsert_dataset(
                DatasetCreate(
                    name=dataset_name,
                    storage_kind=DatasetStorageKind.S3,
                    root_uri=manifest.connection.s3_uri,
                    description="Manifest-backed two-track S3 dataset",
                )
            )
            discovered = discover_manifest_samples(manifest)
            completed_ids = self.repository.completed_quality_sample_ids(
                dataset.id,
                METRIC_VERSION,
            )
            pending = tuple(
                sample for sample in discovered if sample.external_id not in completed_ids
            )
            selected = select_manifest_duration(pending, max_duration_hours)
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.RUNNING,
                (
                    f"Manifest contains {len(manifest.samples)} samples; "
                    f"{len(discovered)} passed duration checks; "
                    f"{len(completed_ids)} already complete; selected {len(selected)}"
                ),
                dataset_id=dataset.id,
                total_samples=len(selected),
                processed_samples=0,
            )
            processed_samples = 0
            analyzed_samples = 0
            language_excluded_samples = 0
            failed_samples = 0
            sample_errors: list[str] = []

            registered_samples: list[ManifestRegisteredSample] = []
            for sample in selected:
                try:
                    registered_samples.append(
                        register_manifest_sample(
                            repository=self.repository,
                            dataset_id=dataset.id,
                            discovered=sample,
                        )
                    )
                except Exception as error:
                    failed_samples += 1
                    sample_errors.append(stage_error("registration", sample.external_id, error))
                    logger.exception("manifest registration failed: %s", sample.external_id)
            self.repository.update_ingestion_job(
                job.id,
                JobStatus.RUNNING,
                (
                    f"Registration complete: {len(registered_samples)} registered; "
                    f"{failed_samples} failed. Starting language pass."
                ),
                processed_samples=processed_samples,
                failed_samples=failed_samples,
            )

            language_ready_samples: list[ManifestLanguageReady] = []
            language_assessed_samples = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.assess_language, registered): registered
                    for registered in registered_samples
                }
                for future in as_completed(futures):
                    registered = futures[future]
                    try:
                        language_ready = future.result()
                        language_assessed_samples += 1
                        if language_ready.assessment.allows_full_analysis:
                            language_ready_samples.append(language_ready)
                        else:
                            language_excluded_samples += 1
                            processed_samples += 1
                    except Exception as error:
                        failed_samples += 1
                        sample_errors.append(
                            stage_error(
                                "language",
                                registered.discovered.external_id,
                                error,
                            )
                        )
                        logger.exception(
                            "manifest language assessment failed: %s",
                            registered.discovered.external_id,
                        )
                    self.repository.update_ingestion_job(
                        job.id,
                        JobStatus.RUNNING,
                        (
                            f"Language pass: {language_assessed_samples} assessed; "
                            f"{len(language_ready_samples)} eligible; "
                            f"{language_excluded_samples} excluded; "
                            f"{failed_samples} failed"
                        ),
                        processed_samples=processed_samples,
                        failed_samples=failed_samples,
                    )

            asr_ready_samples = language_ready_samples
            for model_index, model_id in enumerate(FULL_RECORDING_MODELS, start=1):
                model_ready_samples: list[ManifestLanguageReady] = []
                model_completed_samples = 0
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = {
                        executor.submit(
                            self.transcribe_sample_model,
                            language_ready,
                            model_id,
                        ): language_ready
                        for language_ready in asr_ready_samples
                    }
                    for future in as_completed(futures):
                        language_ready = futures[future]
                        try:
                            model_ready_samples.append(future.result())
                            model_completed_samples += 1
                        except Exception as error:
                            failed_samples += 1
                            sample_errors.append(
                                stage_error(
                                    f"asr-{model_id.value}",
                                    language_ready.discovered.external_id,
                                    error,
                                )
                            )
                            logger.exception(
                                "manifest %s ASR failed: %s",
                                model_id.value,
                                language_ready.discovered.external_id,
                            )
                        self.repository.update_ingestion_job(
                            job.id,
                            JobStatus.RUNNING,
                            (
                                f"ASR model pass {model_index} of "
                                f"{len(FULL_RECORDING_MODELS)} ({model_id.value}): "
                                f"{model_completed_samples} of "
                                f"{len(asr_ready_samples)} samples complete; "
                                f"{failed_samples} failed"
                            ),
                            processed_samples=processed_samples,
                            failed_samples=failed_samples,
                        )
                asr_ready_samples = model_ready_samples

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.analyze_quality, asr_ready): asr_ready
                    for asr_ready in asr_ready_samples
                }
                for future in as_completed(futures):
                    asr_ready = futures[future]
                    try:
                        future.result()
                        processed_samples += 1
                        analyzed_samples += 1
                    except Exception as error:
                        failed_samples += 1
                        sample_errors.append(
                            stage_error(
                                "quality",
                                asr_ready.discovered.external_id,
                                error,
                            )
                        )
                        logger.exception(
                            "manifest quality analysis failed: %s",
                            asr_ready.discovered.external_id,
                        )
                    self.repository.update_ingestion_job(
                        job.id,
                        JobStatus.RUNNING,
                        (
                            f"Quality pass: {analyzed_samples} of "
                            f"{len(asr_ready_samples)} samples complete; "
                            f"{language_excluded_samples} language-excluded; "
                            f"{failed_samples} failed"
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
                "Manifest ingestion failed",
                error=f"{type(error).__name__}: {error}",
            )

    def process_sample(
        self,
        dataset_id: UUID,
        discovered: ManifestDiscoveredSample,
    ) -> SampleProcessingStatus:
        registered = register_manifest_sample(
            repository=self.repository,
            dataset_id=dataset_id,
            discovered=discovered,
        )
        language_ready = self.assess_language(registered)
        if not language_ready.assessment.allows_full_analysis:
            return SampleProcessingStatus.LANGUAGE_EXCLUDED
        self.transcribe_sample(language_ready)
        self.analyze_quality(language_ready)
        return SampleProcessingStatus.ANALYZED

    def assess_language(
        self,
        registered: ManifestRegisteredSample,
    ) -> ManifestLanguageReady:
        return ensure_manifest_language(
            repository=self.repository,
            language_repository=self.language_repository,
            analysis_client=self.analysis_client,
            registered=registered,
        )

    def transcribe_sample(
        self,
        language_ready: ManifestLanguageReady,
    ) -> ManifestLanguageReady:
        for model_id in FULL_RECORDING_MODELS:
            self.transcribe_sample_model(language_ready, model_id)
        return language_ready

    def transcribe_sample_model(
        self,
        language_ready: ManifestLanguageReady,
        model_id: AsrModelId,
    ) -> ManifestLanguageReady:
        ensure_manifest_model_transcripts(
            repository=self.full_asr_repository,
            analysis_client=self.analysis_client,
            registered=language_ready.registered,
            speaker1_source=language_ready.discovered.speaker1,
            speaker2_source=language_ready.discovered.speaker2,
            model_id=model_id,
        )
        return language_ready

    def analyze_quality(
        self,
        language_ready: ManifestLanguageReady,
    ) -> None:
        discovered = language_ready.discovered
        ready = ready_sample(
            full_asr_repository=self.full_asr_repository,
            registered=language_ready.registered,
        )
        quality_response = self.analysis_client.analyze_quality(
            DatasetQualityRequest(
                sample_id=discovered.external_id,
                speaker1=discovered.speaker1,
                speaker2=discovered.speaker2,
                speaker1_transcripts=DatasetTrackTranscripts(
                    parakeet=transcript_result_from_record(ready.transcripts.parakeet.speaker1),
                    canary=transcript_result_from_record(ready.transcripts.canary.speaker1),
                ),
                speaker2_transcripts=DatasetTrackTranscripts(
                    parakeet=transcript_result_from_record(ready.transcripts.parakeet.speaker2),
                    canary=transcript_result_from_record(ready.transcripts.canary.speaker2),
                ),
            )
        )
        validate_materialized_audio(
            quality_response.speaker1_audio,
            language_ready.registered.speaker1_source_audio_sha256,
        )
        validate_materialized_audio(
            quality_response.speaker2_audio,
            language_ready.registered.speaker2_source_audio_sha256,
        )
        self.vad_repository.upsert(
            sample_track_id=language_ready.registered.speaker1_track_id,
            source_audio_sha256=language_ready.registered.speaker1_source_audio_sha256,
            vad_version=quality_response.vad_version,
            result=quality_response.speaker1_vad,
        )
        self.vad_repository.upsert(
            sample_track_id=language_ready.registered.speaker2_track_id,
            source_audio_sha256=language_ready.registered.speaker2_source_audio_sha256,
            vad_version=quality_response.vad_version,
            result=quality_response.speaker2_vad,
        )
        duration_seconds = quality_response.quality_result.duration_seconds
        if duration_seconds is None:
            raise ValueError(
                f"Remote quality analysis failed: {quality_response.quality_result.error}"
            )
        annotation = analyze_conversation_from_remote_evidence(
            speaker1_words=quality_response.speaker1_filtered_words,
            speaker2_words=quality_response.speaker2_filtered_words,
            speaker1_vad=quality_response.speaker1_vad,
            speaker2_vad=quality_response.speaker2_vad,
            duration_seconds=duration_seconds,
        )
        quality_result = quality_result_with_conversation_annotation(
            result=quality_response.quality_result,
            conversation_annotation=annotation,
        )
        persist_processed_ready_sample(
            repository=self.repository,
            processed=ProcessedReadySample(
                ready=ready,
                quality_result=quality_result,
            ),
        )
        self.conversation_region_repository.upsert(
            sample_id=ready.registered.sample_id,
            analysis=analyze_conversation_regions(
                speaker1_vad=quality_response.speaker1_vad,
                speaker2_vad=quality_response.speaker2_vad,
                annotation=annotation,
                config=ConversationRegionConfig(),
            ),
        )


def stage_error(stage: str, external_id: str, error: Exception) -> str:
    return f"{stage} {external_id}: {type(error).__name__}: {error}"


def select_manifest_duration(
    samples: tuple[ManifestDiscoveredSample, ...],
    max_duration_hours: float | None,
) -> tuple[ManifestDiscoveredSample, ...]:
    if max_duration_hours is None:
        return samples
    if max_duration_hours <= 0.0:
        raise ValueError("max_duration_hours must be positive")
    maximum_seconds = max_duration_hours * 3600.0
    selected: list[ManifestDiscoveredSample] = []
    selected_seconds = 0.0
    for sample in samples:
        if selected and selected_seconds + sample.duration_seconds > maximum_seconds:
            break
        selected.append(sample)
        selected_seconds += sample.duration_seconds
    return tuple(selected)


def register_manifest_sample(
    repository: Repository,
    dataset_id: UUID,
    discovered: ManifestDiscoveredSample,
) -> ManifestRegisteredSample:
    sample = repository.upsert_sample(dataset_id, discovered.external_id)
    repository.update_sample_duration(sample.id, discovered.duration_seconds)
    speaker1_track = upsert_manifest_track(
        repository=repository,
        sample_id=sample.id,
        side=TrackSide.SPEAKER1,
        speaker_index=1,
        source=discovered.speaker1,
    )
    speaker2_track = upsert_manifest_track(
        repository=repository,
        sample_id=sample.id,
        side=TrackSide.SPEAKER2,
        speaker_index=2,
        source=discovered.speaker2,
    )
    return ManifestRegisteredSample(
        discovered=discovered,
        sample_id=sample.id,
        speaker1_track=speaker1_track,
        speaker2_track=speaker2_track,
    )


def upsert_manifest_track(
    repository: Repository,
    sample_id: UUID,
    side: TrackSide,
    speaker_index: int,
    source: S3AudioSource,
) -> SampleTrackRecord:
    return repository.upsert_sample_track(
        sample_id,
        SampleTrackInput(
            side=side,
            speaker_index=speaker_index,
            storage_uri=source.uri,
            access_uri=source.uri,
            duration_seconds=source.duration_seconds,
            sample_rate=None,
            channels=None,
            sample_count=None,
            audio_sha256=None,
            source_size_bytes=source.size_bytes,
            source_etag=source.etag,
        ),
    )


def ensure_manifest_language(
    repository: Repository,
    language_repository: LanguageAssessmentRepository,
    analysis_client: DatasetAnalysisClient,
    registered: ManifestRegisteredSample,
) -> ManifestLanguageReady:
    cached = cached_language_assessment(language_repository, registered)
    if cached is not None:
        return ManifestLanguageReady(
            discovered=registered.discovered,
            registered=completed_registered_sample(registered),
            assessment=cached,
        )
    response = analysis_client.assess_language(
        DatasetLanguageRequest(
            speaker1=registered.discovered.speaker1,
            speaker2=registered.discovered.speaker2,
        )
    )
    speaker1_track = update_materialized_track(
        repository=repository,
        existing=registered.speaker1_track,
        audio=response.speaker1.audio,
    )
    speaker2_track = update_materialized_track(
        repository=repository,
        existing=registered.speaker2_track,
        audio=response.speaker2.audio,
    )
    speaker1_assessment = persist_language_track(
        repository=language_repository,
        track=speaker1_track,
        response=response.speaker1,
    )
    speaker2_assessment = persist_language_track(
        repository=language_repository,
        track=speaker2_track,
        response=response.speaker2,
    )
    return ManifestLanguageReady(
        discovered=registered.discovered,
        registered=completed_registered_sample(
            ManifestRegisteredSample(
                discovered=registered.discovered,
                sample_id=registered.sample_id,
                speaker1_track=speaker1_track,
                speaker2_track=speaker2_track,
            )
        ),
        assessment=SampleLanguageAssessment(
            status=sample_language_status(speaker1_assessment, speaker2_assessment),
            speaker1=speaker1_assessment,
            speaker2=speaker2_assessment,
        ),
    )


def cached_language_assessment(
    repository: LanguageAssessmentRepository,
    registered: ManifestRegisteredSample,
) -> SampleLanguageAssessment | None:
    speaker1_sha256 = registered.speaker1_track.audio_sha256
    speaker2_sha256 = registered.speaker2_track.audio_sha256
    if speaker1_sha256 is None or speaker2_sha256 is None:
        return None
    speaker1 = repository.get_current_assessment(
        sample_track_id=registered.speaker1_track.id,
        source_audio_sha256=speaker1_sha256,
        assessment_version=LANGUAGE_ASSESSMENT_VERSION,
    )
    speaker2 = repository.get_current_assessment(
        sample_track_id=registered.speaker2_track.id,
        source_audio_sha256=speaker2_sha256,
        assessment_version=LANGUAGE_ASSESSMENT_VERSION,
    )
    if speaker1 is None or speaker2 is None:
        return None
    return SampleLanguageAssessment(
        status=sample_language_status(speaker1, speaker2),
        speaker1=speaker1,
        speaker2=speaker2,
    )


def update_materialized_track(
    repository: Repository,
    existing: SampleTrackRecord,
    audio: MaterializedAudio,
) -> SampleTrackRecord:
    if existing.storage_uri != audio.source.uri:
        raise ValueError("Materialized audio belongs to a different S3 object.")
    track = repository.upsert_sample_track(
        existing.sample_id,
        SampleTrackInput(
            side=existing.side,
            speaker_index=existing.speaker_index,
            storage_uri=audio.source.uri,
            access_uri=audio.source.uri,
            duration_seconds=audio.metadata.duration_seconds,
            sample_rate=audio.metadata.sample_rate,
            channels=audio.metadata.channels,
            sample_count=audio.metadata.sample_count,
            audio_sha256=audio.content_sha256,
            source_size_bytes=audio.source.size_bytes,
            source_etag=audio.source.etag,
        ),
    )
    repository.upsert_audio_metadata(
        AudioMetadataInput(
            sample_track_id=track.id,
            duration_seconds=audio.metadata.duration_seconds,
            sample_rate=audio.metadata.sample_rate,
            channels=audio.metadata.channels,
            sample_count=audio.metadata.sample_count,
            payload=audio.metadata.model_dump(),
        )
    )
    return track


def persist_language_track(
    repository: LanguageAssessmentRepository,
    track: SampleTrackRecord,
    response: DatasetLanguageTrackResponse,
) -> TrackLanguageAssessment:
    if response.error is not None or response.status is TrackLanguageStatus.FAILED:
        raise ValueError(f"Remote language assessment failed: {response.error}")
    if track.audio_sha256 is None:
        raise ValueError("Materialized track is missing its content SHA-256.")
    return repository.upsert_assessment(
        TrackLanguageAssessment(
            sample_track_id=track.id,
            source_audio_sha256=track.audio_sha256,
            assessment_version=LANGUAGE_ASSESSMENT_VERSION,
            status=response.status,
            language_code=response.language_code,
            confidence=response.confidence,
            transcript_word_count=response.transcript_word_count,
            transcript_text=response.transcript_text,
            probe_windows=response.probe_windows,
            error=response.error,
        )
    )


def completed_registered_sample(registered: ManifestRegisteredSample) -> RegisteredSample:
    speaker1_sha256 = registered.speaker1_track.audio_sha256
    speaker2_sha256 = registered.speaker2_track.audio_sha256
    if speaker1_sha256 is None or speaker2_sha256 is None:
        raise ValueError("Manifest sample audio has not been materialized.")
    return RegisteredSample(
        discovered=DiscoveredSample(
            external_id=registered.discovered.external_id,
            speaker1_path=registered.discovered.speaker1.uri,
            speaker2_path=registered.discovered.speaker2.uri,
        ),
        sample_id=registered.sample_id,
        speaker1_track_id=registered.speaker1_track.id,
        speaker2_track_id=registered.speaker2_track.id,
        speaker1_metadata=track_audio_metadata(registered.speaker1_track),
        speaker2_metadata=track_audio_metadata(registered.speaker2_track),
        speaker1_source_audio_sha256=speaker1_sha256,
        speaker2_source_audio_sha256=speaker2_sha256,
    )


def track_audio_metadata(track: SampleTrackRecord) -> AudioMetadata:
    if (
        track.duration_seconds is None
        or track.sample_rate is None
        or track.channels is None
        or track.sample_count is None
    ):
        raise ValueError(f"Materialized track metadata is incomplete: {track.id}")
    return AudioMetadata(
        duration_seconds=track.duration_seconds,
        sample_rate=track.sample_rate,
        channels=track.channels,
        sample_count=track.sample_count,
    )


def ensure_manifest_transcripts(
    repository: FullRecordingAsrRepository,
    analysis_client: DatasetAnalysisClient,
    registered: RegisteredSample,
    speaker1_source: S3AudioSource,
    speaker2_source: S3AudioSource,
) -> None:
    for model_id in FULL_RECORDING_MODELS:
        ensure_manifest_model_transcripts(
            repository=repository,
            analysis_client=analysis_client,
            registered=registered,
            speaker1_source=speaker1_source,
            speaker2_source=speaker2_source,
            model_id=model_id,
        )


def ensure_manifest_model_transcripts(
    repository: FullRecordingAsrRepository,
    analysis_client: DatasetAnalysisClient,
    registered: RegisteredSample,
    speaker1_source: S3AudioSource,
    speaker2_source: S3AudioSource,
    model_id: AsrModelId,
) -> None:
    for sample_track_id, source_audio_sha256, source in (
        (
            registered.speaker1_track_id,
            registered.speaker1_source_audio_sha256,
            speaker1_source,
        ),
        (
            registered.speaker2_track_id,
            registered.speaker2_source_audio_sha256,
            speaker2_source,
        ),
    ):
        ensure_track_transcripts(
            repository=repository,
            analysis_client=analysis_client,
            sample_track_id=sample_track_id,
            source_audio_sha256=source_audio_sha256,
            source=source,
            model_ids=(model_id,),
        )


def ensure_track_transcripts(
    repository: FullRecordingAsrRepository,
    analysis_client: DatasetAnalysisClient,
    sample_track_id: UUID,
    source_audio_sha256: str,
    source: S3AudioSource,
    model_ids: tuple[AsrModelId, ...],
) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
    cached = repository.get_cached_transcripts(
        sample_track_id=sample_track_id,
        source_audio_sha256=source_audio_sha256,
        model_ids=model_ids,
    )
    missing = missing_model_ids(requested=model_ids, transcripts=cached)
    if missing:
        response = analysis_client.transcribe(DatasetAsrRequest(source=source, models=missing))
        validate_materialized_audio(response.audio, source_audio_sha256)
        for result in requested_results(requested=missing, results=response.results):
            validate_transcript_timestamps(
                transcript=result,
                duration_seconds=response.prepared_duration_seconds,
            )
            repository.upsert_transcript(
                sample_track_id=sample_track_id,
                source_audio_sha256=source_audio_sha256,
                prepared_audio_sha256=response.prepared_audio_sha256,
                audio_filename=source_filename(source.uri),
                source_duration_seconds=response.audio.metadata.duration_seconds,
                prepared_duration_seconds=response.prepared_duration_seconds,
                transcript=result,
            )
    final = repository.get_cached_transcripts(
        sample_track_id=sample_track_id,
        source_audio_sha256=source_audio_sha256,
        model_ids=model_ids,
    )
    remaining = missing_model_ids(requested=model_ids, transcripts=final)
    if remaining:
        labels = ", ".join(model.value for model in remaining)
        raise ValueError(f"Manifest ASR persistence is missing models: {labels}")
    return final


def validate_materialized_audio(audio: MaterializedAudio, expected_sha256: str) -> None:
    if audio.content_sha256 != expected_sha256:
        raise ValueError("Compute cache returned different immutable S3 audio content.")


def source_filename(uri: str) -> str:
    filename = PurePosixPath(urlparse(uri).path).name
    if not filename:
        raise ValueError(f"S3 audio URI has no filename: {uri}")
    return filename

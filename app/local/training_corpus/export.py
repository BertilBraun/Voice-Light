from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.config import DATABASE_URL
from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
)
from app.local.conversation_regions.repository import ConversationRegionRepository
from app.local.corpus_audit.models import CorpusAuditRequest
from app.local.corpus_audit.repository import CorpusAuditEvidence, CorpusAuditRepository
from app.local.corpus_audit.service import (
    _audit_events,
    _audit_window,
    _floor_validity_index,
    _oriented_speakers,
    _window_starts,
)
from app.local.corpus_review.exclusions import CorpusIntervalExclusion
from app.local.corpus_review.models import CorpusReviewDatasetSelection
from app.local.corpus_review.repository import CorpusReviewRepository
from app.local.corpus_review.service import corpus_review_readiness
from app.local.db.models import (
    DashboardSample,
    SampleTrackRecord,
    TrackLanguageAssessment,
    TrackSide,
)
from app.local.db.repository import Repository
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.source_annotations.models import SourceAnnotationImport
from app.local.source_annotations.repository import SourceAnnotationRepository
from app.local.training_corpus.audio_assets import (
    CorpusAudioAssetCatalog,
    load_audio_asset_catalog,
)
from app.local.training_corpus.splits import (
    ConversationSplitCandidate,
    ConversationSplitPlan,
    TrainingCorpusSplit,
    assign_conversation_splits,
)
from app.local.training_samples.models import TrainingFramePreview
from app.local.training_samples.service import (
    FRAME_SECONDS,
    INPUT_DURATION_SECONDS,
    TRAINING_LABEL_VERSION,
    build_frame_previews,
)
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import (
    METRIC_VERSION,
    AudioMetadata,
    AudioQualityMetrics,
    ConversationAnnotation,
    ConversationCountEstimate,
    InteractionDensityMetrics,
    QualityResult,
    SpeakerSide,
    TimingReliabilityMetrics,
    TrackVadResult,
)

SCHEMA_VERSION = "voice-light-turn-taking-v1"
FRAMES_PER_SAMPLE = int(round(INPUT_DURATION_SECONDS / FRAME_SECONDS))
PARQUET_ROWS_PER_SHARD = 1000


class AudioReference(FrozenBaseModel):
    side: TrackSide
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    audio_sha256: str


class RecordingMetadata(FrozenBaseModel):
    schema_version: str
    dataset_name: str
    dataset_id: UUID
    sample_id: UUID
    external_id: str
    duration_seconds: float
    quality_score: float
    metric_version: str
    quality: RecordingQualityMetadata
    transcripts: tuple[FullRecordingAsrTranscriptRecord, ...]
    language_assessments: tuple[TrackLanguageAssessment, ...]
    source_annotations: tuple[SourceAnnotationImport, ...]
    annotation: ConversationAnnotation
    speaker1_vad: TrackVadResult
    speaker2_vad: TrackVadResult
    conversation_regions: ConversationRegionAnalysis
    review_interval_exclusions: tuple[CorpusIntervalExclusion, ...]
    split: TrainingCorpusSplit
    audio: tuple[AudioReference, AudioReference]


class RecordingQualityMetadata(FrozenBaseModel):
    quality_result_id: UUID
    metric_version: str
    raw_quality_score: float
    total_quality_score: float
    quality_flags: tuple[str, ...]
    interaction_density: InteractionDensityMetrics
    timing_reliability: TimingReliabilityMetrics
    audio_quality: AudioQualityMetrics
    conversation_count_estimate: ConversationCountEstimate


class DatasetMetadata(FrozenBaseModel):
    schema_version: str
    dataset_id: UUID
    dataset_name: str
    corpus_directory: str
    minimum_quality: float = Field(ge=0.0, le=1.0)
    recording_count: int = Field(ge=0)
    source_duration_seconds: float = Field(ge=0.0)
    metric_version: str
    annotation_version: str
    region_analysis_version: str
    training_label_version: str
    review_set_name: str
    known_limitations: tuple[str, ...]


class MaterializedTrainingSample(FrozenBaseModel):
    schema_version: str
    training_label_version: str
    window_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: UUID
    dataset_name: str
    sample_id: UUID
    external_id: str
    user_side: TrackSide
    assistant_side: TrackSide
    split: TrainingCorpusSplit
    user_audio_path: str
    assistant_audio_path: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    category: str
    assistant_has_floor: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    p_user_has_floor: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    p_user_yield: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    p_assistant_backchannel: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    future_activity_0_200: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    future_activity_200_500: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    future_activity_500_1000: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    future_activity_1000_1500: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    turn_completion: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    continuation_pause: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    non_floor_feedback: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )
    floor_take: tuple[float, ...] = Field(
        min_length=FRAMES_PER_SAMPLE, max_length=FRAMES_PER_SAMPLE
    )

    @model_validator(mode="after")
    def validate_label_values(self) -> MaterializedTrainingSample:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Training sample end time must follow its start time.")
        if any(value < 0.0 or value > 1.0 for value in self.assistant_has_floor):
            raise ValueError("Assistant-floor inputs must be probabilities between zero and one.")
        masked_targets = (
            self.p_user_has_floor,
            self.p_user_yield,
            self.p_assistant_backchannel,
            self.future_activity_0_200,
            self.future_activity_200_500,
            self.future_activity_500_1000,
            self.future_activity_1000_1500,
            self.turn_completion,
            self.continuation_pause,
            self.non_floor_feedback,
            self.floor_take,
        )
        if any(
            value != -1.0 and not 0.0 <= value <= 1.0
            for target in masked_targets
            for value in target
        ):
            raise ValueError("Training targets must be masked with -1 or be probabilities.")
        return self


class ExportShard(FrozenBaseModel):
    split: TrainingCorpusSplit
    path: str
    row_count: int = Field(ge=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExportSplitSummary(FrozenBaseModel):
    split: TrainingCorpusSplit
    recording_count: int = Field(ge=0)
    training_sample_count: int = Field(ge=0)
    source_duration_seconds: float = Field(ge=0.0)


class ExportManifest(FrozenBaseModel):
    schema_version: str
    generated_at: datetime
    metric_version: str
    annotation_version: str
    region_analysis_version: str
    training_label_version: str
    input_duration_seconds: float = Field(gt=0.0)
    frame_seconds: float = Field(gt=0.0)
    review_set_name: str
    split_plan: ConversationSplitPlan
    recording_count: int = Field(ge=0)
    training_sample_count: int = Field(ge=0)
    splits: tuple[ExportSplitSummary, ...]
    shards: tuple[ExportShard, ...]


class TrainingCorpusExportRequest(FrozenBaseModel):
    review_set_name: str = Field(min_length=1)
    split_seed: str = Field(min_length=1)
    audio_build_manifest_paths: tuple[Path, ...] = Field(min_length=1)
    output_directory: Path


def export_training_corpus(request: TrainingCorpusExportRequest) -> ExportManifest:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for corpus export.")
    review_plan = CorpusReviewRepository(DATABASE_URL).get(request.review_set_name)
    readiness = corpus_review_readiness(review_plan)
    if not readiness.ready_to_publish:
        raise ValueError(f"Corpus review set {request.review_set_name!r} is not ready to publish.")
    _validate_export_destination(request.output_directory)
    audio_assets = load_audio_asset_catalog(request.audio_build_manifest_paths)
    evidence = tuple(
        item
        for selection in review_plan.review_set.config.datasets
        for item in CorpusAuditRepository(DATABASE_URL).load_evidence(
            dataset_ids=(selection.dataset_id,),
            minimum_quality=selection.minimum_quality,
            metric_version=METRIC_VERSION,
            annotation_version=ANNOTATION_VERSION,
            region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        )
    )
    split_plan = assign_conversation_splits(
        candidates=tuple(
            ConversationSplitCandidate(dataset_id=item.dataset_id, sample_id=item.sample_id)
            for item in evidence
        ),
        seed=request.split_seed,
    )
    split_by_sample = {
        assignment.sample_id: assignment.split for assignment in split_plan.assignments
    }
    dataset_ids = tuple(
        selection.dataset_id for selection in review_plan.review_set.config.datasets
    )
    vad_by_sample = _vad_by_sample_id(database_url=DATABASE_URL, dataset_ids=dataset_ids)
    sample_repository = Repository(DATABASE_URL)
    transcript_repository = FullRecordingAsrRepository(DATABASE_URL)
    source_annotation_repository = SourceAnnotationRepository(DATABASE_URL)
    training_samples: list[MaterializedTrainingSample] = []
    for conversation in evidence:
        speaker1_vad, speaker2_vad = vad_by_sample[conversation.sample_id]
        dashboard_sample = sample_repository.get_dashboard_sample(conversation.sample_id)
        _write_recording_metadata(
            output_directory=request.output_directory,
            conversation=conversation,
            dashboard_sample=dashboard_sample,
            speaker1_vad=speaker1_vad,
            speaker2_vad=speaker2_vad,
            split=split_by_sample[conversation.sample_id],
            transcript_repository=transcript_repository,
            source_annotation_repository=source_annotation_repository,
            audio_assets=audio_assets,
        )
        training_samples.extend(
            _training_samples_for_conversation(
                conversation=conversation,
                dashboard_sample=dashboard_sample,
                split=split_by_sample[conversation.sample_id],
                audio_assets=audio_assets,
            )
        )
    _write_dataset_metadata(
        output_directory=request.output_directory,
        evidence=evidence,
        review_set_name=request.review_set_name,
        selections=review_plan.review_set.config.datasets,
        audio_assets=audio_assets,
    )
    shards = _write_training_shards(
        output_directory=request.output_directory,
        samples=training_samples,
    )
    split_summaries = _split_summaries(
        evidence=evidence,
        samples=training_samples,
        split_by_sample=split_by_sample,
    )
    manifest = ExportManifest(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(UTC),
        metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
        region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        training_label_version=TRAINING_LABEL_VERSION,
        input_duration_seconds=INPUT_DURATION_SECONDS,
        frame_seconds=FRAME_SECONDS,
        review_set_name=request.review_set_name,
        split_plan=split_plan,
        recording_count=len(evidence),
        training_sample_count=len(training_samples),
        splits=split_summaries,
        shards=shards,
    )
    (request.output_directory / "corpus.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def _vad_by_sample_id(
    database_url: str,
    dataset_ids: Sequence[UUID],
) -> dict[UUID, tuple[TrackVadResult, TrackVadResult]]:
    repository = ConversationRegionRepository(database_url)
    evidence = (
        item
        for dataset_id in dataset_ids
        for item in repository.list_evidence(
            dataset_id=dataset_id,
            metric_version=METRIC_VERSION,
            annotation_version=ANNOTATION_VERSION,
        )
    )
    return {item.sample_id: (item.speaker1_vad, item.speaker2_vad) for item in evidence}


def _write_dataset_metadata(
    output_directory: Path,
    evidence: Sequence[CorpusAuditEvidence],
    review_set_name: str,
    selections: Sequence[CorpusReviewDatasetSelection],
    audio_assets: CorpusAudioAssetCatalog,
) -> None:
    minimum_quality_by_dataset = {
        selection.dataset_id: selection.minimum_quality for selection in selections
    }
    dataset_ids = tuple(sorted({item.dataset_id for item in evidence}, key=lambda value: value.hex))
    for dataset_id in dataset_ids:
        dataset_evidence = tuple(item for item in evidence if item.dataset_id == dataset_id)
        dataset_names = {item.dataset_name for item in dataset_evidence}
        assert len(dataset_names) == 1
        dataset_name = next(iter(dataset_names))
        corpus_directory = audio_assets.corpus_dataset_directory(dataset_name)
        metadata = DatasetMetadata(
            schema_version=SCHEMA_VERSION,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            corpus_directory=corpus_directory,
            minimum_quality=minimum_quality_by_dataset[dataset_id],
            recording_count=len(dataset_evidence),
            source_duration_seconds=sum(
                item.represented_duration_seconds for item in dataset_evidence
            ),
            metric_version=METRIC_VERSION,
            annotation_version=ANNOTATION_VERSION,
            region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
            training_label_version=TRAINING_LABEL_VERSION,
            review_set_name=review_set_name,
            known_limitations=(
                "Manual listening review used three accepted clips per source dataset.",
                "Automatic annotations may contain residual ASR or VAD errors outside "
                "reviewed clips.",
            ),
        )
        path = output_directory / corpus_directory / "dataset.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def _write_recording_metadata(
    output_directory: Path,
    conversation: CorpusAuditEvidence,
    dashboard_sample: DashboardSample,
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    split: TrainingCorpusSplit,
    transcript_repository: FullRecordingAsrRepository,
    source_annotation_repository: SourceAnnotationRepository,
    audio_assets: CorpusAudioAssetCatalog,
) -> None:
    if conversation.conversation_regions is None:
        raise ValueError(f"Missing conversation regions for {conversation.external_id}")
    speaker1 = _track(dashboard_sample, TrackSide.SPEAKER1)
    speaker2 = _track(dashboard_sample, TrackSide.SPEAKER2)
    quality_record = dashboard_sample.latest_quality
    if quality_record is None:
        raise ValueError(f"Missing quality result for {conversation.external_id}")
    quality_result = QualityResult.model_validate(quality_record.payload)
    transcript_ids = (
        quality_record.speaker1_parakeet_full_asr_transcript_id,
        quality_record.speaker2_parakeet_full_asr_transcript_id,
        quality_record.speaker1_canary_full_asr_transcript_id,
        quality_record.speaker2_canary_full_asr_transcript_id,
    )
    if any(transcript_id is None for transcript_id in transcript_ids):
        raise ValueError(f"Incomplete transcript provenance for {conversation.external_id}")
    complete_transcript_ids = tuple(
        transcript_id for transcript_id in transcript_ids if transcript_id is not None
    )
    metadata = RecordingMetadata(
        schema_version=SCHEMA_VERSION,
        dataset_name=conversation.dataset_name,
        dataset_id=conversation.dataset_id,
        sample_id=conversation.sample_id,
        external_id=conversation.external_id,
        duration_seconds=conversation.represented_duration_seconds,
        quality_score=conversation.quality_score,
        metric_version=METRIC_VERSION,
        quality=_recording_quality_metadata(
            quality_result_id=quality_record.id,
            quality=quality_result,
        ),
        transcripts=transcript_repository.get_transcripts_by_ids(complete_transcript_ids),
        language_assessments=dashboard_sample.language_assessments,
        source_annotations=source_annotation_repository.list_for_sample(conversation.sample_id),
        annotation=conversation.annotation,
        speaker1_vad=speaker1_vad,
        speaker2_vad=speaker2_vad,
        conversation_regions=conversation.conversation_regions,
        review_interval_exclusions=conversation.interval_exclusions,
        split=split,
        audio=(
            _audio_reference(
                conversation.dataset_name, conversation.external_id, speaker1, audio_assets
            ),
            _audio_reference(
                conversation.dataset_name, conversation.external_id, speaker2, audio_assets
            ),
        ),
    )
    metadata_path = output_directory / _recording_directory(metadata.audio) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def _recording_quality_metadata(
    quality_result_id: UUID,
    quality: QualityResult,
) -> RecordingQualityMetadata:
    if (
        quality.raw_quality_score is None
        or quality.total_quality_score is None
        or quality.interaction_density is None
        or quality.timing_reliability is None
        or quality.audio_quality is None
        or quality.conversation_count_estimate is None
    ):
        raise ValueError(f"Quality result {quality_result_id} is incomplete.")
    return RecordingQualityMetadata(
        quality_result_id=quality_result_id,
        metric_version=quality.metric_version,
        raw_quality_score=quality.raw_quality_score,
        total_quality_score=quality.total_quality_score,
        quality_flags=quality.quality_flags,
        interaction_density=quality.interaction_density,
        timing_reliability=quality.timing_reliability,
        audio_quality=quality.audio_quality,
        conversation_count_estimate=quality.conversation_count_estimate,
    )


def _training_samples_for_conversation(
    conversation: CorpusAuditEvidence,
    dashboard_sample: DashboardSample,
    split: TrainingCorpusSplit,
    audio_assets: CorpusAudioAssetCatalog,
) -> Iterator[MaterializedTrainingSample]:
    if conversation.conversation_regions is None:
        return
    request = CorpusAuditRequest(
        dataset_ids=(conversation.dataset_id,),
        minimum_quality=0.0,
    )
    for user_side in (TrackSide.SPEAKER1, TrackSide.SPEAKER2):
        user, assistant = _oriented_speakers(conversation.annotation, user_side)
        floor_validity = _floor_validity_index(user)
        events = _audit_events(user=user, assistant=assistant)
        for start_seconds in _window_starts(
            duration_seconds=conversation.represented_duration_seconds,
            request=request,
        ):
            end_seconds = start_seconds + INPUT_DURATION_SECONDS
            if end_seconds > conversation.represented_duration_seconds:
                continue
            audit = _audit_window(
                start_seconds=start_seconds,
                duration_seconds=conversation.represented_duration_seconds,
                floor_validity=floor_validity,
                events=events,
                conversation_regions=conversation.conversation_regions,
                interval_exclusions=conversation.interval_exclusions,
                request=request,
            )
            if not audit.accepted:
                continue
            frames = build_frame_previews(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                annotation_end_seconds=conversation.annotation.analyzed_duration_seconds,
                user=user,
                assistant=assistant,
            )
            yield _materialized_training_sample(
                conversation=conversation,
                dashboard_sample=dashboard_sample,
                user_side=user_side,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                category=audit.category.value,
                split=split,
                frames=frames,
                audio_assets=audio_assets,
            )


def _materialized_training_sample(
    conversation: CorpusAuditEvidence,
    dashboard_sample: DashboardSample,
    user_side: TrackSide,
    start_seconds: float,
    end_seconds: float,
    category: str,
    split: TrainingCorpusSplit,
    frames: tuple[TrainingFramePreview, ...],
    audio_assets: CorpusAudioAssetCatalog,
) -> MaterializedTrainingSample:
    if len(frames) != FRAMES_PER_SAMPLE:
        raise ValueError(f"Expected {FRAMES_PER_SAMPLE} frames, got {len(frames)}")
    user_track = _track(dashboard_sample, user_side)
    assistant_side = _other_side(user_side)
    assistant_track = _track(dashboard_sample, assistant_side)
    future = tuple(tuple(frame.future_activity) for frame in frames)
    if any(len(targets) != 4 for targets in future):
        raise ValueError("Expected four future-activity targets per frame.")
    return MaterializedTrainingSample(
        schema_version=SCHEMA_VERSION,
        training_label_version=TRAINING_LABEL_VERSION,
        window_id=_window_id(
            dataset_id=conversation.dataset_id,
            sample_id=conversation.sample_id,
            user_side=user_side,
            start_seconds=start_seconds,
        ),
        dataset_id=conversation.dataset_id,
        dataset_name=conversation.dataset_name,
        sample_id=conversation.sample_id,
        external_id=conversation.external_id,
        user_side=user_side,
        assistant_side=assistant_side,
        split=split,
        user_audio_path=_audio_reference(
            conversation.dataset_name, conversation.external_id, user_track, audio_assets
        ).path,
        assistant_audio_path=_audio_reference(
            conversation.dataset_name, conversation.external_id, assistant_track, audio_assets
        ).path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        quality_score=conversation.quality_score,
        category=category,
        assistant_has_floor=tuple(frame.assistant_has_floor_input for frame in frames),
        p_user_has_floor=tuple(
            _masked(frame.user_has_floor_target, frame.user_has_floor_valid) for frame in frames
        ),
        p_user_yield=tuple(
            _masked(frame.user_yield_target, frame.user_yield_valid) for frame in frames
        ),
        p_assistant_backchannel=tuple(
            _masked(frame.assistant_backchannel_target, frame.assistant_backchannel_valid)
            for frame in frames
        ),
        future_activity_0_200=tuple(
            _masked(targets[0].occupancy, targets[0].valid) for targets in future
        ),
        future_activity_200_500=tuple(
            _masked(targets[1].occupancy, targets[1].valid) for targets in future
        ),
        future_activity_500_1000=tuple(
            _masked(targets[2].occupancy, targets[2].valid) for targets in future
        ),
        future_activity_1000_1500=tuple(
            _masked(targets[3].occupancy, targets[3].valid) for targets in future
        ),
        turn_completion=tuple(
            _masked(
                frame.interaction_auxiliary.turn_completion.target,
                frame.interaction_auxiliary.turn_completion.valid,
            )
            for frame in frames
        ),
        continuation_pause=tuple(
            _masked(
                frame.interaction_auxiliary.continuation_pause.target,
                frame.interaction_auxiliary.continuation_pause.valid,
            )
            for frame in frames
        ),
        non_floor_feedback=tuple(
            _masked(
                frame.interaction_auxiliary.non_floor_feedback.target,
                frame.interaction_auxiliary.non_floor_feedback.valid,
            )
            for frame in frames
        ),
        floor_take=tuple(
            _masked(
                frame.interaction_auxiliary.floor_take.target,
                frame.interaction_auxiliary.floor_take.valid,
            )
            for frame in frames
        ),
    )


def _write_training_shards(
    output_directory: Path,
    samples: Sequence[MaterializedTrainingSample],
) -> tuple[ExportShard, ...]:
    inventory: list[ExportShard] = []
    for split in TrainingCorpusSplit:
        split_samples = tuple(sample for sample in samples if sample.split is split)
        shard_directory = output_directory / "training" / split.value
        shard_directory.mkdir(parents=True, exist_ok=True)
        for shard_index, offset in enumerate(range(0, len(split_samples), PARQUET_ROWS_PER_SHARD)):
            shard = split_samples[offset : offset + PARQUET_ROWS_PER_SHARD]
            table = pa.Table.from_pylist(
                [sample.model_dump(mode="json") for sample in shard],
                schema=_training_arrow_schema(),
            )
            shard_path = shard_directory / f"shard-{shard_index:05d}.parquet"
            pq.write_table(table, shard_path, compression="zstd")
            relative_path = shard_path.relative_to(output_directory).as_posix()
            inventory.append(
                ExportShard(
                    split=split,
                    path=relative_path,
                    row_count=len(shard),
                    size_bytes=shard_path.stat().st_size,
                    sha256=_file_sha256(shard_path),
                )
            )
    return tuple(inventory)


def _training_arrow_schema() -> pa.Schema:
    frame_values = pa.list_(pa.float32(), FRAMES_PER_SAMPLE)
    fields = (
        pa.field("schema_version", pa.string()),
        pa.field("training_label_version", pa.string()),
        pa.field("window_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("dataset_name", pa.string()),
        pa.field("sample_id", pa.string()),
        pa.field("external_id", pa.string()),
        pa.field("user_side", pa.string()),
        pa.field("assistant_side", pa.string()),
        pa.field("split", pa.string()),
        pa.field("user_audio_path", pa.string()),
        pa.field("assistant_audio_path", pa.string()),
        pa.field("start_seconds", pa.float64()),
        pa.field("end_seconds", pa.float64()),
        pa.field("quality_score", pa.float64()),
        pa.field("category", pa.string()),
        *(pa.field(name, frame_values) for name in _frame_field_names()),
    )
    return pa.schema(fields)


def _frame_field_names() -> tuple[str, ...]:
    return (
        "assistant_has_floor",
        "p_user_has_floor",
        "p_user_yield",
        "p_assistant_backchannel",
        "future_activity_0_200",
        "future_activity_200_500",
        "future_activity_500_1000",
        "future_activity_1000_1500",
        "turn_completion",
        "continuation_pause",
        "non_floor_feedback",
        "floor_take",
    )


def _split_summaries(
    evidence: Sequence[CorpusAuditEvidence],
    samples: Sequence[MaterializedTrainingSample],
    split_by_sample: dict[UUID, TrainingCorpusSplit],
) -> tuple[ExportSplitSummary, ...]:
    return tuple(
        ExportSplitSummary(
            split=split,
            recording_count=sum(split_by_sample[item.sample_id] is split for item in evidence),
            training_sample_count=sum(sample.split is split for sample in samples),
            source_duration_seconds=sum(
                item.represented_duration_seconds
                for item in evidence
                if split_by_sample[item.sample_id] is split
            ),
        )
        for split in TrainingCorpusSplit
    )


def _window_id(
    dataset_id: UUID,
    sample_id: UUID,
    user_side: TrackSide,
    start_seconds: float,
) -> str:
    start_frame = round(start_seconds / FRAME_SECONDS)
    payload = (
        f"{SCHEMA_VERSION}:{TRAINING_LABEL_VERSION}:{dataset_id}:"
        f"{sample_id}:{user_side.value}:{start_frame}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_export_destination(output_directory: Path) -> None:
    for managed_path in (
        output_directory / "training",
        output_directory / "corpus.json",
    ):
        if managed_path.exists():
            raise ValueError(f"Export-managed output already exists: {managed_path}")
    output_directory.mkdir(parents=True, exist_ok=True)


def _masked(value: float | None, valid: bool) -> float:
    if not valid:
        return -1.0
    if value is None:
        raise ValueError("A valid training target must have a value.")
    return value


def _track(dashboard_sample: DashboardSample, side: TrackSide) -> SampleTrackRecord:
    for track in dashboard_sample.tracks:
        if track.side is side:
            return track
    raise ValueError(f"Missing {side.value} track for {dashboard_sample.sample.external_id}")


def _other_side(side: TrackSide) -> TrackSide:
    return TrackSide.SPEAKER2 if side is TrackSide.SPEAKER1 else TrackSide.SPEAKER1


def _audio_reference(
    dataset_name: str,
    external_id: str,
    track: SampleTrackRecord,
    audio_assets: CorpusAudioAssetCatalog,
) -> AudioReference:
    asset = audio_assets.resolve(
        dataset_name=dataset_name,
        external_id=external_id,
        side=SpeakerSide(track.side.value),
        source_uri=track.access_uri,
        source_sha256=_source_audio_sha256(track),
        source_audio=_source_audio_metadata(track),
    )
    return AudioReference(
        side=track.side,
        path=asset.corpus_relative_path.as_posix(),
        duration_seconds=asset.corpus_audio.duration_seconds,
        sample_rate=asset.corpus_audio.sample_rate,
        channels=asset.corpus_audio.channels,
        audio_sha256=asset.corpus_sha256,
    )


def _source_audio_sha256(track: SampleTrackRecord) -> str:
    if track.audio_sha256 is None:
        raise ValueError(f"Missing source audio hash for {track.id}")
    return track.audio_sha256


def _source_audio_metadata(track: SampleTrackRecord) -> AudioMetadata:
    if (
        track.duration_seconds is None
        or track.sample_rate is None
        or track.channels is None
        or track.sample_count is None
    ):
        raise ValueError(f"Incomplete source audio metadata for {track.id}")
    return AudioMetadata(
        duration_seconds=track.duration_seconds,
        sample_rate=track.sample_rate,
        channels=track.channels,
        sample_count=track.sample_count,
    )


def _recording_directory(
    audio: tuple[AudioReference, AudioReference],
) -> PurePosixPath:
    directories = tuple(PurePosixPath(reference.path).parent for reference in audio)
    if directories[0] != directories[1]:
        raise ValueError("Recording speaker tracks must share one corpus directory.")
    return directories[0]


def parse_arguments() -> TrainingCorpusExportRequest:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-set-name", required=True)
    parser.add_argument("--split-seed", required=True)
    parser.add_argument("--audio-build-manifest", action="append", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args()
    return TrainingCorpusExportRequest(
        review_set_name=arguments.review_set_name,
        split_seed=arguments.split_seed,
        audio_build_manifest_paths=tuple(arguments.audio_build_manifest),
        output_directory=arguments.output_directory,
    )


def main() -> None:
    manifest = export_training_corpus(parse_arguments())
    print(json.dumps(manifest.model_dump(mode="json"), indent=2), flush=True)


if __name__ == "__main__":
    main()

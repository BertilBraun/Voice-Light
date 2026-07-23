from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

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
from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.db.repository import Repository
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.training_samples.models import TrainingFramePreview
from app.local.training_samples.service import (
    FRAME_SECONDS,
    INPUT_DURATION_SECONDS,
    build_frame_previews,
)
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import METRIC_VERSION, ConversationAnnotation, TrackVadResult

SCHEMA_VERSION = "voice-light-turn-taking-v1"
FRAMES_PER_SAMPLE = int(round(INPUT_DURATION_SECONDS / FRAME_SECONDS))
PARQUET_ROWS_PER_SHARD = 1000
HUGGING_FACE_S3_BUCKET = "liveroom-assets"


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
    sample_id: UUID
    external_id: str
    duration_seconds: float
    quality_score: float
    metric_version: str
    annotation: ConversationAnnotation
    speaker1_vad: TrackVadResult
    speaker2_vad: TrackVadResult
    conversation_regions: ConversationRegionAnalysis
    audio: tuple[AudioReference, AudioReference]


class MaterializedTrainingSample(FrozenBaseModel):
    schema_version: str
    dataset_name: str
    sample_id: UUID
    external_id: str
    user_side: TrackSide
    assistant_side: TrackSide
    user_audio_path: str
    assistant_audio_path: str
    start_seconds: float
    end_seconds: float
    quality_score: float
    category: str
    assistant_has_floor: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    p_user_has_floor: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    p_user_yield: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    p_assistant_backchannel: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    future_activity_0_200: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    future_activity_200_500: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    future_activity_500_1000: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    future_activity_1000_1500: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    turn_completion: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    continuation_pause: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    non_floor_feedback: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)
    floor_take: tuple[float, ...] = Field(min_length=FRAMES_PER_SAMPLE)


class ExportManifest(FrozenBaseModel):
    schema_version: str
    generated_at: datetime
    metric_version: str
    annotation_version: str
    dataset_ids: tuple[UUID, ...]
    minimum_quality: float
    recording_count: int
    training_sample_count: int
    shard_count: int


class TrainingCorpusExportRequest(FrozenBaseModel):
    dataset_ids: tuple[UUID, ...] = Field(min_length=1)
    output_directory: Path
    minimum_quality: float = Field(default=0.95, ge=0.0, le=1.0)


def export_training_corpus(request: TrainingCorpusExportRequest) -> ExportManifest:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for corpus export.")
    if request.output_directory.exists():
        raise ValueError(f"Export output already exists: {request.output_directory}")
    evidence = CorpusAuditRepository(DATABASE_URL).load_evidence(
        dataset_ids=request.dataset_ids,
        minimum_quality=request.minimum_quality,
        metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
        region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
    )
    vad_by_sample = _vad_by_sample_id(database_url=DATABASE_URL, dataset_ids=request.dataset_ids)
    sample_repository = Repository(DATABASE_URL)
    request.output_directory.mkdir(parents=True)
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
        )
        training_samples.extend(
            _training_samples_for_conversation(
                conversation=conversation,
                dashboard_sample=dashboard_sample,
                minimum_quality=request.minimum_quality,
            )
        )
    shard_count = _write_training_shards(
        output_directory=request.output_directory,
        samples=training_samples,
    )
    manifest = ExportManifest(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(UTC),
        metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
        dataset_ids=request.dataset_ids,
        minimum_quality=request.minimum_quality,
        recording_count=len(evidence),
        training_sample_count=len(training_samples),
        shard_count=shard_count,
    )
    (request.output_directory / "metadata" / "manifest.json").parent.mkdir(parents=True)
    (request.output_directory / "metadata" / "manifest.json").write_text(
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


def _write_recording_metadata(
    output_directory: Path,
    conversation: CorpusAuditEvidence,
    dashboard_sample: DashboardSample,
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
) -> None:
    if conversation.conversation_regions is None:
        raise ValueError(f"Missing conversation regions for {conversation.external_id}")
    speaker1 = _track(dashboard_sample, TrackSide.SPEAKER1)
    speaker2 = _track(dashboard_sample, TrackSide.SPEAKER2)
    metadata = RecordingMetadata(
        schema_version=SCHEMA_VERSION,
        dataset_name=conversation.dataset_name,
        sample_id=conversation.sample_id,
        external_id=conversation.external_id,
        duration_seconds=conversation.represented_duration_seconds,
        quality_score=conversation.quality_score,
        metric_version=METRIC_VERSION,
        annotation=conversation.annotation,
        speaker1_vad=speaker1_vad,
        speaker2_vad=speaker2_vad,
        conversation_regions=conversation.conversation_regions,
        audio=(
            _audio_reference(conversation.dataset_name, conversation.external_id, speaker1),
            _audio_reference(conversation.dataset_name, conversation.external_id, speaker2),
        ),
    )
    metadata_path = (
        output_directory
        / _recording_directory(
            dataset_name=conversation.dataset_name,
            external_id=conversation.external_id,
        )
        / "metadata.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def _training_samples_for_conversation(
    conversation: CorpusAuditEvidence,
    dashboard_sample: DashboardSample,
    minimum_quality: float,
) -> Iterator[MaterializedTrainingSample]:
    if conversation.conversation_regions is None:
        return
    request = CorpusAuditRequest(
        dataset_ids=(conversation.dataset_id,),
        minimum_quality=minimum_quality,
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
                frames=frames,
            )


def _materialized_training_sample(
    conversation: CorpusAuditEvidence,
    dashboard_sample: DashboardSample,
    user_side: TrackSide,
    start_seconds: float,
    end_seconds: float,
    category: str,
    frames: tuple[TrainingFramePreview, ...],
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
        dataset_name=conversation.dataset_name,
        sample_id=conversation.sample_id,
        external_id=conversation.external_id,
        user_side=user_side,
        assistant_side=assistant_side,
        user_audio_path=_audio_reference(
            conversation.dataset_name, conversation.external_id, user_track
        ).path,
        assistant_audio_path=_audio_reference(
            conversation.dataset_name, conversation.external_id, assistant_track
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
) -> int:
    shard_directory = output_directory / "training" / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    shard_count = 0
    for offset in range(0, len(samples), PARQUET_ROWS_PER_SHARD):
        shard = samples[offset : offset + PARQUET_ROWS_PER_SHARD]
        table = pa.Table.from_pylist([sample.model_dump(mode="json") for sample in shard])
        pq.write_table(
            table, shard_directory / f"shard-{shard_count:05d}.parquet", compression="zstd"
        )
        shard_count += 1
    return shard_count


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
) -> AudioReference:
    if track.duration_seconds is None or track.sample_rate is None or track.channels is None:
        raise ValueError(f"Incomplete audio metadata for {track.id}")
    if track.audio_sha256 is None:
        raise ValueError(f"Missing materialized audio hash for {track.id}")
    return AudioReference(
        side=track.side,
        path=_audio_path(dataset_name=dataset_name, external_id=external_id, track=track),
        duration_seconds=track.duration_seconds,
        sample_rate=track.sample_rate,
        channels=track.channels,
        audio_sha256=track.audio_sha256,
    )


def _audio_path(dataset_name: str, external_id: str, track: SampleTrackRecord) -> str:
    if dataset_name == "luel-local":
        return f"luel/{external_id}/{external_id}_{track.side.value}.flac"
    prefix = f"s3://{HUGGING_FACE_S3_BUCKET}/"
    if dataset_name == "meetings-s3" and track.storage_uri.startswith(prefix):
        return track.storage_uri.removeprefix(prefix)
    raise ValueError(f"No Hugging Face path mapping for {dataset_name}: {track.storage_uri}")


def _recording_directory(dataset_name: str, external_id: str) -> PurePosixPath:
    if dataset_name == "luel-local":
        return PurePosixPath("luel") / external_id
    if dataset_name == "meetings-s3":
        return PurePosixPath(external_id)
    raise ValueError(f"No metadata path mapping for dataset: {dataset_name}")


def parse_arguments() -> TrainingCorpusExportRequest:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", action="append", required=True, type=UUID)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--minimum-quality", default=0.95, type=float)
    arguments = parser.parse_args()
    return TrainingCorpusExportRequest(
        dataset_ids=tuple(arguments.dataset_id),
        output_directory=arguments.output_directory,
        minimum_quality=arguments.minimum_quality,
    )


def main() -> None:
    manifest = export_training_corpus(parse_arguments())
    print(json.dumps(manifest.model_dump(mode="json"), indent=2), flush=True)


if __name__ == "__main__":
    main()

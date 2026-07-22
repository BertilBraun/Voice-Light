from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import Field

from app.local.alignment_migration.riff import (
    PcmWaveMetadata,
    WaveTimelineSegment,
    inspect_pcm_wave,
    rewrite_pcm_wave_timeline,
    sha256_file,
)
from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.full_recording_service import prepare_full_recording_audio
from app.local.asr.transcript import SpeakerTrack, Word
from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.db.repository import Repository
from app.local.ingestion.local_audio import materialize_sample_track
from app.local.timeline_repair.generation import timeline_repair
from app.local.timeline_repair.models import (
    TIMELINE_MATERIALIZATION_VERSION,
    TimelineRepairPlanRecord,
    TimelineRepairScope,
)
from app.local.timeline_repair.repository import TimelineRepairRepository
from app.local.timeline_repair.transform import transform_words
from app.shared.asr import TimestampedWord
from app.shared.base_model import FrozenBaseModel


class TimelineMaterializationSummary(FrozenBaseModel):
    planned_count: int = Field(ge=0)
    materialized_count: int = Field(ge=0)
    already_materialized_count: int = Field(ge=0)
    transcript_count: int = Field(ge=0)


@dataclass(frozen=True)
class MaterializedTranscript:
    id: UUID
    words: tuple[TimestampedWord, ...]


@dataclass(frozen=True)
class StagedMaterialization:
    path: Path
    audio_sha256: str
    prepared_audio_sha256: str
    prepared_duration_seconds: float
    source_duration_seconds: float
    metadata: PcmWaveMetadata
    transcripts: tuple[MaterializedTranscript, MaterializedTranscript]


class TimelineMaterializationService:
    def __init__(self, database_url: str, sessions_root: Path) -> None:
        self.database_url = database_url
        self.sessions_root = sessions_root.resolve(strict=True)
        self.plan_repository = TimelineRepairRepository(database_url)
        self.sample_repository = Repository(database_url)
        self.transcript_repository = FullRecordingAsrRepository(database_url)

    def dry_run(self) -> TimelineMaterializationSummary:
        plans = self.plan_repository.list_plans()
        transcript_count = 0
        materialized_count = 0
        already_materialized_count = 0
        for plan in plans:
            dashboard_sample = self.sample_repository.get_dashboard_sample(plan.sample_id)
            self._validate_plan(plan=plan, dashboard_sample=dashboard_sample)
            records = self._speaker2_transcripts(plan)
            transcript_count += len(records)
            if plan.is_materialized:
                self._validate_materialized(plan=plan, dashboard_sample=dashboard_sample)
                already_materialized_count += 1
            else:
                self._validate_source(plan=plan, dashboard_sample=dashboard_sample)
                materialized_count += 1
        return TimelineMaterializationSummary(
            planned_count=len(plans),
            materialized_count=materialized_count,
            already_materialized_count=already_materialized_count,
            transcript_count=transcript_count,
        )

    def apply(self) -> TimelineMaterializationSummary:
        plans = self.plan_repository.list_plans()
        materialized_count = 0
        already_materialized_count = 0
        transcript_count = 0
        for plan in plans:
            dashboard_sample = self.sample_repository.get_dashboard_sample(plan.sample_id)
            self._validate_plan(plan=plan, dashboard_sample=dashboard_sample)
            records = self._speaker2_transcripts(plan)
            transcript_count += len(records)
            speaker2_track = _track(dashboard_sample, TrackSide.SPEAKER2)
            canonical_path = self._canonical_path(speaker2_track)
            if plan.is_materialized:
                self._validate_materialized(plan=plan, dashboard_sample=dashboard_sample)
                _cleanup_completed_artifacts(canonical_path)
                already_materialized_count += 1
                continue
            self._materialize_one(
                plan=plan,
                speaker2_track=speaker2_track,
                canonical_path=canonical_path,
                records=records,
            )
            materialized_count += 1
        return TimelineMaterializationSummary(
            planned_count=len(plans),
            materialized_count=materialized_count,
            already_materialized_count=already_materialized_count,
            transcript_count=transcript_count,
        )

    def _materialize_one(
        self,
        plan: TimelineRepairPlanRecord,
        speaker2_track: SampleTrackRecord,
        canonical_path: Path,
        records: tuple[FullRecordingAsrTranscriptRecord, FullRecordingAsrTranscriptRecord],
    ) -> None:
        backup_path = _backup_path(canonical_path)
        canonical_hash = sha256_file(canonical_path)
        replaced = False
        if canonical_hash == plan.speaker2_audio_sha256:
            if backup_path.exists():
                raise ValueError(f"Unexpected materialization backup exists: {backup_path}")
            staged = self._stage(plan=plan, source_path=canonical_path, records=records)
            try:
                os.replace(canonical_path, backup_path)
                os.replace(staged.path, canonical_path)
            except Exception:
                if backup_path.exists() and not canonical_path.exists():
                    os.replace(backup_path, canonical_path)
                raise
            replaced = True
        elif backup_path.is_file() and sha256_file(backup_path) == plan.speaker2_audio_sha256:
            expected = self._stage(plan=plan, source_path=backup_path, records=records)
            if sha256_file(canonical_path) != expected.audio_sha256:
                expected.path.unlink(missing_ok=True)
                raise ValueError(
                    f"Interrupted materialization has an unexpected canonical WAV: {canonical_path}"
                )
            expected.path.unlink(missing_ok=True)
            staged = expected
        else:
            raise ValueError(f"Source WAV provenance changed for {plan.external_id}.")
        try:
            self._reconcile_database(
                plan=plan,
                speaker2_track=speaker2_track,
                staged=staged,
            )
        except Exception:
            if replaced:
                failed_path = _staged_path(canonical_path)
                os.replace(canonical_path, failed_path)
                os.replace(backup_path, canonical_path)
                failed_path.unlink(missing_ok=True)
            raise
        backup_path.unlink(missing_ok=True)

    def _stage(
        self,
        plan: TimelineRepairPlanRecord,
        source_path: Path,
        records: tuple[FullRecordingAsrTranscriptRecord, FullRecordingAsrTranscriptRecord],
    ) -> StagedMaterialization:
        staged_path = _staged_path(source_path)
        staged_path.unlink(missing_ok=True)
        source_metadata = inspect_pcm_wave(source_path)
        segments = materialized_segments(plan=plan, metadata=source_metadata)
        rewritten = rewrite_pcm_wave_timeline(
            source_path=source_path,
            output_path=staged_path,
            segments=segments,
        )
        staged_metadata = inspect_pcm_wave(staged_path)
        source_duration_seconds = staged_metadata.frame_count / staged_metadata.sample_rate
        prepared = prepare_full_recording_audio(
            source_path=staged_path,
            source_audio_sha256=rewritten.sha256,
        )
        try:
            transcripts = tuple(
                MaterializedTranscript(
                    id=record.id,
                    words=materialized_words(
                        record.words,
                        plan=plan,
                        duration_seconds=source_duration_seconds,
                    ),
                )
                for record in records
            )
            assert len(transcripts) == 2
            return StagedMaterialization(
                path=staged_path,
                audio_sha256=rewritten.sha256,
                prepared_audio_sha256=prepared.prepared_audio_sha256,
                prepared_duration_seconds=prepared.prepared_duration_seconds,
                source_duration_seconds=source_duration_seconds,
                metadata=staged_metadata,
                transcripts=(transcripts[0], transcripts[1]),
            )
        finally:
            prepared.path.unlink(missing_ok=True)

    def _reconcile_database(
        self,
        plan: TimelineRepairPlanRecord,
        speaker2_track: SampleTrackRecord,
        staged: StagedMaterialization,
    ) -> None:
        annotation = plan.derived_annotation
        if annotation is None:
            raise ValueError("Cannot materialize a plan without derived annotations.")
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            locked_plan = connection.execute(
                "SELECT * FROM virtual_timeline_repair_plans WHERE id = %s FOR UPDATE",
                (plan.id,),
            ).fetchone()
            if locked_plan is None or locked_plan["materialized_at"] is not None:
                raise ValueError("Timeline repair plan changed during materialization.")
            locked_track = connection.execute(
                "SELECT audio_sha256 FROM sample_tracks WHERE id = %s FOR UPDATE",
                (speaker2_track.id,),
            ).fetchone()
            if locked_track is None or locked_track["audio_sha256"] != plan.speaker2_audio_sha256:
                raise ValueError("Speaker 2 track provenance changed during materialization.")
            expected_transcript_ids = {transcript.id for transcript in staged.transcripts}
            transcript_rows = connection.execute(
                """
                SELECT id, source_audio_sha256
                FROM full_recording_asr_transcripts
                WHERE id = ANY(%s)
                FOR UPDATE
                """,
                (list(expected_transcript_ids),),
            ).fetchall()
            if {row["id"] for row in transcript_rows} != expected_transcript_ids:
                raise ValueError("Speaker 2 transcript rows changed during materialization.")
            if any(
                row["source_audio_sha256"] != plan.speaker2_audio_sha256 for row in transcript_rows
            ):
                raise ValueError("Speaker 2 transcript provenance changed during materialization.")
            for transcript in staged.transcripts:
                connection.execute(
                    """
                    UPDATE full_recording_asr_transcripts
                    SET source_audio_sha256 = %s,
                        prepared_audio_sha256 = %s,
                        source_duration_seconds = %s,
                        prepared_duration_seconds = %s,
                        words = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (
                        staged.audio_sha256,
                        staged.prepared_audio_sha256,
                        staged.source_duration_seconds,
                        staged.prepared_duration_seconds,
                        Jsonb([word.model_dump(mode="json") for word in transcript.words]),
                        transcript.id,
                    ),
                )
            connection.execute(
                """
                UPDATE sample_tracks
                SET duration_seconds = %s,
                    sample_rate = %s,
                    channels = %s,
                    sample_count = %s,
                    audio_sha256 = %s,
                    source_size_bytes = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    staged.source_duration_seconds,
                    staged.metadata.sample_rate,
                    staged.metadata.channel_count,
                    staged.metadata.frame_count,
                    staged.audio_sha256,
                    staged.metadata.file_size,
                    speaker2_track.id,
                ),
            )
            connection.execute(
                """
                UPDATE audio_metadata
                SET duration_seconds = %s,
                    sample_rate = %s,
                    channels = %s,
                    sample_count = %s,
                    payload = %s,
                    updated_at = now()
                WHERE sample_track_id = %s
                """,
                (
                    staged.source_duration_seconds,
                    staged.metadata.sample_rate,
                    staged.metadata.channel_count,
                    staged.metadata.frame_count,
                    Jsonb(
                        {
                            "duration_seconds": staged.source_duration_seconds,
                            "sample_rate": staged.metadata.sample_rate,
                            "channels": staged.metadata.channel_count,
                            "sample_count": staged.metadata.frame_count,
                        }
                    ),
                    speaker2_track.id,
                ),
            )
            connection.execute(
                """
                UPDATE quality_results
                SET payload = jsonb_set(payload, '{conversation_annotation}', %s, true),
                    conversation_quality_score = %s,
                    interaction_count = %s,
                    speech_segment_count = %s,
                    turn_count = %s,
                    turn_taking_count = %s,
                    pause_count = %s,
                    backchannel_count = %s,
                    interruption_count = %s,
                    usable_event_count = %s,
                    conversation_events_per_hour = %s,
                    annotation_duration_seconds = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    Jsonb(annotation.model_dump(mode="json")),
                    annotation.quality_score,
                    annotation.interaction_count,
                    annotation.speech_segment_count,
                    annotation.turn_count,
                    annotation.turn_taking_count,
                    annotation.pause_count,
                    annotation.backchannel_count,
                    annotation.interruption_count,
                    annotation.usable_event_count,
                    annotation.events_per_hour,
                    annotation.analyzed_duration_seconds,
                    plan.quality_result_id,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE virtual_timeline_repair_plans
                SET materialization_version = %s,
                    materialized_speaker2_audio_sha256 = %s,
                    materialized_prepared_audio_sha256 = %s,
                    materialized_at = now()
                WHERE id = %s AND materialized_at IS NULL
                """,
                (
                    TIMELINE_MATERIALIZATION_VERSION,
                    staged.audio_sha256,
                    staged.prepared_audio_sha256,
                    plan.id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Failed to record timeline materialization.")

    def _speaker2_transcripts(
        self,
        plan: TimelineRepairPlanRecord,
    ) -> tuple[FullRecordingAsrTranscriptRecord, FullRecordingAsrTranscriptRecord]:
        records = self.transcript_repository.get_transcripts_by_ids(
            (plan.speaker2_parakeet_transcript_id, plan.speaker2_canary_transcript_id)
        )
        if len(records) != 2 or any(record.side is not TrackSide.SPEAKER2 for record in records):
            raise ValueError(f"Repair transcripts are invalid for {plan.external_id}.")
        return records[0], records[1]

    def _validate_plan(
        self,
        plan: TimelineRepairPlanRecord,
        dashboard_sample: DashboardSample,
    ) -> None:
        if dashboard_sample.sample.is_unusable:
            raise ValueError(f"Unusable sample has a repair plan: {plan.external_id}.")
        if plan.derived_annotation is None or plan.conversation_regions is None:
            raise ValueError(f"Repair plan is incomplete: {plan.external_id}.")
        if len(dashboard_sample.tracks) != 2:
            raise ValueError(f"Repair sample does not have exactly two tracks: {plan.external_id}.")
        speaker1 = _track(dashboard_sample, TrackSide.SPEAKER1)
        if speaker1.audio_sha256 != plan.speaker1_audio_sha256:
            raise ValueError(f"Speaker 1 provenance changed for {plan.external_id}.")

    def _validate_source(
        self,
        plan: TimelineRepairPlanRecord,
        dashboard_sample: DashboardSample,
    ) -> None:
        speaker2 = _track(dashboard_sample, TrackSide.SPEAKER2)
        if speaker2.audio_sha256 != plan.speaker2_audio_sha256:
            raise ValueError(f"Speaker 2 provenance changed for {plan.external_id}.")
        if sha256_file(self._canonical_path(speaker2)) != plan.speaker2_audio_sha256:
            raise ValueError(f"Speaker 2 WAV hash changed for {plan.external_id}.")

    def _validate_materialized(
        self,
        plan: TimelineRepairPlanRecord,
        dashboard_sample: DashboardSample,
    ) -> None:
        expected_hash = plan.materialized_speaker2_audio_sha256
        if expected_hash is None:
            raise ValueError(f"Materialized plan lacks an audio hash: {plan.external_id}.")
        speaker2 = _track(dashboard_sample, TrackSide.SPEAKER2)
        if speaker2.audio_sha256 != expected_hash:
            raise ValueError(f"Materialized track hash changed for {plan.external_id}.")
        if sha256_file(self._canonical_path(speaker2)) != expected_hash:
            raise ValueError(f"Materialized WAV hash changed for {plan.external_id}.")

    def _canonical_path(self, track: SampleTrackRecord) -> Path:
        path = materialize_sample_track(track).resolve(strict=True)
        if self.sessions_root not in path.parents:
            raise ValueError(f"Track path is outside the sessions root: {path}")
        return path


def materialized_segments(
    plan: TimelineRepairPlanRecord,
    metadata: PcmWaveMetadata,
) -> tuple[WaveTimelineSegment, ...]:
    frame_count = metadata.frame_count
    sample_rate = metadata.sample_rate
    if plan.repair_scope is TimelineRepairScope.GLOBAL_OFFSET:
        return _shifted_interval_segments(
            output_start_frame=0,
            output_end_frame=frame_count,
            shift_frame_count=round(plan.second_part_shift_seconds * sample_rate),
            source_frame_count=frame_count,
        )
    assert plan.exclusion_start_seconds is not None
    assert plan.exclusion_end_seconds is not None
    exclusion_start_frame = min(frame_count, round(plan.exclusion_start_seconds * sample_rate))
    exclusion_end_frame = min(frame_count, round(plan.exclusion_end_seconds * sample_rate))
    segments = (
        *_shifted_interval_segments(
            output_start_frame=0,
            output_end_frame=exclusion_start_frame,
            shift_frame_count=round(plan.first_part_shift_seconds * sample_rate),
            source_frame_count=frame_count,
        ),
        WaveTimelineSegment(
            output_frame_count=exclusion_end_frame - exclusion_start_frame,
            source_start_frame=None,
        ),
        *_shifted_interval_segments(
            output_start_frame=exclusion_end_frame,
            output_end_frame=frame_count,
            shift_frame_count=round(plan.second_part_shift_seconds * sample_rate),
            source_frame_count=frame_count,
        ),
    )
    return _merge_segments(segments)


def materialized_words(
    words: tuple[TimestampedWord, ...],
    plan: TimelineRepairPlanRecord,
    duration_seconds: float,
) -> tuple[TimestampedWord, ...]:
    transformed = transform_words(
        tuple(Word.model_validate(word.model_dump()) for word in words),
        track=SpeakerTrack.SPEAKER2,
        repair=timeline_repair(plan),
    )
    materialized: list[TimestampedWord] = []
    for word in transformed:
        assert word.start_seconds is not None
        assert word.end_seconds is not None
        start_seconds = max(0.0, word.start_seconds)
        end_seconds = min(duration_seconds, word.end_seconds)
        if end_seconds <= start_seconds:
            continue
        materialized.append(
            TimestampedWord(
                text=word.text,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                confidence=word.confidence,
            )
        )
    return tuple(materialized)


def _shifted_interval_segments(
    output_start_frame: int,
    output_end_frame: int,
    shift_frame_count: int,
    source_frame_count: int,
) -> tuple[WaveTimelineSegment, ...]:
    if output_end_frame < output_start_frame:
        raise ValueError("Output interval end precedes its start.")
    valid_start = max(output_start_frame, shift_frame_count)
    valid_end = min(output_end_frame, source_frame_count + shift_frame_count)
    segments: list[WaveTimelineSegment] = []
    if valid_start > output_start_frame:
        segments.append(
            WaveTimelineSegment(
                output_frame_count=valid_start - output_start_frame,
                source_start_frame=None,
            )
        )
    if valid_end > valid_start:
        segments.append(
            WaveTimelineSegment(
                output_frame_count=valid_end - valid_start,
                source_start_frame=valid_start - shift_frame_count,
            )
        )
    if output_end_frame > max(valid_end, output_start_frame):
        segments.append(
            WaveTimelineSegment(
                output_frame_count=output_end_frame - max(valid_end, output_start_frame),
                source_start_frame=None,
            )
        )
    return tuple(segments)


def _merge_segments(
    segments: tuple[WaveTimelineSegment, ...],
) -> tuple[WaveTimelineSegment, ...]:
    merged: list[WaveTimelineSegment] = []
    for segment in segments:
        if segment.output_frame_count == 0:
            continue
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        contiguous_source = (
            previous.source_start_frame is not None
            and segment.source_start_frame
            == previous.source_start_frame + previous.output_frame_count
        )
        if (previous.source_start_frame is None and segment.source_start_frame is None) or (
            contiguous_source
        ):
            merged[-1] = WaveTimelineSegment(
                output_frame_count=previous.output_frame_count + segment.output_frame_count,
                source_start_frame=previous.source_start_frame,
            )
        else:
            merged.append(segment)
    return tuple(merged)


def _track(dashboard_sample: DashboardSample, side: TrackSide) -> SampleTrackRecord:
    matches = tuple(track for track in dashboard_sample.tracks if track.side is side)
    if len(matches) != 1:
        raise ValueError(f"Expected one {side.value} track.")
    return matches[0]


def _staged_path(canonical_path: Path) -> Path:
    return canonical_path.with_name(f"{canonical_path.name}.timeline-repair.staged")


def _backup_path(canonical_path: Path) -> Path:
    return canonical_path.with_name(f"{canonical_path.name}.timeline-repair.bak")


def _cleanup_completed_artifacts(canonical_path: Path) -> None:
    _staged_path(canonical_path).unlink(missing_ok=True)
    _backup_path(canonical_path).unlink(missing_ok=True)

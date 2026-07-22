from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from app.local.analyses.asr.merger import parakeet_canary_union_words
from app.local.analyses.asr.service import word_from_timestamped_word
from app.local.analyses.end_of_turn.detectors.naive_vad import (
    run_naive_vad_floor_full_recording,
)
from app.local.analyses.end_of_turn.service import SpeechSegment as DetectorSpeechSegment
from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.transcript import SpeakerTrack, Word
from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionConfig,
)
from app.local.conversation_regions.service import analyze_conversation_regions
from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.ingestion.conversation import analyze_conversation_evidence
from app.local.ingestion.local_audio import materialize_sample_track
from app.local.timeline_repair.models import (
    TimelineRepairDerivedArtifacts,
    TimelineRepairPlanRecord,
    TimelineRepairScope,
)
from app.local.timeline_repair.transform import (
    GlobalTimelineRepair,
    PiecewiseTimelineRepair,
    TimelineRepair,
    transform_speech_segments,
    transform_words,
)
from app.shared.asr import AsrModelId
from app.shared.quality import (
    SpeakerSide,
    SpeechSegment,
    TrackVadResult,
)

DERIVATION_VERSION = "virtual-timeline-annotation-v1"
VAD_FRAME_SECONDS = 0.03
VAD_MINIMUM_SPEECH_SECONDS = 0.12
VAD_MINIMUM_SILENCE_SECONDS = 0.4


@dataclass(frozen=True)
class RepairTranscripts:
    speaker1_parakeet: FullRecordingAsrTranscriptRecord
    speaker2_parakeet: FullRecordingAsrTranscriptRecord
    speaker1_canary: FullRecordingAsrTranscriptRecord
    speaker2_canary: FullRecordingAsrTranscriptRecord


def generate_derived_artifacts(
    plan: TimelineRepairPlanRecord,
    dashboard_sample: DashboardSample,
    transcript_repository: FullRecordingAsrRepository,
) -> TimelineRepairDerivedArtifacts:
    repair = timeline_repair(plan)
    tracks = _validated_tracks(plan=plan, dashboard_sample=dashboard_sample)
    transcripts = _load_validated_transcripts(
        plan=plan,
        transcript_repository=transcript_repository,
    )
    speaker1_words = _merged_words(
        parakeet=transcripts.speaker1_parakeet,
        canary=transcripts.speaker1_canary,
        track=SpeakerTrack.SPEAKER1,
        repair=repair,
        duration_seconds=plan.duration_seconds,
    )
    speaker2_words = _merged_words(
        parakeet=transcripts.speaker2_parakeet,
        canary=transcripts.speaker2_canary,
        track=SpeakerTrack.SPEAKER2,
        repair=repair,
        duration_seconds=plan.duration_seconds,
    )
    speaker1_segments = _activity_segments(
        path=materialize_sample_track(tracks[TrackSide.SPEAKER1]),
        track=SpeakerTrack.SPEAKER1,
        repair=repair,
        duration_seconds=plan.duration_seconds,
    )
    speaker2_segments = _activity_segments(
        path=materialize_sample_track(tracks[TrackSide.SPEAKER2]),
        track=SpeakerTrack.SPEAKER2,
        repair=repair,
        duration_seconds=plan.duration_seconds,
    )
    annotation = analyze_conversation_evidence(
        speaker1_words=speaker1_words,
        speaker2_words=speaker2_words,
        speaker1_activity_segments=_detector_segments(speaker1_segments),
        speaker2_activity_segments=_detector_segments(speaker2_segments),
        duration_seconds=plan.duration_seconds,
    )
    annotation_version = f"{DERIVATION_VERSION}:{plan.plan_fingerprint}"
    repaired_annotation = annotation.model_copy(update={"annotation_version": annotation_version})
    regions = analyze_conversation_regions(
        speaker1_vad=_track_vad(
            side=SpeakerSide.SPEAKER1,
            segments=speaker1_segments,
            duration_seconds=plan.duration_seconds,
        ),
        speaker2_vad=_track_vad(
            side=SpeakerSide.SPEAKER2,
            segments=speaker2_segments,
            duration_seconds=plan.duration_seconds,
        ),
        annotation=repaired_annotation,
        config=ConversationRegionConfig(),
    )
    return TimelineRepairDerivedArtifacts(
        annotation_version=annotation_version,
        annotation=repaired_annotation,
        conversation_regions_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        conversation_regions=regions,
    )


def timeline_repair(plan: TimelineRepairPlanRecord) -> TimelineRepair:
    if plan.is_materialized:
        if plan.repair_scope is TimelineRepairScope.GLOBAL_OFFSET:
            return GlobalTimelineRepair(shift_seconds=0.0)
        assert plan.change_point_seconds is not None
        assert plan.exclusion_start_seconds is not None
        assert plan.exclusion_end_seconds is not None
        return PiecewiseTimelineRepair(
            first_shift_seconds=0.0,
            second_shift_seconds=0.0,
            canonical_transition_seconds=plan.change_point_seconds,
            exclusion_start_seconds=plan.exclusion_start_seconds,
            exclusion_end_seconds=plan.exclusion_end_seconds,
        )
    match plan.repair_scope:
        case TimelineRepairScope.GLOBAL_OFFSET:
            return GlobalTimelineRepair(shift_seconds=plan.second_part_shift_seconds)
        case TimelineRepairScope.AFTER_CHANGE_POINT:
            assert plan.change_point_seconds is not None
            assert plan.exclusion_start_seconds is not None
            assert plan.exclusion_end_seconds is not None
            return PiecewiseTimelineRepair(
                first_shift_seconds=plan.first_part_shift_seconds,
                second_shift_seconds=plan.second_part_shift_seconds,
                canonical_transition_seconds=plan.change_point_seconds,
                exclusion_start_seconds=plan.exclusion_start_seconds,
                exclusion_end_seconds=plan.exclusion_end_seconds,
            )


def _validated_tracks(
    plan: TimelineRepairPlanRecord,
    dashboard_sample: DashboardSample,
) -> dict[TrackSide, SampleTrackRecord]:
    if dashboard_sample.sample.id != plan.sample_id:
        raise ValueError("Repair plan belongs to a different sample.")
    if dashboard_sample.sample.is_unusable:
        raise ValueError("Unusable samples cannot produce repaired artifacts.")
    quality = dashboard_sample.latest_quality
    if quality is None or quality.id != plan.quality_result_id:
        raise ValueError("Repair plan quality provenance is no longer current.")
    tracks = {track.side: track for track in dashboard_sample.tracks}
    if set(tracks) != {TrackSide.SPEAKER1, TrackSide.SPEAKER2}:
        raise ValueError("Repair requires exactly one track for each speaker.")
    expected_speaker2_hash = (
        plan.materialized_speaker2_audio_sha256
        if plan.is_materialized
        else plan.speaker2_audio_sha256
    )
    assert expected_speaker2_hash is not None
    expected_hashes = {
        TrackSide.SPEAKER1: plan.speaker1_audio_sha256,
        TrackSide.SPEAKER2: expected_speaker2_hash,
    }
    for side, expected_hash in expected_hashes.items():
        if tracks[side].audio_sha256 != expected_hash:
            raise ValueError(f"Repair plan audio provenance is stale for {side.value}.")
    return tracks


def _load_validated_transcripts(
    plan: TimelineRepairPlanRecord,
    transcript_repository: FullRecordingAsrRepository,
) -> RepairTranscripts:
    transcript_ids = (
        plan.speaker1_parakeet_transcript_id,
        plan.speaker2_parakeet_transcript_id,
        plan.speaker1_canary_transcript_id,
        plan.speaker2_canary_transcript_id,
    )
    records = transcript_repository.get_transcripts_by_ids(transcript_ids)
    expected_speaker2_hash = (
        plan.materialized_speaker2_audio_sha256
        if plan.is_materialized
        else plan.speaker2_audio_sha256
    )
    assert expected_speaker2_hash is not None
    expected = (
        (TrackSide.SPEAKER1, AsrModelId.PARAKEET_TDT, plan.speaker1_audio_sha256),
        (TrackSide.SPEAKER2, AsrModelId.PARAKEET_TDT, expected_speaker2_hash),
        (TrackSide.SPEAKER1, AsrModelId.CANARY, plan.speaker1_audio_sha256),
        (TrackSide.SPEAKER2, AsrModelId.CANARY, expected_speaker2_hash),
    )
    for record, (side, model_id, audio_sha256) in zip(records, expected, strict=True):
        if record.sample_id != plan.sample_id:
            raise ValueError("Repair transcript belongs to a different sample.")
        if record.side is not side or record.model_id is not model_id:
            raise ValueError("Repair transcript provenance has the wrong side or model.")
        if record.source_audio_sha256 != audio_sha256:
            raise ValueError("Repair transcript audio provenance is stale.")
        if record.error is not None:
            raise ValueError("Repair transcript contains an ASR error.")
    return RepairTranscripts(*records)


def _merged_words(
    parakeet: FullRecordingAsrTranscriptRecord,
    canary: FullRecordingAsrTranscriptRecord,
    track: SpeakerTrack,
    repair: TimelineRepair,
    duration_seconds: float,
) -> tuple[Word, ...]:
    parakeet_words = transform_words(
        tuple(word_from_timestamped_word(word) for word in parakeet.words),
        track=track,
        repair=repair,
    )
    canary_words = transform_words(
        tuple(word_from_timestamped_word(word) for word in canary.words),
        track=track,
        repair=repair,
    )
    merged = parakeet_canary_union_words(
        parakeet_words=parakeet_words,
        canary_words=canary_words,
    )
    return tuple(
        clipped for word in merged if (clipped := _clip_word(word, duration_seconds)) is not None
    )


def _clip_word(word: Word, duration_seconds: float) -> Word | None:
    if word.start_seconds is None or word.end_seconds is None:
        raise ValueError(f"Cannot use untimed repaired word: {word.text}")
    start_seconds = max(0.0, word.start_seconds)
    end_seconds = min(duration_seconds, word.end_seconds)
    if end_seconds <= start_seconds:
        return None
    return word.model_copy(update={"start_seconds": start_seconds, "end_seconds": end_seconds})


def _activity_segments(
    path: Path,
    track: SpeakerTrack,
    repair: TimelineRepair,
    duration_seconds: float,
) -> tuple[SpeechSegment, ...]:
    result = run_naive_vad_floor_full_recording(
        wave_path=path,
        result_name=DERIVATION_VERSION,
        description="Energy activity for immutable virtual timeline repair.",
        frame_seconds=VAD_FRAME_SECONDS,
        min_speech_seconds=VAD_MINIMUM_SPEECH_SECONDS,
        min_silence_seconds=VAD_MINIMUM_SILENCE_SECONDS,
    )
    raw_segments = tuple(
        SpeechSegment(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        for segment in result.speech_segments
    )
    return tuple(
        clipped
        for segment in transform_speech_segments(raw_segments, track=track, repair=repair)
        if (clipped := _clip_segment(segment, duration_seconds)) is not None
    )


def _clip_segment(
    segment: SpeechSegment,
    duration_seconds: float,
) -> SpeechSegment | None:
    start_seconds = max(0.0, segment.start_seconds)
    end_seconds = min(duration_seconds, segment.end_seconds)
    if end_seconds <= start_seconds:
        return None
    return SpeechSegment(start_seconds=start_seconds, end_seconds=end_seconds)


def _detector_segments(
    segments: tuple[SpeechSegment, ...],
) -> list[DetectorSpeechSegment]:
    return [
        DetectorSpeechSegment(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        for segment in segments
    ]


def _track_vad(
    side: SpeakerSide,
    segments: tuple[SpeechSegment, ...],
    duration_seconds: float,
) -> TrackVadResult:
    segment_durations = tuple(segment.duration_seconds for segment in segments)
    speech_time_seconds = sum(segment_durations)
    segment_count = len(segment_durations)
    return TrackVadResult(
        side=side,
        speech_segments=segments,
        speech_time_seconds=speech_time_seconds,
        speech_ratio=speech_time_seconds / duration_seconds,
        median_segment_duration_seconds=(
            statistics.median(segment_durations) if segment_durations else None
        ),
        tiny_fragment_ratio=(
            sum(duration < 0.25 for duration in segment_durations) / segment_count
            if segment_count
            else 0.0
        ),
        long_segment_ratio=(
            sum(duration > 20.0 for duration in segment_durations) / segment_count
            if segment_count
            else 0.0
        ),
    )

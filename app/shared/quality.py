from __future__ import annotations

from enum import StrEnum

from pydantic import computed_field

from app.shared.base_model import FrozenBaseModel

METRIC_VERSION = "quality-conversation-v3"
QUALITY_SAMPLE_RATE = 16_000


class SpeakerSide(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


class ProcessingStatus(StrEnum):
    COMPLETED = "completed"
    INVALID = "invalid"
    FAILED = "failed"


class EventType(StrEnum):
    TURN_COMPLETION = "turn_completion"
    PAUSE = "pause"
    START_RESPONSE = "start_response"
    INTERRUPTION = "interruption"
    BACKCHANNEL = "backchannel"
    OVERLAP = "overlap"


class AnnotationEvidenceSource(StrEnum):
    TRANSCRIPT = "transcript"
    AUDIO_ACTIVITY = "audio_activity"


class AudioMetadata(FrozenBaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_count: int


class SpeechSegment(FrozenBaseModel):
    start_seconds: float
    end_seconds: float

    @computed_field
    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class TrackVadResult(FrozenBaseModel):
    side: SpeakerSide
    speech_segments: tuple[SpeechSegment, ...]
    speech_time_seconds: float
    speech_ratio: float
    median_segment_duration_seconds: float | None
    tiny_fragment_ratio: float
    long_segment_ratio: float


class TrackAudioQuality(FrozenBaseModel):
    side: SpeakerSide
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_dbfs: float
    peak_amplitude: float
    clipping_ratio: float
    near_zero_ratio: float
    silence_ratio: float
    speech_ratio: float
    speech_silence_entropy: float
    low_information: bool
    quality_score: float
    flags: tuple[str, ...]


class InteractionDensityMetrics(FrozenBaseModel):
    speech_ratio: float
    silence_ratio: float
    overlap_ratio: float
    turn_completions_per_hour: float
    pause_events_per_hour: float
    start_responses_per_hour: float
    interruptions_per_hour: float
    backchannels_per_hour: float
    overlaps_per_hour: float
    usable_candidate_windows_per_hour: float
    quality_score: float


class TimingReliabilityMetrics(FrozenBaseModel):
    median_segment_duration_seconds: float | None
    tiny_fragment_ratio: float
    long_segment_ratio: float
    median_pause_duration_seconds: float | None
    median_turn_gap_seconds: float | None
    median_overlap_duration_seconds: float | None
    plausible_segment_duration_score: float
    event_density_stability_score: float
    quality_score: float


class AudioQualityMetrics(FrozenBaseModel):
    speaker1: TrackAudioQuality
    speaker2: TrackAudioQuality
    duration_gap_seconds: float
    duration_gap_ratio: float
    track_correlation: float | None
    energy_envelope_correlation: float | None
    speaker1_leakage_db: float | None
    speaker2_leakage_db: float | None
    track_leakage_risk: bool
    quality_score: float
    flags: tuple[str, ...]


class EventCandidate(FrozenBaseModel):
    event_type: EventType
    primary_speaker: SpeakerSide
    secondary_speaker: SpeakerSide | None
    start_seconds: float
    end_seconds: float
    gap_seconds: float | None
    overlap_seconds: float | None


class AnnotationSpan(FrozenBaseModel):
    start_seconds: float
    end_seconds: float
    text: str | None


class AnnotationPoint(FrozenBaseModel):
    time_seconds: float
    confidence: float | None
    text: str | None


class SegmentAnnotationTarget(FrozenBaseModel):
    start_seconds: float
    end_seconds: float
    text: str
    evidence_source: AnnotationEvidenceSource
    keep_playing_confidence: float
    turn_confidence: float
    interruption_confidence: float


class ConnectionAnnotationTarget(FrozenBaseModel):
    earlier_end_seconds: float
    later_start_seconds: float
    gap_seconds: float
    pause_confidence: float
    merge_confidence: float


class SpeakerConversationAnnotation(FrozenBaseModel):
    side: SpeakerSide
    speech_segments: tuple[AnnotationSpan, ...]
    pauses: tuple[AnnotationSpan, ...]
    backchannels: tuple[AnnotationSpan, ...]
    turns: tuple[AnnotationPoint, ...]
    interruptions: tuple[AnnotationPoint, ...]
    segment_targets: tuple[SegmentAnnotationTarget, ...]
    connection_targets: tuple[ConnectionAnnotationTarget, ...]
    speech_duration_seconds: float
    pause_duration_seconds: float
    backchannel_duration_seconds: float


class ConversationAnnotation(FrozenBaseModel):
    annotation_version: str
    analyzed_duration_seconds: float
    speaker1: SpeakerConversationAnnotation
    speaker2: SpeakerConversationAnnotation
    speech_segment_count: int
    turn_count: int
    turn_taking_count: int
    interaction_count: int
    pause_count: int
    backchannel_count: int
    interruption_count: int
    usable_event_count: int
    events_per_hour: float
    speaker_balance_score: float
    quality_score: float


class QualityWeights(FrozenBaseModel):
    interaction_density: float = 0.15
    timing_reliability: float = 0.10
    audio_quality: float = 0.25
    conversation_annotation: float = 0.50


class QualityResult(FrozenBaseModel):
    metric_version: str
    sample_id: str
    status: ProcessingStatus
    speaker1_uri: str
    speaker2_uri: str
    duration_seconds: float | None
    interaction_density: InteractionDensityMetrics | None
    timing_reliability: TimingReliabilityMetrics | None
    audio_quality: AudioQualityMetrics | None
    conversation_annotation: ConversationAnnotation | None
    event_candidates: tuple[EventCandidate, ...]
    raw_quality_score: float | None
    calibrated_quality_score: float | None
    calibration_flags: tuple[str, ...]
    total_quality_score: float | None
    error: str | None


class RunConfig(FrozenBaseModel):
    metric_version: str = METRIC_VERSION
    weights: QualityWeights = QualityWeights()
    max_events_per_sample: int = 200

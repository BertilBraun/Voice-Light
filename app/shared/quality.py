from __future__ import annotations

from enum import StrEnum

from pydantic import Field, computed_field, model_validator

from app.shared.base_model import FrozenBaseModel

METRIC_VERSION = "quality-conversation-full-parakeet-canary-v6"
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
    duration_seconds: float = Field(gt=0.0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    sample_count: int = Field(gt=0)


class SpeechSegment(FrozenBaseModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> SpeechSegment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Speech segment end must follow its start.")
        return self

    @computed_field
    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class TrackVadResult(FrozenBaseModel):
    side: SpeakerSide
    speech_segments: tuple[SpeechSegment, ...]
    speech_time_seconds: float = Field(ge=0.0)
    speech_ratio: float = Field(ge=0.0, le=1.0)
    median_segment_duration_seconds: float | None = Field(gt=0.0)
    tiny_fragment_ratio: float = Field(ge=0.0, le=1.0)
    long_segment_ratio: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_segments(self) -> TrackVadResult:
        previous_end_seconds = 0.0
        for segment in self.speech_segments:
            if segment.start_seconds < previous_end_seconds:
                raise ValueError("VAD speech segments must be ordered and disjoint.")
            previous_end_seconds = segment.end_seconds
        calculated_speech_seconds = sum(
            segment.duration_seconds for segment in self.speech_segments
        )
        if abs(calculated_speech_seconds - self.speech_time_seconds) > 0.08:
            raise ValueError("VAD speech time does not match its segments.")
        return self


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
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str | None

    @model_validator(mode="after")
    def validate_interval(self) -> AnnotationSpan:
        if self.end_seconds < self.start_seconds:
            raise ValueError("Annotation span end must not precede its start.")
        return self


class AnnotationPoint(FrozenBaseModel):
    time_seconds: float = Field(ge=0.0)
    confidence: float | None = Field(ge=0.0, le=1.0)
    text: str | None


class SegmentAnnotationTarget(FrozenBaseModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str
    evidence_source: AnnotationEvidenceSource
    keep_playing_confidence: float = Field(ge=0.0, le=1.0)
    turn_confidence: float = Field(ge=0.0, le=1.0)
    interruption_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> SegmentAnnotationTarget:
        if self.end_seconds < self.start_seconds:
            raise ValueError("Segment target end must not precede its start.")
        return self


class ConnectionAnnotationTarget(FrozenBaseModel):
    earlier_end_seconds: float = Field(ge=0.0)
    later_start_seconds: float = Field(ge=0.0)
    gap_seconds: float
    pause_confidence: float = Field(ge=0.0, le=1.0)
    merge_confidence: float = Field(ge=0.0, le=1.0)


class SpeakerConversationAnnotation(FrozenBaseModel):
    side: SpeakerSide
    speech_segments: tuple[AnnotationSpan, ...]
    pauses: tuple[AnnotationSpan, ...]
    backchannels: tuple[AnnotationSpan, ...]
    turns: tuple[AnnotationPoint, ...]
    interruptions: tuple[AnnotationPoint, ...]
    segment_targets: tuple[SegmentAnnotationTarget, ...]
    connection_targets: tuple[ConnectionAnnotationTarget, ...]
    speech_duration_seconds: float = Field(ge=0.0)
    pause_duration_seconds: float = Field(ge=0.0)
    backchannel_duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> SpeakerConversationAnnotation:
        for spans in (self.speech_segments, self.pauses, self.backchannels):
            if tuple(span.start_seconds for span in spans) != tuple(
                sorted(span.start_seconds for span in spans)
            ):
                raise ValueError("Speaker annotation spans must be ordered by start time.")
        for points in (self.turns, self.interruptions):
            if tuple(point.time_seconds for point in points) != tuple(
                sorted(point.time_seconds for point in points)
            ):
                raise ValueError("Speaker annotation points must be ordered by time.")
        return self


class ConversationAnnotation(FrozenBaseModel):
    annotation_version: str
    analyzed_duration_seconds: float = Field(gt=0.0)
    speaker1: SpeakerConversationAnnotation
    speaker2: SpeakerConversationAnnotation
    speech_segment_count: int = Field(ge=0)
    turn_count: int = Field(ge=0)
    turn_taking_count: int = Field(ge=0)
    interaction_count: int = Field(ge=0)
    pause_count: int = Field(ge=0)
    backchannel_count: int = Field(ge=0)
    interruption_count: int = Field(ge=0)
    usable_event_count: int = Field(ge=0)
    events_per_hour: float = Field(ge=0.0)
    speaker_balance_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_structure(self) -> ConversationAnnotation:
        if self.speaker1.side is not SpeakerSide.SPEAKER1:
            raise ValueError("speaker1 annotation must identify speaker1.")
        if self.speaker2.side is not SpeakerSide.SPEAKER2:
            raise ValueError("speaker2 annotation must identify speaker2.")
        speakers = (self.speaker1, self.speaker2)
        times = tuple(
            span.end_seconds
            for speaker in speakers
            for spans in (speaker.speech_segments, speaker.pauses, speaker.backchannels)
            for span in spans
        )
        times += tuple(
            point.time_seconds
            for speaker in speakers
            for points in (speaker.turns, speaker.interruptions)
            for point in points
        )
        times += tuple(
            target.end_seconds for speaker in speakers for target in speaker.segment_targets
        )
        times += tuple(
            target.later_start_seconds
            for speaker in speakers
            for target in speaker.connection_targets
        )
        if times and max(times) > self.analyzed_duration_seconds + 0.08:
            raise ValueError("Conversation annotation exceeds its analyzed duration.")
        expected_counts = (
            (self.speech_segment_count, sum(len(speaker.speech_segments) for speaker in speakers)),
            (self.turn_count, sum(len(speaker.turns) for speaker in speakers)),
            (self.pause_count, sum(len(speaker.pauses) for speaker in speakers)),
            (self.backchannel_count, sum(len(speaker.backchannels) for speaker in speakers)),
            (
                self.interruption_count,
                sum(len(speaker.interruptions) for speaker in speakers),
            ),
        )
        if any(declared != observed for declared, observed in expected_counts):
            raise ValueError("Conversation annotation event counts do not match its evidence.")
        return self


class ConversationEventCounts(FrozenBaseModel):
    speech_segment_count: int
    interaction_count: int
    turn_count: int
    turn_taking_count: int
    pause_count: int
    backchannel_count: int
    interruption_count: int
    usable_event_count: int


class ConversationCountEstimate(FrozenBaseModel):
    annotation_duration_seconds: float
    represented_duration_seconds: float
    scale_factor: float
    observed: ConversationEventCounts
    estimated: ConversationEventCounts


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
    conversation_count_estimate: ConversationCountEstimate | None
    event_candidates: tuple[EventCandidate, ...]
    raw_quality_score: float | None
    quality_flags: tuple[str, ...]
    total_quality_score: float | None
    error: str | None


class RunConfig(FrozenBaseModel):
    metric_version: str = METRIC_VERSION
    weights: QualityWeights = QualityWeights()
    max_events_per_sample: int = 200

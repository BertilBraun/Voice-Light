from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, computed_field

METRIC_VERSION = "quality-calibrated-v2"


class SpeakerSide(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


class ProcessingStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(StrEnum):
    TURN_COMPLETION = "turn_completion"
    PAUSE = "pause"
    START_RESPONSE = "start_response"
    INTERRUPTION = "interruption"
    BACKCHANNEL = "backchannel"
    OVERLAP = "overlap"


class AudioMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    duration_seconds: float
    sample_rate: int
    channels: int
    sample_count: int


class SpeechSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_seconds: float
    end_seconds: float

    @computed_field
    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class TrackVadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    side: SpeakerSide
    speech_segments: tuple[SpeechSegment, ...]
    speech_time_seconds: float
    speech_ratio: float
    median_segment_duration_seconds: float | None
    tiny_fragment_ratio: float
    long_segment_ratio: float


class TrackAudioQuality(BaseModel):
    model_config = ConfigDict(frozen=True)

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


class InteractionDensityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

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


class TimingReliabilityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    median_segment_duration_seconds: float | None
    tiny_fragment_ratio: float
    long_segment_ratio: float
    median_pause_duration_seconds: float | None
    median_turn_gap_seconds: float | None
    median_overlap_duration_seconds: float | None
    plausible_segment_duration_score: float
    event_density_stability_score: float
    quality_score: float


class AudioQualityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

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


class EventCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: EventType
    primary_speaker: SpeakerSide
    secondary_speaker: SpeakerSide | None
    start_seconds: float
    end_seconds: float
    gap_seconds: float | None
    overlap_seconds: float | None


class QualityWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    interaction_density: float = 0.5
    timing_reliability: float = 0.2
    audio_quality: float = 0.3


class QualityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_version: str
    sample_id: str
    status: ProcessingStatus
    speaker1_uri: str
    speaker2_uri: str
    duration_seconds: float | None
    interaction_density: InteractionDensityMetrics | None
    timing_reliability: TimingReliabilityMetrics | None
    audio_quality: AudioQualityMetrics | None
    event_candidates: tuple[EventCandidate, ...]
    raw_quality_score: float | None
    calibrated_quality_score: float | None
    calibration_flags: tuple[str, ...]
    total_quality_score: float | None
    error: str | None


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_version: str = METRIC_VERSION
    weights: QualityWeights = QualityWeights()
    max_events_per_sample: int = 200

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.shared.base_model import FrozenBaseModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class BenchmarkModel(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CandidateTargetPoint(BenchmarkModel):
    absolute_time_seconds: float = Field(ge=0.0)
    silence_duration_seconds: float = Field(gt=0.0)
    yield_probability: float = Field(ge=0.0, le=1.0)


class SilenceCandidate(BenchmarkModel):
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    dataset_id: str
    dataset_name: str
    conversation_id: str
    external_id: str
    user_side: str
    user_audio_path: str
    preceding_speech_start_seconds: float = Field(ge=0.0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    target_points: tuple[CandidateTargetPoint, ...] = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=1)
    source_window_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> SilenceCandidate:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Candidate end_seconds must follow start_seconds.")
        if self.preceding_speech_start_seconds >= self.start_seconds:
            raise ValueError("Candidate preceding speech must start before its silence.")
        times = tuple(point.absolute_time_seconds for point in self.target_points)
        if times != tuple(sorted(set(times))):
            raise ValueError("Candidate target points must have unique increasing timestamps.")
        if times[0] <= self.start_seconds or times[-1] > self.end_seconds:
            raise ValueError("Candidate target points must fall after start and at or before end.")
        durations = tuple(point.silence_duration_seconds for point in self.target_points)
        if durations != tuple(sorted(durations)):
            raise ValueError("Candidate silence durations must be increasing.")
        return self


class CandidateInventoryManifest(BenchmarkModel):
    schema_version: Literal["voice-light-causal-candidate-inventory-v1"] = (
        "voice-light-causal-candidate-inventory-v1"
    )
    corpus_repository: str
    corpus_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: TrainingCorpusSplit
    frame_seconds: float = Field(gt=0.0)
    silence_floor_threshold: float = Field(ge=0.0, le=1.0)
    minimum_silence_seconds: float = Field(gt=0.0)
    sample_window_count: int = Field(gt=0)
    conversation_count: int = Field(gt=0)
    candidate_count: int = Field(ge=0)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)


class CandidateInventory(BenchmarkModel):
    manifest: CandidateInventoryManifest
    candidates: tuple[SilenceCandidate, ...]

    @model_validator(mode="after")
    def validate_inventory(self) -> CandidateInventory:
        if len(self.candidates) != self.manifest.candidate_count:
            raise ValueError("Candidate count does not match the inventory manifest.")
        if candidate_rows_sha256(self.candidates) != self.manifest.candidate_sha256:
            raise ValueError("Candidate rows do not match the inventory manifest hash.")
        return self


class CompletionBoundaryKind(StrEnum):
    CONTINUATION = "continuation"
    TERMINAL = "terminal"


class CompletionTargetPoint(BenchmarkModel):
    absolute_time_seconds: float = Field(ge=0.0)
    elapsed_seconds: float = Field(gt=0.0)
    completion_probability: float = Field(ge=0.0, le=1.0)


class TurnCompletionCandidate(BenchmarkModel):
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    dataset_id: str
    dataset_name: str
    conversation_id: str
    external_id: str
    user_side: str
    user_audio_path: str
    preceding_speech_start_seconds: float = Field(ge=0.0)
    anchor_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    boundary_kind: CompletionBoundaryKind
    continuation_probability: float | None = Field(ge=0.0, le=1.0)
    target_points: tuple[CompletionTargetPoint, ...] = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=1)
    source_window_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> TurnCompletionCandidate:
        if self.end_seconds <= self.anchor_seconds:
            raise ValueError("Candidate end_seconds must follow anchor_seconds.")
        if self.preceding_speech_start_seconds >= self.anchor_seconds:
            raise ValueError("Candidate preceding speech must start before its anchor.")
        times = tuple(point.absolute_time_seconds for point in self.target_points)
        if times != tuple(sorted(set(times))):
            raise ValueError("Candidate target points must have unique increasing timestamps.")
        if times[0] <= self.anchor_seconds or times[-1] > self.end_seconds:
            raise ValueError("Candidate target points must fall after anchor and at or before end.")
        elapsed = tuple(point.elapsed_seconds for point in self.target_points)
        if elapsed != tuple(sorted(elapsed)):
            raise ValueError("Candidate elapsed times must be increasing.")
        probabilities = {point.completion_probability for point in self.target_points}
        if len(probabilities) != 1:
            raise ValueError("A sparse completion target must remain fixed across the candidate.")
        return self


class TurnCompletionInventoryManifest(BenchmarkModel):
    schema_version: Literal["voice-light-turn-completion-candidate-inventory-v2"] = (
        "voice-light-turn-completion-candidate-inventory-v2"
    )
    corpus_repository: str
    corpus_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: TrainingCorpusSplit
    frame_seconds: float = Field(gt=0.0)
    user_floor_threshold: float = Field(ge=0.0, le=1.0)
    assistant_active_threshold: float = Field(ge=0.0, le=1.0)
    causal_horizon_seconds: float = Field(gt=0.0)
    sample_window_count: int = Field(gt=0)
    conversation_count: int = Field(gt=0)
    candidate_count: int = Field(ge=0)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)


class TurnCompletionInventory(BenchmarkModel):
    manifest: TurnCompletionInventoryManifest
    candidates: tuple[TurnCompletionCandidate, ...]

    @model_validator(mode="after")
    def validate_inventory(self) -> TurnCompletionInventory:
        if len(self.candidates) != self.manifest.candidate_count:
            raise ValueError("Candidate count does not match the inventory manifest.")
        if turn_completion_candidate_rows_sha256(self.candidates) != (
            self.manifest.candidate_sha256
        ):
            raise ValueError("Candidate rows do not match the inventory manifest hash.")
        return self


class DetectorKind(StrEnum):
    VOICE_LIGHT = "voice_light"
    SILERO_TIMEOUT = "silero_timeout"
    PIPECAT_SMART_TURN_V3_2 = "pipecat_smart_turn_v3_2"
    LIVEKIT_V1_MINI = "livekit_v1_mini"


class CompletionDetectorKind(StrEnum):
    VOICE_LIGHT_COMPLETION = "voice_light_turn_completion"
    SILERO_SILENCE = "silero_silence"
    PIPECAT_SMART_TURN_V3_2 = "pipecat_smart_turn_v3_2_completion"
    LIVEKIT_V1_MINI = "livekit_v1_mini_completion"


class VoiceLightCompletionDetectorConfiguration(BenchmarkModel):
    model_identifier: str
    lookahead_tokens: int = Field(ge=0)
    encoder_frame_seconds: float = Field(gt=0.0)
    optimizer_step: int = Field(gt=0)
    event_head_index: Literal[0] = 0
    boundary_lookahead_seconds: Literal[0.08] = 0.08
    score_semantic: Literal["turn_completion_probability"] = "turn_completion_probability"
    assistant_history: Literal["causal_through_boundary"] = "causal_through_boundary"
    score_schedule: Literal["boundary_only"] = "boundary_only"
    score_persistence: Literal["latched"] = "latched"
    causal_candidate_coverage_seconds: Literal[2.0] = 2.0
    assistant_active_anchors_excluded: Literal[True] = True
    insufficient_horizon_candidates_censored: Literal[True] = True


class VoiceLightCompletionDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[CompletionDetectorKind.VOICE_LIGHT_COMPLETION] = (
        CompletionDetectorKind.VOICE_LIGHT_COMPLETION
    )
    display_name: str
    implementation_version: str
    model_repository: str
    model_revision: str
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration: VoiceLightCompletionDetectorConfiguration


class VoiceLightDetectorConfiguration(BenchmarkModel):
    model_identifier: str
    lookahead_tokens: int = Field(ge=0)
    encoder_frame_seconds: float = Field(gt=0.0)
    optimizer_step: int = Field(gt=0)


class VoiceLightDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[DetectorKind.VOICE_LIGHT] = DetectorKind.VOICE_LIGHT
    display_name: str
    implementation_version: str
    model_repository: str
    model_revision: str
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration: VoiceLightDetectorConfiguration


class SileroDetectorConfiguration(BenchmarkModel):
    speech_threshold: float = Field(ge=0.0, le=1.0)
    minimum_speech_seconds: float = Field(gt=0.0)
    minimum_silence_seconds: float = Field(gt=0.0)
    use_onnx: bool


class SileroDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[DetectorKind.SILERO_TIMEOUT] = DetectorKind.SILERO_TIMEOUT
    display_name: str
    implementation_version: str
    package_name: str
    package_version: str
    configuration: SileroDetectorConfiguration


class SmartTurnDetectorConfiguration(BenchmarkModel):
    sample_rate_hz: int = Field(gt=0)
    maximum_window_seconds: float = Field(gt=0.0)
    candidate_silence_seconds: float = Field(gt=0.0)
    quantization: Literal["int8"] = "int8"


class SmartTurnDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[DetectorKind.PIPECAT_SMART_TURN_V3_2] = (
        DetectorKind.PIPECAT_SMART_TURN_V3_2
    )
    display_name: str
    implementation_version: str
    runtime_package_name: str
    runtime_package_version: str
    model_repository: str
    model_revision: str
    model_filename: str
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration: SmartTurnDetectorConfiguration


class LiveKitDetectorConfiguration(BenchmarkModel):
    sample_rate_hz: int = Field(gt=0)
    vad_speech_threshold: float = Field(ge=0.0, le=1.0)
    candidate_silence_seconds: float = Field(gt=0.0)
    model_version: Literal["v1-mini"] = "v1-mini"


class LiveKitDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[DetectorKind.LIVEKIT_V1_MINI] = DetectorKind.LIVEKIT_V1_MINI
    display_name: str
    implementation_version: str
    package_name: str
    package_version: str
    model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    configuration: LiveKitDetectorConfiguration


class SileroCompletionDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[CompletionDetectorKind.SILERO_SILENCE] = (
        CompletionDetectorKind.SILERO_SILENCE
    )
    display_name: str
    implementation_version: str
    package_name: str
    package_version: str
    configuration: SileroDetectorConfiguration


class SmartTurnCompletionDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[CompletionDetectorKind.PIPECAT_SMART_TURN_V3_2] = (
        CompletionDetectorKind.PIPECAT_SMART_TURN_V3_2
    )
    display_name: str
    implementation_version: str
    runtime_package_name: str
    runtime_package_version: str
    model_repository: str
    model_revision: str
    model_filename: str
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration: SmartTurnDetectorConfiguration


class LiveKitCompletionDetectorProvenance(BenchmarkModel):
    detector_kind: Literal[CompletionDetectorKind.LIVEKIT_V1_MINI] = (
        CompletionDetectorKind.LIVEKIT_V1_MINI
    )
    display_name: str
    implementation_version: str
    package_name: str
    package_version: str
    model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    configuration: LiveKitDetectorConfiguration


DetectorProvenance = Annotated[
    VoiceLightDetectorProvenance
    | SileroDetectorProvenance
    | SmartTurnDetectorProvenance
    | LiveKitDetectorProvenance,
    Field(discriminator="detector_kind"),
]

CompletionDetectorProvenance = Annotated[
    VoiceLightCompletionDetectorProvenance
    | SileroCompletionDetectorProvenance
    | SmartTurnCompletionDetectorProvenance
    | LiveKitCompletionDetectorProvenance,
    Field(discriminator="detector_kind"),
]


class CandidatePrediction(BenchmarkModel):
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    absolute_time_seconds: float = Field(ge=0.0)
    silence_duration_seconds: float = Field(gt=0.0)
    yield_probability: float = Field(ge=0.0, le=1.0)
    inference_duration_seconds: float = Field(ge=0.0)


class PredictionManifest(BenchmarkModel):
    schema_version: Literal["voice-light-causal-candidate-predictions-v1"] = (
        "voice-light-causal-candidate-predictions-v1"
    )
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    split: TrainingCorpusSplit
    detector: DetectorProvenance
    prediction_count: int = Field(ge=0)
    predictions_sha256: str = Field(pattern=SHA256_PATTERN)


class AudioProvenanceRecord(BenchmarkModel):
    source_name: str
    external_id: str | None
    audio_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    pcm_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class ExactOverlap(BenchmarkModel):
    local_source_name: str
    local_external_id: str | None
    external_source_name: str
    external_external_id: str | None
    matched_hash_kind: str
    matched_sha256: str = Field(pattern=SHA256_PATTERN)


class OverlapAuditReport(BenchmarkModel):
    schema_version: str = "voice-light-smart-turn-overlap-audit-v1"
    external_repository: str
    external_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    local_record_count: int = Field(ge=0)
    external_record_count: int = Field(ge=0)
    exact_overlaps: tuple[ExactOverlap, ...]
    provenance_risk_sources: tuple[str, ...]
    unverified_local_record_count: int = Field(ge=0)
    exact_hash_comparison_performed: bool
    clean_comparative_claim_permitted: bool


def candidate_rows_sha256(candidates: tuple[SilenceCandidate, ...]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def turn_completion_candidate_rows_sha256(
    candidates: tuple[TurnCompletionCandidate, ...],
) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prediction_rows_sha256(predictions: tuple[CandidatePrediction, ...]) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        digest.update(prediction.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def completion_prediction_rows_sha256(
    predictions: tuple[CompletionCandidatePrediction, ...],
) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        digest.update(prediction.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class PredictionArtifact(BenchmarkModel):
    manifest: PredictionManifest
    predictions: tuple[CandidatePrediction, ...]

    @model_validator(mode="after")
    def validate_predictions(self) -> PredictionArtifact:
        if len(self.predictions) != self.manifest.prediction_count:
            raise ValueError("Prediction count does not match the prediction manifest.")
        if prediction_rows_sha256(self.predictions) != self.manifest.predictions_sha256:
            raise ValueError("Prediction rows do not match the prediction manifest hash.")
        ordered = tuple(
            sorted(
                self.predictions,
                key=lambda prediction: (
                    prediction.candidate_id,
                    prediction.absolute_time_seconds,
                ),
            )
        )
        if ordered != self.predictions:
            raise ValueError("Predictions must be ordered by candidate and timestamp.")
        return self


class CompletionCandidatePrediction(BenchmarkModel):
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    absolute_time_seconds: float = Field(ge=0.0)
    elapsed_seconds: float = Field(gt=0.0)
    completion_probability: float = Field(ge=0.0, le=1.0)
    inference_duration_seconds: float = Field(ge=0.0)


class CompletionPredictionManifest(BenchmarkModel):
    schema_version: Literal["voice-light-turn-completion-predictions-v2"] = (
        "voice-light-turn-completion-predictions-v2"
    )
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    split: TrainingCorpusSplit
    detector: CompletionDetectorProvenance
    prediction_count: int = Field(ge=0)
    predictions_sha256: str = Field(pattern=SHA256_PATTERN)


class CompletionPredictionArtifact(BenchmarkModel):
    manifest: CompletionPredictionManifest
    predictions: tuple[CompletionCandidatePrediction, ...]

    @model_validator(mode="after")
    def validate_predictions(self) -> CompletionPredictionArtifact:
        if len(self.predictions) != self.manifest.prediction_count:
            raise ValueError("Prediction count does not match the prediction manifest.")
        if completion_prediction_rows_sha256(self.predictions) != (
            self.manifest.predictions_sha256
        ):
            raise ValueError("Prediction rows do not match the prediction manifest hash.")
        ordered = tuple(
            sorted(
                self.predictions,
                key=lambda prediction: (
                    prediction.candidate_id,
                    prediction.absolute_time_seconds,
                ),
            )
        )
        if ordered != self.predictions:
            raise ValueError("Predictions must be ordered by candidate and timestamp.")
        return self


def write_inventory(path: Path, inventory: CandidateInventory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")


def read_inventory(path: Path) -> CandidateInventory:
    return CandidateInventory.model_validate_json(path.read_text(encoding="utf-8"))


def write_turn_completion_inventory(
    path: Path,
    inventory: TurnCompletionInventory,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")


def read_turn_completion_inventory(path: Path) -> TurnCompletionInventory:
    return TurnCompletionInventory.model_validate_json(path.read_text(encoding="utf-8"))


def write_predictions(path: Path, artifact: PredictionArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def read_predictions(path: Path) -> PredictionArtifact:
    return PredictionArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def write_completion_predictions(path: Path, artifact: CompletionPredictionArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def read_completion_predictions(path: Path) -> CompletionPredictionArtifact:
    return CompletionPredictionArtifact.model_validate_json(path.read_text(encoding="utf-8"))

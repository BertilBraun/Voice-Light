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
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    target_points: tuple[CandidateTargetPoint, ...] = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=1)
    source_window_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> SilenceCandidate:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Candidate end_seconds must follow start_seconds.")
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
    schema_version: str = "voice-light-causal-candidate-inventory-v1"
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


class DetectorKind(StrEnum):
    VOICE_LIGHT = "voice_light"
    SILERO_TIMEOUT = "silero_timeout"
    PIPECAT_SMART_TURN_V3_2 = "pipecat_smart_turn_v3_2"
    LIVEKIT_V1_MINI = "livekit_v1_mini"


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


DetectorProvenance = Annotated[
    VoiceLightDetectorProvenance
    | SileroDetectorProvenance
    | SmartTurnDetectorProvenance
    | LiveKitDetectorProvenance,
    Field(discriminator="detector_kind"),
]


class CandidatePrediction(BenchmarkModel):
    candidate_id: str = Field(pattern=SHA256_PATTERN)
    absolute_time_seconds: float = Field(ge=0.0)
    silence_duration_seconds: float = Field(gt=0.0)
    yield_probability: float = Field(ge=0.0, le=1.0)
    inference_duration_seconds: float = Field(ge=0.0)


class PredictionManifest(BenchmarkModel):
    schema_version: str = "voice-light-causal-candidate-predictions-v1"
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
    local_record_count: int = Field(ge=0)
    external_record_count: int = Field(ge=0)
    exact_overlaps: tuple[ExactOverlap, ...]
    provenance_risk_sources: tuple[str, ...]
    unverified_local_record_count: int = Field(ge=0)
    clean_comparative_claim_permitted: bool


def candidate_rows_sha256(candidates: tuple[SilenceCandidate, ...]) -> str:
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


def write_inventory(path: Path, inventory: CandidateInventory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")


def read_inventory(path: Path) -> CandidateInventory:
    return CandidateInventory.model_validate_json(path.read_text(encoding="utf-8"))


def write_predictions(path: Path, artifact: PredictionArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def read_predictions(path: Path) -> PredictionArtifact:
    return PredictionArtifact.model_validate_json(path.read_text(encoding="utf-8"))

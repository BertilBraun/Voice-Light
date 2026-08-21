from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

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


class DetectorProvenance(BenchmarkModel):
    detector_kind: DetectorKind
    display_name: str
    implementation_version: str
    package_name: str | None
    package_version: str | None
    model_repository: str | None
    model_revision: str | None
    model_filename: str | None
    model_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    checkpoint_path: str | None
    checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    configuration_json: str


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


def write_inventory(path: Path, inventory: CandidateInventory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")


def read_inventory(path: Path) -> CandidateInventory:
    return CandidateInventory.model_validate_json(path.read_text(encoding="utf-8"))

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpeakerPrompt(FrozenModel):
    text: str = Field(min_length=1)
    instruction: str | None = None


class BackchannelPlacementDefinition(FrozenModel):
    type: Literal["backchannel"] = "backchannel"
    anchor_text: str = Field(min_length=1)
    delay_ms: int = 300


class BackchannelPlacementOutput(FrozenModel):
    type: Literal["backchannel"] = "backchannel"
    anchor_text: str
    transcript_text: str
    matched_text: str
    score: float
    anchor_start_seconds: float
    anchor_end_seconds: float
    delay_ms: int
    speaker_b_start_seconds: float


class InterruptionPlacementDefinition(FrozenModel):
    type: Literal["interruption"] = "interruption"
    anchor_text: str = Field(min_length=1)
    mode: Literal["before_last_word", "at_anchor_start", "at_anchor_end"] = "before_last_word"
    lead_ms: int = 50


class InterruptionPlacementOutput(FrozenModel):
    type: Literal["interruption"] = "interruption"
    anchor_text: str
    transcript_text: str
    matched_text: str
    score: float
    anchor_start_seconds: float
    anchor_end_seconds: float
    speaker_b_start_seconds: float
    mode: str
    lead_ms: int


class CompletionPlacementDefinition(FrozenModel):
    type: Literal["completion"] = "completion"
    pause: Literal["short", "medium", "long"] = "short"
    min_delay_ms: int | None = None
    max_delay_ms: int | None = None

    @model_validator(mode="after")
    def validate_delay_range(self) -> CompletionPlacementDefinition:
        if (self.min_delay_ms is None) != (self.max_delay_ms is None):
            raise ValueError("Set both min_delay_ms and max_delay_ms, or neither.")
        if (
            self.min_delay_ms is not None
            and self.max_delay_ms is not None
            and self.min_delay_ms > self.max_delay_ms
        ):
            raise ValueError("min_delay_ms must be <= max_delay_ms.")
        return self


class CompletionPlacementOutput(FrozenModel):
    type: Literal["completion"] = "completion"
    pause: str
    min_delay_ms: int
    max_delay_ms: int
    sampled_delay_ms: int
    speaker_a_end_seconds: float
    speaker_b_start_seconds: float


class InternalPausePlacementDefinition(FrozenModel):
    type: Literal["internal_pause"] = "internal_pause"
    anchor_text: str = Field(min_length=1)
    pause: Literal["short", "medium", "long"] = "medium"


class InternalPausePlacementOutput(FrozenModel):
    type: Literal["internal_pause"] = "internal_pause"
    anchor_text: str
    transcript_text: str
    matched_text: str
    score: float
    anchor_start_seconds: float
    anchor_end_seconds: float
    next_word_text: str
    next_word_start_seconds: float
    measured_pause_ms: int
    pause: str


PlacementDefinition = Annotated[
    BackchannelPlacementDefinition
    | InterruptionPlacementDefinition
    | CompletionPlacementDefinition
    | InternalPausePlacementDefinition,
    Field(discriminator="type"),
]

PlacementOutput = Annotated[
    BackchannelPlacementOutput
    | InterruptionPlacementOutput
    | CompletionPlacementOutput
    | InternalPausePlacementOutput,
    Field(discriminator="type"),
]


class InteractionCase(FrozenModel):
    case_id: str | None = Field(default=None, min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = ""
    description: str = ""
    speaker_a: SpeakerPrompt
    speaker_b: SpeakerPrompt
    placement: PlacementDefinition | None = None
    alignment_notes: str = ""
    tags: list[str] = Field(default_factory=list)
    default_b_offset_seconds: float | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_placement(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "placement" in data:
            return data
        legacy_field_names = [
            "backchannel_placement",
            "interruption_placement",
            "completion_placement",
            "internal_pause_placement",
        ]
        present_field_names = [
            field_name for field_name in legacy_field_names if field_name in data
        ]
        if len(present_field_names) == 0:
            return data
        if len(present_field_names) > 1:
            raise ValueError("Set exactly one placement field per case.")
        legacy_field_name = present_field_names[0]
        legacy_value = data[legacy_field_name]
        if not isinstance(legacy_value, dict):
            raise ValueError(f"{legacy_field_name} must be an object.")
        migrated_data = dict(data)
        placement_data = dict(legacy_value)
        match legacy_field_name:
            case "backchannel_placement":
                placement_data["type"] = "backchannel"
            case "interruption_placement":
                placement_data["type"] = "interruption"
            case "completion_placement":
                placement_data["type"] = "completion"
            case "internal_pause_placement":
                placement_data["type"] = "internal_pause"
        for field_name in legacy_field_names:
            migrated_data.pop(field_name, None)
        migrated_data["placement"] = placement_data
        return migrated_data


class InteractionCaseFile(FrozenModel):
    experiment_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = ""
    cases: list[InteractionCase]

    @model_validator(mode="after")
    def populate_and_validate_case_ids(self) -> InteractionCaseFile:
        normalized_cases: list[InteractionCase] = []
        for index, interaction_case in enumerate(self.cases, start=1):
            case_id = interaction_case.case_id or f"case_{index:03d}"
            title = interaction_case.title or case_id.replace("_", " ").title()
            normalized_cases.append(
                interaction_case.model_copy(update={"case_id": case_id, "title": title})
            )
        case_ids = [interaction_case.case_id for interaction_case in normalized_cases]
        duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Duplicate case_id values: {', '.join(duplicate_ids)}")
        object.__setattr__(self, "cases", normalized_cases)
        return self


class BackendConfig(FrozenModel):
    backend: str = "qwen-voice-design"
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    options: dict[str, object] = Field(default_factory=dict)


class PreparedRequest(FrozenModel):
    text: str
    instruction: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)


class VariantManifest(FrozenModel):
    id: str
    audio_url: str | None = None
    raw_audio_url: str | None = None
    filename: str
    duration_seconds: float | None = None
    generation_seconds: float | None = None
    real_time_factor: float | None = None
    seed: int | None = None
    sample_rate: int | None = None
    audio_onset_seconds: float | None = None
    audio_offset_seconds: float | None = None
    original_duration_seconds: float | None = None
    trimmed_leading_seconds: float | None = None
    trimmed_trailing_seconds: float | None = None
    status: Literal["success", "failed"]
    error: str | None = None
    backend_id: str
    model_id: str
    request: SpeakerPrompt
    prepared_request: PreparedRequest
    backend_metadata: dict[str, object] = Field(default_factory=dict)
    suggested_b_offset_seconds: float | None = None
    placement_output: PlacementOutput | None = None

    @model_validator(mode="after")
    def validate_success_audio(self) -> VariantManifest:
        if self.status == "success" and self.audio_url is None:
            raise ValueError("Successful variants must include audio_url")
        if self.status == "failed" and self.error is None:
            raise ValueError("Failed variants must include error")
        return self


class SpeakerManifest(FrozenModel):
    text: str
    instruction: str | None = None
    variants: list[VariantManifest]

    @model_validator(mode="after")
    def ensure_unique_variant_ids(self) -> SpeakerManifest:
        variant_ids = [variant.id for variant in self.variants]
        duplicate_ids = sorted(
            {variant_id for variant_id in variant_ids if variant_ids.count(variant_id) > 1}
        )
        if duplicate_ids:
            raise ValueError(f"Duplicate variant ids: {', '.join(duplicate_ids)}")
        return self


class CaseManifest(FrozenModel):
    case_id: str
    title: str
    description: str
    alignment_notes: str = ""
    tags: list[str] = Field(default_factory=list)
    default_b_offset_seconds: float | None = None
    placement: PlacementDefinition | None = None
    speaker_a: SpeakerManifest
    speaker_b: SpeakerManifest


class GenerationBackendManifest(FrozenModel):
    backend_id: str
    model_id: str
    options: dict[str, object] = Field(default_factory=dict)


class ExperimentManifest(FrozenModel):
    schema_version: int = 1
    experiment_id: str
    description: str = ""
    generation_backend: GenerationBackendManifest
    cases: list[CaseManifest]

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, experiment_id: str) -> str:
        if not experiment_id:
            raise ValueError("experiment_id is required")
        return experiment_id

    @model_validator(mode="after")
    def ensure_unique_case_ids(self) -> ExperimentManifest:
        case_ids = [interaction_case.case_id for interaction_case in self.cases]
        duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Duplicate case_id values: {', '.join(duplicate_ids)}")
        return self


def manifest_path(experiment_dir: Path) -> Path:
    return experiment_dir / "experiment.json"

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from app.training.tool_use.schema import ToolUseBaseModel

QWEN3_1_7B_MODEL_IDENTIFIER = "Qwen/Qwen3-1.7B"
QWEN3_1_7B_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


class LoraTrainingConfig(ToolUseBaseModel):
    schema_version: Literal["voice-light.lora-training/v1"] = "voice-light.lora-training/v1"
    records_path: Path
    output_directory: Path
    source_commit: str = Field(pattern=r"^[a-f0-9]{7,40}$")
    model_identifier: str
    model_revision: str
    random_seed: int
    holdout_record_count: int = Field(ge=1)
    logical_epochs: int = Field(ge=1)
    minimum_user_turns: int = Field(ge=1)
    maximum_user_turns: int = Field(ge=1)
    maximum_sequence_tokens: int = Field(ge=128)
    per_device_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    maximum_gradient_norm: float = Field(gt=0.0)
    lora_rank: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    lora_dropout: float = Field(ge=0.0, lt=1.0)
    lora_target_modules: tuple[str, ...] = Field(min_length=1)
    save_steps: int = Field(ge=1)
    gradient_checkpointing: bool

    @model_validator(mode="after")
    def validate_turn_bounds(self) -> LoraTrainingConfig:
        if self.maximum_user_turns < self.minimum_user_turns:
            raise ValueError("Maximum user turns cannot be smaller than the minimum.")
        return self


def pilot_training_config(
    records_path: Path,
    output_directory: Path,
    source_commit: str,
) -> LoraTrainingConfig:
    return LoraTrainingConfig(
        records_path=records_path,
        output_directory=output_directory,
        source_commit=source_commit,
        model_identifier=QWEN3_1_7B_MODEL_IDENTIFIER,
        model_revision=QWEN3_1_7B_REVISION,
        random_seed=20260729,
        holdout_record_count=80,
        logical_epochs=16,
        minimum_user_turns=8,
        maximum_user_turns=16,
        maximum_sequence_tokens=4096,
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        maximum_gradient_norm=1.0,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        save_steps=200,
        gradient_checkpointing=False,
    )

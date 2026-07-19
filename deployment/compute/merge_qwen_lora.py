from __future__ import annotations

import argparse
import importlib.metadata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import torch
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_ADAPTER_NAME,
    LANGUAGE_MODEL_ADAPTER_REVISION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
)
from app.compute.voice.qwen_config import HUGGING_FACE_REPOSITORY_PATTERN
from app.shared.base_model import FrozenBaseModel

PROVENANCE_FILENAME = "merge-provenance.json"
MODEL_CARD_FILENAME = "README.md"
MAXIMUM_SHARD_SIZE: Final = "5GB"


class HuggingFaceRevision(FrozenBaseModel):
    repository_id: str
    revision: str


class MergeImplementation(FrozenBaseModel):
    dtype: Literal["bfloat16"]
    device: Literal["cpu"]
    safe_merge: Literal[True]
    safe_serialization: Literal[True]
    maximum_shard_size: Literal["5GB"]
    huggingface_hub_version: str
    safetensors_version: str
    transformers_version: str
    peft_version: str
    torch_version: str


class MergedModelProvenance(FrozenBaseModel):
    schema_version: Literal[1]
    destination_repository_id: str
    base_model: HuggingFaceRevision
    adapter: HuggingFaceRevision
    merge: MergeImplementation


@dataclass(frozen=True)
class MergeOptions:
    output_directory: Path
    destination_repository_id: str


def main(arguments: Sequence[str] | None = None) -> None:
    options = parse_options(arguments)
    provenance = merge_qwen_lora(options)
    print(f"Merged model written to {options.output_directory.resolve()}.")
    print(provenance.model_dump_json(indent=2))


def parse_options(arguments: Sequence[str] | None = None) -> MergeOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the pinned Voice Light Qwen3-1.7B LoRA into its pinned base model and "
            "write a standalone BF16 safetensors Hugging Face checkpoint."
        )
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--destination-repository-id", required=True)
    options = parser.parse_args(arguments)
    if not HUGGING_FACE_REPOSITORY_PATTERN.fullmatch(options.destination_repository_id):
        parser.error("--destination-repository-id must have the form owner/model")
    return MergeOptions(
        output_directory=options.output_directory,
        destination_repository_id=options.destination_repository_id,
    )


def merge_qwen_lora(options: MergeOptions) -> MergedModelProvenance:
    output_directory = prepare_output_directory(options.output_directory)
    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        LANGUAGE_MODEL_NAME,
        revision=LANGUAGE_MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    adapter_model = PeftModelForCausalLM.from_pretrained(
        base_model,
        LANGUAGE_MODEL_ADAPTER_NAME,
        revision=LANGUAGE_MODEL_ADAPTER_REVISION,
        is_trainable=False,
    )
    merged_model: PreTrainedModel = adapter_model.merge_and_unload(
        progressbar=True,
        safe_merge=True,
    )
    validate_bfloat16_checkpoint(merged_model)
    merged_model.save_pretrained(
        output_directory,
        safe_serialization=True,
        max_shard_size=MAXIMUM_SHARD_SIZE,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        LANGUAGE_MODEL_NAME,
        revision=LANGUAGE_MODEL_REVISION,
    )
    tokenizer.save_pretrained(output_directory)
    validate_safetensors_artifact(output_directory)

    provenance = create_provenance(options.destination_repository_id)
    (output_directory / PROVENANCE_FILENAME).write_text(
        f"{provenance.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    (output_directory / MODEL_CARD_FILENAME).write_text(
        render_model_card(provenance),
        encoding="utf-8",
    )
    return provenance


def prepare_output_directory(output_directory: Path) -> Path:
    resolved_directory = output_directory.expanduser().resolve()
    if resolved_directory.exists():
        if not resolved_directory.is_dir():
            raise ValueError(f"Merge output path is not a directory: {resolved_directory}")
        if next(resolved_directory.iterdir(), None) is not None:
            raise ValueError(f"Merge output directory must be empty: {resolved_directory}")
    else:
        resolved_directory.mkdir(parents=True)
    return resolved_directory


def validate_bfloat16_checkpoint(model: PreTrainedModel) -> None:
    invalid_tensors = tuple(
        name
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16
    )
    if invalid_tensors:
        names = ", ".join(invalid_tensors[:5])
        raise RuntimeError(f"Merged checkpoint contains non-BF16 floating tensors: {names}")


def validate_safetensors_artifact(output_directory: Path) -> None:
    safetensors_files = tuple(output_directory.glob("*.safetensors"))
    if not safetensors_files:
        raise RuntimeError("Merged model did not produce a safetensors checkpoint.")
    pytorch_binary_files = tuple(output_directory.glob("*.bin"))
    if pytorch_binary_files:
        raise RuntimeError("Merged model unexpectedly produced PyTorch binary weights.")


def create_provenance(destination_repository_id: str) -> MergedModelProvenance:
    return MergedModelProvenance(
        schema_version=1,
        destination_repository_id=destination_repository_id,
        base_model=HuggingFaceRevision(
            repository_id=LANGUAGE_MODEL_NAME,
            revision=LANGUAGE_MODEL_REVISION,
        ),
        adapter=HuggingFaceRevision(
            repository_id=LANGUAGE_MODEL_ADAPTER_NAME,
            revision=LANGUAGE_MODEL_ADAPTER_REVISION,
        ),
        merge=MergeImplementation(
            dtype="bfloat16",
            device="cpu",
            safe_merge=True,
            safe_serialization=True,
            maximum_shard_size=MAXIMUM_SHARD_SIZE,
            huggingface_hub_version=importlib.metadata.version("huggingface-hub"),
            safetensors_version=importlib.metadata.version("safetensors"),
            transformers_version=importlib.metadata.version("transformers"),
            peft_version=importlib.metadata.version("peft"),
            torch_version=torch.__version__,
        ),
    )


def render_model_card(provenance: MergedModelProvenance) -> str:
    base_model = provenance.base_model
    adapter = provenance.adapter
    merge = provenance.merge
    return f"""---
base_model: {base_model.repository_id}
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- voice-assistant
- tool-use
- lora
- merged
---

# Voice Light Qwen3 1.7B Tool-Use - Merged

This repository is the standalone BF16 inference checkpoint for the Voice Light conversational
model. It merges the Voice Light tool-use LoRA into the original Qwen3-1.7B weights. It does not
require PEFT or dynamic LoRA loading at inference time.

## Sources

- Base model: [{base_model.repository_id}](https://huggingface.co/{base_model.repository_id}/tree/{base_model.revision})
  at revision `{base_model.revision}`
- LoRA adapter: [{adapter.repository_id}](https://huggingface.co/{adapter.repository_id}/tree/{adapter.revision})
  at revision `{adapter.revision}`

Refer to those repositories for the original model documentation, adapter training information,
licenses, and intended-use constraints.

## Merge

- Method: PEFT `merge_and_unload(safe_merge=True)`
- Merge device: CPU
- Weight format: safetensors with a `{merge.maximum_shard_size}` maximum shard size
- Weight dtype: BF16
- Hugging Face Hub: `{merge.huggingface_hub_version}`
- Safetensors: `{merge.safetensors_version}`
- Transformers: `{merge.transformers_version}`
- PEFT: `{merge.peft_version}`
- PyTorch: `{merge.torch_version}`

The exact machine-readable source revisions and merge settings are included in
`{PROVENANCE_FILENAME}`. The destination repository expected when this artifact was built is
`{provenance.destination_repository_id}`.
"""


if __name__ == "__main__":
    main()

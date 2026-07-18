from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.training.tool_use.composition import (
    CompositionPlan,
    build_training_compositions,
)
from app.training.tool_use.renderer import (
    ChatTemplateTokenizer,
    QwenTrainingExample,
    render_qwen_training_records,
)
from app.training.tool_use.schema import DatasetSplit, ToolDialogueRecord, ToolUseBaseModel


@dataclass(frozen=True)
class PreparedSplit:
    plans: tuple[CompositionPlan, ...]
    records: tuple[ToolDialogueRecord, ...]
    examples: tuple[QwenTrainingExample, ...]
    omitted_source_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedTrainingCorpus:
    training: PreparedSplit
    validation: PreparedSplit


class TokenLengthSummary(ToolUseBaseModel):
    minimum: int = Field(ge=1)
    maximum: int = Field(ge=1)
    mean: float = Field(gt=0.0)
    total: int = Field(ge=1)


class PreparedSplitSummary(ToolUseBaseModel):
    source_record_count: int = Field(ge=1)
    composition_count: int = Field(ge=1)
    omitted_source_record_ids: tuple[str, ...]
    token_lengths: TokenLengthSummary
    assistant_target_tokens: int = Field(ge=1)


class TrainingCorpusManifest(ToolUseBaseModel):
    schema_version: Literal["voice-light.training-corpus-manifest/v1"] = (
        "voice-light.training-corpus-manifest/v1"
    )
    source_records_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training: PreparedSplitSummary
    validation: PreparedSplitSummary


def prepare_training_corpus(
    tokenizer: ChatTemplateTokenizer,
    source_records: tuple[ToolDialogueRecord, ...],
    logical_epochs: int,
    random_seed: int,
    minimum_user_turns: int,
    maximum_user_turns: int,
    maximum_sequence_tokens: int,
) -> PreparedTrainingCorpus:
    training = _prepare_split(
        tokenizer=tokenizer,
        source_records=source_records,
        split=DatasetSplit.TRAIN,
        logical_epochs=logical_epochs,
        random_seed=random_seed,
        minimum_user_turns=minimum_user_turns,
        maximum_user_turns=maximum_user_turns,
        maximum_sequence_tokens=maximum_sequence_tokens,
    )
    validation = _prepare_split(
        tokenizer=tokenizer,
        source_records=source_records,
        split=DatasetSplit.VALIDATION,
        logical_epochs=1,
        random_seed=random_seed + 50_000_000,
        minimum_user_turns=minimum_user_turns,
        maximum_user_turns=maximum_user_turns,
        maximum_sequence_tokens=maximum_sequence_tokens,
    )
    return PreparedTrainingCorpus(training=training, validation=validation)


def training_corpus_manifest(
    records_path: Path,
    source_records: tuple[ToolDialogueRecord, ...],
    corpus: PreparedTrainingCorpus,
) -> TrainingCorpusManifest:
    return TrainingCorpusManifest(
        source_records_sha256=_sha256_file(records_path),
        training=_split_summary(source_records, DatasetSplit.TRAIN, corpus.training),
        validation=_split_summary(
            source_records,
            DatasetSplit.VALIDATION,
            corpus.validation,
        ),
    )


def write_prepared_split(
    directory: Path,
    name: str,
    prepared_split: PreparedSplit,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        directory / f"{name}-composition-plans.jsonl",
        prepared_split.plans,
    )
    _write_jsonl(
        directory / f"{name}-composed-records.jsonl",
        prepared_split.records,
    )


def _prepare_split(
    tokenizer: ChatTemplateTokenizer,
    source_records: tuple[ToolDialogueRecord, ...],
    split: DatasetSplit,
    logical_epochs: int,
    random_seed: int,
    minimum_user_turns: int,
    maximum_user_turns: int,
    maximum_sequence_tokens: int,
) -> PreparedSplit:
    pairs = tuple(
        pair
        for logical_epoch in range(logical_epochs)
        for pair in build_training_compositions(
            source_records=source_records,
            split=split,
            logical_epoch=logical_epoch,
            random_seed=random_seed,
            minimum_user_turns=minimum_user_turns,
            maximum_user_turns=maximum_user_turns,
        )
    )
    plans = tuple(plan for plan, _ in pairs)
    records = tuple(record for _, record in pairs)
    examples = render_qwen_training_records(tokenizer, records)
    oversized = tuple(
        example.record_id
        for example in examples
        if len(example.input_ids) > maximum_sequence_tokens
    )
    if oversized:
        preview = ", ".join(oversized[:5])
        raise ValueError(
            f"{len(oversized)} composed {split.value} records exceed "
            f"{maximum_sequence_tokens} tokens; first records: {preview}."
        )
    source_ids = {
        record.record_id for record in source_records if record.metadata.split.name is split
    }
    selected_ids = {record_id for plan in plans for record_id in plan.segment_record_ids}
    return PreparedSplit(
        plans=plans,
        records=records,
        examples=examples,
        omitted_source_record_ids=tuple(sorted(source_ids - selected_ids)),
    )


def _split_summary(
    source_records: tuple[ToolDialogueRecord, ...],
    split: DatasetSplit,
    prepared_split: PreparedSplit,
) -> PreparedSplitSummary:
    lengths = tuple(len(example.input_ids) for example in prepared_split.examples)
    return PreparedSplitSummary(
        source_record_count=sum(record.metadata.split.name is split for record in source_records),
        composition_count=len(prepared_split.examples),
        omitted_source_record_ids=prepared_split.omitted_source_record_ids,
        token_lengths=TokenLengthSummary(
            minimum=min(lengths),
            maximum=max(lengths),
            mean=sum(lengths) / len(lengths),
            total=sum(lengths),
        ),
        assistant_target_tokens=sum(
            sum(example.assistant_loss_mask) for example in prepared_split.examples
        ),
    )


def _write_jsonl(
    path: Path,
    models: Sequence[ToolUseBaseModel],
) -> None:
    content = "\n".join(model.model_dump_json() for model in models)
    path.write_text(f"{content}\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import random
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from pydantic import Field
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from app.training.tool_use.renderer import (
    IGNORE_INDEX,
    ChatTemplateTokenizer,
    QwenTrainingExample,
)
from app.training.tool_use.schema import ToolUseBaseModel, read_records, validate_records
from app.training.tool_use.training_config import LoraTrainingConfig
from app.training.tool_use.training_data import (
    PreparedTrainingCorpus,
    TrainingCorpusManifest,
    prepare_training_corpus,
    training_corpus_manifest,
    write_prepared_split,
)


class TrainingBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class TrainingMetric(ToolUseBaseModel):
    schema_version: Literal["voice-light.training-metric/v1"] = "voice-light.training-metric/v1"
    phase: Literal["train", "validation"]
    optimizer_step: int = Field(ge=0)
    loss: float = Field(ge=0.0)
    learning_rate: float = Field(ge=0.0)
    elapsed_seconds: float = Field(ge=0.0)


class HardwareSummary(ToolUseBaseModel):
    device_name: str
    total_vram_bytes: int = Field(ge=1)
    peak_allocated_vram_bytes: int = Field(ge=0)
    peak_reserved_vram_bytes: int = Field(ge=0)
    torch_version: str
    transformers_version: str
    peft_version: str
    cuda_version: str


class TrainingRunSummary(ToolUseBaseModel):
    schema_version: Literal["voice-light.training-run-summary/v1"] = (
        "voice-light.training-run-summary/v1"
    )
    source_commit: str
    optimizer_steps: int = Field(ge=1)
    training_examples: int = Field(ge=1)
    validation_examples: int = Field(ge=1)
    training_tokens: int = Field(ge=1)
    assistant_target_tokens: int = Field(ge=1)
    final_training_loss: float = Field(ge=0.0)
    validation_loss: float = Field(ge=0.0)
    elapsed_seconds: float = Field(gt=0.0)
    trainable_parameters: int = Field(ge=1)
    total_parameters: int = Field(ge=1)
    hardware: HardwareSummary


class ArtifactHash(ToolUseBaseModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)


class ArtifactHashManifest(ToolUseBaseModel):
    schema_version: Literal["voice-light.artifact-hashes/v1"] = "voice-light.artifact-hashes/v1"
    files: tuple[ArtifactHash, ...]


class QwenExampleDataset(Dataset[QwenTrainingExample]):
    def __init__(self, examples: tuple[QwenTrainingExample, ...]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> QwenTrainingExample:
        return self._examples[index]


@dataclass(frozen=True)
class QwenBatchCollator:
    pad_token_id: int

    def __call__(self, examples: Sequence[QwenTrainingExample]) -> TrainingBatch:
        maximum_length = max(len(example.input_ids) for example in examples)
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for example in examples:
            padding_length = maximum_length - len(example.input_ids)
            input_rows.append([*example.input_ids, *([self.pad_token_id] * padding_length)])
            attention_rows.append([*([1] * len(example.input_ids)), *([0] * padding_length)])
            label_rows.append([*example.labels, *([IGNORE_INDEX] * padding_length)])
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


def run_lora_training(
    config: LoraTrainingConfig,
    prepare_only: bool,
) -> TrainingRunSummary | None:
    _prepare_clean_output_directory(config.output_directory)
    _write_model(config.output_directory / "training-config.json", config)
    source_records = read_records(config.records_path)
    validate_records(source_records)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_identifier,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("The target tokenizer has no padding token.")
    corpus = prepare_training_corpus(
        tokenizer=cast(ChatTemplateTokenizer, tokenizer),
        source_records=source_records,
        logical_epochs=config.logical_epochs,
        random_seed=config.random_seed,
        minimum_user_turns=config.minimum_user_turns,
        maximum_user_turns=config.maximum_user_turns,
        maximum_sequence_tokens=config.maximum_sequence_tokens,
    )
    corpus_manifest = training_corpus_manifest(
        records_path=config.records_path,
        source_records=source_records,
        corpus=corpus,
    )
    _write_corpus_artifacts(config.output_directory, corpus, corpus_manifest)
    tokenizer.save_pretrained(config.output_directory / "tokenizer")
    if prepare_only:
        _write_artifact_hashes(config.output_directory)
        return None
    if not torch.cuda.is_available():
        raise ValueError("BF16 LoRA training requires an explicitly allocated CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("The allocated CUDA GPU does not support BF16 training.")

    _set_random_seeds(config.random_seed)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_identifier,
        revision=config.model_revision,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.to(torch.device("cuda"))
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        ),
    )
    if config.gradient_checkpointing:
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    trainable_parameters, total_parameters = peft_model.get_nb_trainable_parameters()
    summary = _train_model(
        model=peft_model,
        tokenizer=tokenizer,
        corpus=corpus,
        corpus_manifest=corpus_manifest,
        config=config,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )
    _write_model(config.output_directory / "training-summary.json", summary)
    _write_artifact_hashes(config.output_directory)
    return summary


def _train_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    corpus: PreparedTrainingCorpus,
    corpus_manifest: TrainingCorpusManifest,
    config: LoraTrainingConfig,
    trainable_parameters: int,
    total_parameters: int,
) -> TrainingRunSummary:
    collator = QwenBatchCollator(pad_token_id=cast(int, tokenizer.pad_token_id))
    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    training_loader = DataLoader(
        QwenExampleDataset(corpus.training.examples),
        batch_size=config.per_device_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        QwenExampleDataset(corpus.validation.examples),
        batch_size=config.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer_steps = math.ceil(len(training_loader) / config.gradient_accumulation_steps)
    warmup_steps = max(1, round(optimizer_steps * config.warmup_ratio))
    optimizer = AdamW(
        tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        fused=True,
    )
    scheduler = LambdaLR(
        optimizer,
        _learning_rate_schedule(
            warmup_steps=warmup_steps,
            total_steps=optimizer_steps,
        ),
    )
    metrics_path = config.output_directory / "metrics.jsonl"
    started_at = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    accumulated_training_loss = 0.0
    accumulated_micro_batches = 0
    optimizer_step = 0
    final_training_loss = 0.0
    model.train()
    for micro_batch_index, cpu_batch in enumerate(training_loader, start=1):
        batch = _move_batch_to_cuda(cpu_batch)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None:
            raise AssertionError("Causal language model training always returns a loss.")
        accumulated_training_loss += float(loss.detach().item())
        accumulated_micro_batches += 1
        group_start = (
            (micro_batch_index - 1) // config.gradient_accumulation_steps
        ) * config.gradient_accumulation_steps
        group_size = min(
            config.gradient_accumulation_steps,
            len(training_loader) - group_start,
        )
        (loss / group_size).backward()
        should_step = (
            micro_batch_index % config.gradient_accumulation_steps == 0
            or micro_batch_index == len(training_loader)
        )
        if not should_step:
            continue
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.maximum_gradient_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        final_training_loss = accumulated_training_loss / accumulated_micro_batches
        _append_metric(
            metrics_path,
            TrainingMetric(
                phase="train",
                optimizer_step=optimizer_step,
                loss=final_training_loss,
                learning_rate=scheduler.get_last_lr()[0],
                elapsed_seconds=time.monotonic() - started_at,
            ),
        )
        accumulated_training_loss = 0.0
        accumulated_micro_batches = 0
        if optimizer_step % config.save_steps == 0:
            _save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                directory=config.output_directory / f"checkpoint-{optimizer_step:05d}",
            )

    validation_loss = _evaluate_model(model, validation_loader)
    elapsed_seconds = time.monotonic() - started_at
    _append_metric(
        metrics_path,
        TrainingMetric(
            phase="validation",
            optimizer_step=optimizer_step,
            loss=validation_loss,
            learning_rate=scheduler.get_last_lr()[0],
            elapsed_seconds=elapsed_seconds,
        ),
    )
    _save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        directory=config.output_directory / "adapter",
    )
    device_properties = torch.cuda.get_device_properties(0)
    return TrainingRunSummary(
        source_commit=_repository_commit(),
        optimizer_steps=optimizer_step,
        training_examples=len(corpus.training.examples),
        validation_examples=len(corpus.validation.examples),
        training_tokens=corpus_manifest.training.token_lengths.total,
        assistant_target_tokens=corpus_manifest.training.assistant_target_tokens,
        final_training_loss=final_training_loss,
        validation_loss=validation_loss,
        elapsed_seconds=elapsed_seconds,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        hardware=HardwareSummary(
            device_name=device_properties.name,
            total_vram_bytes=device_properties.total_memory,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
            torch_version=torch.__version__,
            transformers_version=importlib.metadata.version("transformers"),
            peft_version=importlib.metadata.version("peft"),
            cuda_version=torch.version.cuda or "unknown",
        ),
    )


def _evaluate_model(
    model: PeftModel,
    validation_loader: DataLoader[TrainingBatch],
) -> float:
    model.eval()
    weighted_loss = 0.0
    target_token_count = 0
    with torch.no_grad():
        for cpu_batch in validation_loader:
            batch = _move_batch_to_cuda(cpu_batch)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise AssertionError("Causal language model evaluation always returns a loss.")
            batch_target_tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())
            weighted_loss += float(loss.item()) * batch_target_tokens
            target_token_count += batch_target_tokens
    if target_token_count == 0:
        raise AssertionError("Validation data always contains assistant target tokens.")
    return weighted_loss / target_token_count


def _move_batch_to_cuda(batch: TrainingBatch) -> TrainingBatch:
    return {
        "input_ids": batch["input_ids"].to("cuda", non_blocking=True),
        "attention_mask": batch["attention_mask"].to("cuda", non_blocking=True),
        "labels": batch["labels"].to("cuda", non_blocking=True),
    }


def _learning_rate_schedule(
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    def multiplier(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return multiplier


def _save_checkpoint(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)


def _write_corpus_artifacts(
    output_directory: Path,
    corpus: PreparedTrainingCorpus,
    manifest: TrainingCorpusManifest,
) -> None:
    write_prepared_split(output_directory, "train", corpus.training)
    write_prepared_split(output_directory, "validation", corpus.validation)
    _write_model(output_directory / "training-corpus-manifest.json", manifest)


def _append_metric(path: Path, metric: TrainingMetric) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(metric.model_dump_json())
        output_file.write("\n")


def _prepare_clean_output_directory(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Training output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def _write_model(path: Path, model: ToolUseBaseModel) -> None:
    path.write_text(f"{model.model_dump_json(indent=2)}\n", encoding="utf-8")


def _write_artifact_hashes(directory: Path) -> None:
    hashes_path = directory / "artifact-hashes.json"
    files = tuple(
        ArtifactHash(
            relative_path=path.relative_to(directory).as_posix(),
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != hashes_path
    )
    _write_model(hashes_path, ArtifactHashManifest(files=files))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_commit() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _set_random_seeds(random_seed: int) -> None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True

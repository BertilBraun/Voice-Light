from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from app.local.synthetic_generation.labels import (
    SYNTHETIC_DATASET_ID,
    materialize_training_sample,
)
from app.local.synthetic_generation.models import SyntheticModel
from app.local.synthetic_generation.rendering import (
    SyntheticRenderRequest,
    render_conversation,
)
from app.local.training_corpus.export import (
    SCHEMA_VERSION,
    ExportManifest,
    ExportSplitSummary,
    write_training_shards,
)
from app.local.training_corpus.splits import (
    ConversationSplitAssignment,
    ConversationSplitPlan,
    TrainingCorpusSplit,
)
from app.local.training_samples.service import (
    FRAME_SECONDS,
    INPUT_DURATION_SECONDS,
    TRAINING_LABEL_VERSION,
)

SYNTHETIC_ANNOTATION_VERSION = "planned-event-timeline-v1"
SYNTHETIC_REGION_VERSION = "planned-two-party-channel-v1"
SYNTHETIC_METRIC_VERSION = "synthetic-source-provenance-v1"


class SyntheticCorpusRequest(SyntheticModel):
    corpus_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    plans: tuple[SyntheticRenderRequest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_plan_ids(self) -> SyntheticCorpusRequest:
        plan_ids = tuple(request.plan.plan_id for request in self.plans)
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("Synthetic corpus plan IDs must be unique.")
        return self


class SyntheticCorpusItem(SyntheticModel):
    plan_id: str
    render_manifest_path: str = Field(min_length=1)
    render_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_window_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class SyntheticCorpusBuildManifest(SyntheticModel):
    schema_version: Literal["voice-light-synthetic-corpus-v1"] = "voice-light-synthetic-corpus-v1"
    corpus_id: str
    description: str
    generated_at: datetime
    label_source: Literal["planned_event_timeline"] = "planned_event_timeline"
    evaluation_policy: Literal["train_only_real_validation_required"] = (
        "train_only_real_validation_required"
    )
    items: tuple[SyntheticCorpusItem, ...]


def build_synthetic_corpus(
    request: SyntheticCorpusRequest,
    output_directory: Path,
) -> SyntheticCorpusBuildManifest:
    output_directory.mkdir(parents=True, exist_ok=True)
    samples = []
    items = []
    assignments = []
    for render_request in request.plans:
        plan_id = render_request.plan.plan_id
        render_directory = output_directory / "audio" / plan_id
        render_manifest = render_conversation(render_request, render_directory)
        user_relative_path = render_manifest.user_audio_path.relative_to(output_directory)
        assistant_relative_path = render_manifest.assistant_audio_path.relative_to(output_directory)
        sample = materialize_training_sample(
            plan=render_request.plan,
            user_audio_path=user_relative_path,
            assistant_audio_path=assistant_relative_path,
        )
        samples.append(sample)
        assignments.append(
            ConversationSplitAssignment(
                dataset_id=SYNTHETIC_DATASET_ID,
                sample_id=sample.sample_id,
                split=TrainingCorpusSplit.TRAIN,
            )
        )
        render_manifest_path = render_directory / "render.json"
        items.append(
            SyntheticCorpusItem(
                plan_id=plan_id,
                render_manifest_path=render_manifest_path.relative_to(output_directory).as_posix(),
                render_manifest_sha256=_file_sha256(render_manifest_path),
                training_window_id=sample.window_id,
            )
        )
    shards = write_training_shards(output_directory, samples)
    split_plan = ConversationSplitPlan(
        seed=f"synthetic:{request.corpus_id}",
        assignments=tuple(assignments),
    )
    split_summaries = tuple(
        ExportSplitSummary(
            split=split,
            recording_count=len(samples) if split is TrainingCorpusSplit.TRAIN else 0,
            training_sample_count=len(samples) if split is TrainingCorpusSplit.TRAIN else 0,
            source_duration_seconds=(
                len(samples) * INPUT_DURATION_SECONDS if split is TrainingCorpusSplit.TRAIN else 0.0
            ),
        )
        for split in TrainingCorpusSplit
    )
    export_manifest = ExportManifest(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(UTC),
        metric_version=SYNTHETIC_METRIC_VERSION,
        annotation_version=SYNTHETIC_ANNOTATION_VERSION,
        region_analysis_version=SYNTHETIC_REGION_VERSION,
        training_label_version=TRAINING_LABEL_VERSION,
        input_duration_seconds=INPUT_DURATION_SECONDS,
        frame_seconds=FRAME_SECONDS,
        review_set_name=f"synthetic:{request.corpus_id}:train-only",
        split_plan=split_plan,
        recording_count=len(samples),
        training_sample_count=len(samples),
        splits=split_summaries,
        shards=shards,
    )
    (output_directory / "corpus.json").write_text(
        export_manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    build_manifest = SyntheticCorpusBuildManifest(
        corpus_id=request.corpus_id,
        description=request.description,
        generated_at=datetime.now(UTC),
        items=tuple(items),
    )
    (output_directory / "synthetic.json").write_text(
        build_manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return build_manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

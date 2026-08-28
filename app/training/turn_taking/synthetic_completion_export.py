from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid5

import soundfile as sf

from app.local.db.models import TrackSide
from app.local.synthetic_generation.conversation_compiler import ConversationCompilerConfig
from app.local.training_corpus.export import (
    ExportManifest,
    ExportSplitSummary,
    MaterializedTrainingSample,
    write_training_shards,
)
from app.local.training_corpus.splits import (
    ConversationSplitAssignment,
    ConversationSplitPlan,
    TrainingCorpusSplit,
)
from app.training.turn_taking.config import SyntheticAnchorSamplingConfig
from app.training.turn_taking.data import TrainingItem
from app.training.turn_taking.synthetic_dataset import AnchoredSyntheticTurnTakingDataset

MASKED_TARGET = -1.0


def export_synthetic_completion_validation(
    source_root: Path,
    output_root: Path,
    random_seed: int,
) -> ExportManifest:
    if output_root.exists():
        raise ValueError(f"Synthetic completion export already exists: {output_root}")
    source_manifest = ExportManifest.model_validate_json(
        (source_root / "corpus.json").read_text(encoding="utf-8")
    )
    compiler_config = ConversationCompilerConfig(
        sample_rate_hz=16_000,
        crop_duration_seconds=20.0,
        crop_variant_count=1,
    )
    dataset = AnchoredSyntheticTurnTakingDataset(
        root=source_root,
        split=TrainingCorpusSplit.VALIDATION,
        compiler_config=compiler_config,
        sampling_config=SyntheticAnchorSamplingConfig(),
        augmenter=None,
        random_seed=random_seed,
        randomize=False,
        completion_only=True,
    )
    dataset_id = uuid5(
        UUID("ed157dee-136d-5c11-86b4-1b3b319a826a"),
        source_manifest.review_set_name,
    )
    audio_directory = output_root / "audio"
    audio_directory.mkdir(parents=True)
    samples: list[MaterializedTrainingSample] = []
    conversation_ids: set[str] = set()
    for index, anchor_entry in enumerate(dataset.anchors):
        item = dataset[index]
        conversation = dataset.conversations[anchor_entry.conversation_index]
        conversation_id = conversation.plan.conversation_id
        conversation_ids.add(conversation_id)
        window_id = hashlib.sha256(item.sample_id.encode("utf-8")).hexdigest()
        relative_audio_path = Path("audio") / f"{window_id}.wav"
        sf.write(
            output_root / relative_audio_path,
            item.waveform.numpy(),
            compiler_config.sample_rate_hz,
            format="WAV",
            subtype="PCM_16",
        )
        anchor_frames = item.targets.event_mask[:, 0].nonzero().flatten().tolist()
        if len(anchor_frames) != 1:
            raise ValueError("A completion validation crop must contain exactly one anchor frame.")
        anchor_frame = anchor_frames[0]
        completion_target = float(item.targets.event_targets[anchor_frame, 0])
        samples.append(
            _materialized_sample(
                item=item,
                dataset_id=dataset_id,
                dataset_name=source_manifest.review_set_name,
                conversation_id=conversation_id,
                window_id=window_id,
                relative_audio_path=relative_audio_path,
                anchor_frame=anchor_frame,
                completion_target=completion_target,
                schema_version=source_manifest.schema_version,
                training_label_version=source_manifest.training_label_version,
            )
        )
    shards = write_training_shards(output_root, samples)
    assignments = tuple(
        ConversationSplitAssignment(
            dataset_id=dataset_id,
            sample_id=uuid5(dataset_id, conversation_id),
            split=TrainingCorpusSplit.VALIDATION,
        )
        for conversation_id in sorted(conversation_ids)
    )
    split_plan = ConversationSplitPlan(
        seed=f"synthetic-completion-validation:{random_seed}",
        assignments=assignments,
    )
    manifest = ExportManifest(
        schema_version=source_manifest.schema_version,
        generated_at=datetime.now(UTC),
        metric_version=source_manifest.metric_version,
        annotation_version=source_manifest.annotation_version,
        region_analysis_version=source_manifest.region_analysis_version,
        training_label_version=source_manifest.training_label_version,
        input_duration_seconds=20.0,
        frame_seconds=0.08,
        review_set_name=f"{source_manifest.review_set_name}:completion-validation",
        split_plan=split_plan,
        recording_count=len(conversation_ids),
        training_sample_count=len(samples),
        splits=tuple(
            ExportSplitSummary(
                split=split,
                recording_count=(
                    len(conversation_ids) if split is TrainingCorpusSplit.VALIDATION else 0
                ),
                training_sample_count=(
                    len(samples) if split is TrainingCorpusSplit.VALIDATION else 0
                ),
                source_duration_seconds=(
                    len(samples) * 20.0 if split is TrainingCorpusSplit.VALIDATION else 0.0
                ),
            )
            for split in TrainingCorpusSplit
        ),
        shards=shards,
    )
    (output_root / "corpus.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _materialized_sample(
    *,
    item: TrainingItem,
    dataset_id: UUID,
    dataset_name: str,
    conversation_id: str,
    window_id: str,
    relative_audio_path: Path,
    anchor_frame: int,
    completion_target: float,
    schema_version: str,
    training_label_version: str,
) -> MaterializedTrainingSample:
    frame_count = item.targets.yield_probability.shape[0]
    masked = [MASKED_TARGET] * frame_count
    completion = masked.copy()
    completion[anchor_frame] = completion_target
    continuation = _continuation_targets(
        item=item,
        anchor_frame=anchor_frame,
        completion_target=completion_target,
    )
    user_floor = tuple(
        1.0 - float(value) if bool(valid) else MASKED_TARGET
        for value, valid in zip(
            item.targets.yield_probability,
            item.targets.primary_mask,
            strict=True,
        )
    )
    return MaterializedTrainingSample(
        schema_version=schema_version,
        training_label_version=training_label_version,
        window_id=window_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        sample_id=uuid5(dataset_id, conversation_id),
        external_id=conversation_id,
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=TrainingCorpusSplit.VALIDATION,
        user_audio_path=relative_audio_path.as_posix(),
        assistant_audio_path=None,
        start_seconds=0.0,
        end_seconds=20.0,
        quality_score=1.0,
        category=f"synthetic_completion:{'eot' if completion_target == 1.0 else 'hold'}",
        assistant_has_floor=tuple(float(value) for value in item.assistant_speaking),
        assistant_speaking_probability=tuple(float(value) for value in item.assistant_speaking),
        p_user_has_floor=user_floor,
        p_user_floor_now=user_floor,
        p_user_yield=tuple(
            float(value) if bool(valid) else MASKED_TARGET
            for value, valid in zip(
                item.targets.yield_probability,
                item.targets.primary_mask,
                strict=True,
            )
        ),
        p_assistant_backchannel=tuple(masked),
        future_activity_0_200=tuple(masked),
        future_activity_200_500=tuple(masked),
        future_activity_500_1000=tuple(masked),
        future_activity_1000_1500=tuple(masked),
        turn_completion=tuple(completion),
        continuation_pause=continuation,
        non_floor_feedback=tuple(masked),
        floor_take=tuple(masked),
    )


def _continuation_targets(
    item: TrainingItem,
    anchor_frame: int,
    completion_target: float,
) -> tuple[float, ...]:
    values = [
        float(value) if bool(valid) else MASKED_TARGET
        for value, valid in zip(
            item.targets.event_targets[:, 1],
            item.targets.event_mask[:, 1],
            strict=True,
        )
    ]
    if completion_target == 1.0:
        return tuple(values)
    positive_frames = [index for index, value in enumerate(values) if value == 1.0]
    if not positive_frames:
        raise ValueError("A synthetic hold crop must contain its continuation interval.")
    closest_frame = min(positive_frames, key=lambda index: abs(index - anchor_frame))
    interval_end = closest_frame
    while interval_end + 1 < len(values) and values[interval_end + 1] == 1.0:
        interval_end += 1
    if interval_end < anchor_frame:
        raise ValueError("A synthetic hold interval must not end before its anchor.")
    for frame_index in range(anchor_frame, interval_end + 1):
        values[frame_index] = 1.0
    return tuple(values)

from io import StringIO
from pathlib import Path
from uuid import UUID

import torch

from app.local.db.models import TrackSide
from app.local.training_corpus.export import FRAMES_PER_SAMPLE, MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.backbone import BackboneFeatures
from app.training.turn_taking.benchmark_inventory import build_candidate_inventory
from app.training.turn_taking.benchmark_voice_light import (
    LoadedVoiceLightCheckpoint,
    predict_voice_light_checkpoints,
)
from app.training.turn_taking.config import AdapterConfig, TrainingConfig
from app.training.turn_taking.data import FrameTargets, TrainingBatch
from app.training.turn_taking.hub import frame_targets_from_sample
from app.training.turn_taking.model import TurnTakingAdapter


class FixedBackbone:
    def extract(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> BackboneFeatures:
        batch_size = waveforms.shape[0]
        return BackboneFeatures(
            taps=(torch.ones((batch_size, FRAMES_PER_SAMPLE, 2)),),
            frame_mask=torch.ones((batch_size, FRAMES_PER_SAMPLE), dtype=torch.bool),
        )


def test_voice_light_checkpoints_share_features_and_cache_only_candidate_points() -> None:
    sample = _sample()
    inventory = build_candidate_inventory(
        samples=(sample,),
        corpus_repository="owner/corpus",
        corpus_revision="1" * 40,
        split=TrainingCorpusSplit.VALIDATION,
    )
    adapter_config = AdapterConfig(
        feature_dimension=2,
        tap_layer_indices=(0,),
        tap_projection_dimension=2,
        fused_dimension=2,
        recurrent_dimension=2,
        dropout=0.0,
    )
    config = TrainingConfig(
        model_identifier="model",
        adapter=adapter_config,
    )
    checkpoints = tuple(
        LoadedVoiceLightCheckpoint(
            path=Path(f"checkpoint-{step}.pt"),
            sha256=str(step)[0] * 64,
            optimizer_step=step,
            config=config,
            adapter=TurnTakingAdapter(adapter_config),
        )
        for step in (3_500, 7_000)
    )
    targets = frame_targets_from_sample(sample)
    batch = TrainingBatch(
        sample_ids=(sample.window_id,),
        waveforms=torch.zeros((1, 320_000)),
        waveform_lengths=torch.tensor([320_000]),
        assistant_speaking=torch.zeros((1, FRAMES_PER_SAMPLE)),
        targets=FrameTargets(
            yield_probability=targets.yield_probability.unsqueeze(0),
            primary_weight=targets.primary_weight.unsqueeze(0),
            primary_mask=targets.primary_mask.unsqueeze(0),
            event_targets=targets.event_targets.unsqueeze(0),
            event_mask=targets.event_mask.unsqueeze(0),
            future_activity=targets.future_activity.unsqueeze(0),
            future_activity_mask=targets.future_activity_mask.unsqueeze(0),
        ),
    )

    artifacts = predict_voice_light_checkpoints(
        backbone=FixedBackbone(),
        checkpoints=checkpoints,
        batches=(batch,),
        samples=(sample,),
        inventory=inventory,
        model_repository="model",
        model_revision="2" * 40,
        device=torch.device("cpu"),
    )

    assert len(artifacts) == 2
    assert [artifact.manifest.prediction_count for artifact in artifacts] == [3, 3]
    assert [artifact.manifest.detector.configuration.optimizer_step for artifact in artifacts] == [
        3_500,
        7_000,
    ]
    assert all(
        artifact.manifest.inventory_sha256 == inventory.manifest.candidate_sha256
        for artifact in artifacts
    )


def test_voice_light_prediction_reports_batch_progress_and_eta() -> None:
    sample = _sample()
    inventory = build_candidate_inventory(
        samples=(sample,),
        corpus_repository="owner/corpus",
        corpus_revision="1" * 40,
        split=TrainingCorpusSplit.VALIDATION,
    )
    adapter_config = AdapterConfig(
        feature_dimension=2,
        tap_layer_indices=(0,),
        tap_projection_dimension=2,
        fused_dimension=2,
        recurrent_dimension=2,
        dropout=0.0,
    )
    checkpoint = LoadedVoiceLightCheckpoint(
        path=Path("checkpoint.pt"),
        sha256="3" * 64,
        optimizer_step=3_500,
        config=TrainingConfig(model_identifier="model", adapter=adapter_config),
        adapter=TurnTakingAdapter(adapter_config),
    )
    targets = frame_targets_from_sample(sample)
    batch = TrainingBatch(
        sample_ids=(sample.window_id,),
        waveforms=torch.zeros((1, 320_000)),
        waveform_lengths=torch.tensor([320_000]),
        assistant_speaking=torch.zeros((1, FRAMES_PER_SAMPLE)),
        targets=FrameTargets(
            yield_probability=targets.yield_probability.unsqueeze(0),
            primary_weight=targets.primary_weight.unsqueeze(0),
            primary_mask=targets.primary_mask.unsqueeze(0),
            event_targets=targets.event_targets.unsqueeze(0),
            event_mask=targets.event_mask.unsqueeze(0),
            future_activity=targets.future_activity.unsqueeze(0),
            future_activity_mask=targets.future_activity_mask.unsqueeze(0),
        ),
    )
    output = StringIO()

    predict_voice_light_checkpoints(
        backbone=FixedBackbone(),
        checkpoints=(checkpoint,),
        batches=(batch,),
        samples=(sample,),
        inventory=inventory,
        model_repository="model",
        model_revision="2" * 40,
        device=torch.device("cpu"),
        total_batch_count=1,
        progress_output=output,
    )

    lines = output.getvalue().splitlines()
    assert lines[0] == (
        "Voice Light inference: 0/1 batches (0.0%), elapsed 00:00:00, ETA ?, average ?"
    )
    assert "Voice Light inference: 1/1 batches (100.0%)" in lines[1]
    assert "ETA 00:00:00" in lines[1]


def _sample() -> MaterializedTrainingSample:
    zeros = tuple(0.0 for _ in range(FRAMES_PER_SAMPLE))
    floor = [1.0] * FRAMES_PER_SAMPLE
    floor[2:5] = [0.0, 0.0, 0.0]
    user_yield = [0.0] * FRAMES_PER_SAMPLE
    user_yield[2:5] = [0.2, 0.7, 0.9]
    return MaterializedTrainingSample(
        schema_version="voice-light-turn-taking-v1",
        training_label_version="turn-taking-frame-labels-v1",
        window_id="a" * 64,
        dataset_id=UUID("00000000-0000-0000-0000-000000000001"),
        dataset_name="dataset",
        sample_id=UUID("00000000-0000-0000-0000-000000000002"),
        external_id="conversation",
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=TrainingCorpusSplit.VALIDATION,
        user_audio_path="audio/speaker_1.flac",
        assistant_audio_path="audio/speaker_2.flac",
        start_seconds=0.0,
        end_seconds=20.0,
        quality_score=0.9,
        category="hold_pause",
        assistant_has_floor=zeros,
        p_user_has_floor=tuple(floor),
        p_user_yield=tuple(user_yield),
        p_assistant_backchannel=zeros,
        future_activity_0_200=zeros,
        future_activity_200_500=zeros,
        future_activity_500_1000=zeros,
        future_activity_1000_1500=zeros,
        turn_completion=zeros,
        continuation_pause=zeros,
        non_floor_feedback=zeros,
        floor_take=zeros,
    )

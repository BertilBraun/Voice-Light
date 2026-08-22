from io import StringIO
from pathlib import Path
from uuid import UUID

import pytest
import torch

from app.local.db.models import TrackSide
from app.local.training_corpus.export import FRAMES_PER_SAMPLE, MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.backbone import BackboneFeatures
from app.training.turn_taking.benchmark_completion_inventory import (
    build_turn_completion_inventory,
)
from app.training.turn_taking.benchmark_inventory import build_candidate_inventory
from app.training.turn_taking.benchmark_voice_light import (
    LoadedVoiceLightCheckpoint,
    _earliest_aligned_output_indices,
    predict_voice_light_checkpoints,
    predict_voice_light_completion_checkpoints,
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


class CountingBackbone(FixedBackbone):
    def __init__(self) -> None:
        self.call_count = 0

    def extract(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> BackboneFeatures:
        self.call_count += 1
        return super().extract(waveforms, waveform_lengths)


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


def test_completion_head_scores_boundary_once_without_post_boundary_assistant_leakage() -> None:
    sample = _completion_sample()
    inventory = build_turn_completion_inventory(
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
    checkpoints = tuple(_completion_checkpoint(step, adapter_config) for step in (3_500, 7_000))
    backbone = CountingBackbone()

    artifacts = predict_voice_light_completion_checkpoints(
        backbone=backbone,
        checkpoints=checkpoints,
        batches=(_batch(sample, assistant_after_boundary=False),),
        samples=(sample,),
        inventory=inventory,
        model_repository="model",
        model_revision="2" * 40,
        device=torch.device("cpu"),
    )
    leaked = predict_voice_light_completion_checkpoints(
        backbone=backbone,
        checkpoints=(checkpoints[0],),
        batches=(_batch(sample, assistant_after_boundary=True),),
        samples=(sample,),
        inventory=inventory,
        model_repository="model",
        model_revision="2" * 40,
        device=torch.device("cpu"),
    )[0]

    assert backbone.call_count == 2
    assert [artifact.manifest.prediction_count for artifact in artifacts] == [1, 1]
    prediction = artifacts[0].predictions[0]
    assert prediction.absolute_time_seconds == pytest.approx(0.24)
    assert prediction.elapsed_seconds == pytest.approx(0.08)
    assert prediction.completion_probability == pytest.approx(0.25)
    assert leaked.predictions[0].completion_probability == pytest.approx(
        prediction.completion_probability
    )
    assert artifacts[0].manifest.detector.configuration.event_head_index == 0
    assert artifacts[0].manifest.detector.configuration.score_semantic == (
        "turn_completion_probability"
    )
    assert artifacts[0].manifest.detector.configuration.score_schedule == "boundary_only"
    assert artifacts[0].manifest.detector.configuration.score_persistence == "latched"
    assert artifacts[0].manifest.detector.configuration.causal_candidate_coverage_seconds == 2.0
    assert artifacts[0].manifest.detector.configuration.assistant_active_anchors_excluded
    assert artifacts[0].manifest.detector.configuration.insufficient_horizon_candidates_censored
    taps = (torch.ones((1, FRAMES_PER_SAMPLE, 2)),)
    without_future_assistant = (
        checkpoints[0]
        .adapter(
            taps,
            torch.zeros((1, FRAMES_PER_SAMPLE)),
        )
        .event_logits[..., 0]
    )
    assistant_after_boundary = torch.zeros((1, FRAMES_PER_SAMPLE))
    assistant_after_boundary[:, 3:] = 1.0
    with_future_assistant = (
        checkpoints[0]
        .adapter(
            taps,
            assistant_after_boundary,
        )
        .event_logits[..., 0]
    )
    assert with_future_assistant[0, 2].item() == pytest.approx(
        without_future_assistant[0, 2].item()
    )
    assert with_future_assistant[0, 3].item() > without_future_assistant[0, 3].item()


def test_completion_alignment_uses_earliest_output_for_251_to_250_mapping() -> None:
    pairs = _earliest_aligned_output_indices(251, 250)

    assert len(pairs) == 250
    assert pairs[124] == (124, 124)
    assert (125, 124) not in pairs


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


def _completion_sample() -> MaterializedTrainingSample:
    sample = _sample()
    completion = [-1.0] * FRAMES_PER_SAMPLE
    completion[2] = 0.1
    continuation = [-1.0] * FRAMES_PER_SAMPLE
    continuation[2] = 0.9
    return sample.model_copy(
        update={
            "turn_completion": tuple(completion),
            "continuation_pause": tuple(continuation),
        }
    )


def _completion_checkpoint(
    step: int,
    adapter_config: AdapterConfig,
) -> LoadedVoiceLightCheckpoint:
    adapter = TurnTakingAdapter(adapter_config)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        adapter.yield_head.weight.zero_()
        adapter.yield_head.bias.fill_(20.0)
        hidden_size = adapter_config.recurrent_dimension
        assistant_input_index = adapter_config.fused_dimension
        adapter.recurrent.weight_ih_l0[
            2 * hidden_size,
            assistant_input_index,
        ] = 2.0
        adapter.recurrent.bias_ih_l0[hidden_size : 2 * hidden_size] = -10.0
        adapter.event_head.weight[0, 0] = 1.0
        adapter.event_head.bias[0] = torch.logit(torch.tensor(0.25))
    return LoadedVoiceLightCheckpoint(
        path=Path(f"completion-{step}.pt"),
        sha256=str(step)[0] * 64,
        optimizer_step=step,
        config=TrainingConfig(model_identifier="model", adapter=adapter_config),
        adapter=adapter,
    )


def _batch(
    sample: MaterializedTrainingSample,
    assistant_after_boundary: bool,
) -> TrainingBatch:
    targets = frame_targets_from_sample(sample)
    assistant = torch.zeros((1, FRAMES_PER_SAMPLE))
    if assistant_after_boundary:
        assistant[:, 3:] = 1.0
    return TrainingBatch(
        sample_ids=(sample.window_id,),
        waveforms=torch.zeros((1, 320_000)),
        waveform_lengths=torch.tensor([320_000]),
        assistant_speaking=assistant,
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

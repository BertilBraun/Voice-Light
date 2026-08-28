from pathlib import Path
from uuid import UUID

import torch

from app.training.turn_taking.data import FrameTargets, TrainingItem
from app.training.turn_taking.synthetic_completion_export import (
    _continuation_targets,
    _materialized_sample,
)


def test_materialized_completion_sample_keeps_only_selected_boundary() -> None:
    frame_count = 250
    event_targets = torch.zeros((frame_count, 5), dtype=torch.float32)
    event_mask = torch.zeros((frame_count, 5), dtype=torch.bool)
    event_targets[100, 0] = 1.0
    event_mask[100, 0] = True
    event_targets[100, 1] = 0.0
    event_mask[100, 1] = True
    item = TrainingItem(
        sample_id="sample",
        waveform=torch.zeros(320_000),
        assistant_speaking=torch.zeros(frame_count),
        targets=FrameTargets(
            yield_probability=torch.zeros(frame_count),
            primary_weight=torch.ones(frame_count),
            primary_mask=torch.ones(frame_count, dtype=torch.bool),
            event_targets=event_targets,
            event_mask=event_mask,
            future_activity=torch.zeros((frame_count, 4)),
            future_activity_mask=torch.zeros((frame_count, 4), dtype=torch.bool),
        ),
    )

    sample = _materialized_sample(
        item=item,
        dataset_id=UUID("00000000-0000-0000-0000-000000000001"),
        dataset_name="synthetic",
        conversation_id="conversation",
        window_id="a" * 64,
        relative_audio_path=Path("audio/sample.flac"),
        anchor_frame=100,
        completion_target=1.0,
        schema_version="voice-light-synthetic-turn-taking-v2",
        training_label_version="semantic-user-floor-v1",
    )

    assert sample.turn_completion[100] == 1.0
    assert sample.continuation_pause[100] == 0.0
    assert sum(value >= 0.0 for value in sample.turn_completion) == 1
    assert sum(value >= 0.0 for value in sample.continuation_pause) == 1
    assert sample.p_user_has_floor == (1.0,) * frame_count


def test_hold_continuation_interval_is_aligned_to_anchor() -> None:
    item = _training_item()
    item.targets.event_targets[102:106, 1] = 1.0
    item.targets.event_mask[102:106, 1] = True

    continuation = _continuation_targets(item, anchor_frame=100, completion_target=0.0)

    assert continuation[99] == -1.0
    assert continuation[100:106] == (1.0,) * 6
    assert continuation[106] == -1.0


def _training_item() -> TrainingItem:
    frame_count = 250
    return TrainingItem(
        sample_id="sample",
        waveform=torch.zeros(320_000),
        assistant_speaking=torch.zeros(frame_count),
        targets=FrameTargets(
            yield_probability=torch.zeros(frame_count),
            primary_weight=torch.ones(frame_count),
            primary_mask=torch.ones(frame_count, dtype=torch.bool),
            event_targets=torch.zeros((frame_count, 5), dtype=torch.float32),
            event_mask=torch.zeros((frame_count, 5), dtype=torch.bool),
            future_activity=torch.zeros((frame_count, 4)),
            future_activity_mask=torch.zeros((frame_count, 4), dtype=torch.bool),
        ),
    )

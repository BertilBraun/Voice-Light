from uuid import UUID

import pytest
import torch
from torch.utils.data import Dataset

from app.local.db.models import TrackSide
from app.local.training_corpus.export import FRAMES_PER_SAMPLE, MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.completion_training import (
    CompletionBoundaryDataset,
    CompletionClass,
    balanced_completion_weights,
    build_completion_boundaries,
)
from app.training.turn_taking.config import TurnCompletionObjectiveConfig
from app.training.turn_taking.data import FrameTargets, TrainingItem


class _ItemDataset(Dataset[TrainingItem]):
    def __init__(self, item: TrainingItem) -> None:
        self.item = item

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> TrainingItem:
        assert index == 0
        return self.item


def test_completion_boundaries_select_clean_causal_labels() -> None:
    completion = [-1.0] * FRAMES_PER_SAMPLE
    continuation = [-1.0] * FRAMES_PER_SAMPLE
    user_floor = [0.0] * FRAMES_PER_SAMPLE
    assistant_floor = [0.0] * FRAMES_PER_SAMPLE
    completion[5] = 0.1
    continuation[5] = 0.9
    completion[10] = 0.9
    completion[15] = 0.5
    completion[20] = 0.9
    assistant_floor[20] = 1.0
    completion[25] = 0.1
    continuation[25] = 0.9
    user_floor[25] = 1.0

    boundaries = build_completion_boundaries(
        (_sample(completion, continuation, user_floor, assistant_floor),),
        TurnCompletionObjectiveConfig(),
    )

    assert [(item.frame_index, item.completion_class) for item in boundaries] == [
        (5, CompletionClass.HOLD),
        (10, CompletionClass.EOT),
    ]
    assert balanced_completion_weights(boundaries).tolist() == pytest.approx([0.5, 0.5])


def test_boundary_dataset_exposes_only_selected_completion_target() -> None:
    event_mask = torch.ones((30, 5), dtype=torch.bool)
    item = TrainingItem(
        sample_id="sample",
        waveform=torch.zeros(320),
        assistant_speaking=torch.zeros(30),
        targets=FrameTargets(
            yield_probability=torch.zeros(30),
            primary_weight=torch.ones(30),
            primary_mask=torch.ones(30, dtype=torch.bool),
            event_targets=torch.zeros((30, 5)),
            event_mask=event_mask,
            future_activity=torch.zeros((30, 4)),
            future_activity_mask=torch.ones((30, 4), dtype=torch.bool),
        ),
    )
    source_sample = _sample(
        [-1.0] * FRAMES_PER_SAMPLE,
        [-1.0] * FRAMES_PER_SAMPLE,
        [0.0] * FRAMES_PER_SAMPLE,
        [0.0] * FRAMES_PER_SAMPLE,
    )
    boundary = build_completion_boundaries(
        (
            source_sample.model_copy(
                update={
                    "turn_completion": tuple(
                        0.1 if index == 7 else -1.0 for index in range(FRAMES_PER_SAMPLE)
                    ),
                    "continuation_pause": tuple(
                        0.9 if index == 7 else -1.0 for index in range(FRAMES_PER_SAMPLE)
                    ),
                }
            ),
        ),
        TurnCompletionObjectiveConfig(),
    )[0]

    selected = CompletionBoundaryDataset(_ItemDataset(item), (boundary,))[0]

    assert selected.sample_id == "sample:7"
    assert torch.count_nonzero(selected.targets.event_mask[:, 0]) == 1
    assert selected.targets.event_mask[7, 0]
    assert torch.all(selected.targets.event_mask[:, 1:])


def _sample(
    completion: list[float],
    continuation: list[float],
    user_floor: list[float],
    assistant_floor: list[float],
) -> MaterializedTrainingSample:
    zeros = (0.0,) * FRAMES_PER_SAMPLE
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
        split=TrainingCorpusSplit.TRAIN,
        user_audio_path="speaker-1.flac",
        assistant_audio_path="speaker-2.flac",
        start_seconds=0.0,
        end_seconds=20.0,
        quality_score=1.0,
        category="turn_shift",
        assistant_has_floor=tuple(assistant_floor),
        p_user_has_floor=tuple(user_floor),
        p_user_yield=zeros,
        p_assistant_backchannel=zeros,
        future_activity_0_200=zeros,
        future_activity_200_500=zeros,
        future_activity_500_1000=zeros,
        future_activity_1000_1500=zeros,
        turn_completion=tuple(completion),
        continuation_pause=tuple(continuation),
        non_floor_feedback=zeros,
        floor_take=zeros,
    )

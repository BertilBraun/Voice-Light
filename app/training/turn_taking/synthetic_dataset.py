from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from app.local.synthetic_generation.conversation_compiler import (
    AssistantStateAnchor,
    BackchannelAnchor,
    CompiledConversation,
    ConversationAnchor,
    ConversationCompilerConfig,
    ConversationFrameTracks,
    EotAnchor,
    HoldAnchor,
    InterruptionAnchor,
    ResponseAnchor,
    UserStateAnchor,
    compile_anchored_crop,
    conversation_anchors,
    render_crop_audio,
)
from app.local.synthetic_generation.conversation_pipeline import (
    SyntheticConversationCorpusManifest,
)
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.config import SyntheticAnchorKind, SyntheticAnchorSamplingConfig
from app.training.turn_taking.data import FrameTargets, TrainingItem

MASKED_TARGET = -1.0


@dataclass(frozen=True)
class SyntheticAnchorIndex:
    conversation_index: int
    anchor: ConversationAnchor


@dataclass(frozen=True)
class AnchoredSyntheticTrainingItem:
    item: TrainingItem
    anchor: ConversationAnchor
    anchor_frame_index: int


class AnchoredSyntheticTurnTakingDataset(Dataset[TrainingItem]):
    def __init__(
        self,
        root: Path,
        split: TrainingCorpusSplit,
        compiler_config: ConversationCompilerConfig,
        sampling_config: SyntheticAnchorSamplingConfig,
        augmenter: Callable[[Tensor, random.Random], Tensor] | None,
        random_seed: int,
        randomize: bool,
        completion_only: bool = False,
    ) -> None:
        manifest_path = root / "synthetic-corpus.json"
        if not manifest_path.is_file():
            raise ValueError(f"Dynamic synthetic corpus has no synthetic-corpus.json: {root}")
        manifest = SyntheticConversationCorpusManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        references = tuple(
            reference for reference in manifest.conversations if reference.split is split
        )
        self.conversations = tuple(
            CompiledConversation.model_validate_json(
                (root / reference.manifest_path).read_text(encoding="utf-8")
            )
            for reference in references
        )
        self.anchors = tuple(
            SyntheticAnchorIndex(conversation_index=conversation_index, anchor=anchor)
            for conversation_index, conversation in enumerate(self.conversations)
            for anchor in conversation_anchors(conversation)
            if not completion_only or _is_completion_anchor(anchor)
        )
        if not self.anchors:
            raise ValueError(f"Dynamic synthetic corpus has no eligible {split.value} anchors.")
        self.compiler_config = compiler_config
        self.sampling_config = sampling_config
        self.augmenter = augmenter
        self.random_seed = random_seed
        self.randomize = randomize
        self.worker_seed: int | None = None
        self.generator = random.Random()

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> TrainingItem:
        return self.anchored_item(index).item

    def anchored_item(self, index: int) -> AnchoredSyntheticTrainingItem:
        entry = self.anchors[index]
        crop_seed = (
            self._worker_generator().randrange(2**31)
            if self.randomize
            else _stable_anchor_seed(self.random_seed, entry)
        )
        anchored = compile_anchored_crop(
            conversation=self.conversations[entry.conversation_index],
            anchor=entry.anchor,
            random_seed=crop_seed,
            config=self.compiler_config,
        )
        waveform = torch.from_numpy(
            render_crop_audio(self.conversations[entry.conversation_index], anchored.crop).copy()
        )
        if self.augmenter is not None:
            waveform = self.augmenter(waveform, self._worker_generator())
        labels = anchored.crop.labels
        event_targets, event_mask = _event_targets(labels)
        event_mask[:, 0] = False
        if _is_completion_anchor(entry.anchor):
            event_mask[anchored.anchor_frame_index, 0] = True
        user_floor = torch.tensor(labels.p_user_floor_now, dtype=torch.float32)
        primary_mask = user_floor >= 0.0
        future_activity = torch.zeros((len(user_floor), 4), dtype=torch.float32)
        return AnchoredSyntheticTrainingItem(
            item=TrainingItem(
                sample_id=(
                    f"{self.conversations[entry.conversation_index].plan.conversation_id}:"
                    f"{entry.anchor.kind}:{crop_seed}"
                ),
                waveform=waveform,
                assistant_speaking=torch.tensor(
                    labels.assistant_speaking_probability, dtype=torch.float32
                ),
                targets=FrameTargets(
                    yield_probability=(1.0 - user_floor).masked_fill(~primary_mask, 0.0),
                    primary_weight=primary_mask.float(),
                    primary_mask=primary_mask,
                    event_targets=event_targets,
                    event_mask=event_mask,
                    future_activity=future_activity,
                    future_activity_mask=torch.zeros_like(future_activity, dtype=torch.bool),
                ),
            ),
            anchor=entry.anchor,
            anchor_frame_index=anchored.anchor_frame_index,
        )

    def sampling_weights(self) -> Tensor:
        counts = {
            kind: sum(_anchor_kind(entry.anchor) is kind for entry in self.anchors)
            for kind in SyntheticAnchorKind
        }
        missing = tuple(kind for kind, count in counts.items() if count == 0)
        if missing:
            raise ValueError(f"Synthetic corpus is missing configured anchor kinds: {missing}")
        return torch.tensor(
            [
                self.sampling_config.fraction(_anchor_kind(entry.anchor))
                / counts[_anchor_kind(entry.anchor)]
                for entry in self.anchors
            ],
            dtype=torch.double,
        )

    def _worker_generator(self) -> random.Random:
        worker_seed = torch.initial_seed() ^ self.random_seed
        if self.worker_seed != worker_seed:
            self.generator.seed(worker_seed)
            self.worker_seed = worker_seed
        return self.generator


class AnchoredSyntheticDatasetCollection(Dataset[TrainingItem]):
    def __init__(self, datasets: tuple[AnchoredSyntheticTurnTakingDataset, ...]) -> None:
        if not datasets:
            raise ValueError("Synthetic dataset collection requires at least one corpus.")
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index: int) -> TrainingItem:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise AssertionError("Synthetic dataset collection index resolution failed.")

    def sampling_weights(self) -> Tensor:
        corpus_weight = 1.0 / len(self.datasets)
        return torch.cat(
            tuple(dataset.sampling_weights() * corpus_weight for dataset in self.datasets)
        )


def _event_targets(labels: ConversationFrameTracks) -> tuple[Tensor, Tensor]:
    columns = (
        labels.turn_completion,
        labels.continuation_pause,
        tuple(MASKED_TARGET for _ in labels.turn_completion),
        labels.non_floor_feedback,
        labels.floor_take,
    )
    raw = torch.tensor(columns, dtype=torch.float32).transpose(0, 1)
    mask = raw >= 0.0
    return raw.masked_fill(~mask, 0.0), mask


def _anchor_kind(anchor: ConversationAnchor) -> SyntheticAnchorKind:
    match anchor:
        case EotAnchor():
            return SyntheticAnchorKind.EOT
        case HoldAnchor():
            return SyntheticAnchorKind.HOLD
        case BackchannelAnchor():
            return SyntheticAnchorKind.BACKCHANNEL
        case InterruptionAnchor():
            return SyntheticAnchorKind.INTERRUPTION
        case ResponseAnchor():
            return SyntheticAnchorKind.RESPONSE
        case AssistantStateAnchor():
            return SyntheticAnchorKind.ASSISTANT_STATE
        case UserStateAnchor():
            return SyntheticAnchorKind.USER_STATE


def _is_completion_anchor(anchor: ConversationAnchor) -> bool:
    match anchor:
        case EotAnchor() | HoldAnchor():
            return True
        case _:
            return False


def _stable_anchor_seed(random_seed: int, entry: SyntheticAnchorIndex) -> int:
    serialized = f"{random_seed}:{entry.conversation_index}:{entry.anchor.model_dump_json()}"
    return int.from_bytes(hashlib.sha256(serialized.encode()).digest()[:4], "big")

"""Frozen-ASR turn-taking adapter training."""

from app.training.turn_taking.config import TrainingConfig
from app.training.turn_taking.model import TurnTakingAdapter

__all__ = ["TrainingConfig", "TurnTakingAdapter"]

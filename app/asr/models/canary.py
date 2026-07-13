from __future__ import annotations

from pathlib import Path
from threading import Lock

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from app.asr.models.base import cuda_device, load_time_seconds, timestamped_word_from_word
from app.asr.models.parsing import (
    CANARY_IDENTIFIER,
    CANARY_REVISION,
    words_from_nemo_timestamps,
)
from app.asr.schemas import AsrModelId, TimestampedWord


class CanaryAsrModel:
    model_id = AsrModelId.CANARY
    package_names = ("torch", "nemo_toolkit")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.device = cuda_device()
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=CANARY_IDENTIFIER,
            revision=CANARY_REVISION,
        )
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: Path) -> tuple[TimestampedWord, ...]:
        with self.inference_lock:
            hypotheses: list[Hypothesis] = self.model.transcribe(
                [str(audio_path)],
                source_lang="en",
                target_lang="en",
                timestamps=True,
            )
        return tuple(
            timestamped_word_from_word(word)
            for word in words_from_nemo_timestamps(hypotheses[0].timestamp["word"])
        )

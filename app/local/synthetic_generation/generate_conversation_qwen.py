from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from app.local.synthetic_generation.conversation_prompts import EnglishConversationPromptSet
from app.local.synthetic_generation.conversation_tts import (
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    TtsBackendIdentity,
    render_conversation_user_audio,
)


class QwenVoiceDesignBatchSynthesizer:
    def __init__(
        self,
        model: Qwen3TTSModel,
        model_id: str,
        model_revision: str,
    ) -> None:
        self._model = model
        self._identity = TtsBackendIdentity(
            backend_id="qwen3_voice_design",
            model_id=model_id,
            model_revision=model_revision,
            runtime_version=importlib.metadata.version("qwen-tts"),
            model_license="Apache-2.0",
        )

    @property
    def identity(self) -> TtsBackendIdentity:
        return self._identity

    def generate_batch(
        self,
        requests: tuple[SpeechSynthesisRequest, ...],
    ) -> tuple[SpeechSynthesisResult, ...]:
        if not requests:
            raise ValueError("Qwen generation requires at least one request.")
        batch_seed = _batch_seed(requests)
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        started_at = time.monotonic()
        waveforms, sample_rate_hz = self._model.generate_voice_design(
            text=[request.text for request in requests],
            language=["English" for _ in requests],
            instruct=[request.voice_instruction for request in requests],
        )
        generation_seconds = time.monotonic() - started_at
        if len(waveforms) != len(requests):
            raise ValueError(
                f"Qwen returned {len(waveforms)} waveforms for {len(requests)} requests."
            )
        attributed_seconds = generation_seconds / len(requests)
        return tuple(
            SpeechSynthesisResult(
                clause_id=request.clause_id,
                samples=np.asarray(waveform, dtype=np.float32).reshape(-1),
                sample_rate_hz=int(sample_rate_hz),
                generation_seconds=attributed_seconds,
                batch_seed=batch_seed,
            )
            for request, waveform in zip(requests, waveforms, strict=True)
        )


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        parsed.prompts.read_text(encoding="utf-8")
    )
    model = Qwen3TTSModel.from_pretrained(
        parsed.model,
        revision=parsed.model_revision,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    synthesizer = QwenVoiceDesignBatchSynthesizer(
        model=model,
        model_id=parsed.model,
        model_revision=parsed.model_revision,
    )
    manifest = render_conversation_user_audio(
        prompt_set=prompt_set,
        prompt_set_path=parsed.prompts,
        output_directory=parsed.output,
        synthesizer=synthesizer,
        batch_size=parsed.batch_size,
    )
    print(manifest.model_dump_json(indent=2), flush=True)


def _batch_seed(requests: tuple[SpeechSynthesisRequest, ...]) -> int:
    content = ":".join(f"{request.clause_id}:{request.seed}" for request in requests)
    return int.from_bytes(hashlib.sha256(content.encode()).digest()[:4], "big")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render English conversation user units with Qwen VoiceDesign."
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


if __name__ == "__main__":
    main()

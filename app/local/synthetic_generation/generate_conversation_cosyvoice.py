from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import torch
from cosyvoice.cli.cosyvoice import AutoModel

from app.local.synthetic_generation.completion_dataset import DEFAULT_SILENCE_DETECTION
from app.local.synthetic_generation.conversation_prompts import EnglishConversationPromptSet
from app.local.synthetic_generation.conversation_tts import (
    SpeechSynthesisAttemptProvenance,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    VoiceClonePromptProvenance,
    cosyvoice3_reference_prompt,
    render_conversation_user_audio,
)
from app.local.synthetic_generation.conversation_voice_references import (
    ConversationVoiceReference,
    TtsBackendIdentity,
    load_voice_reference_manifest,
    trim_generated_speech,
)


class CosyVoiceChunk(TypedDict):
    tts_speech: torch.Tensor


class CosyVoiceConversationSynthesizer:
    def __init__(
        self,
        model: AutoModel,
        model_id: str,
        model_revision: str,
        runtime_revision: str,
    ) -> None:
        self._model = model
        self._identity = TtsBackendIdentity(
            backend_id="cosyvoice3_zero_shot_clone",
            model_id=model_id,
            model_revision=model_revision,
            runtime_version=f"CosyVoice repository {runtime_revision}",
            model_license="Apache-2.0",
        )

    @property
    def identity(self) -> TtsBackendIdentity:
        return self._identity

    def generate_batch(
        self,
        reference: ConversationVoiceReference,
        requests: tuple[SpeechSynthesisRequest, ...],
    ) -> tuple[SpeechSynthesisResult, ...]:
        if not requests:
            raise ValueError("CosyVoice generation requires at least one request.")
        clone_prompt = _clone_prompt_provenance(reference, self.identity)
        results = []
        for request in requests:
            samples, generation_seconds, batch_seed, attempts = self._generate_request(
                reference, request
            )
            results.append(
                SpeechSynthesisResult(
                    clause_id=request.clause_id,
                    samples=samples,
                    sample_rate_hz=int(self._model.sample_rate),
                    generation_seconds=generation_seconds,
                    batch_seed=batch_seed,
                    clone_prompt=clone_prompt,
                    attempts=attempts,
                )
            )
        return tuple(results)

    def _generate_request(
        self,
        reference: ConversationVoiceReference,
        request: SpeechSynthesisRequest,
    ) -> tuple[np.ndarray, float, int, tuple[SpeechSynthesisAttemptProvenance, ...]]:
        attempts = []
        total_generation_seconds = 0.0
        for attempt_index in range(1, 3):
            seed = (request.seed + attempt_index - 1) % (2**32)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            started_at = time.monotonic()
            chunks = tuple(
                cast(
                    Iterable[CosyVoiceChunk],
                    self._model.inference_zero_shot(
                        request.text,
                        cosyvoice3_reference_prompt(reference.reference_text),
                        str(reference.audio_path),
                        stream=False,
                        speed=request.speed,
                        text_frontend=True,
                    ),
                )
            )
            generation_seconds = time.monotonic() - started_at
            total_generation_seconds += generation_seconds
            if not chunks:
                raise ValueError(f"CosyVoice returned no audio for {request.clause_id}.")
            samples = np.concatenate(
                tuple(
                    chunk["tts_speech"].detach().cpu().to(torch.float32).numpy().reshape(-1)
                    for chunk in chunks
                )
            )
            outcome: Literal["accepted", "no_speech_like_energy"] = "accepted"
            try:
                trim_generated_speech(
                    samples,
                    int(self._model.sample_rate),
                    request.clause_id,
                    DEFAULT_SILENCE_DETECTION,
                )
            except ValueError as error:
                if "no speech-like energy" not in str(error):
                    raise
                outcome = "no_speech_like_energy"
            attempts.append(
                SpeechSynthesisAttemptProvenance(
                    attempt_index=attempt_index,
                    seed=seed,
                    generated_duration_seconds=samples.size / int(self._model.sample_rate),
                    generation_seconds=generation_seconds,
                    outcome=outcome,
                )
            )
            if outcome == "accepted":
                return samples, total_generation_seconds, seed, tuple(attempts)
        raise ValueError(
            f"CosyVoice returned no speech-like energy after two attempts for {request.clause_id}."
        )


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        parsed.prompts.read_text(encoding="utf-8")
    )
    reference_manifest = load_voice_reference_manifest(parsed.references)
    plan_ids = _shard_plan_ids(prompt_set, parsed.shard_index, parsed.shard_count)
    model = AutoModel(model_dir=str(parsed.model_directory))
    synthesizer = CosyVoiceConversationSynthesizer(
        model=model,
        model_id=parsed.model_id,
        model_revision=parsed.model_revision,
        runtime_revision=parsed.runtime_revision,
    )
    manifest = render_conversation_user_audio(
        prompt_set=prompt_set,
        prompt_set_path=parsed.prompts,
        reference_manifest=reference_manifest,
        reference_manifest_path=parsed.references,
        output_directory=parsed.output,
        synthesizer=synthesizer,
        batch_size=parsed.batch_size,
        plan_ids=plan_ids,
    )
    print(manifest.model_dump_json(indent=2), flush=True)


def _clone_prompt_provenance(
    reference: ConversationVoiceReference,
    backend: TtsBackendIdentity,
) -> VoiceClonePromptProvenance:
    prompt_content = (
        f"cosyvoice3:inference_zero_shot:{reference.audio_sha256}:{reference.reference_text_sha256}"
    )
    return VoiceClonePromptProvenance(
        plan_id=reference.plan_id,
        prompt_sha256=hashlib.sha256(prompt_content.encode()).hexdigest(),
        reference_audio_sha256=reference.audio_sha256,
        reference_text_sha256=reference.reference_text_sha256,
        backend=backend,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render English conversation user units with CosyVoice 3 zero-shot voices."
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def _shard_plan_ids(
    prompt_set: EnglishConversationPromptSet,
    shard_index: int,
    shard_count: int,
) -> frozenset[str]:
    if shard_count <= 0:
        raise ValueError("CosyVoice shard count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("CosyVoice shard index must be within the shard count.")
    plan_ids = frozenset(
        plan.plan_id
        for index, plan in enumerate(prompt_set.plans)
        if index % shard_count == shard_index
    )
    if not plan_ids:
        raise ValueError("CosyVoice shard assignment produced an empty shard.")
    return plan_ids


if __name__ == "__main__":
    main()

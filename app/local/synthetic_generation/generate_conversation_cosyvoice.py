from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from cosyvoice.cli.cosyvoice import AutoModel

from app.local.synthetic_generation.conversation_prompts import EnglishConversationPromptSet
from app.local.synthetic_generation.conversation_tts import (
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    VoiceClonePromptProvenance,
    render_conversation_user_audio,
)
from app.local.synthetic_generation.conversation_voice_references import (
    ConversationVoiceReference,
    TtsBackendIdentity,
    load_voice_reference_manifest,
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
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
            started_at = time.monotonic()
            chunks = tuple(
                cast(
                    Iterable[CosyVoiceChunk],
                    self._model.inference_instruct2(
                        request.text,
                        (
                            "You are a helpful conversational speaker. "
                            f"{request.delivery_instruction}<|endofprompt|>"
                        ),
                        str(reference.audio_path),
                        stream=False,
                        text_frontend=False,
                    ),
                )
            )
            generation_seconds = time.monotonic() - started_at
            if not chunks:
                raise ValueError(f"CosyVoice returned no audio for {request.clause_id}.")
            samples = np.concatenate(
                tuple(
                    chunk["tts_speech"].detach().cpu().to(torch.float32).numpy().reshape(-1)
                    for chunk in chunks
                )
            )
            results.append(
                SpeechSynthesisResult(
                    clause_id=request.clause_id,
                    samples=samples,
                    sample_rate_hz=int(self._model.sample_rate),
                    generation_seconds=generation_seconds,
                    batch_seed=request.seed,
                    clone_prompt=clone_prompt,
                )
            )
        return tuple(results)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        parsed.prompts.read_text(encoding="utf-8")
    )
    reference_manifest = load_voice_reference_manifest(parsed.references)
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
    )
    print(manifest.model_dump_json(indent=2), flush=True)


def _clone_prompt_provenance(
    reference: ConversationVoiceReference,
    backend: TtsBackendIdentity,
) -> VoiceClonePromptProvenance:
    prompt_content = (
        f"cosyvoice3:inference_instruct2:{reference.audio_sha256}:{reference.reference_text_sha256}"
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
    return parser


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

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


class QwenBaseVoiceCloneSynthesizer:
    def __init__(
        self,
        model: Qwen3TTSModel,
        model_id: str,
        model_revision: str,
    ) -> None:
        self._model = model
        self._identity = TtsBackendIdentity(
            backend_id="qwen3_base_icl_voice_clone",
            model_id=model_id,
            model_revision=model_revision,
            runtime_version=importlib.metadata.version("qwen-tts"),
            model_license="Apache-2.0",
        )
        self._voice_clone_prompts: dict[str, list[VoiceClonePromptItem]] = {}
        self._clone_prompt_provenance: dict[str, VoiceClonePromptProvenance] = {}

    @property
    def identity(self) -> TtsBackendIdentity:
        return self._identity

    def generate_batch(
        self,
        reference: ConversationVoiceReference,
        requests: tuple[SpeechSynthesisRequest, ...],
    ) -> tuple[SpeechSynthesisResult, ...]:
        if not requests:
            raise ValueError("Qwen generation requires at least one request.")
        batch_seed = _batch_seed(requests)
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        started_at = time.monotonic()
        voice_clone_prompt = self._voice_clone_prompts.get(reference.plan_id)
        if voice_clone_prompt is None:
            voice_clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=str(reference.audio_path),
                ref_text=reference.reference_text,
                x_vector_only_mode=False,
            )
            _validate_icl_prompt(voice_clone_prompt, reference)
            self._voice_clone_prompts[reference.plan_id] = voice_clone_prompt
            self._clone_prompt_provenance[reference.plan_id] = VoiceClonePromptProvenance(
                plan_id=reference.plan_id,
                prompt_sha256=_clone_prompt_sha256(voice_clone_prompt),
                reference_audio_sha256=reference.audio_sha256,
                reference_text_sha256=reference.reference_text_sha256,
                backend=self.identity,
            )
        waveforms, sample_rate_hz = self._model.generate_voice_clone(
            text=[request.text for request in requests],
            language=["English" for _ in requests],
            voice_clone_prompt=voice_clone_prompt,
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
                clone_prompt=self._clone_prompt_provenance[reference.plan_id],
            )
            for request, waveform in zip(requests, waveforms, strict=True)
        )


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        parsed.prompts.read_text(encoding="utf-8")
    )
    reference_manifest = load_voice_reference_manifest(parsed.references)
    model = Qwen3TTSModel.from_pretrained(
        parsed.model,
        revision=parsed.model_revision,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    synthesizer = QwenBaseVoiceCloneSynthesizer(
        model=model,
        model_id=parsed.model,
        model_revision=parsed.model_revision,
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


def _batch_seed(requests: tuple[SpeechSynthesisRequest, ...]) -> int:
    content = ":".join(f"{request.clause_id}:{request.seed}" for request in requests)
    return int.from_bytes(hashlib.sha256(content.encode()).digest()[:4], "big")


def _validate_icl_prompt(
    prompt_items: list[VoiceClonePromptItem],
    reference: ConversationVoiceReference,
) -> None:
    if len(prompt_items) != 1:
        raise ValueError(f"Qwen returned multiple clone prompts for {reference.plan_id}.")
    prompt = prompt_items[0]
    if prompt.x_vector_only_mode or not prompt.icl_mode:
        raise ValueError(f"Qwen did not create an ICL clone prompt for {reference.plan_id}.")
    if prompt.ref_text != reference.reference_text:
        raise ValueError(f"Qwen clone prompt changed reference text for {reference.plan_id}.")


def _clone_prompt_sha256(prompt_items: list[VoiceClonePromptItem]) -> str:
    digest = hashlib.sha256()
    for prompt in prompt_items:
        if prompt.ref_code is None:
            digest.update(b"ref-code:none\n")
        else:
            digest.update(_tensor_hash_content("ref-code", prompt.ref_code))
        digest.update(_tensor_hash_content("speaker-embedding", prompt.ref_spk_embedding))
        digest.update(f"x-vector-only:{prompt.x_vector_only_mode}\n".encode())
        digest.update(f"icl-mode:{prompt.icl_mode}\n".encode())
        digest.update(f"reference-text:{prompt.ref_text}\n".encode())
    return digest.hexdigest()


def _tensor_hash_content(
    name: str,
    tensor: torch.Tensor,
) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    header = f"{name}:{contiguous.dtype}:{tuple(contiguous.shape)}\n".encode()
    return header + contiguous.view(torch.uint8).numpy().tobytes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render English conversation user units with reusable Qwen Base ICL voices."
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


if __name__ == "__main__":
    main()

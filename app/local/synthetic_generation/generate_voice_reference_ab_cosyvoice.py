from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from cosyvoice.cli.cosyvoice import AutoModel

from app.local.synthetic_generation.conversation_voice_references import (
    TtsBackendIdentity,
    file_sha256,
    write_pcm16_wave,
)
from app.local.synthetic_generation.voice_reference_ab_pilot import (
    AuditionUtterance,
    RenderedAuditionUtterance,
    VoiceReferenceAbRenderManifest,
    audition_utterances,
    compose_candidate_conversation,
    load_reference_manifest,
)


class CosyVoiceChunk(TypedDict):
    tts_speech: torch.Tensor


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    references = load_reference_manifest(parsed.references)
    model = AutoModel(model_dir=str(parsed.model_directory))
    output_directory: Path = parsed.output
    audio_directory = output_directory / "audio"
    conversation_directory = output_directory / "conversations"
    audio_directory.mkdir(parents=True, exist_ok=True)
    rendered = []
    utterances = audition_utterances()
    total_count = len(references.candidates) * len(utterances)
    for candidate in references.candidates:
        reference_path = parsed.references.parent / candidate.audio_path
        for utterance in utterances:
            rendered.append(
                _render_utterance(
                    model=model,
                    candidate_id=candidate.candidate_id,
                    reference_path=reference_path,
                    utterance=utterance,
                    audio_directory=audio_directory,
                )
            )
            print(f"cosyvoice_units={len(rendered)}/{total_count}", flush=True)
    rendered_utterances = tuple(rendered)
    conversations = tuple(
        compose_candidate_conversation(
            candidate_id=candidate.candidate_id,
            rendered_utterances=tuple(
                rendered_unit
                for rendered_unit in rendered_utterances
                if rendered_unit.candidate_id == candidate.candidate_id
            ),
            output_path=conversation_directory / f"{candidate.candidate_id}.wav",
        ).model_copy(update={"audio_path": Path("conversations") / f"{candidate.candidate_id}.wav"})
        for candidate in references.candidates
    )
    relative_rendered = tuple(
        rendered_unit.model_copy(
            update={"audio_path": rendered_unit.audio_path.relative_to(output_directory)}
        )
        for rendered_unit in rendered_utterances
    )
    manifest = VoiceReferenceAbRenderManifest(
        references_sha256=file_sha256(parsed.references),
        backend=TtsBackendIdentity(
            backend_id="cosyvoice3_instruct2_clone",
            model_id=parsed.model_id,
            model_revision=parsed.model_revision,
            runtime_version=(f"CosyVoice repository {parsed.runtime_revision}"),
            model_license="Apache-2.0",
        ),
        utterances=utterances,
        rendered_utterances=relative_rendered,
        conversations=conversations,
    )
    manifest_path = output_directory / "render.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(manifest_path, flush=True)


def _render_utterance(
    model: AutoModel,
    candidate_id: str,
    reference_path: Path,
    utterance: AuditionUtterance,
    audio_directory: Path,
) -> RenderedAuditionUtterance:
    started_at = time.monotonic()
    chunks: tuple[CosyVoiceChunk, ...] = tuple(
        model.inference_instruct2(
            utterance.text,
            f"You are speaking English. {utterance.delivery_instruction}<|endofprompt|>",
            str(reference_path),
            stream=False,
            text_frontend=False,
        )
    )
    generation_seconds = time.monotonic() - started_at
    if not chunks:
        raise ValueError(
            f"CosyVoice returned no audio for {candidate_id}/{utterance.utterance_id}."
        )
    samples = np.concatenate(
        tuple(
            chunk["tts_speech"].detach().cpu().to(torch.float32).numpy().reshape(-1)
            for chunk in chunks
        )
    )
    sample_rate_hz = int(model.sample_rate)
    audio_path = audio_directory / f"{candidate_id}_{utterance.utterance_id}.wav"
    write_pcm16_wave(audio_path, samples, sample_rate_hz)
    return RenderedAuditionUtterance(
        candidate_id=candidate_id,
        utterance=utterance,
        audio_path=audio_path,
        audio_sha256=file_sha256(audio_path),
        sample_rate_hz=sample_rate_hz,
        duration_seconds=samples.size / sample_rate_hz,
        generation_seconds=generation_seconds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone one identical brisk conversation from every Qwen reference candidate."
    )
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    return parser


if __name__ == "__main__":
    main()

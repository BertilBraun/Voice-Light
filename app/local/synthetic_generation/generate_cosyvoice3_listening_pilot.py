from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from cosyvoice.cli.cosyvoice import AutoModel

from app.local.synthetic_generation.provider_listening_pilot import (
    ListeningProvider,
    ProviderListeningManifest,
    RenderedListeningUtterance,
    VoiceKey,
    cloning_provider_listening_plan,
    compose_listening_conversations,
    file_sha256,
    write_pcm16_wave,
    write_provider_listening_manifest,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    output_directory: Path = parsed.output
    audio_directory = output_directory / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    plan = cloning_provider_listening_plan(ListeningProvider.COSYVOICE3)
    references = {
        VoiceKey.REFERENCE_ONE: parsed.reference_one,
        VoiceKey.REFERENCE_TWO: parsed.reference_two,
    }
    model = AutoModel(model_dir=str(parsed.model_directory))
    rendered = []
    for utterance in plan.utterances:
        started_at = time.monotonic()
        chunks = tuple(
            model.inference_instruct2(
                utterance.text,
                f"You are a helpful assistant. {utterance.delivery_instruction}<|endofprompt|>",
                str(references[utterance.voice]),
                stream=False,
            )
        )
        generation_seconds = time.monotonic() - started_at
        if not chunks:
            raise ValueError(f"CosyVoice returned no audio for {utterance.utterance_id}.")
        samples = np.concatenate(
            [
                chunk["tts_speech"].detach().cpu().to(torch.float32).numpy().reshape(-1)
                for chunk in chunks
            ]
        )
        audio_path = audio_directory / f"{utterance.utterance_id}.wav"
        write_pcm16_wave(audio_path, samples, int(model.sample_rate))
        duration_seconds = samples.size / int(model.sample_rate)
        rendered.append(
            RenderedListeningUtterance(
                utterance=utterance,
                speaker_label=utterance.voice.value,
                audio_path=audio_path,
                audio_sha256=file_sha256(audio_path),
                sample_rate_hz=int(model.sample_rate),
                duration_seconds=duration_seconds,
                generation_seconds=generation_seconds,
                real_time_factor=generation_seconds / duration_seconds,
            )
        )
    rendered_utterances = tuple(rendered)
    rendered_conversations = compose_listening_conversations(
        plan,
        rendered_utterances,
        output_directory / "conversations",
    )
    manifest = ProviderListeningManifest(
        plan=plan,
        model_id=parsed.model_id,
        model_revision=parsed.model_revision,
        runtime=f"CosyVoice repository {parsed.runtime_revision}",
        rendered_utterances=rendered_utterances,
        rendered_conversations=rendered_conversations,
    )
    manifest_path = write_provider_listening_manifest(manifest, output_directory)
    print(manifest_path, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the CosyVoice 3 English audition.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--reference-one", required=True, type=Path)
    parser.add_argument("--reference-two", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()

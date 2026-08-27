from __future__ import annotations

import argparse
import importlib.metadata
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from app.local.synthetic_generation.provider_listening_pilot import (
    ProviderListeningManifest,
    RenderedListeningUtterance,
    VoiceKey,
    file_sha256,
    qwen_custom_listening_plan,
    write_pcm16_wave,
    write_provider_listening_manifest,
)

QWEN_SPEAKERS = {
    VoiceKey.QWEN_RYAN: "Ryan",
    VoiceKey.QWEN_AIDEN: "Aiden",
    VoiceKey.QWEN_VIVIAN: "Vivian",
    VoiceKey.QWEN_SOHEE: "Sohee",
}


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    output_directory: Path = parsed.output
    audio_directory = output_directory / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    plan = qwen_custom_listening_plan()
    model = Qwen3TTSModel.from_pretrained(
        parsed.model,
        revision=parsed.model_revision,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    torch.manual_seed(parsed.seed)
    torch.cuda.manual_seed_all(parsed.seed)
    started_at = time.monotonic()
    waveforms, sample_rate_hz = model.generate_custom_voice(
        text=[utterance.text for utterance in plan.utterances],
        language=["English" for _ in plan.utterances],
        speaker=[QWEN_SPEAKERS[utterance.voice] for utterance in plan.utterances],
        instruct=[utterance.delivery_instruction for utterance in plan.utterances],
    )
    generation_seconds = time.monotonic() - started_at
    if len(waveforms) != len(plan.utterances):
        raise ValueError("Qwen returned the wrong number of listening samples.")
    attributed_generation_seconds = generation_seconds / len(plan.utterances)
    rendered = []
    for utterance, waveform in zip(plan.utterances, waveforms, strict=True):
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        audio_path = audio_directory / f"{utterance.utterance_id}.wav"
        write_pcm16_wave(audio_path, samples, int(sample_rate_hz))
        duration_seconds = samples.size / int(sample_rate_hz)
        rendered.append(
            RenderedListeningUtterance(
                utterance=utterance,
                speaker_label=QWEN_SPEAKERS[utterance.voice],
                audio_path=audio_path,
                audio_sha256=file_sha256(audio_path),
                sample_rate_hz=int(sample_rate_hz),
                duration_seconds=duration_seconds,
                generation_seconds=attributed_generation_seconds,
                real_time_factor=attributed_generation_seconds / duration_seconds,
            )
        )
    manifest = ProviderListeningManifest(
        plan=plan,
        model_id=parsed.model,
        model_revision=parsed.model_revision,
        runtime=f"qwen-tts {importlib.metadata.version('qwen-tts')}",
        rendered_utterances=tuple(rendered),
        rendered_conversations=(),
    )
    manifest_path = write_provider_listening_manifest(manifest, output_directory)
    print(manifest_path, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Qwen CustomVoice English audition.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


if __name__ == "__main__":
    main()

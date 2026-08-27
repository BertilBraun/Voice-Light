from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
from indextts.infer_v2_5 import IndexTTS2

from app.local.synthetic_generation.provider_listening_pilot import (
    ListeningProvider,
    ListeningPurpose,
    ListeningUtterance,
    ProviderListeningManifest,
    RenderedListeningUtterance,
    VoiceKey,
    cloning_provider_listening_plan,
    compose_listening_conversations,
    file_sha256,
    write_pcm16_wave,
    write_provider_listening_manifest,
)


@dataclass(frozen=True)
class IndexGenerationControl:
    emotion_vector: tuple[float, float, float, float, float, float, float, float]
    duration_factor: float


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    output_directory: Path = parsed.output
    audio_directory = output_directory / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    plan = cloning_provider_listening_plan(ListeningProvider.INDEXTTS25)
    references = {
        VoiceKey.REFERENCE_ONE: parsed.reference_one,
        VoiceKey.REFERENCE_TWO: parsed.reference_two,
    }
    model = IndexTTS2(
        cfg_path=str(parsed.configuration),
        model_dir=str(parsed.model_directory),
        use_bf16=True,
        use_qwen_emo=False,
    )
    rendered = []
    for utterance in plan.utterances:
        control = _generation_control(utterance)
        audio_path = audio_directory / f"{utterance.utterance_id}.wav"
        started_at = time.monotonic()
        model.infer(
            spk_audio_prompt=str(references[utterance.voice]),
            text=utterance.text,
            lang="EN",
            output_path=str(audio_path),
            emo_vector=list(control.emotion_vector),
            emo_alpha=0.6,
            use_random=False,
            duration_factor=control.duration_factor,
            verbose=False,
        )
        generation_seconds = time.monotonic() - started_at
        samples, sample_rate_hz = soundfile.read(audio_path, dtype="float32", always_2d=False)
        normalized_samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        write_pcm16_wave(audio_path, normalized_samples, int(sample_rate_hz))
        duration_seconds = normalized_samples.size / int(sample_rate_hz)
        rendered.append(
            RenderedListeningUtterance(
                utterance=utterance,
                speaker_label=utterance.voice.value,
                audio_path=audio_path,
                audio_sha256=file_sha256(audio_path),
                sample_rate_hz=int(sample_rate_hz),
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
        runtime=f"IndexTTS repository {parsed.runtime_revision}",
        rendered_utterances=rendered_utterances,
        rendered_conversations=rendered_conversations,
    )
    manifest_path = write_provider_listening_manifest(manifest, output_directory)
    print(manifest_path, flush=True)


def _generation_control(utterance: ListeningUtterance) -> IndexGenerationControl:
    match utterance.purpose:
        case ListeningPurpose.BACKCHANNEL:
            return IndexGenerationControl(
                emotion_vector=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
                duration_factor=0.78,
            )
        case ListeningPurpose.LONG_TURN:
            return IndexGenerationControl(
                emotion_vector=(0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2),
                duration_factor=0.9,
            )
        case ListeningPurpose.SHORT_TURN | ListeningPurpose.NORMAL_TURN:
            return IndexGenerationControl(
                emotion_vector=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4),
                duration_factor=1.0,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the IndexTTS 2.5 English audition.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--reference-one", required=True, type=Path)
    parser.add_argument("--reference-two", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()

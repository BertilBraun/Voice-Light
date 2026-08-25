from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import repeat
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pydantic import Field
from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import SpeechGenerator
from voxtream.utils.generator import set_seed, text_generator

from app.local.synthetic_generation.completion_dataset import (
    GeneratedUtteranceAnnotation,
    PromptLanguage,
    Voxtream2Provenance,
    analyze_generated_samples,
    completion_window_plans,
)
from app.local.synthetic_generation.completion_prompts import SyntheticSpeechPromptSet
from app.local.synthetic_generation.models import SyntheticModel

VOXTREAM_RUNTIME_VERSION = importlib.metadata.version("voxtream")


class VoxtreamCompletionRunManifest(SyntheticModel):
    schema_version: str = "voice-light-voxtream-completion-run-v1"
    started_at: datetime
    finished_at: datetime
    prompt_set_path: Path
    output_directory: Path
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str
    runtime_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    requested_run_seconds: float = Field(gt=0.0)
    variants_per_prompt: int = Field(gt=0)
    reference_voice_count: int = Field(gt=0)
    utterance_count: int = Field(ge=0)
    hold_boundary_count: int = Field(ge=0)
    generated_audio_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    aggregate_real_time_factor: float | None = Field(default=None, gt=0.0)
    peak_allocated_vram_bytes: int = Field(ge=0)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    prompt_set = SyntheticSpeechPromptSet.model_validate_json(
        parsed.prompts.read_text(encoding="utf-8")
    )
    prompts = tuple(
        prompt for prompt in prompt_set.prompts if prompt.language is PromptLanguage.ENGLISH
    )
    if not prompts:
        raise ValueError("VoXtream requires at least one English prompt.")
    reference_audio_paths = tuple(sorted(parsed.reference_voices.glob("*.wav")))
    if not reference_audio_paths:
        raise ValueError("VoXtream requires at least one reference voice WAV.")
    configuration = SpeechGeneratorConfig(
        **json.loads(parsed.configuration.read_text(encoding="utf-8"))
    )
    configuration.cache_prompt = False
    set_seed(parsed.seed)
    model = SpeechGenerator(
        configuration,
        compile=parsed.compile,
        cache_prompt_in_memory=False,
    )
    parsed.output.mkdir(parents=True, exist_ok=True)
    utterance_manifest_path = parsed.output / "utterances.jsonl"
    window_manifest_path = parsed.output / "windows.jsonl"
    completed_utterance_ids = _completed_utterance_ids(utterance_manifest_path)
    jobs = tuple(
        (prompt, variant_index)
        for variant_index in range(parsed.variants_per_prompt)
        for prompt in prompts
        if _utterance_id(prompt.prompt_id, variant_index) not in completed_utterance_ids
    )
    torch.cuda.reset_peak_memory_stats()
    started_at = datetime.now(UTC)
    deadline = time.monotonic() + parsed.run_seconds
    for prompt, variant_index in jobs:
        if time.monotonic() >= deadline:
            break
        utterance_id = _utterance_id(prompt.prompt_id, variant_index)
        variant_seed = prompt.seed + variant_index * 1_000_003
        generator = random.Random(variant_seed)
        reference_audio_path = reference_audio_paths[
            generator.randrange(len(reference_audio_paths))
        ]
        speaking_rate = generator.uniform(2.4, 5.2)
        set_seed(variant_seed)
        generation_started = time.monotonic()
        stream = model.generate_stream(
            prompt_audio_path=reference_audio_path,
            text=text_generator(prompt.text),
            speaking_rate=repeat(speaking_rate),
        )
        frames = [np.asarray(frame, dtype=np.float32).reshape(-1) for frame, _ in stream]
        generation_seconds = time.monotonic() - generation_started
        if not frames:
            raise ValueError(f"VoXtream returned no audio for {utterance_id}.")
        samples = np.concatenate(frames)
        raw_duration_seconds = samples.size / configuration.mimi_sr
        audio_relative_path = Path("audio") / f"{utterance_id}.wav"
        provenance = Voxtream2Provenance(
            model_id=parsed.model,
            model_revision=parsed.model_revision,
            runtime_version=VOXTREAM_RUNTIME_VERSION,
            runtime_revision=parsed.runtime_revision,
            model_license="CC-BY-4.0",
            generation_seconds=generation_seconds,
            real_time_factor=generation_seconds / raw_duration_seconds,
            batch_size=1,
            configuration_sha256=_sha256(parsed.configuration),
            reference_audio_path=reference_audio_path,
            reference_audio_sha256=_sha256(reference_audio_path),
            speaking_rate_syllables_per_second=speaking_rate,
        )
        trimmed, annotation = analyze_generated_samples(
            samples=samples,
            sample_rate_hz=configuration.mimi_sr,
            utterance_id=utterance_id,
            prompt=prompt,
            audio_path=audio_relative_path,
            provenance=provenance,
        )
        audio_path = parsed.output / audio_relative_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, trimmed, configuration.mimi_sr, subtype="PCM_16")
        _append_line(utterance_manifest_path, annotation.model_dump_json())
        for window in completion_window_plans(annotation, seed=variant_seed):
            _append_line(window_manifest_path, window.model_dump_json())
        print(
            f"generated={utterance_id} duration={annotation.trimmed_duration_seconds:.2f}s "
            f"rtf={provenance.real_time_factor:.3f} "
            f"holds={len(annotation.boundaries) - 1}",
            flush=True,
        )
    annotations = _read_annotations(utterance_manifest_path)
    total_audio_seconds = sum(item.trimmed_duration_seconds for item in annotations)
    total_generation_seconds = sum(item.provenance.generation_seconds for item in annotations)
    manifest = VoxtreamCompletionRunManifest(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        prompt_set_path=parsed.prompts,
        output_directory=parsed.output,
        model_id=parsed.model,
        model_revision=parsed.model_revision,
        runtime_version=VOXTREAM_RUNTIME_VERSION,
        runtime_revision=parsed.runtime_revision,
        requested_run_seconds=parsed.run_seconds,
        variants_per_prompt=parsed.variants_per_prompt,
        reference_voice_count=len(reference_audio_paths),
        utterance_count=len(annotations),
        hold_boundary_count=sum(len(item.boundaries) - 1 for item in annotations),
        generated_audio_seconds=total_audio_seconds,
        generation_seconds=total_generation_seconds,
        aggregate_real_time_factor=(
            total_generation_seconds / total_audio_seconds if total_audio_seconds > 0.0 else None
        ),
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
    )
    (parsed.output / "run.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(manifest.model_dump_json(indent=2), flush=True)


def _utterance_id(prompt_id: str, variant_index: int) -> str:
    return f"{prompt_id}_variant_{variant_index:02d}"


def _completed_utterance_ids(path: Path) -> set[str]:
    return {annotation.utterance_id for annotation in _read_annotations(path)}


def _read_annotations(path: Path) -> tuple[GeneratedUtteranceAnnotation, ...]:
    if not path.exists():
        return ()
    return tuple(
        GeneratedUtteranceAnnotation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _append_line(path: Path, content: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(f"{content}\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and label a time-bounded VoXtream2 completion corpus."
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--reference-voices", required=True, type=Path)
    parser.add_argument("--model", default="herimor/voxtream2")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--run-seconds", type=float, default=7200.0)
    parser.add_argument("--variants-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main()

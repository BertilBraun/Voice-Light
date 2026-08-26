from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import snapshot_download
from pydantic import Field

from app.local.synthetic_generation.completion_dataset import (
    ChatterboxMultilingualProvenance,
    GeneratedUtteranceAnnotation,
    analyze_generated_samples,
    completion_window_plans,
)
from app.local.synthetic_generation.completion_prompts import SyntheticSpeechPromptSet
from app.local.synthetic_generation.models import SyntheticModel

CHATTERBOX_RUNTIME_VERSION = importlib.metadata.version("chatterbox-tts")
CHATTERBOX_T3_FILENAME = "t3_mtl23ls_v3.safetensors"


class ChatterboxCompletionRunManifest(SyntheticModel):
    schema_version: str = "voice-light-chatterbox-completion-run-v1"
    started_at: datetime
    finished_at: datetime
    prompt_set_path: Path
    output_directory: Path
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str
    runtime_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    requested_run_seconds: float = Field(gt=0.0)
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
    reference_audio_paths = tuple(sorted(parsed.reference_voices.glob("*.wav")))
    if not reference_audio_paths:
        raise ValueError("Chatterbox requires at least one reference voice WAV.")
    checkpoint_directory = Path(
        snapshot_download(
            repo_id=parsed.model,
            revision=parsed.model_revision,
            allow_patterns=(
                "ve.pt",
                CHATTERBOX_T3_FILENAME,
                "s3gen.pt",
                "grapheme_mtl_merged_expanded_v1.json",
                "conds.pt",
                "Cangjie5_TC.json",
            ),
        )
    )
    model = ChatterboxMultilingualTTS.from_local(
        checkpoint_directory,
        device="cuda",
        t3_model=CHATTERBOX_T3_FILENAME,
    )
    parsed.output.mkdir(parents=True, exist_ok=True)
    utterance_manifest_path = parsed.output / "utterances.jsonl"
    window_manifest_path = parsed.output / "windows.jsonl"
    completed_prompt_ids = _completed_prompt_ids(utterance_manifest_path)
    remaining_prompts = tuple(
        prompt for prompt in prompt_set.prompts if prompt.prompt_id not in completed_prompt_ids
    )
    torch.cuda.reset_peak_memory_stats()
    started_at = datetime.now(UTC)
    deadline = time.monotonic() + parsed.run_seconds
    for prompt in remaining_prompts:
        if time.monotonic() >= deadline:
            break
        reference_audio_path = reference_audio_paths[prompt.seed % len(reference_audio_paths)]
        generator = random.Random(prompt.seed)
        exaggeration = generator.uniform(0.35, 0.75)
        guidance_weight = generator.uniform(0.3, 0.6)
        temperature = generator.uniform(0.65, 0.9)
        generation_started = time.monotonic()
        waveform = model.generate(
            prompt.text,
            language_id="en",
            audio_prompt_path=str(reference_audio_path),
            exaggeration=exaggeration,
            cfg_weight=guidance_weight,
            temperature=temperature,
        )
        generation_seconds = time.monotonic() - generation_started
        samples = np.asarray(waveform.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        raw_duration_seconds = samples.size / model.sr
        audio_relative_path = Path("audio") / f"{prompt.prompt_id}.wav"
        provenance = ChatterboxMultilingualProvenance(
            model_id=parsed.model,
            model_revision=parsed.model_revision,
            runtime_version=CHATTERBOX_RUNTIME_VERSION,
            runtime_revision=parsed.runtime_revision,
            model_license="MIT",
            generation_seconds=generation_seconds,
            real_time_factor=generation_seconds / raw_duration_seconds,
            batch_size=1,
            reference_audio_path=reference_audio_path,
            reference_audio_sha256=_sha256(reference_audio_path),
            exaggeration=exaggeration,
            classifier_free_guidance_weight=guidance_weight,
            temperature=temperature,
        )
        trimmed, annotation = analyze_generated_samples(
            samples=samples,
            sample_rate_hz=model.sr,
            utterance_id=prompt.prompt_id,
            prompt=prompt,
            audio_path=audio_relative_path,
            provenance=provenance,
        )
        audio_path = parsed.output / audio_relative_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, trimmed, model.sr, subtype="PCM_16")
        _append_line(utterance_manifest_path, annotation.model_dump_json())
        for window in completion_window_plans(annotation, seed=prompt.seed):
            _append_line(window_manifest_path, window.model_dump_json())
        print(
            f"generated={prompt.prompt_id} duration={annotation.trimmed_duration_seconds:.2f}s "
            f"rtf={provenance.real_time_factor:.3f} "
            f"holds={len(annotation.boundaries) - 1}",
            flush=True,
        )
    annotations = _read_annotations(utterance_manifest_path)
    total_audio_seconds = sum(item.trimmed_duration_seconds for item in annotations)
    total_generation_seconds = sum(item.provenance.generation_seconds for item in annotations)
    manifest = ChatterboxCompletionRunManifest(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        prompt_set_path=parsed.prompts,
        output_directory=parsed.output,
        model_id=parsed.model,
        model_revision=parsed.model_revision,
        runtime_version=CHATTERBOX_RUNTIME_VERSION,
        runtime_revision=parsed.runtime_revision,
        requested_run_seconds=parsed.run_seconds,
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


def _completed_prompt_ids(path: Path) -> set[str]:
    return {annotation.prompt.prompt_id for annotation in _read_annotations(path)}


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
        description="Generate and label a time-bounded Chatterbox V3 completion corpus."
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-voices", required=True, type=Path)
    parser.add_argument("--model", default="ResembleAI/chatterbox")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--run-seconds", type=float, default=7200.0)
    return parser


if __name__ == "__main__":
    main()

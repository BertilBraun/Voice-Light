from __future__ import annotations

import argparse
import importlib.metadata
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pydantic import Field
from qwen_tts import Qwen3TTSModel

from app.local.synthetic_generation.completion_dataset import (
    GeneratedUtteranceAnnotation,
    TtsGenerationProvenance,
    TtsProvider,
    analyze_generated_samples,
    completion_window_plans,
)
from app.local.synthetic_generation.completion_prompts import SyntheticSpeechPromptSet
from app.local.synthetic_generation.models import SyntheticModel

QWEN_TTS_RUNTIME_VERSION = importlib.metadata.version("qwen-tts")


class QwenCompletionRunManifest(SyntheticModel):
    schema_version: str = "voice-light-qwen-completion-run-v1"
    started_at: datetime
    finished_at: datetime
    prompt_set_path: Path
    output_directory: Path
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str
    requested_run_seconds: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
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
    parsed.output.mkdir(parents=True, exist_ok=True)
    utterance_manifest_path = parsed.output / "utterances.jsonl"
    window_manifest_path = parsed.output / "windows.jsonl"
    completed_prompt_ids = _completed_prompt_ids(utterance_manifest_path)
    remaining_prompts = tuple(
        prompt for prompt in prompt_set.prompts if prompt.prompt_id not in completed_prompt_ids
    )
    model = Qwen3TTSModel.from_pretrained(
        parsed.model,
        revision=parsed.model_revision,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    torch.cuda.reset_peak_memory_stats()
    started_at = datetime.now(UTC)
    deadline = time.monotonic() + parsed.run_seconds
    for offset in range(0, len(remaining_prompts), parsed.batch_size):
        if time.monotonic() >= deadline:
            break
        batch = remaining_prompts[offset : offset + parsed.batch_size]
        generation_started = time.monotonic()
        waveforms, sample_rate_hz = model.generate_voice_design(
            text=[prompt.text for prompt in batch],
            language=[prompt.language.value for prompt in batch],
            instruct=[prompt.voice_instruction for prompt in batch],
        )
        batch_generation_seconds = time.monotonic() - generation_started
        if len(waveforms) != len(batch):
            raise ValueError(
                f"Qwen returned {len(waveforms)} waveforms for a batch of {len(batch)}."
            )
        for prompt, waveform in zip(batch, waveforms, strict=True):
            samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
            raw_duration_seconds = samples.size / sample_rate_hz
            attributed_generation_seconds = batch_generation_seconds / len(batch)
            audio_relative_path = Path("audio") / f"{prompt.prompt_id}.wav"
            audio_path = parsed.output / audio_relative_path
            provenance = TtsGenerationProvenance(
                provider=TtsProvider.QWEN3_VOICE_DESIGN,
                model_id=parsed.model,
                model_revision=parsed.model_revision,
                runtime_version=QWEN_TTS_RUNTIME_VERSION,
                model_license="Apache-2.0",
                generation_seconds=attributed_generation_seconds,
                real_time_factor=attributed_generation_seconds / raw_duration_seconds,
                batch_size=len(batch),
            )
            trimmed, annotation = analyze_generated_samples(
                samples=samples,
                sample_rate_hz=sample_rate_hz,
                prompt=prompt,
                audio_path=audio_relative_path,
                provenance=provenance,
            )
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(audio_path, trimmed, sample_rate_hz, subtype="PCM_16")
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
    manifest = QwenCompletionRunManifest(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        prompt_set_path=parsed.prompts,
        output_directory=parsed.output,
        model_id=parsed.model,
        model_revision=parsed.model_revision,
        runtime_version=QWEN_TTS_RUNTIME_VERSION,
        requested_run_seconds=parsed.run_seconds,
        batch_size=parsed.batch_size,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and label a time-bounded Qwen3-TTS completion corpus."
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--run-seconds", type=float, default=7200.0)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


if __name__ == "__main__":
    main()

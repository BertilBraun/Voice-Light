from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio_utils import to_mono_float32, write_mono_wav
from src.models import CaseManifest, ExperimentManifest, VariantManifest, manifest_path
from src.package_experiment import resolve_manifest_audio_path


@dataclass(frozen=True)
class MixRenderResult:
    mix_path: Path
    metadata_path: Path
    duration_seconds: float
    speaker_b_start_seconds: float
    normalization_applied: bool
    normalization_gain: float
    source: str


def main() -> None:
    arguments = parse_arguments()
    manifest = load_manifest(arguments.experiment_dir)
    output_dir = arguments.output_dir or arguments.experiment_dir / "mixes"
    results = export_all_mixes(
        experiment_dir=arguments.experiment_dir,
        manifest=manifest,
        output_dir=output_dir,
        speaker_a_gain=arguments.speaker_a_gain,
        speaker_b_gain=arguments.speaker_b_gain,
        source=arguments.source,
    )
    print(f"Wrote {len(results)} mixes to {output_dir}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render every successful A/B variant combination in an experiment."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--speaker-a-gain", type=float, default=1.0)
    parser.add_argument("--speaker-b-gain", type=float, default=1.0)
    parser.add_argument("--source", choices=["trimmed", "raw"], default="trimmed")
    return parser.parse_args()


def load_manifest(experiment_dir: Path) -> ExperimentManifest:
    return ExperimentManifest.model_validate_json(
        manifest_path(experiment_dir).read_text(encoding="utf-8")
    )


def export_all_mixes(
    experiment_dir: Path,
    manifest: ExperimentManifest,
    output_dir: Path,
    speaker_a_gain: float,
    speaker_b_gain: float,
    source: str,
) -> list[MixRenderResult]:
    clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[MixRenderResult] = []
    for case_manifest in manifest.cases:
        speaker_a_variants = successful_variants(case_manifest.speaker_a.variants)
        speaker_b_variants = successful_variants(case_manifest.speaker_b.variants)
        for speaker_a_variant in speaker_a_variants:
            for speaker_b_variant in speaker_b_variants:
                result = render_mix(
                    experiment_dir=experiment_dir,
                    output_dir=output_dir,
                    manifest=manifest,
                    case_manifest=case_manifest,
                    speaker_a_variant=speaker_a_variant,
                    speaker_b_variant=speaker_b_variant,
                    speaker_a_gain=speaker_a_gain,
                    speaker_b_gain=speaker_b_gain,
                    source=source,
                )
                results.append(result)
    write_index(output_dir, results)
    return results


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for output_path in output_dir.iterdir():
        if output_path.is_file() and output_path.suffix in {".wav", ".json"}:
            output_path.unlink()


def successful_variants(variants: list[VariantManifest]) -> list[VariantManifest]:
    return [variant for variant in variants if variant.status == "success" and variant.audio_url]


def render_mix(
    experiment_dir: Path,
    output_dir: Path,
    manifest: ExperimentManifest,
    case_manifest: CaseManifest,
    speaker_a_variant: VariantManifest,
    speaker_b_variant: VariantManifest,
    speaker_a_gain: float,
    speaker_b_gain: float,
    source: str,
) -> MixRenderResult:
    speaker_a_audio, sample_rate = read_variant_audio(experiment_dir, speaker_a_variant, source)
    speaker_b_audio, speaker_b_sample_rate = read_variant_audio(
        experiment_dir,
        speaker_b_variant,
        source,
    )
    if speaker_b_sample_rate != sample_rate:
        raise ValueError(
            "Cannot mix variants with different sample rates: "
            f"{speaker_a_variant.id} has {sample_rate}, "
            f"{speaker_b_variant.id} has {speaker_b_sample_rate}."
        )
    speaker_b_start_seconds = speaker_b_offset_seconds(case_manifest, speaker_a_variant)
    mix = mix_tracks(
        speaker_a_audio=speaker_a_audio,
        speaker_b_audio=speaker_b_audio,
        sample_rate=sample_rate,
        speaker_b_start_seconds=speaker_b_start_seconds,
        speaker_a_gain=speaker_a_gain,
        speaker_b_gain=speaker_b_gain,
    )
    peak = float(np.max(np.abs(mix))) if mix.size > 0 else 0.0
    normalization_gain = 1.0 / peak if peak > 1.0 else 1.0
    if normalization_gain != 1.0:
        mix = mix * normalization_gain
    filename_stem = mix_filename_stem(
        case_id=case_manifest.case_id,
        speaker_a_variant_id=speaker_a_variant.id,
        speaker_b_variant_id=speaker_b_variant.id,
        speaker_b_start_seconds=speaker_b_start_seconds,
    )
    mix_path = output_dir / f"{filename_stem}.wav"
    metadata_path = output_dir / f"{filename_stem}.json"
    write_mono_wav(mix_path, mix, sample_rate)
    metadata = mix_metadata(
        manifest=manifest,
        case_manifest=case_manifest,
        speaker_a_variant=speaker_a_variant,
        speaker_b_variant=speaker_b_variant,
        speaker_a_duration_seconds=float(speaker_a_audio.size) / float(sample_rate),
        speaker_b_duration_seconds=float(speaker_b_audio.size) / float(sample_rate),
        speaker_b_start_seconds=speaker_b_start_seconds,
        speaker_a_gain=speaker_a_gain,
        speaker_b_gain=speaker_b_gain,
        duration_seconds=float(mix.size) / float(sample_rate),
        normalization_gain=normalization_gain,
        source=source,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return MixRenderResult(
        mix_path=mix_path,
        metadata_path=metadata_path,
        duration_seconds=float(mix.size) / float(sample_rate),
        speaker_b_start_seconds=speaker_b_start_seconds,
        normalization_applied=normalization_gain != 1.0,
        normalization_gain=normalization_gain,
        source=source,
    )


def read_variant_audio(
    experiment_dir: Path,
    variant: VariantManifest,
    source: str,
) -> tuple[np.ndarray, int]:
    audio_url = variant_audio_url(variant, source)
    audio_path = resolve_manifest_audio_path(experiment_dir, audio_url)
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    return to_mono_float32(samples), int(sample_rate)


def variant_audio_url(variant: VariantManifest, source: str) -> str:
    if source == "raw":
        if variant.raw_audio_url is None:
            raise ValueError(f"Variant has no raw_audio_url: {variant.id}")
        return variant.raw_audio_url
    if variant.audio_url is None:
        raise ValueError(f"Variant has no audio_url: {variant.id}")
    return variant.audio_url


def speaker_b_offset_seconds(
    case_manifest: CaseManifest,
    speaker_a_variant: VariantManifest,
) -> float:
    if speaker_a_variant.suggested_b_offset_seconds is not None:
        return speaker_a_variant.suggested_b_offset_seconds
    if case_manifest.default_b_offset_seconds is not None:
        return case_manifest.default_b_offset_seconds
    if speaker_a_variant.duration_seconds is not None:
        return speaker_a_variant.duration_seconds + 0.3
    return 0.3


def mix_tracks(
    speaker_a_audio: np.ndarray,
    speaker_b_audio: np.ndarray,
    sample_rate: int,
    speaker_b_start_seconds: float,
    speaker_a_gain: float,
    speaker_b_gain: float,
) -> np.ndarray:
    speaker_a_start_sample = 0
    speaker_b_start_sample = int(round(speaker_b_start_seconds * sample_rate))
    earliest_sample = min(speaker_a_start_sample, speaker_b_start_sample)
    speaker_a_output_start = speaker_a_start_sample - earliest_sample
    speaker_b_output_start = speaker_b_start_sample - earliest_sample
    output_length = max(
        speaker_a_output_start + speaker_a_audio.size,
        speaker_b_output_start + speaker_b_audio.size,
    )
    mix = np.zeros(output_length, dtype=np.float32)
    mix[speaker_a_output_start : speaker_a_output_start + speaker_a_audio.size] += (
        speaker_a_audio * speaker_a_gain
    )
    mix[speaker_b_output_start : speaker_b_output_start + speaker_b_audio.size] += (
        speaker_b_audio * speaker_b_gain
    )
    return mix


def mix_filename_stem(
    case_id: str,
    speaker_a_variant_id: str,
    speaker_b_variant_id: str,
    speaker_b_start_seconds: float,
) -> str:
    offset_ms = int(round(speaker_b_start_seconds * 1000.0))
    sign = "neg" if offset_ms < 0 else "pos"
    return (
        f"{case_id}_A-{variant_number(speaker_a_variant_id)}_"
        f"B-{variant_number(speaker_b_variant_id)}_offset-{sign}{abs(offset_ms)}ms"
    )


def variant_number(variant_id: str) -> str:
    prefix = "variant_"
    if variant_id.startswith(prefix):
        return variant_id.removeprefix(prefix)
    return variant_id


def mix_metadata(
    manifest: ExperimentManifest,
    case_manifest: CaseManifest,
    speaker_a_variant: VariantManifest,
    speaker_b_variant: VariantManifest,
    speaker_a_duration_seconds: float,
    speaker_b_duration_seconds: float,
    speaker_b_start_seconds: float,
    speaker_a_gain: float,
    speaker_b_gain: float,
    duration_seconds: float,
    normalization_gain: float,
    source: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "case_id": case_manifest.case_id,
        "placement": case_manifest.placement.model_dump() if case_manifest.placement else None,
        "source": source,
        "speaker_a": {
            "variant_id": speaker_a_variant.id,
            "audio_url": speaker_a_variant.audio_url,
            "raw_audio_url": speaker_a_variant.raw_audio_url,
            "filename": speaker_a_variant.filename,
            "start_seconds": 0.0,
            "end_seconds": speaker_a_duration_seconds,
            "gain": speaker_a_gain,
            "placement_output": (
                speaker_a_variant.placement_output.model_dump()
                if speaker_a_variant.placement_output
                else None
            ),
        },
        "speaker_b": {
            "variant_id": speaker_b_variant.id,
            "audio_url": speaker_b_variant.audio_url,
            "raw_audio_url": speaker_b_variant.raw_audio_url,
            "filename": speaker_b_variant.filename,
            "start_seconds": speaker_b_start_seconds,
            "end_seconds": speaker_b_start_seconds + speaker_b_duration_seconds,
            "gain": speaker_b_gain,
        },
        "export": {
            "duration_seconds": duration_seconds,
            "normalization_applied": normalization_gain != 1.0,
            "normalization_gain": normalization_gain,
            "source": source,
        },
    }


def write_index(output_dir: Path, results: list[MixRenderResult]) -> None:
    index_data = [
        {
            "mix_path": result.mix_path.name,
            "metadata_path": result.metadata_path.name,
            "duration_seconds": result.duration_seconds,
            "speaker_b_start_seconds": result.speaker_b_start_seconds,
            "normalization_applied": result.normalization_applied,
            "normalization_gain": result.normalization_gain,
            "source": result.source,
        }
        for result in results
    ]
    (output_dir / "index.json").write_text(json.dumps(index_data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

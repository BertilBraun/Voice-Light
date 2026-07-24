from __future__ import annotations

import argparse
from pathlib import Path

from src.audio_utils import detect_major_audio_onset_seconds
from src.models import ExperimentManifest, SpeakerManifest, VariantManifest, manifest_path


def main() -> None:
    arguments = parse_arguments()
    experiment_dir = arguments.experiment_dir
    manifest = ExperimentManifest.model_validate_json(
        manifest_path(experiment_dir).read_text(encoding="utf-8")
    )
    updated_manifest = update_manifest(experiment_dir, manifest)
    manifest_path(experiment_dir).write_text(
        updated_manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"Updated backchannel onset metadata: {manifest_path(experiment_dir)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update B onset-corrected backchannel offsets.")
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def update_manifest(experiment_dir: Path, manifest: ExperimentManifest) -> ExperimentManifest:
    updated_cases = []
    for case_manifest in manifest.cases:
        updated_speaker_a = update_audio_onsets(experiment_dir, case_manifest.speaker_a)
        updated_speaker_b = update_audio_onsets(experiment_dir, case_manifest.speaker_b)
        if case_manifest.placement is not None and case_manifest.placement.type == "backchannel":
            updated_speaker_b = update_b_offsets(updated_speaker_a, updated_speaker_b)
        updated_cases.append(
            case_manifest.model_copy(
                update={
                    "speaker_a": updated_speaker_a,
                    "speaker_b": updated_speaker_b,
                }
            )
        )
    return manifest.model_copy(update={"cases": updated_cases})


def update_audio_onsets(experiment_dir: Path, speaker_manifest: SpeakerManifest) -> SpeakerManifest:
    updated_variants: list[VariantManifest] = []
    for variant in speaker_manifest.variants:
        if variant.status != "success" or variant.audio_url is None:
            updated_variants.append(variant)
            continue
        audio_onset_seconds = detect_major_audio_onset_seconds(experiment_dir / variant.audio_url)
        updated_variants.append(
            variant.model_copy(update={"audio_onset_seconds": audio_onset_seconds})
        )
    return speaker_manifest.model_copy(update={"variants": updated_variants})


def update_b_offsets(
    speaker_a_manifest: SpeakerManifest,
    speaker_b_manifest: SpeakerManifest,
) -> SpeakerManifest:
    target_variants = [
        variant
        for variant in speaker_a_manifest.variants
        if variant.status == "success" and variant.suggested_b_offset_seconds is not None
    ]
    if len(target_variants) == 0:
        return speaker_b_manifest
    updated_variants: list[VariantManifest] = []
    for index, variant in enumerate(speaker_b_manifest.variants):
        if variant.status != "success" or variant.audio_onset_seconds is None:
            updated_variants.append(variant)
            continue
        target_variant = target_variants[min(index, len(target_variants) - 1)]
        assert target_variant.suggested_b_offset_seconds is not None
        suggested_file_start_seconds = (
            target_variant.suggested_b_offset_seconds - variant.audio_onset_seconds
        )
        backend_metadata = dict(variant.backend_metadata)
        backend_metadata["backchannel_onset_alignment"] = {
            "paired_speaker_a_variant_id": target_variant.id,
            "target_backchannel_onset_seconds": target_variant.suggested_b_offset_seconds,
            "speaker_b_audio_onset_seconds": variant.audio_onset_seconds,
            "suggested_file_start_seconds": suggested_file_start_seconds,
        }
        updated_variants.append(
            variant.model_copy(
                update={
                    "suggested_b_offset_seconds": suggested_file_start_seconds,
                    "backend_metadata": backend_metadata,
                }
            )
        )
    return speaker_b_manifest.model_copy(update={"variants": updated_variants})


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from pydantic import ValidationError

from src.audio_utils import validate_decodable_audio
from src.models import ExperimentManifest, manifest_path


def main() -> None:
    arguments = parse_arguments()
    experiment_dir = arguments.experiment_dir
    manifest = load_manifest(experiment_dir)
    validate_experiment(experiment_dir, manifest)
    archive_path = experiment_dir.with_suffix(".zip")
    create_archive(experiment_dir, archive_path)
    print(f"Wrote package: {archive_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a generated experiment directory.")
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def load_manifest(experiment_dir: Path) -> ExperimentManifest:
    experiment_manifest_path = manifest_path(experiment_dir)
    if not experiment_manifest_path.exists():
        raise ValueError(f"Missing experiment manifest: {experiment_manifest_path}")
    try:
        return ExperimentManifest.model_validate_json(
            experiment_manifest_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError(f"Invalid experiment manifest:\n{error}") from error


def validate_experiment(experiment_dir: Path, manifest: ExperimentManifest) -> None:
    seen_case_ids: set[str] = set()
    for case_manifest in manifest.cases:
        if case_manifest.case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id: {case_manifest.case_id}")
        seen_case_ids.add(case_manifest.case_id)
        for speaker_manifest in [case_manifest.speaker_a, case_manifest.speaker_b]:
            seen_variant_ids: set[str] = set()
            for variant in speaker_manifest.variants:
                if variant.id in seen_variant_ids:
                    raise ValueError(
                        f"Duplicate variant id in case {case_manifest.case_id}: {variant.id}"
                    )
                seen_variant_ids.add(variant.id)
                if variant.status != "success":
                    continue
                if variant.audio_url is None:
                    raise ValueError(f"Successful variant lacks audio_url: {variant.id}")
                audio_path = resolve_manifest_audio_path(experiment_dir, variant.audio_url)
                if not audio_path.exists():
                    raise ValueError(f"Manifest audio path does not exist: {audio_path}")
                validate_decodable_audio(audio_path)


def resolve_manifest_audio_path(experiment_dir: Path, audio_url: str) -> Path:
    relative_path = Path(audio_url)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Manifest audio_url must be relative and contained: {audio_url}")
    return experiment_dir / relative_path


def create_archive(experiment_dir: Path, archive_path: Path) -> None:
    excluded_suffixes = {".zip", ".tmp", ".part"}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path in sorted(experiment_dir.rglob("*")):
            if not source_path.is_file():
                continue
            if source_path.suffix in excluded_suffixes:
                continue
            archive.write(source_path, source_path.relative_to(experiment_dir.parent))


if __name__ == "__main__":
    main()

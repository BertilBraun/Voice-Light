from __future__ import annotations

import argparse
from pathlib import Path

from app.local.config import DATABASE_URL
from app.local.source_annotations.repository import SourceAnnotationRepository
from app.local.source_annotations.service import (
    import_dataset_source_annotations_after_registration,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import prepared human source annotations after local sample registration."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    imported_sample_count = import_dataset_source_annotations_after_registration(
        repository=SourceAnnotationRepository(DATABASE_URL),
        dataset_name=arguments.dataset_name,
        samples_root=arguments.samples_root,
        source_name=arguments.source_name,
    )
    print(f"Imported source annotations for {imported_sample_count} registered samples.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from uuid import UUID

from app.local.source_annotations.models import (
    RegisteredSourceAnnotationSample,
    SourceAnnotationImport,
    SourceSampleAnnotations,
)


class SourceAnnotationWriter(Protocol):
    def import_after_sample_registration(
        self, annotations: Sequence[SourceAnnotationImport]
    ) -> None: ...


class PreparedSourceAnnotationRepository(SourceAnnotationWriter, Protocol):
    def list_registered_samples(
        self, dataset_name: str
    ) -> tuple[RegisteredSourceAnnotationSample, ...]: ...


def imports_after_sample_registration(
    speaker_1_track_id: UUID,
    speaker_2_track_id: UUID,
    source_name: str,
    annotations: SourceSampleAnnotations,
) -> tuple[SourceAnnotationImport, ...]:
    return tuple(
        SourceAnnotationImport(
            sample_track_id=speaker_1_track_id,
            source_name=source_name,
            annotation_version=annotations.annotation_version,
            annotator_id=annotation.annotator_id,
            annotation=annotation,
        )
        for annotation in annotations.speaker_1
    ) + tuple(
        SourceAnnotationImport(
            sample_track_id=speaker_2_track_id,
            source_name=source_name,
            annotation_version=annotations.annotation_version,
            annotator_id=annotation.annotator_id,
            annotation=annotation,
        )
        for annotation in annotations.speaker_2
    )


def import_prepared_sample_annotations_after_registration(
    repository: SourceAnnotationWriter,
    registered: RegisteredSourceAnnotationSample,
    sample_root: Path,
    source_name: str,
) -> None:
    annotation_path = sample_root / "source_annotations.json"
    if not annotation_path.is_file():
        raise ValueError(f"Source annotations do not exist: {annotation_path}")
    annotations = SourceSampleAnnotations.model_validate_json(
        annotation_path.read_text(encoding="utf-8")
    )
    repository.import_after_sample_registration(
        imports_after_sample_registration(
            speaker_1_track_id=registered.speaker_1_track_id,
            speaker_2_track_id=registered.speaker_2_track_id,
            source_name=source_name,
            annotations=annotations,
        )
    )


def import_dataset_source_annotations_after_registration(
    repository: PreparedSourceAnnotationRepository,
    dataset_name: str,
    samples_root: Path,
    source_name: str,
) -> int:
    if not samples_root.is_dir():
        raise ValueError(f"Prepared samples root does not exist: {samples_root}")
    registered_samples = repository.list_registered_samples(dataset_name)
    if not registered_samples:
        raise ValueError(f"No registered samples found for dataset: {dataset_name}")
    imported_sample_count = 0
    for registered in registered_samples:
        sample_root = samples_root / registered.external_id
        import_prepared_sample_annotations_after_registration(
            repository=repository,
            registered=registered,
            sample_root=sample_root,
            source_name=source_name,
        )
        imported_sample_count += 1
    return imported_sample_count

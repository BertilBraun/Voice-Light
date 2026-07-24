from __future__ import annotations

import argparse
from pathlib import Path

from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.config import DATABASE_URL
from app.local.db.models import TrackSide
from app.local.db.repository import Repository
from app.local.ingestion.discovery import DiscoveredSample
from app.local.ingestion.service import (
    persist_processed_ready_sample,
    process_ready_sample,
    ready_sample,
    register_sample,
)
from app.local.source_annotations.asr_materialization import (
    materialize_source_transcripts,
    sample_track_for_side,
)
from app.local.source_annotations.models import SourceSampleAnnotations
from app.local.source_annotations.repository import SourceAnnotationRepository
from app.shared.storage.local import LocalStorageBackend


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize supplied source transcripts into the full-recording transcript store."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--refresh-quality", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    sample_repository = Repository(DATABASE_URL)
    annotation_repository = SourceAnnotationRepository(DATABASE_URL)
    transcript_repository = FullRecordingAsrRepository(DATABASE_URL)
    registered_samples = annotation_repository.list_registered_samples(arguments.dataset_name)
    if not registered_samples:
        raise ValueError(f"No registered samples found for dataset: {arguments.dataset_name}")
    storage = LocalStorageBackend()
    for registered_sample in registered_samples:
        annotation_path = (
            arguments.samples_root / registered_sample.external_id / "source_annotations.json"
        )
        annotations = SourceSampleAnnotations.model_validate_json(
            annotation_path.read_text(encoding="utf-8")
        )
        materialize_source_transcripts(
            repository=transcript_repository,
            dashboard_sample=sample_repository.get_dashboard_sample(registered_sample.sample_id),
            annotations=annotations,
        )
        if arguments.refresh_quality:
            dashboard_sample = sample_repository.get_dashboard_sample(registered_sample.sample_id)
            speaker_1_track = sample_track_for_side(dashboard_sample, TrackSide.SPEAKER1)
            speaker_2_track = sample_track_for_side(dashboard_sample, TrackSide.SPEAKER2)
            registered_ingestion_sample = register_sample(
                repository=sample_repository,
                dataset_id=dashboard_sample.sample.dataset_id,
                storage=storage,
                discovered=DiscoveredSample(
                    external_id=registered_sample.external_id,
                    speaker1_path=speaker_1_track.storage_uri,
                    speaker2_path=speaker_2_track.storage_uri,
                ),
            )
            ready = ready_sample(
                full_asr_repository=transcript_repository,
                registered=registered_ingestion_sample,
            )
            persist_processed_ready_sample(
                repository=sample_repository,
                processed=process_ready_sample(storage=storage, ready=ready),
            )
    print(f"Materialized source transcripts for {len(registered_samples)} samples.")


if __name__ == "__main__":
    main()

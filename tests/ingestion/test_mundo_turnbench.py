from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from app.local.ingestion.mundo_turnbench import (
    SOURCE_ANNOTATION_VERSION,
    apply_preparation,
    build_preparation_plan,
)
from app.local.source_annotations.models import (
    RegisteredSourceAnnotationSample,
    SourceAnnotationImport,
)
from app.local.source_annotations.service import (
    import_dataset_source_annotations_after_registration,
    imports_after_sample_registration,
)


def test_turnbench_preparation_uses_flac_columns_and_preserves_all_human_tracks(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.parquet"
    write_source_parquet(source_path)

    plan = build_preparation_plan(source_path)
    sample = plan.samples[0]

    assert sample.sample_id == "sample_001"
    assert sample.speaker_1.content == b"fLaCspeaker-1"
    assert sample.speaker_2.content == b"fLaCspeaker-2"
    assert [track.annotator_id for track in sample.annotations.speaker_1] == ["a", "b", "c"]
    assert sample.annotations.speaker_2[0].events[0].text == "speaker two"


def test_turnbench_preparation_combines_parquet_shards_in_path_order(tmp_path: Path) -> None:
    first_source_path = tmp_path / "dev-00000-of-00002.parquet"
    second_source_path = tmp_path / "dev-00001-of-00002.parquet"
    write_source_parquet(first_source_path)
    write_source_parquet(second_source_path)

    plan = build_preparation_plan((second_source_path, first_source_path))

    assert [sample.sample_id for sample in plan.samples] == ["sample_001", "sample_002"]


def test_turnbench_preparation_writes_generic_layout_and_restricted_mapping(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    target_root = tmp_path / "data" / "dataset_3" / "samples"
    restricted_path = tmp_path / "restricted" / "mapping.jsonl"
    write_source_parquet(source_path)

    apply_preparation(source_path, target_root, restricted_path)

    sample_root = target_root / "sample_001"
    assert {path.name for path in sample_root.iterdir()} == {
        "speaker_1.flac",
        "speaker_2.flac",
        "metadata.json",
        "source_annotations.json",
    }
    assert json.loads((sample_root / "metadata.json").read_text(encoding="utf-8")) == {
        "name": "sample_001",
        "speaker1_audioURL": "speaker_1.flac",
        "speaker2_audioURL": "speaker_2.flac",
    }
    assert "source-conversation-1" not in (sample_root / "source_annotations.json").read_text(
        encoding="utf-8"
    )
    assert "source-conversation-1" in restricted_path.read_text(encoding="utf-8")


def test_turnbench_rejects_preview_audio_in_place_of_flac(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    write_source_parquet(source_path, speaker_1_audio=b"OggSpreview")

    with pytest.raises(ValueError, match="not FLAC"):
        build_preparation_plan(source_path)


def test_source_annotation_imports_are_separate_from_asr_and_cover_both_tracks(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.parquet"
    write_source_parquet(source_path)
    annotations = build_preparation_plan(source_path).samples[0].annotations

    imports = imports_after_sample_registration(
        speaker_1_track_id=uuid4(),
        speaker_2_track_id=uuid4(),
        source_name="dataset_3",
        annotations=annotations,
    )

    assert len(imports) == 6
    assert {item.annotation_version for item in imports} == {SOURCE_ANNOTATION_VERSION}
    assert {item.source_name for item in imports} == {"dataset_3"}
    assert len({item.sample_track_id for item in imports[:3]}) == 1
    assert len({item.sample_track_id for item in imports[3:]}) == 1


def test_annotation_only_workflow_imports_prepared_annotations_after_registration(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.parquet"
    samples_root = tmp_path / "data" / "dataset_3" / "samples"
    write_source_parquet(source_path)
    apply_preparation(source_path, samples_root, tmp_path / "restricted.jsonl")
    repository = RecordingSourceAnnotationRepository()

    imported_sample_count = import_dataset_source_annotations_after_registration(
        repository=repository,
        dataset_name="dataset_3",
        samples_root=samples_root,
        source_name="dataset_3",
    )

    assert imported_sample_count == 1
    assert len(repository.imported) == 6
    assert {item.annotator_id for item in repository.imported} == {"a", "b", "c"}
    assert all(item.annotation.events for item in repository.imported[::3])


class RecordingSourceAnnotationRepository:
    def __init__(self) -> None:
        self.sample = RegisteredSourceAnnotationSample(
            sample_id=uuid4(),
            external_id="sample_001",
            speaker_1_track_id=uuid4(),
            speaker_2_track_id=uuid4(),
        )
        self.imported: tuple[SourceAnnotationImport, ...] = ()

    def list_registered_samples(
        self, dataset_name: str
    ) -> tuple[RegisteredSourceAnnotationSample, ...]:
        assert dataset_name == "dataset_3"
        return (self.sample,)

    def import_after_sample_registration(
        self, annotations: tuple[SourceAnnotationImport, ...]
    ) -> None:
        self.imported = annotations


def write_source_parquet(source_path: Path, speaker_1_audio: bytes = b"fLaCspeaker-1") -> None:
    event_1 = [{"start_s": 0.0, "end_s": 1.0, "label": "[turn]", "text": "speaker one"}]
    event_2 = [{"start_s": 1.1, "end_s": 2.0, "label": "[turn]", "text": "speaker two"}]
    table = arrow.Table.from_pylist(
        [
            {
                "conversation_id": "source-conversation-1",
                "speaker_1_audio": {"bytes": speaker_1_audio, "path": "source-1.flac"},
                "speaker_2_audio": {"bytes": b"fLaCspeaker-2", "path": "source-2.flac"},
                "speaker_1_audio_preview": {"bytes": b"OggSpreview", "path": "source-1.opus"},
                "speaker_2_audio_preview": {"bytes": b"OggSpreview", "path": "source-2.opus"},
                "speaker_1_annotation_a": event_1,
                "speaker_1_annotation_b": [],
                "speaker_1_annotation_c": [],
                "speaker_2_annotation_a": event_2,
                "speaker_2_annotation_b": [],
                "speaker_2_annotation_c": [],
            }
        ]
    )
    parquet.write_table(table, source_path)

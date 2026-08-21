from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from app.local.training_corpus.hub_upload_plan import (
    CorpusUploadPlan,
    DiscoveredPublicationFiles,
    ExcludedLocalFile,
    LocalPublicationFile,
    RemoteRepositoryFile,
    UploadPlanAction,
    UploadPlanReason,
    build_upload_plan,
    discover_publication_files,
    write_upload_plan,
)

BASE_REVISION = "1" * 40
GENERATED_AT = datetime(2026, 8, 21, tzinfo=UTC)


def test_discovery_allows_only_staged_d2_d3_audio_and_public_export(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    export_root = tmp_path / "export"
    write(audio_root / "dataset_2/samples/sample_001/speaker_1.flac", b"magic")
    write(audio_root / "dataset_3/samples/sample_038/speaker_2.flac", b"turnbench")
    write(audio_root / ".build/dataset_2-audio-assets.json", b"restricted")
    write(export_root / "corpus.json", b"manifest")
    write(export_root / "dataset_1/dataset.json", b"dataset")
    write(export_root / "dataset_4/samples/sample_111/metadata.json", b"recording")
    write(export_root / "training/test/shard-00000.parquet", b"shard")

    discovered = discover_publication_files(audio_root, export_root)

    assert tuple(item.path.as_posix() for item in discovered.publishable) == (
        "corpus.json",
        "dataset_1/dataset.json",
        "dataset_2/samples/sample_001/speaker_1.flac",
        "dataset_3/samples/sample_038/speaker_2.flac",
        "dataset_4/samples/sample_111/metadata.json",
        "training/test/shard-00000.parquet",
    )
    assert len(discovered.excluded) == 1
    assert discovered.excluded[0].path == PurePosixPath(".build/dataset_2-audio-assets.json")
    assert discovered.excluded[0].reason is UploadPlanReason.RESTRICTED_BUILD_PROVENANCE


def test_discovery_supports_canonical_common_staging_tree(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    write(root / "dataset_2/samples/sample_001/speaker_1.flac", b"magic")
    write(root / "dataset_3/samples/sample_038/metadata.json", b"recording")
    write(root / "training/train/shard-00000.parquet", b"shard")
    write(root / "corpus.json", b"manifest")
    write(root / ".build/dataset_2-audio-assets.json", b"restricted")

    discovered = discover_publication_files(root, root)

    assert tuple(item.path.as_posix() for item in discovered.publishable) == (
        "corpus.json",
        "dataset_2/samples/sample_001/speaker_1.flac",
        "dataset_3/samples/sample_038/metadata.json",
        "training/train/shard-00000.parquet",
    )
    assert tuple(item.path.as_posix() for item in discovered.excluded) == (
        ".build/dataset_2-audio-assets.json",
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        "dataset_1/samples/sample_001/speaker_1.flac",
        "dataset_4/samples/sample_001/speaker_1.flac",
        "dataset_2/samples/sample_001/source.wav",
        "private-provenance.json",
    ),
)
def test_discovery_rejects_unsupported_staging_files(
    tmp_path: Path,
    relative_path: str,
) -> None:
    audio_root = tmp_path / "audio"
    export_root = tmp_path / "export"
    export_root.mkdir()
    write(audio_root / relative_path, b"forbidden")

    with pytest.raises(ValueError, match="unsupported publication file"):
        discover_publication_files(audio_root, export_root)


def test_discovery_rejects_audio_inside_public_export(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    export_root = tmp_path / "export"
    audio_root.mkdir()
    write(export_root / "dataset_2/samples/sample_001/speaker_1.flac", b"audio")

    with pytest.raises(ValueError, match="unsupported publication file"):
        discover_publication_files(audio_root, export_root)


def test_plan_classifies_add_update_unchanged_and_excluded(tmp_path: Path) -> None:
    added = local_file(tmp_path, "corpus.json", b"new")
    changed = local_file(tmp_path, "dataset_1/dataset.json", b"changed")
    unchanged = local_file(tmp_path, "training/train/shard-00000.parquet", b"same")
    restricted_path = tmp_path / ".build/manifest.json"
    write(restricted_path, b"secret")
    discovered = DiscoveredPublicationFiles(
        publishable=(added, changed, unchanged),
        excluded=(
            ExcludedLocalFile(
                path=PurePosixPath(".build/manifest.json"),
                local_path=restricted_path,
                size_bytes=restricted_path.stat().st_size,
                reason=UploadPlanReason.RESTRICTED_BUILD_PROVENANCE,
            ),
        ),
    )
    remote = (
        remote_file(changed.path, b"old"),
        remote_file(unchanged.path, b"same"),
    )

    plan = build_upload_plan(
        repository_id="BertilBraun/voice-light-audio",
        base_revision=BASE_REVISION,
        current_head_revision=BASE_REVISION,
        discovered=discovered,
        remote_files=remote,
        generated_at=GENERATED_AT,
    )

    action_by_path = {entry.path.as_posix(): entry.action for entry in plan.entries}
    assert action_by_path == {
        ".build/manifest.json": UploadPlanAction.EXCLUDED,
        "corpus.json": UploadPlanAction.ADD,
        "dataset_1/dataset.json": UploadPlanAction.UPDATE,
        "training/train/shard-00000.parquet": UploadPlanAction.UNCHANGED,
    }
    assert plan.summary.add_file_count == 1
    assert plan.summary.update_file_count == 1
    assert plan.summary.unchanged_file_count == 1
    assert plan.summary.excluded_file_count == 1
    assert plan.summary.upload_file_count == 2
    assert plan.summary.upload_bytes == added.size_bytes + changed.size_bytes
    assert plan.delete_file_count == 0
    assert plan.dry_run is True
    excluded_entry = plan.entries[0]
    assert excluded_entry.local_sha256 is None
    assert excluded_entry.remote_sha256 is None


def test_plan_refuses_when_remote_head_moved(tmp_path: Path) -> None:
    discovered = DiscoveredPublicationFiles(
        publishable=(local_file(tmp_path, "corpus.json", b"new"),),
        excluded=(),
    )

    with pytest.raises(ValueError, match="head moved"):
        build_upload_plan(
            repository_id="BertilBraun/voice-light-audio",
            base_revision=BASE_REVISION,
            current_head_revision="2" * 40,
            discovered=discovered,
            remote_files=(),
            generated_at=GENERATED_AT,
        )


def test_plan_rejects_duplicate_remote_paths(tmp_path: Path) -> None:
    local = local_file(tmp_path, "corpus.json", b"new")
    duplicate = remote_file(local.path, b"old")

    with pytest.raises(ValueError, match="duplicate path"):
        build_upload_plan(
            repository_id="BertilBraun/voice-light-audio",
            base_revision=BASE_REVISION,
            current_head_revision=BASE_REVISION,
            discovered=DiscoveredPublicationFiles(publishable=(local,), excluded=()),
            remote_files=(duplicate, duplicate),
            generated_at=GENERATED_AT,
        )


def test_plan_report_is_reviewable_and_refuses_overwrite(tmp_path: Path) -> None:
    local = local_file(tmp_path, "corpus.json", b"new")
    plan = build_upload_plan(
        repository_id="BertilBraun/voice-light-audio",
        base_revision=BASE_REVISION,
        current_head_revision=BASE_REVISION,
        discovered=DiscoveredPublicationFiles(publishable=(local,), excluded=()),
        remote_files=(),
        generated_at=GENERATED_AT,
    )
    output_path = tmp_path / "plan/upload-plan.json"

    write_upload_plan(plan, output_path)

    loaded = CorpusUploadPlan.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert loaded == plan
    with pytest.raises(ValueError, match="already exists"):
        write_upload_plan(plan, output_path)


def local_file(root: Path, relative_path: str, content: bytes) -> LocalPublicationFile:
    path = root / relative_path
    write(path, content)
    return LocalPublicationFile(
        path=PurePosixPath(relative_path),
        local_path=path.resolve(),
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def remote_file(path: PurePosixPath, content: bytes) -> RemoteRepositoryFile:
    return RemoteRepositoryFile(
        path=path,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

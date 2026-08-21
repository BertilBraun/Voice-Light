from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download
from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"
PUBLIC_DATASET_DIRECTORIES = frozenset(("dataset_1", "dataset_2", "dataset_3", "dataset_4"))
LOCAL_AUDIO_DATASET_DIRECTORIES = frozenset(("dataset_2", "dataset_3"))
SPLIT_DIRECTORIES = frozenset(("train", "validation", "test"))
SPEAKER_FILENAMES = frozenset(("speaker_1.flac", "speaker_2.flac"))


class UploadPlanAction(StrEnum):
    ADD = "add"
    UPDATE = "update"
    UNCHANGED = "unchanged"
    EXCLUDED = "excluded"


class UploadPlanReason(StrEnum):
    MISSING_REMOTELY = "missing_remotely"
    CONTENT_CHANGED = "content_changed"
    CONTENT_IDENTICAL = "content_identical"
    RESTRICTED_BUILD_PROVENANCE = "restricted_build_provenance"


class CorpusUploadPlanRequest(FrozenBaseModel):
    repository_id: str = Field(min_length=1)
    base_revision: str = Field(pattern=REVISION_PATTERN)
    audio_staging_root: Path
    export_root: Path
    output_path: Path


class LocalPublicationFile(FrozenBaseModel):
    path: PurePosixPath
    local_path: Path
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class ExcludedLocalFile(FrozenBaseModel):
    path: PurePosixPath
    local_path: Path
    size_bytes: int = Field(ge=0)
    reason: UploadPlanReason


class RemoteRepositoryFile(FrozenBaseModel):
    path: PurePosixPath
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class CorpusUploadPlanEntry(FrozenBaseModel):
    path: PurePosixPath
    local_path: Path
    action: UploadPlanAction
    reason: UploadPlanReason
    local_size_bytes: int = Field(ge=0)
    local_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    remote_size_bytes: int | None = Field(default=None, ge=0)
    remote_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_action_evidence(self) -> CorpusUploadPlanEntry:
        if self.action is UploadPlanAction.EXCLUDED:
            if self.local_sha256 is not None or self.remote_sha256 is not None:
                raise ValueError("Excluded upload-plan entries must not expose content hashes.")
            return self
        if self.local_sha256 is None:
            raise ValueError("Publishable upload-plan entries require a local SHA-256.")
        if self.action is UploadPlanAction.ADD:
            if self.remote_sha256 is not None or self.remote_size_bytes is not None:
                raise ValueError("Added upload-plan entries must not have remote evidence.")
            return self
        if self.remote_sha256 is None or self.remote_size_bytes is None:
            raise ValueError("Existing upload-plan entries require remote evidence.")
        if self.action is UploadPlanAction.UNCHANGED:
            if self.local_sha256 != self.remote_sha256:
                raise ValueError("Unchanged upload-plan entries must have identical hashes.")
        elif self.local_sha256 == self.remote_sha256:
            raise ValueError("Updated upload-plan entries must have different hashes.")
        return self


class CorpusUploadPlanSummary(FrozenBaseModel):
    add_file_count: int = Field(ge=0)
    update_file_count: int = Field(ge=0)
    unchanged_file_count: int = Field(ge=0)
    excluded_file_count: int = Field(ge=0)
    upload_file_count: int = Field(ge=0)
    upload_bytes: int = Field(ge=0)


class CorpusUploadPlan(FrozenBaseModel):
    schema_version: Literal["voice-light-hub-upload-plan-v1"] = "voice-light-hub-upload-plan-v1"
    generated_at: datetime
    repository_id: str
    base_revision: str = Field(pattern=REVISION_PATTERN)
    verified_head_revision: str = Field(pattern=REVISION_PATTERN)
    dry_run: Literal[True] = True
    delete_file_count: Literal[0] = 0
    summary: CorpusUploadPlanSummary
    entries: tuple[CorpusUploadPlanEntry, ...]

    @model_validator(mode="after")
    def validate_publication_guardrails(self) -> CorpusUploadPlan:
        if self.verified_head_revision != self.base_revision:
            raise ValueError("Upload plan head revision must equal its pinned base revision.")
        calculated = summarize_entries(self.entries)
        if calculated != self.summary:
            raise ValueError("Upload-plan summary does not match its entries.")
        return self


class DiscoveredPublicationFiles(FrozenBaseModel):
    publishable: tuple[LocalPublicationFile, ...]
    excluded: tuple[ExcludedLocalFile, ...]


def discover_publication_files(
    audio_staging_root: Path,
    export_root: Path,
) -> DiscoveredPublicationFiles:
    if not audio_staging_root.is_dir():
        raise ValueError(f"Audio staging root does not exist: {audio_staging_root}")
    if not export_root.is_dir():
        raise ValueError(f"Corpus export root does not exist: {export_root}")
    publishable: list[LocalPublicationFile] = []
    excluded: list[ExcludedLocalFile] = []
    for local_path in _ordered_files(audio_staging_root):
        relative_path = PurePosixPath(local_path.relative_to(audio_staging_root).as_posix())
        if relative_path.parts and relative_path.parts[0] == ".build":
            excluded.append(
                ExcludedLocalFile(
                    path=relative_path,
                    local_path=local_path.resolve(),
                    size_bytes=local_path.stat().st_size,
                    reason=UploadPlanReason.RESTRICTED_BUILD_PROVENANCE,
                )
            )
            continue
        if not _is_allowed_staged_audio(relative_path):
            raise ValueError(
                f"Audio staging root contains unsupported publication file {relative_path}."
            )
        publishable.append(_local_publication_file(audio_staging_root, local_path))
    for local_path in _ordered_files(export_root):
        relative_path = PurePosixPath(local_path.relative_to(export_root).as_posix())
        if relative_path.parts and relative_path.parts[0] == ".build":
            excluded.append(
                ExcludedLocalFile(
                    path=relative_path,
                    local_path=local_path.resolve(),
                    size_bytes=local_path.stat().st_size,
                    reason=UploadPlanReason.RESTRICTED_BUILD_PROVENANCE,
                )
            )
            continue
        if not _is_allowed_public_export(relative_path):
            raise ValueError(
                f"Corpus export contains unsupported publication file {relative_path}."
            )
        publishable.append(_local_publication_file(export_root, local_path))
    _validate_unique_target_paths(publishable, excluded)
    return DiscoveredPublicationFiles(
        publishable=tuple(sorted(publishable, key=lambda item: item.path.as_posix())),
        excluded=tuple(sorted(excluded, key=lambda item: item.path.as_posix())),
    )


def build_upload_plan(
    repository_id: str,
    base_revision: str,
    current_head_revision: str,
    discovered: DiscoveredPublicationFiles,
    remote_files: Sequence[RemoteRepositoryFile],
    generated_at: datetime,
) -> CorpusUploadPlan:
    if current_head_revision != base_revision:
        raise ValueError(
            "Hugging Face repository head moved after the pinned base revision: "
            f"expected {base_revision}, found {current_head_revision}."
        )
    remote_by_path = _unique_remote_files(remote_files)
    entries: list[CorpusUploadPlanEntry] = []
    for local_file in discovered.publishable:
        remote_file = remote_by_path.get(local_file.path)
        if remote_file is None:
            action = UploadPlanAction.ADD
            reason = UploadPlanReason.MISSING_REMOTELY
        elif remote_file.sha256 == local_file.sha256:
            action = UploadPlanAction.UNCHANGED
            reason = UploadPlanReason.CONTENT_IDENTICAL
        else:
            action = UploadPlanAction.UPDATE
            reason = UploadPlanReason.CONTENT_CHANGED
        entries.append(
            CorpusUploadPlanEntry(
                path=local_file.path,
                local_path=local_file.local_path,
                action=action,
                reason=reason,
                local_size_bytes=local_file.size_bytes,
                local_sha256=local_file.sha256,
                remote_size_bytes=None if remote_file is None else remote_file.size_bytes,
                remote_sha256=None if remote_file is None else remote_file.sha256,
            )
        )
    entries.extend(
        CorpusUploadPlanEntry(
            path=excluded.path,
            local_path=excluded.local_path,
            action=UploadPlanAction.EXCLUDED,
            reason=excluded.reason,
            local_size_bytes=excluded.size_bytes,
        )
        for excluded in discovered.excluded
    )
    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.path.as_posix()))
    return CorpusUploadPlan(
        generated_at=generated_at,
        repository_id=repository_id,
        base_revision=base_revision,
        verified_head_revision=current_head_revision,
        summary=summarize_entries(ordered_entries),
        entries=ordered_entries,
    )


def build_live_upload_plan(request: CorpusUploadPlanRequest) -> CorpusUploadPlan:
    discovered = discover_publication_files(
        audio_staging_root=request.audio_staging_root,
        export_root=request.export_root,
    )
    api = HfApi()
    head_info = api.dataset_info(request.repository_id)
    if head_info.sha != request.base_revision:
        raise ValueError(
            "Hugging Face repository head moved after the pinned base revision: "
            f"expected {request.base_revision}, found {head_info.sha}."
        )
    remote_files = load_remote_repository_files(
        repository_id=request.repository_id,
        revision=request.base_revision,
        paths=tuple(item.path for item in discovered.publishable),
        api=api,
    )
    plan = build_upload_plan(
        repository_id=request.repository_id,
        base_revision=request.base_revision,
        current_head_revision=head_info.sha,
        discovered=discovered,
        remote_files=remote_files,
        generated_at=datetime.now(UTC),
    )
    write_upload_plan(plan, request.output_path)
    return plan


def load_remote_repository_files(
    repository_id: str,
    revision: str,
    paths: Sequence[PurePosixPath],
    api: HfApi,
) -> tuple[RemoteRepositoryFile, ...]:
    info = api.dataset_info(
        repo_id=repository_id,
        revision=revision,
        files_metadata=True,
    )
    if info.sha != revision:
        raise ValueError(
            f"Hub resolved revision {info.sha!r}, expected exact revision {revision!r}."
        )
    sibling_by_path = {PurePosixPath(sibling.rfilename): sibling for sibling in info.siblings}
    remote_files: list[RemoteRepositoryFile] = []
    for path in sorted(set(paths), key=PurePosixPath.as_posix):
        sibling = sibling_by_path.get(path)
        if sibling is None:
            continue
        if sibling.size is None:
            raise ValueError(f"Hub file has no size metadata: {path}")
        if sibling.lfs is not None:
            sha256 = sibling.lfs.sha256
        else:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=repository_id,
                    filename=path.as_posix(),
                    repo_type="dataset",
                    revision=revision,
                )
            )
            sha256 = _file_sha256(downloaded_path)
        remote_files.append(RemoteRepositoryFile(path=path, size_bytes=sibling.size, sha256=sha256))
    return tuple(remote_files)


def summarize_entries(entries: Sequence[CorpusUploadPlanEntry]) -> CorpusUploadPlanSummary:
    add_count = sum(entry.action is UploadPlanAction.ADD for entry in entries)
    update_count = sum(entry.action is UploadPlanAction.UPDATE for entry in entries)
    unchanged_count = sum(entry.action is UploadPlanAction.UNCHANGED for entry in entries)
    excluded_count = sum(entry.action is UploadPlanAction.EXCLUDED for entry in entries)
    return CorpusUploadPlanSummary(
        add_file_count=add_count,
        update_file_count=update_count,
        unchanged_file_count=unchanged_count,
        excluded_file_count=excluded_count,
        upload_file_count=add_count + update_count,
        upload_bytes=sum(
            entry.local_size_bytes
            for entry in entries
            if entry.action in {UploadPlanAction.ADD, UploadPlanAction.UPDATE}
        ),
    )


def write_upload_plan(plan: CorpusUploadPlan, output_path: Path) -> None:
    if output_path.exists():
        raise ValueError(f"Upload plan output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _is_allowed_staged_audio(path: PurePosixPath) -> bool:
    return (
        len(path.parts) == 4
        and path.parts[0] in LOCAL_AUDIO_DATASET_DIRECTORIES
        and path.parts[1] == "samples"
        and _is_sample_directory(path.parts[2])
        and path.parts[3] in SPEAKER_FILENAMES
    )


def _is_allowed_public_export(path: PurePosixPath) -> bool:
    if path == PurePosixPath("corpus.json"):
        return True
    if (
        len(path.parts) == 3
        and path.parts[0] == "training"
        and path.parts[1] in SPLIT_DIRECTORIES
        and re.fullmatch(r"shard-\d{5}\.parquet", path.parts[2]) is not None
    ):
        return True
    if path.parts[0] not in PUBLIC_DATASET_DIRECTORIES:
        return False
    if len(path.parts) == 2 and path.parts[1] == "dataset.json":
        return True
    return (
        len(path.parts) == 4
        and path.parts[1] == "samples"
        and _is_sample_directory(path.parts[2])
        and path.parts[3] == "metadata.json"
    )


def _is_sample_directory(value: str) -> bool:
    return re.fullmatch(r"sample_\d{3}", value) is not None


def _local_publication_file(root: Path, path: Path) -> LocalPublicationFile:
    return LocalPublicationFile(
        path=PurePosixPath(path.relative_to(root).as_posix()),
        local_path=path.resolve(),
        size_bytes=path.stat().st_size,
        sha256=_file_sha256(path),
    )


def _ordered_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=str))


def _validate_unique_target_paths(
    publishable: Sequence[LocalPublicationFile],
    excluded: Sequence[ExcludedLocalFile],
) -> None:
    paths = tuple(item.path for item in (*publishable, *excluded))
    if len(set(paths)) != len(paths):
        raise ValueError("Publication sources contain duplicate target paths.")


def _unique_remote_files(
    remote_files: Sequence[RemoteRepositoryFile],
) -> Mapping[PurePosixPath, RemoteRepositoryFile]:
    by_path: dict[PurePosixPath, RemoteRepositoryFile] = {}
    for remote_file in remote_files:
        if remote_file.path in by_path:
            raise ValueError(
                f"Remote repository inventory contains duplicate path {remote_file.path}."
            )
        by_path[remote_file.path] = remote_file
    return by_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> CorpusUploadPlanRequest:
    parser = argparse.ArgumentParser(description="Build a dry-run Hugging Face corpus upload plan.")
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--audio-staging-root", required=True, type=Path)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    arguments = parser.parse_args()
    return CorpusUploadPlanRequest(
        repository_id=arguments.repository_id,
        base_revision=arguments.base_revision,
        audio_staging_root=arguments.audio_staging_root,
        export_root=arguments.export_root,
        output_path=arguments.output_path,
    )


def main() -> None:
    plan = build_live_upload_plan(parse_arguments())
    print(plan.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from pydantic import Field

from app.local.training_corpus.hub_upload_plan import (
    CorpusUploadPlan,
    CorpusUploadPlanEntry,
    UploadPlanAction,
)
from app.shared.base_model import FrozenBaseModel

HASH_CHUNK_BYTES = 1024 * 1024


class CorpusPublicationRequest(FrozenBaseModel):
    upload_plan_path: Path
    commit_message: str = Field(min_length=1)


class CorpusPublicationResult(FrozenBaseModel):
    repository_id: str
    base_revision: str
    published_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    uploaded_file_count: int = Field(gt=0)
    uploaded_bytes: int = Field(gt=0)


def publish_corpus(request: CorpusPublicationRequest) -> CorpusPublicationResult:
    plan = CorpusUploadPlan.model_validate_json(
        request.upload_plan_path.read_text(encoding="utf-8")
    )
    upload_entries = verify_upload_entries(plan)
    api = HfApi()
    current_revision = api.repo_info(
        repo_id=plan.repository_id,
        repo_type="dataset",
        revision="main",
    ).sha
    if current_revision != plan.verified_head_revision:
        raise ValueError(
            "Hugging Face repository head moved after upload planning: "
            f"expected {plan.verified_head_revision}, found {current_revision}."
        )
    operations = tuple(
        CommitOperationAdd(
            path_in_repo=entry.path.as_posix(),
            path_or_fileobj=entry.local_path,
        )
        for entry in upload_entries
    )
    commit = api.create_commit(
        repo_id=plan.repository_id,
        repo_type="dataset",
        revision="main",
        parent_commit=plan.base_revision,
        operations=operations,
        commit_message=request.commit_message,
    )
    return CorpusPublicationResult(
        repository_id=plan.repository_id,
        base_revision=plan.base_revision,
        published_revision=commit.oid,
        uploaded_file_count=len(upload_entries),
        uploaded_bytes=sum(entry.local_size_bytes for entry in upload_entries),
    )


def verify_upload_entries(
    plan: CorpusUploadPlan,
) -> tuple[CorpusUploadPlanEntry, ...]:
    if not plan.dry_run or plan.delete_file_count != 0:
        raise ValueError("Corpus publication requires a dry-run plan with zero deletions.")
    entries = tuple(
        entry
        for entry in plan.entries
        if entry.action in (UploadPlanAction.ADD, UploadPlanAction.UPDATE)
    )
    if not entries:
        raise ValueError("Corpus upload plan contains no files to publish.")
    for entry in entries:
        if entry.local_sha256 is None:
            raise ValueError(f"Upload entry has no local SHA-256: {entry.path}")
        if not entry.local_path.is_file():
            raise ValueError(f"Upload entry file is missing: {entry.local_path}")
        if entry.local_path.stat().st_size != entry.local_size_bytes:
            raise ValueError(f"Upload entry size changed after planning: {entry.path}")
        if _file_sha256(entry.local_path) != entry.local_sha256:
            raise ValueError(f"Upload entry hash changed after planning: {entry.path}")
    return entries


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> CorpusPublicationRequest:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-plan", required=True, type=Path)
    parser.add_argument("--commit-message", required=True)
    arguments = parser.parse_args()
    return CorpusPublicationRequest(
        upload_plan_path=arguments.upload_plan,
        commit_message=arguments.commit_message,
    )


def main() -> None:
    result = publish_corpus(parse_arguments())
    print(json.dumps(result.model_dump(mode="json"), indent=2), flush=True)


if __name__ == "__main__":
    main()

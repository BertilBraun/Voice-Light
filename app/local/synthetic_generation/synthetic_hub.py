from __future__ import annotations

import hashlib
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol

from huggingface_hub import snapshot_download
from pydantic import Field, StringConstraints, model_validator

from app.local.synthetic_generation.conversation_compiler import ConversationCompilerConfig
from app.local.synthetic_generation.conversation_pipeline import (
    SyntheticConversationCorpusManifest,
    build_conversation_corpus,
)
from app.local.synthetic_generation.conversation_tts import ConversationTtsManifest
from app.local.synthetic_generation.models import SyntheticModel
from app.local.synthetic_generation.synthetic_publication import PublishedUnitArchiveManifest

DEFAULT_SYNTHETIC_HUB_REPOSITORY = "BertilBraun/voice-light-synthetic-audio"
SyntheticRunId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class SnapshotDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: Literal["dataset"],
        revision: str,
        cache_dir: str | Path | None,
        allow_patterns: tuple[str, ...],
    ) -> str: ...


class SyntheticHubPreparationRequest(SyntheticModel):
    repository_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    run_ids: tuple[SyntheticRunId, ...] = Field(min_length=1)
    cache_directory: Path | None
    output_directory: Path
    split_seed: str = Field(min_length=1)
    compiler: ConversationCompilerConfig
    enforce_sampling_gates: bool = True

    @model_validator(mode="after")
    def validate_unique_runs(self) -> SyntheticHubPreparationRequest:
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("Synthetic Hub run IDs must be unique.")
        return self


class PreparedSyntheticHubRun(SyntheticModel):
    run_id: SyntheticRunId
    source_prompt_path: Path
    source_tts_manifest_path: Path
    materialized_directory: Path
    prompt_set_id: str
    conversation_count: int = Field(gt=0)
    crop_count: int = Field(gt=0)


class SyntheticHubPreparationManifest(SyntheticModel):
    schema_version: Literal["voice-light-synthetic-hub-preparation-v1"] = (
        "voice-light-synthetic-hub-preparation-v1"
    )
    generated_at: datetime
    request: SyntheticHubPreparationRequest
    runs: tuple[PreparedSyntheticHubRun, ...]


def prepare_synthetic_hub_corpora(
    request: SyntheticHubPreparationRequest,
    downloader: SnapshotDownloader = snapshot_download,
) -> SyntheticHubPreparationManifest:
    manifest_path = request.output_directory / "synthetic-hub-preparation.json"
    if manifest_path.exists():
        raise ValueError(f"Synthetic preparation manifest already exists: {manifest_path}")
    snapshot_root = Path(
        downloader(
            repo_id=request.repository_id,
            repo_type="dataset",
            revision=request.revision,
            cache_dir=request.cache_directory,
            allow_patterns=tuple(
                pattern
                for run_id in request.run_ids
                for pattern in (
                    f"runs/{run_id}/prompts.json",
                    f"runs/{run_id}/units/render.json",
                    f"runs/{run_id}/units/unit-shards.json",
                    f"runs/{run_id}/units/shards/*.tar",
                )
            ),
        )
    )
    prepared_runs = tuple(
        _prepare_run(
            request=request,
            snapshot_root=snapshot_root,
            run_id=run_id,
        )
        for run_id in request.run_ids
    )
    manifest = SyntheticHubPreparationManifest(
        generated_at=datetime.now(UTC),
        request=request,
        runs=prepared_runs,
    )
    request.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def _prepare_run(
    request: SyntheticHubPreparationRequest,
    snapshot_root: Path,
    run_id: SyntheticRunId,
) -> PreparedSyntheticHubRun:
    source_directory = snapshot_root / "runs" / run_id
    prompt_path = source_directory / "prompts.json"
    downloaded_tts_manifest_path = source_directory / "units" / "render.json"
    if not prompt_path.is_file():
        raise ValueError(f"Synthetic Hub run has no prompt set: {prompt_path}")
    if not downloaded_tts_manifest_path.is_file():
        raise ValueError(f"Synthetic Hub run has no TTS manifest: {downloaded_tts_manifest_path}")
    materialized_directory = request.output_directory / run_id
    if materialized_directory.exists():
        raise ValueError(
            f"Synthetic materialization destination already exists: {materialized_directory}"
        )
    tts_manifest_path = extract_synthetic_unit_archives(
        downloaded_units_directory=source_directory / "units",
        destination=request.output_directory / ".sources" / run_id / "units",
    )
    corpus = build_conversation_corpus(
        prompt_set_path=prompt_path,
        tts_manifest_path=tts_manifest_path,
        output_directory=materialized_directory,
        split_seed=f"{request.split_seed}:{run_id}",
        compiler_config=request.compiler,
        enforce_sampling_gates=request.enforce_sampling_gates,
    )
    return _prepared_run(
        run_id=run_id,
        prompt_path=prompt_path,
        tts_manifest_path=tts_manifest_path,
        materialized_directory=materialized_directory,
        corpus=corpus,
    )


def _prepared_run(
    run_id: SyntheticRunId,
    prompt_path: Path,
    tts_manifest_path: Path,
    materialized_directory: Path,
    corpus: SyntheticConversationCorpusManifest,
) -> PreparedSyntheticHubRun:
    return PreparedSyntheticHubRun(
        run_id=run_id,
        source_prompt_path=prompt_path,
        source_tts_manifest_path=tts_manifest_path,
        materialized_directory=materialized_directory,
        prompt_set_id=corpus.prompt_set_id,
        conversation_count=len(corpus.conversations),
        crop_count=corpus.sampling_summary.total_crop_count,
    )


def extract_synthetic_unit_archives(
    downloaded_units_directory: Path,
    destination: Path,
) -> Path:
    archive_manifest_path = downloaded_units_directory / "unit-shards.json"
    if not archive_manifest_path.is_file():
        raise ValueError(f"Synthetic Hub run has no unit archive manifest: {archive_manifest_path}")
    archive_manifest = PublishedUnitArchiveManifest.model_validate_json(
        archive_manifest_path.read_text(encoding="utf-8")
    )
    tts_manifest_source = downloaded_units_directory / "render.json"
    tts_manifest = ConversationTtsManifest.model_validate_json(
        tts_manifest_source.read_text(encoding="utf-8")
    )
    expected_paths = {unit.clip.audio_path for unit in tts_manifest.rendered_units}
    archived_paths = {
        audio_path for archive in archive_manifest.archives for audio_path in archive.audio_paths
    }
    if archived_paths != expected_paths:
        raise ValueError("Synthetic unit archives do not exactly cover the TTS manifest.")
    destination.mkdir(parents=True)
    extracted_paths: set[Path] = set()
    for archive_record in archive_manifest.archives:
        archive_path = downloaded_units_directory / archive_record.path
        if archive_path.stat().st_size != archive_record.size_bytes:
            raise ValueError(f"Synthetic unit archive size changed: {archive_path}")
        if _file_sha256(archive_path) != archive_record.sha256:
            raise ValueError(f"Synthetic unit archive hash changed: {archive_path}")
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if (
                    not member.isfile()
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or member_path not in expected_paths
                    or member_path in extracted_paths
                ):
                    raise ValueError(f"Invalid synthetic unit archive member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Synthetic unit archive member has no content: {member.name}")
                target = destination / member_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                extracted_paths.add(member_path)
    if extracted_paths != expected_paths:
        raise ValueError("Synthetic unit archive extraction is incomplete.")
    tts_manifest_path = destination / "render.json"
    shutil.copy2(tts_manifest_source, tts_manifest_path)
    return tts_manifest_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

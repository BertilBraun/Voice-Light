from __future__ import annotations

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
from app.local.synthetic_generation.models import SyntheticModel

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
                    f"runs/{run_id}/units/audio/*.flac",
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
    tts_manifest_path = source_directory / "units" / "render.json"
    if not prompt_path.is_file():
        raise ValueError(f"Synthetic Hub run has no prompt set: {prompt_path}")
    if not tts_manifest_path.is_file():
        raise ValueError(f"Synthetic Hub run has no TTS manifest: {tts_manifest_path}")
    materialized_directory = request.output_directory / run_id
    if materialized_directory.exists():
        raise ValueError(
            f"Synthetic materialization destination already exists: {materialized_directory}"
        )
    corpus = build_conversation_corpus(
        prompt_set_path=prompt_path,
        tts_manifest_path=tts_manifest_path,
        output_directory=materialized_directory,
        split_seed=f"{request.split_seed}:{run_id}",
        compiler_config=request.compiler,
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

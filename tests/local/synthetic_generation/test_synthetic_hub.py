from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.conversation_compiler import ConversationCompilerConfig
from app.local.synthetic_generation.conversation_pipeline import (
    CompiledConversationReference,
    CropSamplingSummary,
    SyntheticConversationCorpusManifest,
)
from app.local.synthetic_generation.synthetic_hub import (
    SyntheticHubPreparationRequest,
    prepare_synthetic_hub_corpora,
)
from app.local.training_corpus.splits import (
    ConversationSplitAssignment,
    ConversationSplitPlan,
    TrainingCorpusSplit,
)


def test_prepare_hub_corpora_downloads_only_canonical_units_and_materializes_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    observed_patterns: list[tuple[str, ...]] = []

    def download_snapshot(
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        cache_dir: str | Path | None,
        allow_patterns: tuple[str, ...],
    ) -> str:
        assert repo_id == "test/synthetic"
        assert repo_type == "dataset"
        assert revision == "1" * 40
        assert cache_dir == tmp_path / "cache"
        observed_patterns.append(allow_patterns)
        for run_id in ("v4", "v5"):
            (snapshot_root / "runs" / run_id / "units").mkdir(parents=True)
            (snapshot_root / "runs" / run_id / "prompts.json").write_text("{}")
            (snapshot_root / "runs" / run_id / "units" / "render.json").write_text("{}")
        return str(snapshot_root)

    observed_outputs: list[Path] = []

    def extract_archives(downloaded_units_directory: Path, destination: Path) -> Path:
        assert downloaded_units_directory.name == "units"
        destination.mkdir(parents=True)
        path = destination / "render.json"
        path.write_text("{}")
        return path

    def build_corpus(
        prompt_set_path: Path,
        tts_manifest_path: Path,
        output_directory: Path,
        split_seed: str,
        compiler_config: ConversationCompilerConfig,
        enforce_sampling_gates: bool = True,
    ) -> SyntheticConversationCorpusManifest:
        del prompt_set_path, tts_manifest_path, compiler_config
        assert not enforce_sampling_gates
        assert split_seed in {"synthetic-seed:v4", "synthetic-seed:v5"}
        observed_outputs.append(output_directory)
        return _corpus(output_directory.name)

    monkeypatch.setattr(
        "app.local.synthetic_generation.synthetic_hub.build_conversation_corpus",
        build_corpus,
    )
    monkeypatch.setattr(
        "app.local.synthetic_generation.synthetic_hub.extract_synthetic_unit_archives",
        extract_archives,
    )
    request = SyntheticHubPreparationRequest(
        repository_id="test/synthetic",
        revision="1" * 40,
        run_ids=("v4", "v5"),
        cache_directory=tmp_path / "cache",
        output_directory=tmp_path / "prepared",
        split_seed="synthetic-seed",
        compiler=ConversationCompilerConfig(crop_variant_count=4),
        enforce_sampling_gates=False,
    )

    manifest = prepare_synthetic_hub_corpora(request, downloader=download_snapshot)

    assert observed_patterns == [
        (
            "runs/v4/prompts.json",
            "runs/v4/units/render.json",
            "runs/v4/units/unit-shards.json",
            "runs/v4/units/shards/*.tar",
            "runs/v5/prompts.json",
            "runs/v5/units/render.json",
            "runs/v5/units/unit-shards.json",
            "runs/v5/units/shards/*.tar",
        )
    ]
    assert observed_outputs == [tmp_path / "prepared" / "v4", tmp_path / "prepared" / "v5"]
    assert tuple(run.crop_count for run in manifest.runs) == (10, 10)
    assert (tmp_path / "prepared" / "synthetic-hub-preparation.json").is_file()


def test_synthetic_hub_request_rejects_duplicate_runs(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="run IDs must be unique"):
        SyntheticHubPreparationRequest(
            repository_id="test/synthetic",
            revision="1" * 40,
            run_ids=("v4", "v4"),
            cache_directory=None,
            output_directory=tmp_path,
            split_seed="synthetic-seed",
            compiler=ConversationCompilerConfig(),
        )


def _corpus(run_id: str) -> SyntheticConversationCorpusManifest:
    dataset_id = uuid4()
    return SyntheticConversationCorpusManifest(
        prompt_set_id=f"synthetic_{run_id}",
        prompt_set_sha256="1" * 64,
        tts_manifest_sha256="2" * 64,
        dataset_id=dataset_id,
        split_plan=ConversationSplitPlan(
            seed="synthetic-seed",
            assignments=(
                ConversationSplitAssignment(
                    dataset_id=dataset_id,
                    sample_id=uuid4(),
                    split=TrainingCorpusSplit.TRAIN,
                ),
            ),
        ),
        conversations=(
            CompiledConversationReference(
                conversation_id="pilot_1",
                split=TrainingCorpusSplit.TRAIN,
                source_duration_seconds=60.0,
                plan_path="conversations/pilot_1/composition.json",
                manifest_path="conversations/pilot_1/compiled.json",
                crop_audio_paths=("conversations/pilot_1/crops/000.flac",),
            ),
        ),
        sampling_summary=CropSamplingSummary(
            total_crop_count=10,
            event_focused_count=7,
            assistant_only_count=1,
            user_only_count=1,
            event_light_count=1,
            padded_count=0,
            assistant_only_fraction=0.1,
            user_only_fraction=0.1,
            event_light_fraction=0.1,
            padded_fraction=0.0,
            control_quotas_satisfied=True,
            padding_limit_satisfied=True,
        ),
    )

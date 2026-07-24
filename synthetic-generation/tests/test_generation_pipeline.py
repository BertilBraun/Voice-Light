from __future__ import annotations

from pathlib import Path

from src.backends.fake import FakeBackend
from src.export_mixes import export_all_mixes
from src.generate_samples import generate_experiment, load_case_file
from src.interaction_cases import DEFAULT_CASES
from src.models import BackendConfig
from src.package_experiment import create_archive, load_manifest, validate_experiment


def test_fake_backend_manifest_records_success_and_failure(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "outputs" / "fake_experiment"
    backend_config = BackendConfig(
        backend="fake",
        model="fake-tone",
        options={"fail_variant": 2},
    )
    backend = FakeBackend(backend_config.model, backend_config.options)
    backend.load()

    manifest = generate_experiment(
        backend=backend,
        backend_config=backend_config,
        case_file=DEFAULT_CASES,
        experiment_id="fake_experiment",
        output_dir=experiment_dir,
        variants=3,
        seed=42,
    )

    first_case = manifest.cases[0]
    speaker_a_variants = first_case.speaker_a.variants
    assert [variant.id for variant in speaker_a_variants] == [
        "variant_001",
        "variant_002",
        "variant_003",
    ]
    assert speaker_a_variants[0].status == "success"
    assert speaker_a_variants[0].audio_url == (
        "cases/corrective_interruption/speaker_a/variant_001.wav"
    )
    assert speaker_a_variants[0].raw_audio_url == (
        "cases/corrective_interruption/speaker_a/raw/variant_001.wav"
    )
    assert speaker_a_variants[0].audio_onset_seconds == 0.0
    assert speaker_a_variants[0].audio_offset_seconds is not None
    assert speaker_a_variants[0].original_duration_seconds is not None
    assert speaker_a_variants[0].trimmed_leading_seconds is not None
    assert speaker_a_variants[0].trimmed_trailing_seconds is not None
    assert speaker_a_variants[1].status == "failed"
    assert speaker_a_variants[1].audio_url is None
    assert speaker_a_variants[1].error == "Simulated fake backend failure."
    assert speaker_a_variants[2].seed == 44


def test_package_validation_accepts_generated_fake_experiment(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "outputs" / "fake_experiment"
    backend_config = BackendConfig(backend="fake", model="fake-tone")
    backend = FakeBackend(backend_config.model, backend_config.options)
    backend.load()
    manifest = generate_experiment(
        backend=backend,
        backend_config=backend_config,
        case_file=DEFAULT_CASES,
        experiment_id="fake_experiment",
        output_dir=experiment_dir,
        variants=2,
        seed=10,
    )
    manifest_path = experiment_dir / "experiment.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    loaded_manifest = load_manifest(experiment_dir)
    validate_experiment(experiment_dir, loaded_manifest)
    archive_path = experiment_dir.with_suffix(".zip")
    create_archive(experiment_dir, archive_path)

    assert archive_path.exists()


def test_minimal_llm_case_file_populates_case_ids_and_typed_completion_placement(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        """
{
  "cases": [
    {
      "speaker_a": {"text": "I finished the review."},
      "speaker_b": {"text": "Great, thanks."},
      "placement": {"type": "completion", "pause": "short"}
    }
  ]
}
""",
        encoding="utf-8",
    )

    case_file = load_case_file(cases_path)

    assert case_file.cases[0].case_id == "case_001"
    assert case_file.cases[0].title == "Case 001"
    assert case_file.cases[0].placement is not None
    assert case_file.cases[0].placement.type == "completion"
    assert case_file.cases[0].placement.pause == "short"


def test_typed_completion_placement_generates_typed_output(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        """
{
  "cases": [
    {
      "speaker_a": {"text": "The export finished cleanly."},
      "speaker_b": {"text": "Great, I'll send it to the client."},
      "placement": {"type": "completion", "pause": "short"}
    }
  ]
}
""",
        encoding="utf-8",
    )
    case_file = load_case_file(cases_path)
    experiment_dir = tmp_path / "outputs" / "fake_completion"
    backend_config = BackendConfig(backend="fake", model="fake-tone")
    backend = FakeBackend(backend_config.model, backend_config.options)
    backend.load()

    manifest = generate_experiment(
        backend=backend,
        backend_config=backend_config,
        case_file=case_file,
        experiment_id="fake_completion",
        output_dir=experiment_dir,
        variants=1,
        seed=100,
    )

    speaker_a_variant = manifest.cases[0].speaker_a.variants[0]
    assert manifest.cases[0].placement is not None
    assert manifest.cases[0].placement.type == "completion"
    assert speaker_a_variant.placement_output is not None
    assert speaker_a_variant.placement_output.type == "completion"
    assert speaker_a_variant.suggested_b_offset_seconds == (
        speaker_a_variant.placement_output.speaker_b_start_seconds
    )


def test_export_all_mixes_writes_every_successful_variant_pair(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        """
{
  "cases": [
    {
      "speaker_a": {"text": "The package is ready for review."},
      "speaker_b": {"text": "Great, I'll send it over."},
      "placement": {"type": "completion", "pause": "short"}
    }
  ]
}
""",
        encoding="utf-8",
    )
    case_file = load_case_file(cases_path)
    experiment_dir = tmp_path / "outputs" / "fake_mixes"
    backend_config = BackendConfig(backend="fake", model="fake-tone")
    backend = FakeBackend(backend_config.model, backend_config.options)
    backend.load()
    manifest = generate_experiment(
        backend=backend,
        backend_config=backend_config,
        case_file=case_file,
        experiment_id="fake_mixes",
        output_dir=experiment_dir,
        variants=2,
        seed=200,
    )
    manifest_path = experiment_dir / "experiment.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    results = export_all_mixes(
        experiment_dir=experiment_dir,
        manifest=manifest,
        output_dir=experiment_dir / "mixes",
        speaker_a_gain=1.0,
        speaker_b_gain=1.0,
        source="trimmed",
    )
    raw_results = export_all_mixes(
        experiment_dir=experiment_dir,
        manifest=manifest,
        output_dir=experiment_dir / "raw_mixes",
        speaker_a_gain=1.0,
        speaker_b_gain=1.0,
        source="raw",
    )

    assert len(results) == 4
    assert len(raw_results) == 4
    assert all(result.mix_path.exists() for result in results)
    assert all(result.mix_path.exists() for result in raw_results)
    assert all(result.metadata_path.exists() for result in results)
    assert (experiment_dir / "mixes" / "index.json").exists()
    assert (experiment_dir / "raw_mixes" / "index.json").exists()

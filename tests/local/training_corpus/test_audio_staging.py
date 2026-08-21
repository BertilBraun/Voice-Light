from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import pytest

from app.local.training_corpus.audio_staging import (
    CorpusAudioPreparation,
    CorpusAudioVerification,
    ExistingFlacStagingRequest,
    LocalSourceAudio,
    WaveToFlacStagingRequest,
    corpus_track_relative_path,
    decoded_pcm_sha256,
    deterministic_verification_indices,
    stage_dataset_audio,
    transcode_lossless_flac,
    validate_decoded_pcm_equal,
)
from app.shared.quality import SpeakerSide


def test_wave_staging_preserves_sources_and_samples_only_two_pcm_checks(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_samples(source_root, sample_count=2, extension=".wav")
    corpus_root = tmp_path / "corpus"
    original_source_hashes = source_hashes(source_root)

    manifest = stage_dataset_audio(
        WaveToFlacStagingRequest(
            dataset_name="dataset_2",
            source_samples_root=source_root,
            corpus_root=corpus_root,
            exact_verification_file_count=2,
        )
    )

    assert source_hashes(source_root) == original_source_hashes
    assert len(manifest.assets) == 4
    assert [asset.verification for asset in manifest.assets] == [
        CorpusAudioVerification.DECODED_PCM,
        CorpusAudioVerification.METADATA,
        CorpusAudioVerification.METADATA,
        CorpusAudioVerification.DECODED_PCM,
    ]
    assert all(
        asset.preparation is CorpusAudioPreparation.LOSSLESS_FLAC_TRANSCODE
        for asset in manifest.assets
    )
    assert all(asset.source_sha256 != asset.corpus_sha256 for asset in manifest.assets)
    assert (corpus_root / "dataset_2" / "samples" / "sample_001" / "speaker_1.flac").is_file()
    build_manifest = json.loads(
        (corpus_root / ".build" / "dataset_2-audio-assets.json").read_text(encoding="utf-8")
    )
    assert len(build_manifest["assets"]) == 4


def test_existing_flac_staging_uses_hard_links(tmp_path: Path) -> None:
    wave_root = tmp_path / "wave"
    write_samples(wave_root, sample_count=1, extension=".wav")
    source_root = tmp_path / "source"
    transcode_samples(wave_root, source_root)
    corpus_root = tmp_path / "corpus"

    manifest = stage_dataset_audio(
        ExistingFlacStagingRequest(
            dataset_name="dataset_3",
            source_samples_root=source_root,
            corpus_root=corpus_root,
        )
    )

    assert len(manifest.assets) == 2
    for asset in manifest.assets:
        staged_path = corpus_root.joinpath(*asset.corpus_relative_path.parts)
        assert isinstance(asset.source, LocalSourceAudio)
        source_path = source_root.joinpath(*asset.source.sample_relative_path.parts)
        assert os.path.samefile(source_path, staged_path)
        assert asset.source_sha256 == asset.corpus_sha256
        assert asset.verification is CorpusAudioVerification.HARD_LINK_IDENTITY


def test_staging_rejects_existing_dataset_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_samples(source_root, sample_count=1, extension=".wav")
    corpus_root = tmp_path / "corpus"
    (corpus_root / "dataset_2").mkdir(parents=True)

    with pytest.raises(ValueError, match="already exists"):
        stage_dataset_audio(
            WaveToFlacStagingRequest(
                dataset_name="dataset_2",
                source_samples_root=source_root,
                corpus_root=corpus_root,
            )
        )


def test_decoded_pcm_comparison_detects_different_audio(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    write_wave(first, value=10)
    write_wave(second, value=20)

    assert decoded_pcm_sha256(first) != decoded_pcm_sha256(second)
    with pytest.raises(ValueError, match="Decoded PCM differs"):
        validate_decoded_pcm_equal(first, second)


@pytest.mark.parametrize(
    ("item_count", "verification_count", "expected"),
    [
        (0, 2, frozenset()),
        (4, 0, frozenset()),
        (4, 1, frozenset((0,))),
        (4, 2, frozenset((0, 3))),
    ],
)
def test_deterministic_verification_indices(
    item_count: int,
    verification_count: int,
    expected: frozenset[int],
) -> None:
    assert deterministic_verification_indices(item_count, verification_count) == expected


def test_corpus_path_uses_existing_generic_filename_convention() -> None:
    assert (
        corpus_track_relative_path(
            dataset_name="dataset_2",
            sample_id="sample_008",
            side=SpeakerSide.SPEAKER2,
        ).as_posix()
        == "dataset_2/samples/sample_008/speaker_2.flac"
    )


@pytest.mark.parametrize("invalid_component", ["", ".", "..", "nested/sample"])
def test_corpus_path_rejects_noncanonical_components(invalid_component: str) -> None:
    with pytest.raises(ValueError, match="one non-relative path component"):
        corpus_track_relative_path(
            dataset_name="dataset_2",
            sample_id=invalid_component,
            side=SpeakerSide.SPEAKER1,
        )


def write_samples(root: Path, sample_count: int, extension: str) -> None:
    for sample_index in range(1, sample_count + 1):
        sample_root = root / f"sample_{sample_index:03d}"
        sample_root.mkdir(parents=True)
        (sample_root / "metadata.json").write_text("{}\n", encoding="utf-8")
        for speaker_index in (1, 2):
            path = sample_root / f"speaker_{speaker_index}{extension}"
            write_wave(path, value=sample_index * speaker_index)


def write_wave(path: Path, value: int) -> None:
    with wave.open(str(path), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(16_000)
        wave_file.writeframes(int(value).to_bytes(2, byteorder="little", signed=True) * 1600)


def source_hashes(root: Path) -> tuple[str, ...]:
    return tuple(decoded_pcm_sha256(path) for path in sorted(root.rglob("*.wav")))


def transcode_samples(wave_root: Path, flac_root: Path) -> None:
    for source_path in sorted(wave_root.rglob("*.wav")):
        relative_path = source_path.relative_to(wave_root).with_suffix(".flac")
        output_path = flac_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transcode_lossless_flac(source_path, output_path)
    for metadata_path in wave_root.rglob("metadata.json"):
        output_path = flac_root / metadata_path.relative_to(wave_root)
        output_path.write_text("{}\n", encoding="utf-8")

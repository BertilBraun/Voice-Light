from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.local.conversation_regions.models import ConversationRegionAnalysis
from app.local.db.models import TrackSide
from app.local.training_corpus.audio_staging import (
    AUDIO_STAGING_SCHEMA_VERSION,
    CorpusAudioAsset,
    CorpusAudioPreparation,
    CorpusAudioStagingManifest,
    CorpusAudioVerification,
    LocalSourceAudio,
    RemoteSourceAudio,
)
from app.local.training_corpus.export import (
    AudioReference,
    ExportManifest,
    ExportShard,
    ExportSplitSummary,
    MaterializedTrainingSample,
    RecordingMetadata,
)
from app.local.training_corpus.splits import (
    ConversationSplitAssignment,
    ConversationSplitPlan,
    TrainingCorpusSplit,
)
from app.local.training_corpus.validation import (
    ExportedCorpusValidationRequest,
    RecordingKey,
    validate_exported_corpus,
)
from app.shared.quality import AudioMetadata, ConversationAnnotation, SpeakerSide, TrackVadResult

DATASET_ID = UUID("11111111-1111-1111-1111-111111111111")
SAMPLE_ID = UUID("22222222-2222-2222-2222-222222222222")
GENERATED_AT = datetime(2026, 8, 21, tzinfo=UTC)
LOCAL_HASH = hashlib.sha256(b"local-flac").hexdigest()
HUB_HASH = "b" * 64


def test_validator_checks_split_export_and_reports_audio_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _write_corpus(tmp_path, (_sample(),))
    _use_metadata(monkeypatch)

    report = validate_exported_corpus(request)

    assert report.recording_count == 1
    assert report.training_sample_count == 1
    assert report.local_audio_file_count == 1
    assert report.hub_audio_reference_count == 1
    train = next(item for item in report.concentrations if item.split is TrainingCorpusSplit.TRAIN)
    assert train.recording_count == 1
    assert train.window_count == 1
    assert train.maximum_recording_concentration == 1.0


def test_validator_rejects_changed_shard_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _write_corpus(tmp_path, (_sample(),))
    _use_metadata(monkeypatch)
    shard_path = tmp_path / "corpus" / "training" / "train" / "shard-00000.parquet"
    shard_path.write_bytes(shard_path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="Shard size does not match manifest"):
        validate_exported_corpus(request)


def test_validator_rejects_duplicate_window_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _sample()
    second = first.model_copy(
        update={
            "user_side": TrackSide.SPEAKER2,
            "assistant_side": TrackSide.SPEAKER1,
            "user_audio_path": "dataset/recording/speaker_2.flac",
            "assistant_audio_path": "dataset/recording/speaker_1.flac",
        }
    )
    request = _write_corpus(tmp_path, (first, second))
    _use_metadata(monkeypatch)

    with pytest.raises(ValueError, match="duplicate window IDs"):
        validate_exported_corpus(request)


def test_validator_rejects_duplicate_logical_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _sample()
    second = first.model_copy(update={"window_id": "d" * 64})
    request = _write_corpus(tmp_path, (first, second))
    _use_metadata(monkeypatch)

    with pytest.raises(ValueError, match="duplicate logical training windows"):
        validate_exported_corpus(request)


def test_validator_rejects_sample_in_wrong_split_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = _sample().model_copy(update={"split": TrainingCorpusSplit.VALIDATION})
    request = _write_corpus(
        tmp_path,
        (sample,),
        shard_split=TrainingCorpusSplit.TRAIN,
        metadata_split=TrainingCorpusSplit.VALIDATION,
    )
    _use_metadata(monkeypatch, split=TrainingCorpusSplit.VALIDATION)

    with pytest.raises(ValueError, match="wrong split shard"):
        validate_exported_corpus(request)


def _write_corpus(
    root: Path,
    samples: tuple[MaterializedTrainingSample, ...],
    shard_split: TrainingCorpusSplit = TrainingCorpusSplit.TRAIN,
    metadata_split: TrainingCorpusSplit = TrainingCorpusSplit.TRAIN,
) -> ExportedCorpusValidationRequest:
    corpus_directory = root / "corpus"
    local_audio = corpus_directory / "dataset" / "recording" / "speaker_1.flac"
    local_audio.parent.mkdir(parents=True)
    local_audio.write_bytes(b"local-flac")
    asset_manifest = _audio_manifest(local_audio)
    asset_manifest_path = root / "audio-assets.json"
    asset_manifest_path.write_text(asset_manifest.model_dump_json(), encoding="utf-8")
    shard_path = corpus_directory / "training" / "train" / "shard-00000.parquet"
    shard_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([sample.model_dump(mode="json") for sample in samples]), shard_path
    )
    shard = ExportShard(
        split=shard_split,
        path="training/train/shard-00000.parquet",
        row_count=len(samples),
        size_bytes=shard_path.stat().st_size,
        sha256=_sha256(shard_path),
    )
    manifest = _manifest(samples, shard, metadata_split)
    (corpus_directory / "corpus.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return ExportedCorpusValidationRequest(
        corpus_directory=corpus_directory,
        audio_build_manifest_paths=(asset_manifest_path,),
    )


def _manifest(
    samples: tuple[MaterializedTrainingSample, ...],
    shard: ExportShard,
    metadata_split: TrainingCorpusSplit,
) -> ExportManifest:
    return ExportManifest(
        schema_version="voice-light-turn-taking-v1",
        generated_at=GENERATED_AT,
        metric_version="quality-v6",
        annotation_version="annotation-v1",
        region_analysis_version="regions-v1",
        training_label_version="labels-v1",
        input_duration_seconds=20.0,
        frame_seconds=0.08,
        review_set_name="prepublish-v1",
        split_plan=ConversationSplitPlan(
            seed="test-seed",
            assignments=(
                ConversationSplitAssignment(
                    dataset_id=DATASET_ID,
                    sample_id=SAMPLE_ID,
                    split=metadata_split,
                ),
            ),
        ),
        recording_count=1,
        training_sample_count=len(samples),
        splits=tuple(
            ExportSplitSummary(
                split=split,
                recording_count=int(split is metadata_split),
                training_sample_count=sum(sample.split is split for sample in samples),
                source_duration_seconds=20.0 if split is metadata_split else 0.0,
            )
            for split in TrainingCorpusSplit
        ),
        shards=(shard,),
    )


def _sample() -> MaterializedTrainingSample:
    labels = (0.5,) * 250
    return MaterializedTrainingSample(
        schema_version="voice-light-turn-taking-v1",
        training_label_version="labels-v1",
        window_id="a" * 64,
        dataset_id=DATASET_ID,
        dataset_name="dataset",
        sample_id=SAMPLE_ID,
        external_id="recording",
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=TrainingCorpusSplit.TRAIN,
        user_audio_path="dataset/recording/speaker_1.flac",
        assistant_audio_path="dataset/recording/speaker_2.flac",
        start_seconds=0.0,
        end_seconds=20.0,
        quality_score=0.99,
        category="turn_shift",
        assistant_has_floor=labels,
        p_user_has_floor=labels,
        p_user_yield=labels,
        p_assistant_backchannel=labels,
        future_activity_0_200=labels,
        future_activity_200_500=labels,
        future_activity_500_1000=labels,
        future_activity_1000_1500=labels,
        turn_completion=labels,
        continuation_pause=labels,
        non_floor_feedback=labels,
        floor_take=labels,
    )


def _audio_manifest(local_audio: Path) -> CorpusAudioStagingManifest:
    audio = AudioMetadata(
        duration_seconds=20.0,
        sample_rate=16_000,
        channels=1,
        sample_count=320_000,
    )
    return CorpusAudioStagingManifest(
        schema_version=AUDIO_STAGING_SCHEMA_VERSION,
        generated_at=GENERATED_AT,
        dataset_name="dataset",
        assets=(
            CorpusAudioAsset(
                sample_id="recording",
                side=SpeakerSide.SPEAKER1,
                source=LocalSourceAudio(path=local_audio),
                source_sha256=LOCAL_HASH,
                corpus_relative_path=PurePosixPath("dataset/recording/speaker_1.flac"),
                corpus_sha256=LOCAL_HASH,
                source_audio=audio,
                corpus_audio=audio,
                preparation=CorpusAudioPreparation.HARD_LINKED_FLAC,
                verification=CorpusAudioVerification.HARD_LINK_IDENTITY,
            ),
            CorpusAudioAsset(
                sample_id="recording",
                side=SpeakerSide.SPEAKER2,
                source=RemoteSourceAudio(uri="s3://source/recording/speaker_2.flac"),
                source_sha256=HUB_HASH,
                corpus_relative_path=PurePosixPath("dataset/recording/speaker_2.flac"),
                corpus_sha256=HUB_HASH,
                source_audio=audio,
                corpus_audio=audio,
                preparation=CorpusAudioPreparation.EXISTING_HUB_FLAC,
                verification=CorpusAudioVerification.HUB_LFS_SHA256,
            ),
        ),
    )


def _use_metadata(
    monkeypatch: pytest.MonkeyPatch,
    split: TrainingCorpusSplit = TrainingCorpusSplit.TRAIN,
) -> None:
    metadata = _metadata(split)
    monkeypatch.setattr(
        "app.local.training_corpus.validation._load_recording_metadata",
        lambda _: {RecordingKey(DATASET_ID, SAMPLE_ID): metadata},
    )


def _metadata(split: TrainingCorpusSplit) -> RecordingMetadata:
    audio = (
        AudioReference(
            side=TrackSide.SPEAKER1,
            path="dataset/recording/speaker_1.flac",
            duration_seconds=20.0,
            sample_rate=16_000,
            channels=1,
            audio_sha256=LOCAL_HASH,
        ),
        AudioReference(
            side=TrackSide.SPEAKER2,
            path="dataset/recording/speaker_2.flac",
            duration_seconds=20.0,
            sample_rate=16_000,
            channels=1,
            audio_sha256=HUB_HASH,
        ),
    )
    return RecordingMetadata.model_construct(
        schema_version="voice-light-turn-taking-v1",
        dataset_name="dataset",
        dataset_id=DATASET_ID,
        sample_id=SAMPLE_ID,
        external_id="recording",
        duration_seconds=20.0,
        quality_score=0.99,
        metric_version="quality-v6",
        annotation=ConversationAnnotation.model_construct(
            annotation_version="annotation-v1",
            analyzed_duration_seconds=20.0,
        ),
        speaker1_vad=TrackVadResult.model_construct(speech_segments=()),
        speaker2_vad=TrackVadResult.model_construct(speech_segments=()),
        conversation_regions=ConversationRegionAnalysis.model_construct(
            analysis_version="regions-v1",
            annotation_version="annotation-v1",
            duration_seconds=20.0,
        ),
        review_interval_exclusions=(),
        split=split,
        audio=audio,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

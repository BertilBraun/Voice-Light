from __future__ import annotations

import inspect
import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

import app.local.ingestion.service as ingestion_service_module
from app.local.asr.full_recording_models import (
    FullRecordingAsrTranscriptBundle,
    FullRecordingAsrTranscriptPair,
    FullRecordingAsrTranscriptRecord,
)
from app.local.db.models import SampleRecord, SampleTrackRecord, TrackSide
from app.local.db.repository import AudioMetadataInput, SampleTrackInput
from app.local.ingestion.conversation import ANNOTATION_VERSION, analyze_conversation
from app.local.ingestion.discovery import DiscoveredSample
from app.local.ingestion.language import (
    LANGUAGE_ASSESSMENT_VERSION,
    LanguageTrack,
    SampleLanguageAssessment,
    SampleLanguageStatus,
    TrackLanguageAssessment,
    TrackLanguageStatus,
)
from app.local.ingestion.service import (
    RegisteredSample,
    SampleProcessingStatus,
    process_ingestion_sample,
    ready_sample,
    register_sample,
    track_durations_are_compatible,
)
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.audio import AudioTrack
from app.shared.quality import AudioMetadata
from app.shared.storage.local import LocalStorageBackend


class RegistrationRepository:
    def __init__(self) -> None:
        self.sample_id = uuid4()
        self.track_inputs: list[SampleTrackInput] = []

    def upsert_sample(self, dataset_id: UUID, external_id: str) -> SampleRecord:
        now = datetime.now(tz=UTC)
        return SampleRecord(
            id=self.sample_id,
            dataset_id=dataset_id,
            external_id=external_id,
            duration_seconds=None,
            quality_score=None,
            quality_flags=(),
            is_unusable=False,
            created_at=now,
            updated_at=now,
        )

    def update_sample_duration(
        self,
        sample_id: UUID,
        duration_seconds: float,
    ) -> SampleRecord:
        assert sample_id == self.sample_id
        assert duration_seconds > 0.0
        return self.upsert_sample(dataset_id=uuid4(), external_id="sample_001")

    def upsert_sample_track(
        self,
        sample_id: UUID,
        track: SampleTrackInput,
    ) -> SampleTrackRecord:
        assert sample_id == self.sample_id
        self.track_inputs.append(track)
        now = datetime.now(tz=UTC)
        return SampleTrackRecord(
            id=uuid4(),
            sample_id=sample_id,
            side=track.side,
            speaker_index=track.speaker_index,
            storage_uri=track.storage_uri,
            access_uri=track.access_uri,
            duration_seconds=track.duration_seconds,
            sample_rate=track.sample_rate,
            channels=track.channels,
            sample_count=track.sample_count,
            audio_sha256=track.audio_sha256,
            created_at=now,
            updated_at=now,
        )

    def upsert_audio_metadata(self, metadata: AudioMetadataInput) -> None:
        del metadata


class ReadinessFullAsrRepository:
    def __init__(self, transcripts: FullRecordingAsrTranscriptBundle | None) -> None:
        self.transcripts = transcripts
        self.requests: list[tuple[UUID, UUID, str, UUID, str, AsrModelId]] = []

    def get_exact_transcript_pair(
        self,
        sample_id: UUID,
        speaker1_track_id: UUID,
        speaker1_source_audio_sha256: str,
        speaker2_track_id: UUID,
        speaker2_source_audio_sha256: str,
        model_id: AsrModelId,
    ) -> FullRecordingAsrTranscriptPair | None:
        self.requests.append(
            (
                sample_id,
                speaker1_track_id,
                speaker1_source_audio_sha256,
                speaker2_track_id,
                speaker2_source_audio_sha256,
                model_id,
            )
        )
        if self.transcripts is None:
            return None
        return (
            self.transcripts.parakeet
            if model_id is AsrModelId.PARAKEET_TDT
            else self.transcripts.canary
        )


class NonEnglishLanguagePreflight:
    def assess_sample(
        self,
        speaker1: LanguageTrack,
        speaker2: LanguageTrack,
    ) -> SampleLanguageAssessment:
        return SampleLanguageAssessment(
            status=SampleLanguageStatus.NON_ENGLISH,
            speaker1=language_assessment(
                track=speaker1,
                status=TrackLanguageStatus.ENGLISH,
                language_code="en",
            ),
            speaker2=language_assessment(
                track=speaker2,
                status=TrackLanguageStatus.NON_ENGLISH,
                language_code="de",
            ),
        )


def test_ready_sample_uses_exact_physical_audio_transcripts() -> None:
    registered = registered_sample()
    transcripts = transcript_bundle(registered=registered)

    ready = ready_sample(
        full_asr_repository=ReadinessFullAsrRepository(transcripts=transcripts),
        registered=registered,
    )

    assert ready.registered is registered
    assert ready.transcripts == transcripts


def test_nonzero_stored_review_cannot_influence_production_readiness() -> None:
    stored_review_shift_seconds = -2.5
    registered = registered_sample()
    transcripts = transcript_bundle(registered=registered)

    assert tuple(inspect.signature(ready_sample).parameters) == (
        "full_asr_repository",
        "registered",
    )
    assert stored_review_shift_seconds != 0.0
    ready = ready_sample(
        full_asr_repository=ReadinessFullAsrRepository(transcripts=transcripts),
        registered=registered,
    )
    assert (
        ready.transcripts.parakeet.speaker2.words[0].start_seconds
        == transcripts.parakeet.speaker2.words[0].start_seconds
    )


def test_registration_hashes_each_physical_track_once_and_persists_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker1_path = tmp_path / "sample_001_speaker1.wav"
    speaker2_path = tmp_path / "sample_001_speaker2.wav"
    _write_silent_wave(speaker1_path)
    _write_silent_wave(speaker2_path)
    real_sha256_file = ingestion_service_module.sha256_file
    hashed_paths: list[Path] = []

    def recording_sha256_file(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(ingestion_service_module, "sha256_file", recording_sha256_file)
    repository = RegistrationRepository()

    registered = register_sample(
        repository=repository,
        dataset_id=uuid4(),
        storage=LocalStorageBackend(),
        discovered=DiscoveredSample(
            external_id="sample_001",
            speaker1_path=speaker1_path.as_posix(),
            speaker2_path=speaker2_path.as_posix(),
        ),
    )

    assert hashed_paths == [speaker1_path, speaker2_path]
    assert tuple(track.audio_sha256 for track in repository.track_inputs) == (
        registered.speaker1_source_audio_sha256,
        registered.speaker2_source_audio_sha256,
    )


def test_non_english_preflight_skips_full_asr_and_quality(tmp_path: Path) -> None:
    speaker1_path = tmp_path / "sample_speaker1.wav"
    speaker2_path = tmp_path / "sample_speaker2.wav"
    _write_silent_wave(speaker1_path)
    _write_silent_wave(speaker2_path)
    repository = RegistrationRepository()

    status = process_ingestion_sample(
        repository=repository,
        full_asr_repository=ReadinessFullAsrRepository(transcripts=None),
        dataset_id=uuid4(),
        storage=LocalStorageBackend(),
        discovered=DiscoveredSample(
            external_id="sample",
            speaker1_path=speaker1_path.as_posix(),
            speaker2_path=speaker2_path.as_posix(),
        ),
        remote_asr_client_factory=unexpected_remote_asr_client,
        language_preflight_service=NonEnglishLanguagePreflight(),
    )

    assert status is SampleProcessingStatus.LANGUAGE_EXCLUDED


def test_full_conversation_annotation_uses_words_beyond_three_minutes() -> None:
    registered = registered_sample()
    transcripts = transcript_bundle(registered=registered)
    pair = transcripts.parakeet
    pair = pair.model_copy(
        update={
            "speaker1": pair.speaker1.model_copy(
                update={
                    "words": (
                        TimestampedWord(
                            text="late speaker one",
                            start_seconds=200.0,
                            end_seconds=202.0,
                        ),
                    )
                }
            ),
            "speaker2": pair.speaker2.model_copy(
                update={
                    "words": (
                        TimestampedWord(
                            text="late speaker two",
                            start_seconds=205.0,
                            end_seconds=207.0,
                        ),
                    )
                }
            ),
        }
    )
    audio = silent_audio_track(duration_seconds=240.0)

    annotation = analyze_conversation(
        speaker1_audio=audio,
        speaker2_audio=audio,
        transcripts=transcripts.model_copy(
            update={
                "parakeet": pair,
                "canary": pair.model_copy(update={"model_id": AsrModelId.CANARY}),
            }
        ),
    )

    assert annotation.annotation_version == ANNOTATION_VERSION
    assert annotation.analyzed_duration_seconds == 240.0
    assert any(
        target.start_seconds >= 200.0
        for speaker in (annotation.speaker1, annotation.speaker2)
        for target in speaker.segment_targets
    )


@pytest.mark.parametrize(
    ("speaker1_duration_seconds", "speaker2_duration_seconds", "expected"),
    (
        (1_000.0, 1_010.0, True),
        (1_000.0, 1_010.01, False),
        (2_000.0, 2_020.0, True),
        (2_001.0, 2_021.01, False),
        (10_000.0, 10_020.0, True),
        (10_000.0, 10_020.01, False),
    ),
)
def test_track_duration_compatibility_requires_absolute_and_relative_limits(
    speaker1_duration_seconds: float,
    speaker2_duration_seconds: float,
    expected: bool,
) -> None:
    assert (
        track_durations_are_compatible(
            speaker1_duration_seconds=speaker1_duration_seconds,
            speaker2_duration_seconds=speaker2_duration_seconds,
        )
        is expected
    )


def registered_sample() -> RegisteredSample:
    metadata = AudioMetadata(
        duration_seconds=10.0,
        sample_rate=16_000,
        channels=1,
        sample_count=160_000,
    )
    return RegisteredSample(
        discovered=DiscoveredSample(
            external_id="sample_001",
            speaker1_path="speaker1.wav",
            speaker2_path="speaker2.wav",
        ),
        sample_id=uuid4(),
        speaker1_track_id=uuid4(),
        speaker2_track_id=uuid4(),
        speaker1_metadata=metadata,
        speaker2_metadata=metadata,
        speaker1_source_audio_sha256="a" * 64,
        speaker2_source_audio_sha256="b" * 64,
    )


def transcript_pair(registered: RegisteredSample) -> FullRecordingAsrTranscriptPair:
    return transcript_pair_for_model(
        registered=registered,
        model_id=AsrModelId.PARAKEET_TDT,
    )


def transcript_bundle(registered: RegisteredSample) -> FullRecordingAsrTranscriptBundle:
    return FullRecordingAsrTranscriptBundle(
        parakeet=transcript_pair_for_model(
            registered=registered,
            model_id=AsrModelId.PARAKEET_TDT,
        ),
        canary=transcript_pair_for_model(
            registered=registered,
            model_id=AsrModelId.CANARY,
        ),
    )


def transcript_pair_for_model(
    registered: RegisteredSample,
    model_id: AsrModelId,
) -> FullRecordingAsrTranscriptPair:
    return FullRecordingAsrTranscriptPair(
        sample_id=registered.sample_id,
        sample_external_id=registered.discovered.external_id,
        model_id=model_id,
        speaker1=transcript_record(
            registered=registered,
            side=TrackSide.SPEAKER1,
            track_id=registered.speaker1_track_id,
            model_id=model_id,
        ),
        speaker2=transcript_record(
            registered=registered,
            side=TrackSide.SPEAKER2,
            track_id=registered.speaker2_track_id,
            model_id=model_id,
        ),
    )


def transcript_record(
    registered: RegisteredSample,
    side: TrackSide,
    track_id: UUID,
    model_id: AsrModelId = AsrModelId.PARAKEET_TDT,
) -> FullRecordingAsrTranscriptRecord:
    now = datetime.now(tz=UTC)
    return FullRecordingAsrTranscriptRecord(
        id=uuid4(),
        sample_track_id=track_id,
        sample_id=registered.sample_id,
        sample_external_id=registered.discovered.external_id,
        side=side,
        source_audio_sha256=(
            registered.speaker1_source_audio_sha256
            if side is TrackSide.SPEAKER1
            else registered.speaker2_source_audio_sha256
        ),
        prepared_audio_sha256="c" * 64,
        audio_filename=f"{registered.discovered.external_id}_{side.value}.flac",
        model_id=model_id,
        transcript_text="hello",
        words=(
            TimestampedWord(
                text="hello",
                start_seconds=1.0,
                end_seconds=2.0,
            ),
        ),
        language_estimate=None,
        source_duration_seconds=10.0,
        prepared_duration_seconds=10.0,
        processing_time_seconds=1.0,
        runtime=None,
        error=None,
        created_at=now,
        updated_at=now,
    )


def silent_audio_track(duration_seconds: float) -> AudioTrack:
    sample_rate = 100
    sample_count = round(duration_seconds * sample_rate)
    return AudioTrack(
        samples=np.zeros((sample_count, 1), dtype=np.float32),
        metadata=AudioMetadata(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=1,
            sample_count=sample_count,
        ),
    )


def language_assessment(
    track: LanguageTrack,
    status: TrackLanguageStatus,
    language_code: str,
) -> TrackLanguageAssessment:
    return TrackLanguageAssessment(
        sample_track_id=track.sample_track_id,
        source_audio_sha256=track.source_audio_sha256,
        assessment_version=LANGUAGE_ASSESSMENT_VERSION,
        status=status,
        language_code=language_code,
        confidence=0.99,
        transcript_word_count=4,
        transcript_text="one two three four",
        probe_windows=(),
        error=None,
    )


def unexpected_remote_asr_client() -> None:
    raise AssertionError("Full ASR must not run for a non-English sample.")


def _write_silent_wave(path: Path) -> None:
    with wave.open(str(path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(16_000)
        wave_writer.writeframes(b"\x00\x00" * 160)

from __future__ import annotations

import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from app.local.ingestion.discovery import DiscoveredSample
from app.local.ingestion.service import (
    pending_discovered_samples,
    process_sample,
    remote_audio_source,
)
from app.local.quality.remote_models import (
    LocalAudioSource,
    RemoteQualityRequest,
    UriAudioSource,
)
from app.shared.compute_api import QualityAnalysisResponse
from app.shared.quality import AudioMetadata, ProcessingStatus, QualityResult
from app.shared.storage.local import LocalStorageBackend


@dataclass
class MemoryStorage:
    content: bytes
    uri: str

    def walk(self, root: str) -> Iterator[str]:
        return iter(())

    def open(self, path: str) -> BinaryIO:
        return BytesIO(self.content)

    def read(self, path: str) -> bytes:
        return self.content

    def access_uri(self, path: str) -> str:
        return self.uri


@dataclass
class RecordingQualityClient:
    response: QualityAnalysisResponse
    requests: list[RemoteQualityRequest] = field(default_factory=list)

    def analyze(self, request: RemoteQualityRequest) -> QualityAnalysisResponse:
        self.requests.append(request)
        return self.response


def test_local_audio_source_is_uploaded(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * 16_000)

    source = remote_audio_source(LocalStorageBackend(), str(audio_path))

    assert source == LocalAudioSource(
        filename="audio.wav",
        path=str(audio_path),
        original_metadata=AudioMetadata(
            duration_seconds=1.0,
            sample_rate=16_000,
            channels=1,
            sample_count=16_000,
        ),
    )


def test_presigned_audio_source_is_sent_as_remote_uri() -> None:
    source = remote_audio_source(
        MemoryStorage(b"unused", "https://bucket.example/audio.wav?signature=value"),
        "audio.wav",
    )

    assert source == UriAudioSource(uri="https://bucket.example/audio.wav?signature=value")


def test_process_sample_uses_remote_metadata_and_quality_result() -> None:
    metadata = AudioMetadata(
        duration_seconds=2.0,
        sample_rate=16_000,
        channels=1,
        sample_count=32_000,
    )
    quality_result = failed_quality_result()
    client = RecordingQualityClient(
        QualityAnalysisResponse(
            speaker1_metadata=metadata,
            speaker2_metadata=metadata,
            quality_result=quality_result,
        )
    )

    processed = process_sample(
        MemoryStorage(b"audio", "https://bucket.example/audio.wav"),
        DiscoveredSample("sample", "speaker1.wav", "speaker2.wav"),
        client,
    )

    assert processed.speaker1_metadata == metadata
    assert processed.speaker2_metadata == metadata
    assert processed.quality_result == quality_result
    assert len(client.requests) == 1


def test_pending_discovered_samples_skips_completed_identifiers() -> None:
    samples = [
        DiscoveredSample("completed", "first.wav", "second.wav"),
        DiscoveredSample("pending", "third.wav", "fourth.wav"),
    ]

    assert pending_discovered_samples(samples, {"completed"}) == [samples[1]]


def failed_quality_result() -> QualityResult:
    return QualityResult(
        metric_version="test",
        sample_id="sample",
        status=ProcessingStatus.FAILED,
        speaker1_uri="speaker1",
        speaker2_uri="speaker2",
        duration_seconds=None,
        interaction_density=None,
        timing_reliability=None,
        audio_quality=None,
        conversation_annotation=None,
        conversation_count_estimate=None,
        event_candidates=(),
        raw_quality_score=None,
        quality_flags=(),
        total_quality_score=None,
        error="test",
    )

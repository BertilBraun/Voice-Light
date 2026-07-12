from __future__ import annotations

import base64
from collections.abc import Iterator
from dataclasses import dataclass, field
from io import BytesIO
from typing import BinaryIO

from app.ingestion.discovery import DiscoveredSample
from app.ingestion.service import process_sample, remote_audio_source
from app.quality.models import AudioMetadata, ProcessingStatus, QualityResult
from app.quality.remote_models import (
    RemoteQualityRequest,
    RemoteQualityResponse,
    UploadedAudioSource,
    UriAudioSource,
)


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
    response: RemoteQualityResponse
    requests: list[RemoteQualityRequest] = field(default_factory=list)

    def analyze(self, request: RemoteQualityRequest) -> RemoteQualityResponse:
        self.requests.append(request)
        return self.response


def test_local_audio_source_is_uploaded() -> None:
    source = remote_audio_source(MemoryStorage(b"audio", "C:/audio.wav"), "audio.wav")

    assert isinstance(source, UploadedAudioSource)
    assert source.filename == "audio.wav"
    assert base64.b64decode(source.audio_base64) == b"audio"


def test_presigned_audio_source_is_downloaded_by_modal() -> None:
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
        RemoteQualityResponse(
            speaker1_metadata=metadata,
            speaker2_metadata=metadata,
            quality_result=quality_result,
        )
    )

    processed = process_sample(
        MemoryStorage(b"audio", "C:/audio.wav"),
        DiscoveredSample("sample", "speaker1.wav", "speaker2.wav"),
        client,
    )

    assert processed.speaker1_metadata == metadata
    assert processed.speaker2_metadata == metadata
    assert processed.quality_result == quality_result
    assert len(client.requests) == 1


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
        event_candidates=(),
        raw_quality_score=None,
        calibrated_quality_score=None,
        calibration_flags=(),
        total_quality_score=None,
        error="test",
    )

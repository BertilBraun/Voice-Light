from __future__ import annotations

import math
import wave
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from app.compute.asr.models.base import PreparedAsrAudio, TimedTranscription
from app.compute.dataset_audio import analyze_dataset_language, transcribe_dataset_audio
from app.shared.asr import AsrModelId, LanguageEstimate, TimestampedWord
from app.shared.audio.s3 import S3AudioCache, S3AudioSource
from app.shared.compute_api import DatasetAsrRequest, DatasetLanguageRequest


class MemoryDownloader:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.requests: list[tuple[str, str]] = []

    def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_object: BinaryIO,
    ) -> None:
        self.requests.append((bucket, key))
        file_object.write(self.content)


class FixedModelCache:
    def __init__(self) -> None:
        self.requests: list[tuple[AsrModelId, float]] = []

    def transcribe_prepared(
        self,
        model_id: AsrModelId,
        audio: PreparedAsrAudio,
    ) -> TimedTranscription:
        self.requests.append((model_id, audio.duration_seconds))
        return TimedTranscription(
            model_id=model_id,
            words=(
                TimestampedWord(
                    text="one",
                    start_seconds=0.0,
                    end_seconds=0.2,
                    confidence=0.95,
                ),
                TimestampedWord(
                    text="two",
                    start_seconds=0.2,
                    end_seconds=0.4,
                    confidence=0.95,
                ),
                TimestampedWord(
                    text="three",
                    start_seconds=0.4,
                    end_seconds=0.6,
                    confidence=0.95,
                ),
                TimestampedWord(
                    text="four",
                    start_seconds=0.6,
                    end_seconds=0.8,
                    confidence=0.95,
                ),
            ),
            language_estimate=(
                LanguageEstimate(language_code="en", confidence=0.99)
                if model_id is AsrModelId.WHISPERX
                else None
            ),
            model_loading_time_seconds=0.0,
            inference_time_seconds=0.1,
            package_names=(),
        )


def test_language_and_asr_share_materialized_s3_audio(tmp_path: Path) -> None:
    content = active_wave_bytes()
    downloader = MemoryDownloader(content)
    cache = S3AudioCache(root=tmp_path / "cache", downloader=downloader)
    source = S3AudioSource(
        uri="s3://meetings/sample/speaker.wav",
        size_bytes=len(content),
        etag="immutable",
        duration_seconds=10.0,
    )
    models = FixedModelCache()

    language = analyze_dataset_language(
        request=DatasetLanguageRequest(speaker1=source, speaker2=source),
        audio_cache=cache,
        model_cache=models,
    )
    asr = transcribe_dataset_audio(
        request=DatasetAsrRequest(
            source=source,
            models=(AsrModelId.PARAKEET_TDT, AsrModelId.CANARY),
        ),
        audio_cache=cache,
        model_cache=models,
    )

    assert language.speaker1.status.value == "english"
    assert language.speaker2.status.value == "english"
    assert asr.audio.content_sha256 == language.speaker1.audio.content_sha256
    assert tuple(result.model_id for result in asr.results) == (
        AsrModelId.PARAKEET_TDT,
        AsrModelId.CANARY,
    )
    assert downloader.requests == [("meetings", "sample/speaker.wav")]


def active_wave_bytes() -> bytes:
    sample_rate = 16_000
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        frames = bytearray()
        for sample_index in range(sample_rate * 10):
            value = round(12_000 * math.sin(2.0 * math.pi * 220.0 * sample_index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        writer.writeframes(frames)
    return output.getvalue()

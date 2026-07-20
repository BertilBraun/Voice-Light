from __future__ import annotations

import asyncio
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from app.compute.asr.models.base import PreparedAsrAudio, TimedTranscription, prepare_asr_audio
from app.compute.asr.service import transcribe_requested_models, transcribe_uploaded_request
from app.shared.asr import AsrModelId, LanguageEstimate, RemoteAsrUploadRequest, TimestampedWord
from app.shared.audio import probe_local_audio_metadata
from app.shared.audio.transport import ASR_AUDIO_TRANSPORT_SPEC, prepare_audio_transport
from app.shared.quality import AudioMetadata


@dataclass
class RecordedTranscriptionCall:
    model_id: AsrModelId
    audio_path: Path
    audio_metadata: AudioMetadata
    starts_with_flac_header: bool


@dataclass
class RecordingModelCache:
    calls: list[RecordedTranscriptionCall] = field(default_factory=list)

    def transcribe_prepared(
        self,
        model_id: AsrModelId,
        audio: PreparedAsrAudio,
    ) -> TimedTranscription:
        self.calls.append(
            RecordedTranscriptionCall(
                model_id=model_id,
                audio_path=audio.path,
                audio_metadata=probe_local_audio_metadata(audio.path),
                starts_with_flac_header=audio.path.read_bytes().startswith(b"fLaC"),
            )
        )
        return TimedTranscription(
            model_id=model_id,
            words=(
                TimestampedWord(
                    text=f"{model_id}-word",
                    start_seconds=1.0,
                    end_seconds=2.0,
                ),
            ),
            language_estimate=(
                LanguageEstimate(language_code="en", confidence=0.97)
                if model_id is AsrModelId.WHISPERX
                else None
            ),
            model_loading_time_seconds=3.0,
            inference_time_seconds=4.0,
            package_names=(),
        )


def test_remote_endpoint_transcribes_requested_models_with_one_audio_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    write_wave(source_path=audio_path, sample_rate=16_000, channel_count=1)
    model_cache = RecordingModelCache()
    audio = prepare_asr_audio(audio_path)

    results = transcribe_requested_models(
        model_cache=model_cache,
        model_ids=(AsrModelId.PARAKEET_TDT, AsrModelId.WHISPERX),
        audio=audio,
    )

    assert tuple(call.model_id for call in model_cache.calls) == (
        AsrModelId.PARAKEET_TDT,
        AsrModelId.WHISPERX,
    )
    assert tuple(call.audio_path for call in model_cache.calls) == (audio_path, audio_path)
    assert tuple(result.model_id for result in results) == (
        AsrModelId.PARAKEET_TDT,
        AsrModelId.WHISPERX,
    )
    assert tuple(result.words[0].text for result in results) == (
        f"{AsrModelId.PARAKEET_TDT}-word",
        f"{AsrModelId.WHISPERX}-word",
    )
    assert results[0].language_estimate is None
    assert results[1].language_estimate == LanguageEstimate(
        language_code="en",
        confidence=0.97,
    )
    assert tuple(result.runtime.real_time_factor for result in results if result.runtime) == (
        4.0,
        4.0,
    )


def test_uploaded_opus_is_hash_verified_and_decoded_to_canonical_flac(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "stereo.wav"
    transport_path = tmp_path / "transport.ogg"
    write_wave(source_path=source_path, sample_rate=8_000, channel_count=2)
    transport_metadata = prepare_audio_transport(
        source_path=source_path,
        output_path=transport_path,
        spec=ASR_AUDIO_TRANSPORT_SPEC,
    )
    request = RemoteAsrUploadRequest(
        audio=transport_metadata,
        models=(AsrModelId.PARAKEET_TDT,),
    )
    model_cache = RecordingModelCache()

    response = asyncio.run(
        transcribe_uploaded_request(
            model_cache=model_cache,
            request_json=request.model_dump_json(),
            audio=UploadFile(
                file=BytesIO(transport_path.read_bytes()),
                filename=transport_path.name,
            ),
        )
    )

    assert response.results[0].model_id is AsrModelId.PARAKEET_TDT
    call = model_cache.calls[0]
    assert call.audio_path.suffix == ".flac"
    assert call.starts_with_flac_header
    assert call.audio_metadata.sample_rate == 16_000
    assert call.audio_metadata.channels == 1
    assert call.audio_metadata.duration_seconds == pytest.approx(1.0, abs=0.01)
    assert response.request_stats is not None
    assert response.request_stats.upload_stream_time_seconds >= 0.0
    assert response.request_stats.upload_validation_time_seconds >= 0.0
    assert response.request_stats.audio_preparation_time_seconds >= 0.0
    assert response.request_stats.transcription_time_seconds >= 0.0
    assert response.request_stats.request_time_seconds >= sum(
        (
            response.request_stats.upload_stream_time_seconds,
            response.request_stats.upload_validation_time_seconds,
            response.request_stats.audio_preparation_time_seconds,
            response.request_stats.transcription_time_seconds,
        )
    )


def test_uploaded_asr_rejects_sha256_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    transport_path = tmp_path / "transport.ogg"
    write_wave(source_path=source_path, sample_rate=16_000, channel_count=1)
    transport_metadata = prepare_audio_transport(
        source_path=source_path,
        output_path=transport_path,
        spec=ASR_AUDIO_TRANSPORT_SPEC,
    ).model_copy(update={"sha256": "0" * 64})
    request = RemoteAsrUploadRequest(
        audio=transport_metadata,
        models=(AsrModelId.PARAKEET_TDT,),
    )

    with pytest.raises(HTTPException, match="SHA-256 mismatch") as error:
        asyncio.run(
            transcribe_uploaded_request(
                model_cache=RecordingModelCache(),
                request_json=request.model_dump_json(),
                audio=UploadFile(
                    file=BytesIO(transport_path.read_bytes()),
                    filename=transport_path.name,
                ),
            )
        )

    assert error.value.status_code == 400


def write_wave(source_path: Path, sample_rate: int, channel_count: int) -> None:
    with wave.open(str(source_path), "wb") as wave_writer:
        wave_writer.setnchannels(channel_count)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(b"\x00\x00" * sample_rate * channel_count)

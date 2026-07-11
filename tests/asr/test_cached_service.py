from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.asr.schemas import (
    AsrModelId,
    AsrTranscriptResult,
    RemoteAsrRequest,
    RemoteAsrResponse,
    TimestampedWord,
)
from app.asr.service import cached_asr_transcripts


@dataclass
class MemoryAsrCache:
    stored_results: list[AsrTranscriptResult] = field(default_factory=list)
    upserted_results_are_rewritten: bool = False

    def get_cached_asr_transcripts(
        self,
        audio_sha256: str,
        model_ids: Sequence[AsrModelId],
    ) -> tuple[AsrTranscriptResult, ...]:
        return tuple(result for result in self.stored_results if result.model_id in model_ids)

    def upsert_cached_asr_transcript(
        self,
        audio_sha256: str,
        audio_filename: str,
        transcript: AsrTranscriptResult,
    ) -> AsrTranscriptResult:
        stored_transcript = (
            transcript.model_copy(update={"text": f"stored {transcript.text}"})
            if self.upserted_results_are_rewritten
            else transcript
        )
        self.stored_results = [
            result for result in self.stored_results if result.model_id != transcript.model_id
        ]
        self.stored_results.append(stored_transcript)
        return stored_transcript


@dataclass
class RecordingRemoteAsrClient:
    response: RemoteAsrResponse
    requests: list[RemoteAsrRequest] = field(default_factory=list)

    def transcribe(self, request: RemoteAsrRequest) -> RemoteAsrResponse:
        self.requests.append(request)
        return self.response


def test_fully_cached_request_does_not_call_remote(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    cache = MemoryAsrCache(stored_results=[transcript_result(AsrModelId.PARAKEET_TDT, "cached")])

    response = cached_asr_transcripts(
        audio_path=audio_path,
        requested_models=(AsrModelId.PARAKEET_TDT,),
        cache=cache,
        remote_client_factory=remote_factory_that_fails,
    )

    assert tuple(result.text for result in response.results) == ("cached",)


def test_partial_cache_calls_remote_only_for_missing_models(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_bytes = b"audio"
    audio_path.write_bytes(audio_bytes)
    cache = MemoryAsrCache(stored_results=[transcript_result(AsrModelId.PARAKEET_TDT, "cached")])
    remote_client = RecordingRemoteAsrClient(
        response=RemoteAsrResponse(results=(transcript_result(AsrModelId.WHISPERX, "remote"),))
    )

    response = cached_asr_transcripts(
        audio_path=audio_path,
        requested_models=(AsrModelId.PARAKEET_TDT, AsrModelId.WHISPERX),
        cache=cache,
        remote_client_factory=lambda: remote_client,
    )

    assert len(remote_client.requests) == 1
    assert remote_client.requests[0].models == (AsrModelId.WHISPERX,)
    assert base64.b64decode(remote_client.requests[0].audio_base64) == audio_bytes
    assert tuple(result.text for result in response.results) == ("cached", "remote")


def test_returned_results_are_read_from_cache_after_remote_persistence(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    cache = MemoryAsrCache(upserted_results_are_rewritten=True)
    remote_client = RecordingRemoteAsrClient(
        response=RemoteAsrResponse(results=(transcript_result(AsrModelId.WHISPERX, "remote"),))
    )

    response = cached_asr_transcripts(
        audio_path=audio_path,
        requested_models=(AsrModelId.WHISPERX,),
        cache=cache,
        remote_client_factory=lambda: remote_client,
    )

    assert tuple(result.text for result in response.results) == ("stored remote",)


@pytest.mark.parametrize(
    ("audio_path", "requested_models", "expected_error"),
    [
        (Path("missing.wav"), (AsrModelId.WHISPERX,), "Audio file not found"),
        (Path("unused.wav"), (), "At least one ASR model is required"),
    ],
)
def test_validation_errors(
    tmp_path: Path,
    audio_path: Path,
    requested_models: tuple[AsrModelId, ...],
    expected_error: str,
) -> None:
    resolved_audio_path = tmp_path / audio_path
    cache = MemoryAsrCache()

    with pytest.raises(ValueError, match=expected_error):
        cached_asr_transcripts(
            audio_path=resolved_audio_path,
            requested_models=requested_models,
            cache=cache,
            remote_client_factory=remote_factory_that_fails,
        )


def test_remote_response_must_include_missing_model_result(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    cache = MemoryAsrCache()
    remote_client = RecordingRemoteAsrClient(response=RemoteAsrResponse(results=()))

    with pytest.raises(ValueError, match="Remote ASR response missing requested model results"):
        cached_asr_transcripts(
            audio_path=audio_path,
            requested_models=(AsrModelId.WHISPERX,),
            cache=cache,
            remote_client_factory=lambda: remote_client,
        )


def transcript_result(model_id: AsrModelId, text: str) -> AsrTranscriptResult:
    return AsrTranscriptResult(
        model_id=model_id,
        text=text,
        words=(TimestampedWord(text=text, start_seconds=0.0, end_seconds=1.0),),
    )


def remote_factory_that_fails() -> RecordingRemoteAsrClient:
    raise AssertionError("Remote ASR should not be called.")

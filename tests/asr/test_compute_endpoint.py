from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.asr.models.base import TimedTranscription
from app.asr.schemas import AsrModelId, TimestampedWord
from app.compute.batch_asr import transcribe_requested_models


@dataclass
class RecordedTranscriptionCall:
    model_id: AsrModelId
    audio_path: Path


@dataclass
class RecordingModelCache:
    calls: list[RecordedTranscriptionCall] = field(default_factory=list)

    def transcribe(self, model_id: AsrModelId, audio_path: Path) -> TimedTranscription:
        self.calls.append(RecordedTranscriptionCall(model_id=model_id, audio_path=audio_path))
        return TimedTranscription(
            model_id=model_id,
            words=(
                TimestampedWord(
                    text=f"{model_id}-word",
                    start_seconds=1.0,
                    end_seconds=2.0,
                ),
            ),
            model_loading_time_seconds=3.0,
            inference_time_seconds=4.0,
            package_names=(),
        )


def test_remote_endpoint_transcribes_requested_models_with_one_audio_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    model_cache = RecordingModelCache()

    results = transcribe_requested_models(
        model_cache=model_cache,
        model_ids=(AsrModelId.PARAKEET_TDT, AsrModelId.WHISPERX),
        audio_path=audio_path,
        audio_duration_seconds=8.0,
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
    assert tuple(result.runtime.real_time_factor for result in results if result.runtime) == (
        0.5,
        0.5,
    )

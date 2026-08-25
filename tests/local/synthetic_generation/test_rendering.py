from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

from app.local.synthetic_generation.models import (
    SpeakerRole,
    SpeakerSpecification,
    SpeechEvent,
    SyntheticConversationPlan,
    TurnBoundary,
)
from app.local.synthetic_generation.rendering import (
    AcousticAugmentation,
    RenderedEventSource,
    SyntheticRenderRequest,
    TtsProvenance,
    render_conversation,
)
from app.local.training_samples.service import INPUT_DURATION_SECONDS


def test_render_conversation_fits_sources_to_independent_tracks(tmp_path: Path) -> None:
    user_source = tmp_path / "user-source.wav"
    assistant_source = tmp_path / "assistant-source.wav"
    _write_tone(user_source, 220.0, 0.4)
    _write_tone(assistant_source, 440.0, 0.3)
    request = SyntheticRenderRequest(
        plan=_plan(),
        sources=(
            _source("user_turn", user_source),
            _source("assistant_turn", assistant_source),
        ),
        augmentation=AcousticAugmentation(
            signal_to_noise_db=25.0,
            codec="mulaw_8bit",
            microphone_low_hz=100.0,
            microphone_high_hz=7_000.0,
            user_to_assistant_crosstalk_db=-30.0,
        ),
    )

    manifest = render_conversation(request, tmp_path / "rendered")

    assert _wave_duration(manifest.user_audio_path) == INPUT_DURATION_SECONDS
    assert _wave_duration(manifest.assistant_audio_path) == INPUT_DURATION_SECONDS
    assert manifest.user_audio_sha256 == _sha256(manifest.user_audio_path)
    assert manifest.assistant_audio_sha256 == _sha256(manifest.assistant_audio_path)
    assert (tmp_path / "rendered" / "render.json").is_file()


def _plan() -> SyntheticConversationPlan:
    return SyntheticConversationPlan(
        plan_id="render_test",
        description="Two overlapping synthetic turns.",
        seed=11,
        duration_seconds=INPUT_DURATION_SECONDS,
        speakers=(
            SpeakerSpecification(
                speaker_id="speaker_1",
                role=SpeakerRole.USER,
                voice_id="test_user",
                language="en-US",
            ),
            SpeakerSpecification(
                speaker_id="speaker_2",
                role=SpeakerRole.ASSISTANT,
                voice_id="test_assistant",
                language="en-US",
            ),
        ),
        events=(
            SpeechEvent(
                event_id="user_turn",
                speaker_id="speaker_1",
                text="Could we move it to Tuesday?",
                start_seconds=0.5,
                end_seconds=2.5,
                boundary_after=TurnBoundary.COMPLETION,
            ),
            SpeechEvent(
                event_id="assistant_turn",
                speaker_id="speaker_2",
                text="Tuesday is fine.",
                start_seconds=2.3,
                end_seconds=3.5,
                boundary_after=TurnBoundary.COMPLETION,
            ),
        ),
    )


def _source(event_id: str, path: Path) -> RenderedEventSource:
    return RenderedEventSource(
        event_id=event_id,
        audio_path=path,
        audio_sha256=_sha256(path),
        provenance=TtsProvenance(
            backend_id="test-tone",
            runtime_version="1",
            model_id="deterministic-tone",
            model_revision="test",
            model_license="CC0-1.0",
            seed=3,
            generation_seconds=0.0,
            real_time_factor=0.0,
        ),
    )


def _write_tone(path: Path, frequency_hz: float, duration_seconds: float) -> None:
    sample_rate_hz = 16_000
    times = np.arange(round(sample_rate_hz * duration_seconds)) / sample_rate_hz
    samples = (np.sin(2.0 * np.pi * frequency_hz * times) * 0.2 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(samples.tobytes())


def _wave_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio_file:
        return audio_file.getnframes() / audio_file.getframerate()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

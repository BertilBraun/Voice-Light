from __future__ import annotations

from pathlib import Path

from app.shared.audio.transport import (
    ASR_AUDIO_TRANSPORT_SPEC,
    QUALITY_AUDIO_TRANSPORT_SPEC,
    audio_transport_command,
)
from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS


def test_prepare_analysis_audio_caps_transport_duration(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.flac"

    command = audio_transport_command(
        source_path=input_path,
        output_path=output_path,
        spec=QUALITY_AUDIO_TRANSPORT_SPEC,
    )
    duration_index = command.index("-t") + 1
    assert command[duration_index] == str(ANALYSIS_AUDIO_MAX_DURATION_SECONDS)
    assert command[command.index("-c:a") + 1] == "flac"


def test_asr_transport_uses_full_length_mono_16khz_opus_at_32kbps(
    tmp_path: Path,
) -> None:
    command = audio_transport_command(
        source_path=tmp_path / "input.wav",
        output_path=tmp_path / "output.ogg",
        spec=ASR_AUDIO_TRANSPORT_SPEC,
    )

    assert "-t" not in command
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "libopus"
    assert command[command.index("-b:a") + 1] == "32000"

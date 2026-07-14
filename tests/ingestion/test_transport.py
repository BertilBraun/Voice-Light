from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.shared.audio.transport import prepare_analysis_audio
from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS


def test_prepare_analysis_audio_caps_transport_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_commands: list[tuple[str, ...]] = []

    def record_command(command: Sequence[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        recorded_commands.append(tuple(command))
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(subprocess, "run", record_command)
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.flac"

    prepare_analysis_audio(input_path, output_path)

    command = recorded_commands[0]
    duration_index = command.index("-t") + 1
    assert command[duration_index] == str(ANALYSIS_AUDIO_MAX_DURATION_SECONDS)

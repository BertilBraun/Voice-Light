import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import Response

from app.local.data import sessions
from app.local.main import SESSIONS_CACHE_CONTROL, sessions_api


@pytest.fixture
def isolated_sessions_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(sessions, "SESSIONS_ROOT", tmp_path)
    sessions.list_sessions.cache_clear()
    yield tmp_path
    sessions.list_sessions.cache_clear()


def test_list_sessions_caches_session_metadata_for_process_lifetime(
    isolated_sessions_root: Path,
) -> None:
    session_directory = isolated_sessions_root / "pmt_001"
    session_directory.mkdir()
    (session_directory / "pmt_001_speaker1.wav").touch()
    (session_directory / "pmt_001_speaker2.wav").touch()
    metadata_path = session_directory / "pmt_001.json"
    metadata_path.write_text(
        json.dumps(
            {
                "name": "pmt_001",
                "durationSeconds": 42.5,
                "topic": "Caching",
                "sampleRate": 16_000,
            }
        ),
        encoding="utf-8",
    )

    first_result = sessions.list_sessions()
    metadata_path.unlink()
    second_result = sessions.list_sessions()

    assert second_result is first_result
    assert second_result[0].identifier == "pmt_001"


def test_sessions_api_allows_immutable_browser_caching(isolated_sessions_root: Path) -> None:
    response = Response()

    payload = sessions_api(response=response)

    assert payload.sessions == ()
    assert response.headers["Cache-Control"] == SESSIONS_CACHE_CONTROL

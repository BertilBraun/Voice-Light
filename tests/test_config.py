from __future__ import annotations

from app.config import (
    LOCAL_DATABASE_URL,
    configured_compute_url,
    configured_database_url,
)


def test_database_url_defaults_to_local_compose_postgres() -> None:
    assert configured_database_url({}) == LOCAL_DATABASE_URL
    assert configured_database_url({"VOICE_LIGHT_DATABASE_URL": ""}) == LOCAL_DATABASE_URL


def test_database_url_environment_override_takes_precedence() -> None:
    configured_url = "postgresql://custom:secret@database.example/custom"

    assert configured_database_url({"VOICE_LIGHT_DATABASE_URL": configured_url}) == configured_url


def test_compute_url_is_explicit_and_strips_trailing_slash() -> None:
    assert configured_compute_url({}) == ""
    assert configured_compute_url({"VOICE_LIGHT_COMPUTE_URL": "https://gpu.example/"}) == (
        "https://gpu.example"
    )

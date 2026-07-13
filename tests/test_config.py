from __future__ import annotations

from app.config import (
    DEFAULT_REMOTE_QUALITY_ENDPOINT_URL,
    LOCAL_DATABASE_URL,
    configured_database_url,
    configured_remote_quality_endpoint_url,
)


def test_database_url_defaults_to_local_compose_postgres() -> None:
    assert configured_database_url({}) == LOCAL_DATABASE_URL
    assert configured_database_url({"VOICE_LIGHT_DATABASE_URL": ""}) == LOCAL_DATABASE_URL


def test_database_url_environment_override_takes_precedence() -> None:
    configured_url = "postgresql://custom:secret@database.example/custom"

    assert configured_database_url({"VOICE_LIGHT_DATABASE_URL": configured_url}) == configured_url


def test_remote_quality_endpoint_has_deployed_default() -> None:
    assert configured_remote_quality_endpoint_url({}) == DEFAULT_REMOTE_QUALITY_ENDPOINT_URL
    assert (
        configured_remote_quality_endpoint_url({"VOICE_LIGHT_REMOTE_QUALITY_ENDPOINT_URL": ""})
        == DEFAULT_REMOTE_QUALITY_ENDPOINT_URL
    )

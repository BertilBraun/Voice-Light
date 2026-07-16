import os
from collections.abc import Mapping
from pathlib import Path

LOCAL_DATABASE_URL = "postgresql://voice_light:voice_light@127.0.0.1:5432/voice_light"


def configured_database_url(environment: Mapping[str, str]) -> str:
    return environment.get("VOICE_LIGHT_DATABASE_URL") or LOCAL_DATABASE_URL


def configured_compute_url(environment: Mapping[str, str]) -> str:
    return environment.get("VOICE_LIGHT_COMPUTE_URL", "").rstrip("/")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPOSITORY_ROOT / "data"
SESSIONS_ROOT = DATA_ROOT / "luel" / "sessions"
WEB_ROOT = APP_ROOT / "web"
FUTURE_WORK_ROOT = REPOSITORY_ROOT / "docs" / "future-work"
MIGRATIONS_ROOT = APP_ROOT / "db" / "migrations"
DATABASE_URL = configured_database_url(os.environ)
COMPUTE_BASE_URL = configured_compute_url(os.environ)
COMPUTE_TOKEN = os.environ.get("VOICE_LIGHT_COMPUTE_TOKEN", "")
REMOTE_ASR_ENDPOINT_URL = f"{COMPUTE_BASE_URL}/v1/asr:batch" if COMPUTE_BASE_URL else ""
REMOTE_QUALITY_ENDPOINT_URL = f"{COMPUTE_BASE_URL}/v1/quality:analyze" if COMPUTE_BASE_URL else ""

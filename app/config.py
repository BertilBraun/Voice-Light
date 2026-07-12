import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPOSITORY_ROOT / "data"
SESSIONS_ROOT = DATA_ROOT / "luel" / "sessions"
WEB_ROOT = APP_ROOT / "web"
MIGRATIONS_ROOT = APP_ROOT / "db" / "migrations"
DATABASE_URL = os.environ.get("VOICE_LIGHT_DATABASE_URL", "")
REMOTE_ASR_ENDPOINT_URL = os.environ.get("VOICE_LIGHT_REMOTE_ASR_ENDPOINT_URL", "")
REMOTE_ASR_API_KEY = os.environ.get("VOICE_LIGHT_REMOTE_ASR_API_KEY", "")
REMOTE_QUALITY_ENDPOINT_URL = os.environ.get("VOICE_LIGHT_REMOTE_QUALITY_ENDPOINT_URL", "")
REMOTE_QUALITY_API_KEY = os.environ.get(
    "VOICE_LIGHT_REMOTE_QUALITY_API_KEY",
    REMOTE_ASR_API_KEY,
)

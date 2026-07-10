from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPOSITORY_ROOT / "data"
SESSIONS_ROOT = DATA_ROOT / "luel" / "sessions"
WEB_ROOT = APP_ROOT / "web"

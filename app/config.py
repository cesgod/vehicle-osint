"""Central configuration.

Everything is optional. With no configuration at all the app runs in
"Link mode" (it builds the Google queries and you open them in your browser).
Settings can be changed from the Settings tab in the UI, which writes them to
`.env` and applies them immediately -- no restart, no manual file editing.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv, set_key

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

DATA_DIR = Path(os.getenv("OSINT_DATA_DIR") or BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
CROPS_DIR = DATA_DIR / "crops"
DB_PATH = DATA_DIR / "osint.sqlite3"
TOKEN_PATH = DATA_DIR / "google_token.json"
CLIENT_SECRET_PATH = BASE_DIR / "client_secret.json"

for _d in (DATA_DIR, UPLOAD_DIR, CROPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def save_env(updates: dict[str, str]) -> None:
    """Persist settings to .env and apply them to the running process."""
    ENV_PATH.touch(exist_ok=True)
    for key, value in updates.items():
        set_key(str(ENV_PATH), key, value, quote_mode="never")
        os.environ[key] = value


# --- Quota budgets (see README "Rate limiting & quotas") -------------------
def cse_daily_budget() -> int:
    """Custom Search JSON API: 100 free queries/day. We stop before Google does."""
    return env_int("CSE_DAILY_BUDGET", 100)


def vision_monthly_budget() -> int:
    """Cloud Vision: first 1,000 units/month per feature are free."""
    return env_int("VISION_MONTHLY_BUDGET", 1000)


def cse_min_interval() -> float:
    """Seconds between Custom Search calls (API limit is 100 queries/minute)."""
    try:
        return float(env("CSE_MIN_INTERVAL_SECONDS", "1.0"))
    except ValueError:
        return 1.0


def cache_ttl_hours() -> int:
    """Identical queries inside this window are served from SQLite, costing no quota."""
    return env_int("CSE_CACHE_TTL_HOURS", 24)

"""Centralized configuration loaded from environment variables and `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = PROJECT_ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
JUDGE_SCOREBOARD_PATH = DATA_DIR / "judge_scoreboard.json"

MIN_DEBATE_ROUNDS = 1
MAX_DEBATE_ROUNDS = 3
DEFAULT_META_JUDGE_MODELS: tuple[str, str, str] = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)


def _parse_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _parse_float(value: str | None, default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _parse_meta_judge_models(raw: str | None) -> tuple[str, str, str]:
    if not raw:
        return DEFAULT_META_JUDGE_MODELS
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) < 3:
        padded = parts + list(DEFAULT_META_JUDGE_MODELS[len(parts) :])
        return (padded[0], padded[1], padded[2])
    return (parts[0], parts[1], parts[2])


GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
META_JUDGE_MODELS: tuple[str, str, str] = _parse_meta_judge_models(os.getenv("META_JUDGE_MODELS"))
DEBATE_ROUNDS: int = _parse_int(
    os.getenv("DEBATE_ROUNDS"),
    default=2,
    minimum=MIN_DEBATE_ROUNDS,
    maximum=MAX_DEBATE_ROUNDS,
)
LLM_TIMEOUT_SECONDS: float = _parse_float(os.getenv("LLM_TIMEOUT_SECONDS"), default=60.0, minimum=5.0, maximum=300.0)
LLM_MAX_RETRIES: int = _parse_int(os.getenv("LLM_MAX_RETRIES"), default=2, minimum=0, maximum=5)
META_JUDGE_TIMEOUT_SECONDS: float = _parse_float(
    os.getenv("META_JUDGE_TIMEOUT_SECONDS"), default=40.0, minimum=10.0, maximum=180.0
)
META_JUDGE_MAX_RETRIES: int = _parse_int(os.getenv("META_JUDGE_MAX_RETRIES"), default=1, minimum=0, maximum=3)
MAX_NEWS_ITEMS: int = _parse_int(os.getenv("MAX_NEWS_ITEMS"), default=15, minimum=1, maximum=30)


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings snapshot for a simulation run."""

    gemini_api_key: str = GEMINI_API_KEY
    default_model: str = DEFAULT_MODEL
    meta_judge_models: tuple[str, str, str] = META_JUDGE_MODELS
    debate_rounds: int = DEBATE_ROUNDS
    history_dir: Path = HISTORY_DIR
    judge_scoreboard_path: Path = JUDGE_SCOREBOARD_PATH
    llm_timeout_seconds: float = LLM_TIMEOUT_SECONDS
    llm_max_retries: int = LLM_MAX_RETRIES
    meta_judge_timeout_seconds: float = META_JUDGE_TIMEOUT_SECONDS
    meta_judge_max_retries: int = META_JUDGE_MAX_RETRIES
    max_news_items: int = MAX_NEWS_ITEMS


def get_settings() -> Settings:
    """Re-read env vars (useful if `.env` or environment changed at runtime)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        default_model=os.getenv("DEFAULT_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite",
        meta_judge_models=_parse_meta_judge_models(os.getenv("META_JUDGE_MODELS")),
        debate_rounds=_parse_int(
            os.getenv("DEBATE_ROUNDS"),
            default=2,
            minimum=MIN_DEBATE_ROUNDS,
            maximum=MAX_DEBATE_ROUNDS,
        ),
        history_dir=HISTORY_DIR,
        judge_scoreboard_path=JUDGE_SCOREBOARD_PATH,
        llm_timeout_seconds=_parse_float(os.getenv("LLM_TIMEOUT_SECONDS"), default=60.0, minimum=5.0, maximum=300.0),
        llm_max_retries=_parse_int(os.getenv("LLM_MAX_RETRIES"), default=2, minimum=0, maximum=5),
        meta_judge_timeout_seconds=_parse_float(
            os.getenv("META_JUDGE_TIMEOUT_SECONDS"), default=40.0, minimum=10.0, maximum=180.0
        ),
        meta_judge_max_retries=_parse_int(os.getenv("META_JUDGE_MAX_RETRIES"), default=1, minimum=0, maximum=3),
        max_news_items=_parse_int(os.getenv("MAX_NEWS_ITEMS"), default=15, minimum=1, maximum=30),
    )

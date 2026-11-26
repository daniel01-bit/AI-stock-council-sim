"""Persistent JSON session storage for Stock Council Replay (data/history/)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from config import HISTORY_DIR
from src.schemas import SessionRecord, SessionSummary

logger = logging.getLogger(__name__)

_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_slug(value: str) -> str:
    cleaned = _SAFE_TOKEN_RE.sub("", value.strip())
    return cleaned or "UNKNOWN"


def build_session_filename(record: SessionRecord) -> str:
    """Generate a Windows-safe, chronologically sortable JSON filename."""
    ts_digits = re.sub(r"[^0-9]", "", record.created_at_utc)[:14] or "00000000000000"
    date_part, time_part = ts_digits[:8], ts_digits[8:14]
    ticker_slug = _safe_slug(record.ticker.upper())
    id_slug = _safe_slug(record.session_id)
    return f"{date_part}_{time_part}_{ticker_slug}_{id_slug}.json"


def _atomic_write_json(target: Path, payload: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".json.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(target)
    return target


def save_session(record: SessionRecord, history_dir: Path = HISTORY_DIR) -> Path:
    """Persist a completed SessionRecord as pretty-printed UTF-8 JSON."""
    target = history_dir / build_session_filename(record)
    return _atomic_write_json(target, record.model_dump_json(indent=2))


def update_session(
    record: SessionRecord,
    filename: str,
    history_dir: Path = HISTORY_DIR,
) -> Path:
    """Atomically overwrite an existing session JSON in place (e.g. attaching forward_eval)."""
    resolved_dir = history_dir.resolve()
    target = (resolved_dir / Path(filename).name).resolve()
    if target.parent != resolved_dir:
        raise ValueError(f"Invalid session filename: {filename}")
    return _atomic_write_json(target, record.model_dump_json(indent=2))


def load_session(path_or_filename: Path | str, history_dir: Path = HISTORY_DIR) -> SessionRecord:
    """Load and validate a SessionRecord from a full path or filename inside history_dir."""
    candidate = Path(path_or_filename)
    if not candidate.is_absolute():
        resolved_dir = history_dir.resolve()
        candidate = (resolved_dir / candidate.name).resolve()
        if candidate.parent != resolved_dir:
            raise ValueError(f"Invalid session filename: {path_or_filename}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Session file not found: {candidate}")
    return SessionRecord.model_validate_json(candidate.read_text(encoding="utf-8"))


def list_sessions(history_dir: Path = HISTORY_DIR) -> list[SessionSummary]:
    """Return all valid saved sessions sorted newest-first."""
    if not history_dir.exists():
        return []

    summaries: list[SessionSummary] = []
    for path in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            record = SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping corrupt or unreadable session file %s: %s", path.name, exc)
            continue

        shifts_count = sum(1 for shift in record.opinion_shifts if shift.changed_vote)
        meta_verdict = None
        if record.meta_judge_panel and record.meta_judge_panel.final_decision:
            meta_verdict = record.meta_judge_panel.final_decision.verdict

        forward_5d_status = "PENDING"
        if record.forward_eval and "5D" in record.forward_eval.horizons:
            forward_5d_status = record.forward_eval.horizons["5D"].status

        summaries.append(
            SessionSummary(
                session_id=record.session_id,
                filename=path.name,
                created_at_utc=record.created_at_utc,
                ticker=record.ticker,
                final_verdict=record.final_verdict,
                price_change_7d_pct=record.fact_briefing.price_change_7d_pct,
                model_name=record.model_name,
                debate_rounds=record.debate_rounds,
                shifts_count=shifts_count,
                has_lab_branches=bool(record.counterfactual_branches),
                meta_verdict=meta_verdict,
                forward_5d_status=forward_5d_status,
            )
        )

    summaries.sort(key=lambda item: (item.created_at_utc, item.filename), reverse=True)
    return summaries

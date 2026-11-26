"""Forward-Run Paper Evaluator (T+1D / T+5D / T+10D vs SPY) & Idempotent Judge Scoreboard.

Zero Look-Ahead Bias Guarantee:
- Evaluates only trading sessions strictly AFTER `fact_briefing.price_end_date`.
- Horizons that have not yet occurred in live markets remain `PENDING`.
- Rebuilds `data/judge_scoreboard.json` idempotently from `data/history/*.json` so
  re-running evaluation never double-counts wins or corrupts track records.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import HISTORY_DIR, JUDGE_SCOREBOARD_PATH, META_JUDGE_MODELS
from src.schemas import (
    EntityEvalScore,
    ForwardEvaluation,
    ForwardHorizonResult,
    JudgeScoreboard,
    JudgeTrackRecord,
    SessionRecord,
)
from src.storage import list_sessions, load_session, update_session

logger = logging.getLogger(__name__)

FORWARD_HORIZONS: tuple[int, ...] = (1, 5, 10)
CloseSeriesFetcher = Callable[[str, str], pd.Series]


def fetch_daily_closes_since(ticker: str, start_date: str) -> pd.Series:
    """Fetch daily closing prices starting on/before `start_date` via yfinance."""
    start_dt = datetime.strptime(start_date[:10], "%Y-%m-%d") - timedelta(days=3)
    hist = yf.Ticker(ticker).history(
        start=start_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)
    closes = hist["Close"].dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes


def _compute_horizon_result(
    *,
    entry_date: str,
    entry_price: float,
    stock_closes: pd.Series,
    spy_closes: pd.Series,
    horizon_days: int,
) -> ForwardHorizonResult:
    entry_ts = pd.Timestamp(entry_date[:10]).normalize()
    future_stock = stock_closes.loc[stock_closes.index > entry_ts]
    if len(future_stock) < horizon_days:
        return ForwardHorizonResult(horizon_days=horizon_days, status="PENDING")

    target_ts = future_stock.index[horizon_days - 1]
    target_price = float(future_stock.iloc[horizon_days - 1])
    if entry_price <= 0:
        return ForwardHorizonResult(horizon_days=horizon_days, status="UNAVAILABLE")

    stock_ret = round(((target_price - entry_price) / entry_price) * 100.0, 2)

    spy_ret: float | None = None
    excess_ret: float | None = None
    if len(spy_closes) > 0:
        spy_entry_rows = spy_closes.loc[spy_closes.index <= entry_ts]
        spy_target_rows = spy_closes.loc[spy_closes.index <= target_ts]
        if len(spy_entry_rows) > 0 and len(spy_target_rows) > 0:
            spy_start = float(spy_entry_rows.iloc[-1])
            spy_end = float(spy_target_rows.iloc[-1])
            if spy_start > 0:
                spy_ret = round(((spy_end - spy_start) / spy_start) * 100.0, 2)
                excess_ret = round(stock_ret - spy_ret, 2)

    return ForwardHorizonResult(
        horizon_days=horizon_days,
        target_date=target_ts.strftime("%Y-%m-%d"),
        target_price=round(target_price, 2),
        stock_return_pct=stock_ret,
        spy_return_pct=spy_ret,
        excess_return_pct=excess_ret,
        actual_up=stock_ret > 0,
        status="EVALUATED",
    )


def _score_entity_5d(
    *,
    verdict: str | None,
    prob_up_5d: float | None,
    horizon_5d: ForwardHorizonResult,
) -> EntityEvalScore:
    if horizon_5d.status != "EVALUATED" or horizon_5d.actual_up is None:
        return EntityEvalScore(verdict=verdict, prob_up_5d=prob_up_5d)

    actual_up = horizon_5d.actual_up
    is_directional = verdict in ("BUY", "SELL")
    correct_5d: bool | None = None
    if verdict == "BUY":
        correct_5d = actual_up is True
    elif verdict == "SELL":
        correct_5d = actual_up is False

    brier_5d: float | None = None
    if prob_up_5d is not None:
        p = max(0.0, min(1.0, prob_up_5d / 100.0))
        y = 1.0 if actual_up else 0.0
        brier_5d = round((p - y) ** 2, 4)

    excess_aligned: float | None = None
    if horizon_5d.excess_return_pct is not None and is_directional:
        sign = 1.0 if verdict == "BUY" else -1.0
        excess_aligned = round(sign * horizon_5d.excess_return_pct, 2)

    return EntityEvalScore(
        verdict=verdict,
        prob_up_5d=prob_up_5d,
        is_directional=is_directional,
        correct_5d=correct_5d,
        brier_5d=brier_5d,
        excess_return_5d=excess_aligned,
    )


def evaluate_session_forward(
    record: SessionRecord,
    *,
    close_fetcher: CloseSeriesFetcher = fetch_daily_closes_since,
) -> ForwardEvaluation:
    """Evaluate realized T+1D, T+5D, T+10D returns vs SPY for one SessionRecord."""
    entry_date = record.fact_briefing.price_end_date
    entry_price = record.fact_briefing.price_end

    stock_closes = close_fetcher(record.ticker, entry_date)
    spy_closes = close_fetcher("SPY", entry_date)

    horizons: dict[str, ForwardHorizonResult] = {}
    for h in FORWARD_HORIZONS:
        horizons[f"{h}D"] = _compute_horizon_result(
            entry_date=entry_date,
            entry_price=entry_price,
            stock_closes=stock_closes,
            spy_closes=spy_closes,
            horizon_days=h,
        )

    h5 = horizons["5D"]

    # Main council 5D score
    mean_council_prob = (
        round(sum(v.prob_up_5d for v in record.stage3_votes) / len(record.stage3_votes), 1)
        if record.stage3_votes
        else None
    )
    council_score_5d = _score_entity_5d(
        verdict=record.final_verdict,
        prob_up_5d=mean_council_prob,
        horizon_5d=h5,
    )

    # Per-branch 5D scores
    branch_scores_5d: dict[str, EntityEvalScore] = {}
    for br in record.counterfactual_branches:
        branch_scores_5d[br.branch_id] = _score_entity_5d(
            verdict=br.verdict,
            prob_up_5d=br.mean_prob_up_5d,
            horizon_5d=h5,
        )

    # Per-Meta-Judge shadow 5D scores (ALL 3 scored equally to eliminate selection bias)
    judge_scores_5d: dict[str, EntityEvalScore] = {}
    if record.meta_judge_panel is not None:
        for shadow in record.meta_judge_panel.shadow_decisions:
            if shadow.status == "OK":
                judge_scores_5d[shadow.judge_id] = _score_entity_5d(
                    verdict=shadow.verdict,
                    prob_up_5d=float(shadow.prob_up_5d) if shadow.prob_up_5d is not None else None,
                    horizon_5d=h5,
                )

    return ForwardEvaluation(
        evaluated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        entry_date=entry_date,
        entry_price=entry_price,
        horizons=horizons,
        council_score_5d=council_score_5d,
        branch_scores_5d=branch_scores_5d,
        judge_scores_5d=judge_scores_5d,
    )


def load_judge_scoreboard(
    scoreboard_path: Path = JUDGE_SCOREBOARD_PATH,
    default_models: tuple[str, str, str] = META_JUDGE_MODELS,
) -> JudgeScoreboard:
    """Load persistent judge scoreboard, initializing default slots if absent."""
    if scoreboard_path.is_file():
        try:
            return JudgeScoreboard.model_validate_json(scoreboard_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Corrupt judge scoreboard %s, re-initializing: %s", scoreboard_path, exc)

    judges = {
        f"judge_{idx}": JudgeTrackRecord(judge_id=f"judge_{idx}", model_name=model)
        for idx, model in enumerate(default_models, start=1)
    }
    return JudgeScoreboard(judges=judges)


def save_judge_scoreboard(
    scoreboard: JudgeScoreboard,
    scoreboard_path: Path = JUDGE_SCOREBOARD_PATH,
) -> Path:
    """Atomically persist JudgeScoreboard to JSON."""
    scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = scoreboard_path.with_suffix(".json.tmp")
    tmp_path.write_text(scoreboard.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(scoreboard_path)
    return scoreboard_path


def rebuild_judge_scoreboard_from_history(
    history_dir: Path = HISTORY_DIR,
    scoreboard_path: Path = JUDGE_SCOREBOARD_PATH,
    default_models: tuple[str, str, str] = META_JUDGE_MODELS,
) -> JudgeScoreboard:
    """Idempotently aggregate all Meta-Judge shadow runs and 5D forward scores from history JSONs."""
    # Seed default judge slots so UI/prompts always have all 3 slots represented.
    accum: dict[str, dict[str, object]] = {
        f"judge_{idx}": {
            "model_name": model,
            "total_runs": 0,
            "evaluated_runs": 0,
            "directional_calls": 0,
            "wins": 0,
            "briers": [],
            "excess": [],
            "times_selected": 0,
            "selected_dir": 0,
            "selected_wins": 0,
        }
        for idx, model in enumerate(default_models, start=1)
    }
    evaluated_session_ids: list[str] = []

    for summary in list_sessions(history_dir):
        record = load_session(summary.filename, history_dir=history_dir)
        panel = record.meta_judge_panel
        if panel is None:
            continue

        feval = record.forward_eval
        h5_evaluated = bool(feval and "5D" in feval.horizons and feval.horizons["5D"].status == "EVALUATED")
        if h5_evaluated:
            evaluated_session_ids.append(record.session_id)

        for shadow in panel.shadow_decisions:
            if shadow.status != "OK":
                continue
            slot = accum.setdefault(
                shadow.judge_id,
                {
                    "model_name": shadow.model_name,
                    "total_runs": 0,
                    "evaluated_runs": 0,
                    "directional_calls": 0,
                    "wins": 0,
                    "briers": [],
                    "excess": [],
                    "times_selected": 0,
                    "selected_dir": 0,
                    "selected_wins": 0,
                },
            )
            # Keep the most recent model_name observed for this judge slot
            slot["model_name"] = shadow.model_name
            slot["total_runs"] = int(slot["total_runs"]) + 1

            is_selected = panel.selected_final_judge_id == shadow.judge_id
            if is_selected:
                slot["times_selected"] = int(slot["times_selected"]) + 1

            if h5_evaluated and feval is not None and shadow.judge_id in feval.judge_scores_5d:
                score = feval.judge_scores_5d[shadow.judge_id]
                slot["evaluated_runs"] = int(slot["evaluated_runs"]) + 1
                if score.brier_5d is not None:
                    slot["briers"].append(score.brier_5d)
                if score.excess_return_5d is not None:
                    slot["excess"].append(score.excess_return_5d)
                if score.is_directional:
                    slot["directional_calls"] = int(slot["directional_calls"]) + 1
                    if score.correct_5d is True:
                        slot["wins"] = int(slot["wins"]) + 1
                    if is_selected:
                        slot["selected_dir"] = int(slot["selected_dir"]) + 1
                        if score.correct_5d is True:
                            slot["selected_wins"] = int(slot["selected_wins"]) + 1

    judges_out: dict[str, JudgeTrackRecord] = {}
    for jid, data in sorted(accum.items()):
        dir_calls = int(data["directional_calls"])
        wins = int(data["wins"])
        sel_dir = int(data["selected_dir"])
        sel_wins = int(data["selected_wins"])
        briers: list[float] = data["briers"]
        excess_list: list[float] = data["excess"]

        judges_out[jid] = JudgeTrackRecord(
            judge_id=jid,
            model_name=str(data["model_name"]),
            total_runs=int(data["total_runs"]),
            evaluated_runs=int(data["evaluated_runs"]),
            directional_calls=dir_calls,
            wins=wins,
            win_rate=round(wins / dir_calls, 4) if dir_calls > 0 else None,
            avg_brier_5d=round(sum(briers) / len(briers), 4) if briers else None,
            avg_excess_return_5d=round(sum(excess_list) / len(excess_list), 2) if excess_list else None,
            times_selected_final_judge=int(data["times_selected"]),
            selected_directional_calls=sel_dir,
            selected_wins=sel_wins,
            selected_win_rate=round(sel_wins / sel_dir, 4) if sel_dir > 0 else None,
        )

    scoreboard = JudgeScoreboard(
        updated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        evaluated_sessions=evaluated_session_ids,
        judges=judges_out,
    )
    save_judge_scoreboard(scoreboard, scoreboard_path=scoreboard_path)
    return scoreboard


def _is_fully_evaluated(record: SessionRecord) -> bool:
    """Return True when all forward horizons (1D, 5D, 10D) are already EVALUATED."""
    feval = record.forward_eval
    if feval is None:
        return False
    return all(
        f"{h}D" in feval.horizons and feval.horizons[f"{h}D"].status == "EVALUATED"
        for h in FORWARD_HORIZONS
    )


def evaluate_all_history_sessions(
    history_dir: Path = HISTORY_DIR,
    scoreboard_path: Path = JUDGE_SCOREBOARD_PATH,
    *,
    close_fetcher: CloseSeriesFetcher = fetch_daily_closes_since,
    default_models: tuple[str, str, str] = META_JUDGE_MODELS,
) -> tuple[int, JudgeScoreboard]:
    """Run forward-return evaluation across pending history JSONs and rebuild JudgeScoreboard.

    Skips sessions already evaluated across all horizons and caches close-series lookups per
    (symbol, entry_date) within a single evaluation pass to avoid redundant yfinance calls.
    """
    series_cache: dict[tuple[str, str], pd.Series] = {}

    def _cached_fetcher(symbol: str, start_date: str) -> pd.Series:
        key = (symbol.upper(), start_date[:10])
        if key not in series_cache:
            series_cache[key] = close_fetcher(symbol, start_date)
        return series_cache[key]

    updated_count = 0
    for summary in list_sessions(history_dir):
        record = load_session(summary.filename, history_dir=history_dir)
        if _is_fully_evaluated(record):
            continue
        record.forward_eval = evaluate_session_forward(record, close_fetcher=_cached_fetcher)
        update_session(record, summary.filename, history_dir=history_dir)
        updated_count += 1

    scoreboard = rebuild_judge_scoreboard_from_history(
        history_dir=history_dir,
        scoreboard_path=scoreboard_path,
        default_models=default_models,
    )
    return updated_count, scoreboard

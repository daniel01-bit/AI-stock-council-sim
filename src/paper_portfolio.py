"""Idempotent Standalone T+5D Paper Portfolio Ledger (`data/paper_portfolio.json`).

Rebuilds paper trade history and realized equity/drawdown metrics directly from
`data/history/*.json` so re-evaluating forward returns never double-counts a trade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import config
from src.storage import list_sessions, load_session
from src.strategy import evaluate_record_strategy

DEFAULT_PORTFOLIO_PATH = config.DATA_DIR / "paper_portfolio.json"


class PaperTradeEntry(BaseModel):
    """One paper trade (or abstained signal) derived from a saved SessionRecord."""

    session_id: str
    ticker: str
    entry_date: str
    entry_price: float
    action: Literal["LONG", "SHORT", "NO TRADE"]
    alloc_pct: float
    effective_prob_up_5d: float
    prob_source: str
    status: Literal["ABSTAINED", "OPEN", "SETTLED"]
    exit_date: str | None = None
    exit_price: float | None = None
    raw_stock_return_5d_pct: float | None = None
    spy_return_5d_pct: float | None = None
    strategy_return_5d_pct: float | None = None
    portfolio_pnl_pct: float | None = None


class PaperPortfolioSummary(BaseModel):
    """Aggregate realized performance of the standalone T+5D behavioral strategy."""

    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    total_sessions: int = 0
    actionable_trades: int = 0
    abstained_sessions: int = 0
    open_trades: int = 0
    settled_trades: int = 0
    wins: int = 0
    win_rate: float | None = None
    profit_factor: float | None = None
    cumulative_portfolio_return_pct: float = 0.0
    avg_spy_5d_pct: float | None = None
    max_drawdown_pct: float = 0.0
    trades: list[PaperTradeEntry] = Field(default_factory=list)


def save_paper_portfolio(
    summary: PaperPortfolioSummary,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
) -> Path:
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = portfolio_path.with_suffix(".json.tmp")
    tmp_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(portfolio_path)
    return portfolio_path


def rebuild_paper_portfolio_from_history(
    history_dir: Path = config.HISTORY_DIR,
    portfolio_path: Path = DEFAULT_PORTFOLIO_PATH,
) -> PaperPortfolioSummary:
    """Idempotently compute the standalone T+5D paper portfolio from all saved sessions."""
    summaries = list(reversed(list_sessions(history_dir)))  # chronological order
    entries: list[PaperTradeEntry] = []

    settled_pnls: list[float] = []
    spy_5d_vals: list[float] = []
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    actionable = 0
    abstained = 0
    open_count = 0

    for item in summaries:
        rec = load_session(item.filename, history_dir=history_dir)
        ticket = rec.strategy_ticket or evaluate_record_strategy(rec)
        entry_date = rec.fact_briefing.price_end_date
        entry_price = rec.fact_briefing.price_end

        if ticket.action == "NO TRADE":
            abstained += 1
            entries.append(
                PaperTradeEntry(
                    session_id=rec.session_id,
                    ticker=rec.ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    action="NO TRADE",
                    alloc_pct=0.0,
                    effective_prob_up_5d=ticket.effective_prob_up_5d,
                    prob_source=ticket.prob_source,
                    status="ABSTAINED",
                )
            )
            continue

        actionable += 1
        h5 = rec.forward_eval.horizons.get("5D") if rec.forward_eval else None
        if h5 and h5.status == "EVALUATED" and h5.stock_return_pct is not None:
            sign = 1.0 if ticket.action == "LONG" else -1.0
            strat_ret = round(sign * h5.stock_return_pct, 2)
            port_pnl = round(strat_ret * (ticket.alloc_pct / 100.0), 3)
            settled_pnls.append(port_pnl)
            if h5.spy_return_pct is not None:
                spy_5d_vals.append(h5.spy_return_pct)
            if strat_ret > 0:
                wins += 1
                gross_profit += port_pnl
            elif port_pnl < 0:
                gross_loss += abs(port_pnl)

            entries.append(
                PaperTradeEntry(
                    session_id=rec.session_id,
                    ticker=rec.ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    action=ticket.action,
                    alloc_pct=ticket.alloc_pct,
                    effective_prob_up_5d=ticket.effective_prob_up_5d,
                    prob_source=ticket.prob_source,
                    status="SETTLED",
                    exit_date=h5.target_date,
                    exit_price=h5.target_price,
                    raw_stock_return_5d_pct=h5.stock_return_pct,
                    spy_return_5d_pct=h5.spy_return_pct,
                    strategy_return_5d_pct=strat_ret,
                    portfolio_pnl_pct=port_pnl,
                )
            )
        else:
            open_count += 1
            entries.append(
                PaperTradeEntry(
                    session_id=rec.session_id,
                    ticker=rec.ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    action=ticket.action,
                    alloc_pct=ticket.alloc_pct,
                    effective_prob_up_5d=ticket.effective_prob_up_5d,
                    prob_source=ticket.prob_source,
                    status="OPEN",
                )
            )

    # Compute equity curve & max drawdown across settled trades
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for pnl in settled_pnls:
        equity *= 1.0 + (pnl / 100.0)
        peak = max(peak, equity)
        dd = ((peak - equity) / peak) * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    settled_count = len(settled_pnls)
    pf = None
    if gross_loss > 0:
        pf = round(gross_profit / gross_loss, 2)
    elif gross_profit > 0:
        pf = 99.0

    summary = PaperPortfolioSummary(
        updated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_sessions=len(summaries),
        actionable_trades=actionable,
        abstained_sessions=abstained,
        open_trades=open_count,
        settled_trades=settled_count,
        wins=wins,
        win_rate=round(wins / settled_count, 4) if settled_count > 0 else None,
        profit_factor=pf,
        cumulative_portfolio_return_pct=round((equity - 1.0) * 100.0, 2),
        avg_spy_5d_pct=round(sum(spy_5d_vals) / len(spy_5d_vals), 2) if spy_5d_vals else None,
        max_drawdown_pct=round(max_dd, 2),
        trades=list(reversed(entries)),  # newest first for UI display
    )
    save_paper_portfolio(summary, portfolio_path=portfolio_path)
    return summary

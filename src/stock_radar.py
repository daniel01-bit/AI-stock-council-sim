"""Quick Pick & Hot 7D Behavioral Radar adapted from quant01 for Stock Council Simulator.

Strictly follows MODEL-2 constraints: ranks curated stocks ONLY by 7-calendar-day
price change % (zero RSI, MACD, moving averages, or technical indicators).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import yfinance as yf

# Curated stock groups adapted from mini-project-quant01's Quick Pick & Hot Radar
STOCK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AI & Semiconductors", ("NVDA", "AMD", "AVGO", "TSM", "MU", "ARM", "SMCI", "MRVL")),
    ("Mega-Cap Tech", ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "PLTR")),
    ("Value, Finance & Crypto", ("BRK-B", "JPM", "UNH", "LLY", "COST", "XOM", "BTC-USD", "ETH-USD")),
    ("Indices & Safe-Haven", ("^NDX", "^GSPC", "SPY", "QQQ", "SOXX", "GLD", "SHY", "TLT")),
)

GROUP_BY_TICKER: dict[str, str] = {
    sym: group_name for group_name, symbols in STOCK_GROUPS for sym in symbols
}


@dataclass(frozen=True)
class HotRadarRow:
    """One ranked candidate from the Hot 7D Behavioral Radar (zero technical indicators)."""

    ticker: str
    group: str
    price_start_7d: float
    price_end: float
    price_change_7d_pct: float
    abs_move_7d_pct: float
    heat_tag: str


def classify_7d_heat(pct_change: float) -> str:
    """Label 7-day behavioral intensity to spotlight high-debate stocks."""
    if pct_change >= 5.0:
        return "🔥 Rally Surge"
    if pct_change <= -5.0:
        return "❄️ Panic Dip"
    if pct_change >= 2.0:
        return "📈 Momentum Up"
    if pct_change <= -2.0:
        return "📉 Pullback"
    return "⚖️ Rangebound"


def _compute_7d_from_series(closes: pd.Series) -> tuple[float, float, float] | None:
    clean = closes.dropna()
    if len(clean) < 2:
        return None
    latest_ts = clean.index[-1]
    target_ts = latest_ts - timedelta(days=7)
    older = clean.loc[clean.index <= target_ts]
    price_start = float(older.iloc[-1]) if len(older) > 0 else float(clean.iloc[0])
    price_end = float(clean.iloc[-1])
    if price_start <= 0:
        return None
    pct = round(((price_end - price_start) / price_start) * 100.0, 2)
    return round(price_start, 2), round(price_end, 2), pct


def scan_hot_7d_movers(
    tickers: Sequence[str],
    *,
    top_n: int = 12,
) -> list[HotRadarRow]:
    """Batch-scan tickers for 7-day price change % ONLY and rank by |7D Move %|."""
    unique_tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not unique_tickers:
        return []

    raw = yf.download(
        tickers=unique_tickers,
        period="1mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        return []

    rows: list[HotRadarRow] = []
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance returns MultiIndex (Field, Ticker) by default
        if "Close" in raw.columns.get_level_values(0):
            close_df = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            close_df = raw.xs("Close", axis=1, level=1)
        else:
            return []

        for sym in unique_tickers:
            if sym not in close_df.columns:
                continue
            stats = _compute_7d_from_series(close_df[sym])
            if stats is None:
                continue
            p_start, p_end, pct = stats
            rows.append(
                HotRadarRow(
                    ticker=sym,
                    group=GROUP_BY_TICKER.get(sym, "Custom"),
                    price_start_7d=p_start,
                    price_end=p_end,
                    price_change_7d_pct=pct,
                    abs_move_7d_pct=abs(pct),
                    heat_tag=classify_7d_heat(pct),
                )
            )
    elif "Close" in raw.columns and len(unique_tickers) == 1:
        sym = unique_tickers[0]
        stats = _compute_7d_from_series(raw["Close"])
        if stats is not None:
            p_start, p_end, pct = stats
            rows.append(
                HotRadarRow(
                    ticker=sym,
                    group=GROUP_BY_TICKER.get(sym, "Custom"),
                    price_start_7d=p_start,
                    price_end=p_end,
                    price_change_7d_pct=pct,
                    abs_move_7d_pct=abs(pct),
                    heat_tag=classify_7d_heat(pct),
                )
            )

    rows.sort(key=lambda item: item.abs_move_7d_pct, reverse=True)
    return rows[:top_n]

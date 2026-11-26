"""Market data fetcher: strictly recent news + 7-day price change % via yfinance.

Zero technical indicators (no RSI, MACD, moving averages, or chart patterns) are computed.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import yfinance as yf

from config import MAX_NEWS_ITEMS
from src.schemas import MarketContext, RawNewsItem

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")


class TickerDataError(RuntimeError):
    """Raised when a ticker is invalid or has insufficient price history."""


def _clean_text(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    stripped = _HTML_TAG_RE.sub(" ", html.unescape(raw))
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _format_published_time(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ""


def _parse_news_items(raw_news: Any, max_items: int) -> list[RawNewsItem]:
    """Normalize both yfinance >=0.2.50/1.x nested `content` items and legacy flat items."""
    if not isinstance(raw_news, list):
        return []

    parsed: list[RawNewsItem] = []
    seen_titles: set[str] = set()

    for entry in raw_news:
        if not isinstance(entry, dict):
            continue

        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry

        title = _clean_text(content.get("title", ""))
        if not title:
            continue

        norm_key = title.lower()
        if norm_key in seen_titles:
            continue
        seen_titles.add(norm_key)

        provider = content.get("provider")
        if isinstance(provider, dict):
            publisher = _clean_text(provider.get("displayName", "")) or "Unknown"
        else:
            publisher = _clean_text(content.get("publisher", "")) or "Unknown"

        published_at = _format_published_time(
            content.get("pubDate")
            or content.get("displayTime")
            or content.get("providerPublishTime")
        )

        summary = _clean_text(content.get("summary") or content.get("description") or "")

        canonical = content.get("canonicalUrl")
        click_through = content.get("clickThroughUrl")
        url = ""
        if isinstance(canonical, dict) and isinstance(canonical.get("url"), str):
            url = canonical["url"].strip()
        elif isinstance(click_through, dict) and isinstance(click_through.get("url"), str):
            url = click_through["url"].strip()
        elif isinstance(content.get("link"), str):
            url = content["link"].strip()

        parsed.append(
            RawNewsItem(
                title=title,
                publisher=publisher,
                published_at=published_at,
                summary=summary,
                url=url,
            )
        )
        if len(parsed) >= max_items:
            break

    return parsed


def _extract_7d_price_window(hist: Any) -> tuple[float, str, float, str, float]:
    """Compute ONLY the 7-calendar-day price change % from Close prices."""
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        raise TickerDataError("No historical price data returned.")

    closes = hist["Close"].dropna()
    if len(closes) < 2:
        raise TickerDataError("Insufficient price history to compute a 7-day price change.")

    latest_ts = closes.index[-1]
    target_ts = latest_ts - timedelta(days=7)

    # Pick the closest trading session on or before (latest_date - 7 calendar days).
    older_rows = closes.loc[closes.index <= target_ts]
    if len(older_rows) > 0:
        start_ts = older_rows.index[-1]
        price_start = float(older_rows.iloc[-1])
    else:
        start_ts = closes.index[0]
        price_start = float(closes.iloc[0])

    price_end = float(closes.iloc[-1])
    if price_start <= 0:
        raise TickerDataError("Invalid reference closing price (<= 0).")

    pct_change = round(((price_end - price_start) / price_start) * 100.0, 2)
    start_date = start_ts.strftime("%Y-%m-%d") if hasattr(start_ts, "strftime") else str(start_ts)[:10]
    end_date = latest_ts.strftime("%Y-%m-%d") if hasattr(latest_ts, "strftime") else str(latest_ts)[:10]
    return round(price_start, 2), start_date, round(price_end, 2), end_date, pct_change


def _safe_company_name(ticker_obj: Any, fallback_ticker: str) -> str:
    try:
        info = ticker_obj.info
        if isinstance(info, dict):
            name = info.get("shortName") or info.get("longName")
            if isinstance(name, str) and name.strip():
                return name.strip()
    except Exception:
        # Company display name is non-critical; never fail the run on metadata lookup issues.
        pass
    return fallback_ticker


def fetch_market_context(ticker: str, max_news_items: int = MAX_NEWS_ITEMS) -> MarketContext:
    """Fetch recent news + 7-day price change % for `ticker` using yfinance."""
    normalized = ticker.strip().upper()
    if not normalized or not _TICKER_RE.match(normalized):
        raise TickerDataError(
            f"Invalid stock ticker '{ticker}'. Use standard symbols such as NVDA, AAPL, or TSLA."
        )

    try:
        ticker_obj = yf.Ticker(normalized)
        # Fetch 1 month of daily closes ONLY so weekends/holidays never miss the 7-day anchor.
        hist = ticker_obj.history(period="1mo", interval="1d", auto_adjust=True)
    except Exception as exc:
        raise TickerDataError(f"Failed to retrieve price data for '{normalized}': {exc}") from exc

    try:
        price_start, start_date, price_end, end_date, pct_change = _extract_7d_price_window(hist)
    except TickerDataError as exc:
        raise TickerDataError(
            f"Ticker '{normalized}' could not be loaded ({exc}). Verify the symbol exists on Yahoo Finance."
        ) from exc

    data_warning: str | None = None
    try:
        raw_news = ticker_obj.news
    except Exception as exc:
        raw_news = []
        data_warning = f"News endpoint failed ({exc}); council proceeding with 7-day price change only."

    news_items = _parse_news_items(raw_news, max_items=max_news_items)
    if not news_items and data_warning is None:
        data_warning = (
            f"Yahoo Finance returned 0 recent news articles for {normalized}. "
            "Council decisions reflect limited news visibility + 7-day price change only."
        )

    company_name = _safe_company_name(ticker_obj, normalized)

    return MarketContext(
        ticker=normalized,
        company_name=company_name,
        fetched_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        price_start_7d=price_start,
        price_start_date=start_date,
        price_end=price_end,
        price_end_date=end_date,
        price_change_7d_pct=pct_change,
        news_items=news_items,
        data_warning=data_warning,
    )

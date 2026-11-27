"""Path B — Leakage-Free Masked Point-in-Time Historical Dataset Builder.

Builds historical training/evaluation samples for Stock Council Simulator with
3 strict anti-leakage guarantees:
1. Point-in-Time Price Alignment:
   - Input `price_change_7d_pct` uses ONLY closes `<= event_date`.
   - Ground-truth `T+5D` return vs `SPY` uses ONLY closes strictly `> event_date`.
2. Entity, Date & Price-Level Masking (Anti-Pretraining-Leakage):
   - Replaces ticker symbols, company names, and 4-digit years with `[COMPANY_X]` / `[YEAR]`.
   - Normalizes share price to `$100.00 -> $100*(1 + pct/100)` so the LLM cannot infer
     the company or historical era from raw price levels.
3. Zero-Key & CSV Sources:
   - Built-in official SEC EDGAR 8-K historical material events (free, zero API key).
   - Generic CSV importer (`ticker,date,title,summary,publisher`) for FNSPID / Finnhub / Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from config import DATA_DIR
from src.schemas import MarketContext, RawNewsItem

DATASETS_DIR = DATA_DIR / "datasets"
DEFAULT_DATASET_PATH = DATASETS_DIR / "historical_masked.jsonl"

SEC_USER_AGENT = "StockCouncilResearch/1.0 (academic-behavioral-finance@example.com)"

# Official SEC Form 8-K Item descriptions (material corporate events)
SEC_8K_ITEM_LABELS: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition (Earnings Release)",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities (Restructuring)",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy Listing Rule",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.02": "Departure/Election of Directors or Principal Officers",
    "7.01": "Regulation FD Disclosure (Guidance / Investor Update)",
    "8.01": "Other Material Events",
}

# Pre-mapped CIKs for fast zero-lookup SEC EDGAR ingestion (can also auto-resolve via SEC ticker map)
KNOWN_CIKS: dict[str, tuple[str, str]] = {
    "NVDA": ("0001045810", "NVIDIA"),
    "AAPL": ("0000320193", "Apple"),
    "MSFT": ("0000789019", "Microsoft"),
    "AMD": ("0000002488", "Advanced Micro Devices"),
    "TSLA": ("0001318605", "Tesla"),
    "META": ("0001326801", "Meta Platforms"),
    "AMZN": ("0001018724", "Amazon"),
    "GOOGL": ("0001652044", "Alphabet"),
    "JPM": ("0000019617", "JPMorgan Chase"),
    "NFLX": ("0001065280", "Netflix"),
}

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class HistoricalGroundTruth(BaseModel):
    """Realized T+5D forward outcome (strictly excluded from LLM prompts)."""

    horizon_days: int = 5
    entry_date: str
    entry_price_real: float
    target_date: str
    target_price_real: float
    stock_return_5d_pct: float
    spy_return_5d_pct: float
    excess_return_5d_pct: float
    actual_up_5d: bool


class HistoricalSample(BaseModel):
    """One masked point-in-time historical sample ready for leakage-free Council evaluation."""

    sample_id: str
    source: str
    real_ticker: str
    real_event_date: str
    masked_context: MarketContext
    ground_truth_5d: HistoricalGroundTruth


def mask_sensitive_text(
    text: str,
    *,
    ticker: str,
    company_name: str,
    alias: str = "COMPANY_X",
) -> str:
    """Scrub ticker, company name tokens, ISO dates, and 4-digit years to prevent pretraining leakage."""
    if not text:
        return ""
    masked = text
    # Strip full company name and major name tokens (>= 4 chars)
    tokens: list[str] = [company_name.strip()]
    for part in re.split(r"[\s,./()]+", company_name.strip()):
        cleaned = part.strip()
        if len(cleaned) >= 4 and cleaned.lower() not in {"corp", "inc.", "company", "group", "holdings", "technologies", "platforms"}:
            tokens.append(cleaned)
    tokens.append(ticker.strip().upper())

    for tok in sorted({t for t in tokens if t}, key=len, reverse=True):
        # Match $TICKER or whole-word case-insensitive token
        pattern = re.compile(rf"(?<!\w)\$?{re.escape(tok)}(?!\w)", re.IGNORECASE)
        masked = pattern.sub(alias, masked)

    masked = _ISO_DATE_RE.sub("[DATE]", masked)
    masked = _YEAR_RE.sub("[YEAR]", masked)
    return re.sub(r"\s+", " ", masked).strip()


def fetch_history_series(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch daily adjusted closes via yfinance over [start_date, end_date]."""
    hist = yf.Ticker(symbol).history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist.columns:
        return pd.Series(dtype=float)
    closes = hist["Close"].dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes


def build_masked_sample(
    *,
    ticker: str,
    company_name: str,
    event_date: str,
    raw_items: list[RawNewsItem],
    stock_closes: pd.Series,
    spy_closes: pd.Series,
    source: str = "historical",
    horizon_days: int = 5,
) -> HistoricalSample | None:
    """Construct a point-in-time masked HistoricalSample; returns None if price window/label unavailable."""
    if not raw_items or len(stock_closes) < 7 or len(spy_closes) < 7:
        return None

    event_ts = pd.Timestamp(event_date[:10]).normalize()
    past_stock = stock_closes.loc[stock_closes.index <= event_ts]
    future_stock = stock_closes.loc[stock_closes.index > event_ts]
    if len(past_stock) < 2 or len(future_stock) < horizon_days:
        return None

    entry_ts = past_stock.index[-1]
    entry_price = float(past_stock.iloc[-1])

    # 7-calendar-day input anchor strictly <= entry_ts
    anchor_target = entry_ts - timedelta(days=7)
    older = past_stock.loc[past_stock.index <= anchor_target]
    if len(older) == 0:
        return None
    price_start_7d = float(older.iloc[-1])
    if price_start_7d <= 0 or entry_price <= 0:
        return None
    pct_7d = round(((entry_price - price_start_7d) / price_start_7d) * 100.0, 2)

    # Strictly forward T+5D label (> entry_ts)
    target_ts = future_stock.index[horizon_days - 1]
    target_price = float(future_stock.iloc[horizon_days - 1])
    stock_ret_5d = round(((target_price - entry_price) / entry_price) * 100.0, 2)

    spy_entry_rows = spy_closes.loc[spy_closes.index <= entry_ts]
    spy_target_rows = spy_closes.loc[spy_closes.index <= target_ts]
    if len(spy_entry_rows) == 0 or len(spy_target_rows) == 0:
        return None
    spy_start = float(spy_entry_rows.iloc[-1])
    spy_end = float(spy_target_rows.iloc[-1])
    if spy_start <= 0:
        return None
    spy_ret_5d = round(((spy_end - spy_start) / spy_start) * 100.0, 2)
    excess_5d = round(stock_ret_5d - spy_ret_5d, 2)

    # Deterministic anonymized alias & normalized $100 price scale
    sample_hash = hashlib.sha1(f"{ticker.upper()}:{event_date[:10]}:{source}".encode("utf-8")).hexdigest()[:8]
    alias = f"CORP_{sample_hash[:4].upper()}"
    norm_start = 100.00
    norm_end = round(100.0 * (1.0 + pct_7d / 100.0), 2)

    masked_news = [
        RawNewsItem(
            title=mask_sensitive_text(item.title, ticker=ticker, company_name=company_name, alias=alias),
            publisher=item.publisher or "Wire",
            published_at="T-0",
            summary=mask_sensitive_text(item.summary, ticker=ticker, company_name=company_name, alias=alias),
            url="",
        )
        for item in raw_items
    ]

    masked_ctx = MarketContext(
        ticker=alias,
        company_name=f"Anonymized Issuer ({alias})",
        fetched_at_utc="2000-01-01T00:00:00Z",
        price_start_7d=norm_start,
        price_start_date="T-7D",
        price_end=norm_end,
        price_end_date="T-0",
        price_change_7d_pct=pct_7d,
        news_items=masked_news,
    )

    gt = HistoricalGroundTruth(
        horizon_days=horizon_days,
        entry_date=entry_ts.strftime("%Y-%m-%d"),
        entry_price_real=round(entry_price, 2),
        target_date=target_ts.strftime("%Y-%m-%d"),
        target_price_real=round(target_price, 2),
        stock_return_5d_pct=stock_ret_5d,
        spy_return_5d_pct=spy_ret_5d,
        excess_return_5d_pct=excess_5d,
        actual_up_5d=stock_ret_5d > 0,
    )

    return HistoricalSample(
        sample_id=f"{sample_hash}_{event_ts.strftime('%Y%m%d')}",
        source=source,
        real_ticker=ticker.strip().upper(),
        real_event_date=entry_ts.strftime("%Y-%m-%d"),
        masked_context=masked_ctx,
        ground_truth_5d=gt,
    )


def _fetch_sec_edgar_8k_events(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    max_events: int = 25,
) -> tuple[str, list[tuple[str, list[RawNewsItem]]]]:
    """Fetch historical SEC Form 8-K material corporate events (official, free, point-in-time)."""
    sym = ticker.strip().upper()
    if sym not in KNOWN_CIKS:
        raise ValueError(f"Ticker '{sym}' not in KNOWN_CIKS ({', '.join(sorted(KNOWN_CIKS))}).")
    cik, company_name = KNOWN_CIKS[sym]
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    items_list = recent.get("items") or []
    descs = recent.get("primaryDocDescription") or []

    grouped: dict[str, list[RawNewsItem]] = defaultdict(list)
    for form, fdate, items, desc in zip(forms, dates, items_list, descs):
        if form != "8-K":
            continue
        if not (start_date <= fdate <= end_date):
            continue
        codes = items if isinstance(items, list) else [c.strip() for c in str(items).split(",") if c.strip()]
        material_labels = [SEC_8K_ITEM_LABELS[c] for c in codes if c in SEC_8K_ITEM_LABELS]
        if not material_labels:
            continue
        headline = f"{company_name} files Form 8-K: {'; '.join(material_labels[:3])}"
        summary = (
            f"Official SEC Form 8-K material disclosure covering items {', '.join(codes)} "
            f"({desc or 'Current Report'})."
        )
        grouped[fdate].append(
            RawNewsItem(
                title=headline,
                publisher="SEC EDGAR (Form 8-K)",
                published_at=fdate,
                summary=summary,
                url="",
            )
        )

    ordered = sorted(grouped.items(), key=lambda kv: kv[0], reverse=True)[:max_events]
    return company_name, ordered


def build_from_sec_edgar(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    max_per_ticker: int = 20,
    close_fetcher: Callable[[str, str, str], pd.Series] = fetch_history_series,
) -> list[HistoricalSample]:
    """Build masked historical dataset from SEC 8-K material events + yfinance prices."""
    pad_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=20)).strftime("%Y-%m-%d")
    pad_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=25)).strftime("%Y-%m-%d")
    spy_closes = close_fetcher("SPY", pad_start, pad_end)

    samples: list[HistoricalSample] = []
    for sym in tickers:
        company_name, events = _fetch_sec_edgar_8k_events(
            sym, start_date=start_date, end_date=end_date, max_events=max_per_ticker
        )
        stock_closes = close_fetcher(sym, pad_start, pad_end)
        for fdate, items in events:
            sample = build_masked_sample(
                ticker=sym,
                company_name=company_name,
                event_date=fdate,
                raw_items=items,
                stock_closes=stock_closes,
                spy_closes=spy_closes,
                source="sec_8k",
            )
            if sample is not None:
                samples.append(sample)
        time.sleep(0.25)  # Polite SEC EDGAR rate limit (<10 req/s)

    samples.sort(key=lambda s: (s.real_event_date, s.sample_id))
    return samples


def build_from_csv(
    csv_path: Path,
    *,
    max_samples: int = 200,
    close_fetcher: Callable[[str, str, str], pd.Series] = fetch_history_series,
) -> list[HistoricalSample]:
    """Build masked historical samples from a CSV (`ticker,date,title,summary,publisher`)."""
    grouped: dict[tuple[str, str], list[RawNewsItem]] = defaultdict(list)
    company_names: dict[str, str] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sym = (row.get("ticker") or row.get("symbol") or "").strip().upper()
            raw_date = (row.get("date") or row.get("published_at") or "").strip()[:10]
            title = (row.get("title") or row.get("headline") or "").strip()
            if not (sym and len(raw_date) == 10 and title):
                continue
            company_names.setdefault(sym, (row.get("company_name") or KNOWN_CIKS.get(sym, ("", sym))[1]).strip() or sym)
            grouped[(sym, raw_date)].append(
                RawNewsItem(
                    title=title,
                    publisher=(row.get("publisher") or row.get("source") or "News CSV").strip(),
                    published_at=raw_date,
                    summary=(row.get("summary") or row.get("text") or "").strip()[:400],
                )
            )

    if not grouped:
        return []

    all_dates = [d for _, d in grouped.keys()]
    pad_start = (datetime.strptime(min(all_dates), "%Y-%m-%d") - timedelta(days=20)).strftime("%Y-%m-%d")
    pad_end = (datetime.strptime(max(all_dates), "%Y-%m-%d") + timedelta(days=25)).strftime("%Y-%m-%d")
    spy_closes = close_fetcher("SPY", pad_start, pad_end)
    stock_cache: dict[str, pd.Series] = {}

    samples: list[HistoricalSample] = []
    for (sym, edate), items in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if len(samples) >= max_samples:
            break
        if sym not in stock_cache:
            stock_cache[sym] = close_fetcher(sym, pad_start, pad_end)
        sample = build_masked_sample(
            ticker=sym,
            company_name=company_names.get(sym, sym),
            event_date=edate,
            raw_items=items[:10],
            stock_closes=stock_cache[sym],
            spy_closes=spy_closes,
            source=f"csv:{csv_path.name}",
        )
        if sample is not None:
            samples.append(sample)

    return samples


def save_samples_jsonl(samples: list[HistoricalSample], output_path: Path = DEFAULT_DATASET_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".jsonl.tmp")
    lines = [s.model_dump_json() for s in samples]
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(output_path)
    return output_path


def load_samples_jsonl(path: Path = DEFAULT_DATASET_PATH) -> list[HistoricalSample]:
    if not path.is_file():
        return []
    out: list[HistoricalSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(HistoricalSample.model_validate_json(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Path B: Build Masked Point-in-Time Historical Dataset.")
    parser.add_argument("--sec-edgar", action="store_true", help="Fetch free SEC Form 8-K historical events.")
    parser.add_argument("--csv", type=Path, default=None, help="Import historical news CSV (ticker,date,title,summary,publisher).")
    parser.add_argument("--tickers", type=str, default="NVDA,AAPL,MSFT,AMD,TSLA", help="Tickers for --sec-edgar.")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default="2026-10-31", help="End date YYYY-MM-DD (must be >= 10D before today).")
    parser.add_argument("--max-per-ticker", type=int, default=15, help="Max 8-K events per ticker.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH, help="Output JSONL path.")
    args = parser.parse_args()

    if args.csv is not None:
        samples = build_from_csv(args.csv)
    elif args.sec_edgar:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        samples = build_from_sec_edgar(
            tickers,
            start_date=args.start,
            end_date=args.end,
            max_per_ticker=args.max_per_ticker,
        )
    else:
        parser.error("Specify either --sec-edgar or --csv <path>.")
        return 2

    saved = save_samples_jsonl(samples, args.output)
    up_rate = (sum(1 for s in samples if s.ground_truth_5d.actual_up_5d) / len(samples) * 100.0) if samples else 0.0
    print(f"[DATASET] Saved {len(samples)} masked historical samples -> {saved} (Base Up-Rate 5D: {up_rate:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

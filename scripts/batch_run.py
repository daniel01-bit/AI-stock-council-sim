"""Path A — Idempotent Daily Forward Auto-Collector for Stock Council Simulator.

Runs live Stock Council simulations on a target watchlist after market close,
skips tickers already recorded on the current UTC date, paces calls to stay
within Free-Tier RPM limits, and runs Forward-Run Evaluation (T+1D/5D/10D vs SPY).

Usage:
    python scripts/batch_run.py --dry-run
    python scripts/batch_run.py --tickers NVDA,AAPL,MSFT,AMD,TSLA --rounds 1 --lab
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_fetcher import TickerDataError, fetch_market_context
from src.evaluator import evaluate_all_history_sessions
from src.llm_client import GeminiLLMClient, LLMGenerationError, MissingAPIKeyError
from src.orchestrator import CouncilOrchestrator
from src.storage import list_sessions

DEFAULT_WATCHLIST = ("NVDA", "AAPL", "MSFT", "AMD", "TSLA")


def already_ran_today(ticker: str, history_dir: Path, utc_date: str | None = None) -> bool:
    """Return True if `ticker` already has a saved session on `utc_date` (YYYY-MM-DD)."""
    today = utc_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sym = ticker.strip().upper()
    for item in list_sessions(history_dir):
        if item.ticker == sym and item.created_at_utc[:10] == today:
            return True
    return False


async def run_batch(
    *,
    tickers: list[str],
    model_name: str,
    debate_rounds: int,
    enable_lab: bool,
    delay_seconds: float,
    dry_run: bool,
    force: bool,
    skip_eval: bool,
) -> int:
    settings = config.get_settings()
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    pending: list[str] = []
    for raw in tickers:
        sym = raw.strip().upper()
        if not sym:
            continue
        if not force and already_ran_today(sym, settings.history_dir, utc_date=today_utc):
            print(f"[SKIP] {sym} — already recorded on {today_utc} UTC (use --force to re-run).")
            continue
        pending.append(sym)

    if dry_run:
        print(f"[DRY-RUN] Checking yfinance market context for {len(pending)} ticker(s)...")
        ok = 0
        for sym in pending:
            try:
                ctx = fetch_market_context(sym)
                print(
                    f"  [OK] {sym:6s} | 7D: {ctx.price_change_7d_pct:+6.2f}% "
                    f"| News: {len(ctx.news_items):2d} | End: {ctx.price_end_date}"
                )
                ok += 1
            except TickerDataError as exc:
                print(f"  [ERR] {sym:6s} | {exc}")
        print(f"[DRY-RUN] Complete: {ok}/{len(pending)} tickers valid. Zero LLM calls made.")
        return 0

    failures = 0
    if pending:
        async with AsyncExitStack() as stack:
            try:
                council_client = await stack.enter_async_context(
                    GeminiLLMClient(api_key=settings.gemini_api_key, model_name=model_name)
                )
            except MissingAPIKeyError as exc:
                print(f"[FATAL] {exc}", file=sys.stderr)
                return 2

            judge_clients: dict[str, GeminiLLMClient] = {}
            if enable_lab:
                for idx, jmodel in enumerate(settings.meta_judge_models, start=1):
                    judge_clients[f"judge_{idx}"] = await stack.enter_async_context(
                        GeminiLLMClient(
                            api_key=settings.gemini_api_key,
                            model_name=jmodel,
                            timeout_seconds=settings.meta_judge_timeout_seconds,
                            max_retries=settings.meta_judge_max_retries,
                            fallback_models=(),
                        )
                    )

            orchestrator = CouncilOrchestrator(
                llm_client=council_client,
                history_dir=settings.history_dir,
                judge_clients=judge_clients,
                scoreboard_path=settings.judge_scoreboard_path,
            )

            for idx, sym in enumerate(pending, start=1):
                t0 = time.perf_counter()
                print(f"[{idx}/{len(pending)}] Running {sym} (model={model_name}, rounds={debate_rounds}, lab={enable_lab})...")
                try:
                    record = await orchestrator.run_simulation(
                        ticker=sym,
                        debate_rounds=debate_rounds,
                        enable_counterfactual_lab=enable_lab,
                        save_to_disk=True,
                    )
                    elapsed = time.perf_counter() - t0
                    meta_v = (
                        record.meta_judge_panel.final_decision.verdict
                        if (record.meta_judge_panel and record.meta_judge_panel.final_decision)
                        else "-"
                    )
                    print(
                        f"  [DONE] {sym:6s} | Council={record.final_verdict:21s} | Meta={str(meta_v):5s} "
                        f"| 7D={record.fact_briefing.price_change_7d_pct:+6.2f}% | {elapsed:.1f}s"
                    )
                except (TickerDataError, LLMGenerationError, Exception) as exc:
                    failures += 1
                    print(f"  [FAIL] {sym:6s} | {exc}", file=sys.stderr)

                if idx < len(pending) and delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

    if not skip_eval:
        print("[EVAL] Updating Forward-Run Paper Evaluation (T+1D/5D/10D vs SPY)...")
        updated, scoreboard = evaluate_all_history_sessions(
            history_dir=settings.history_dir,
            scoreboard_path=settings.judge_scoreboard_path,
            default_models=settings.meta_judge_models,
        )
        print(
            f"[EVAL] Updated {updated} pending session(s). "
            f"Evaluated sessions on scoreboard: {len(scoreboard.evaluated_sessions)}."
        )

    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path A: Daily Forward Auto-Collector for Stock Council.")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_WATCHLIST),
        help="Comma-separated stock tickers (default: NVDA,AAPL,MSFT,AMD,TSLA).",
    )
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL, help="Gemini model for Council.")
    parser.add_argument("--rounds", type=int, default=1, help="Stage 2 debate rounds (1-3, default: 1).")
    parser.add_argument("--lab", action="store_true", help="Enable v2.1 Parallel Lab + 3 Meta-Judges.")
    parser.add_argument("--delay", type=float, default=10.0, help="Cooldown seconds between tickers (default: 10).")
    parser.add_argument("--dry-run", action="store_true", help="Verify yfinance & idempotency without LLM calls.")
    parser.add_argument("--force", action="store_true", help="Run even if ticker already ran today (UTC).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip Forward-Run evaluation pass at the end.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    return asyncio.run(
        run_batch(
            tickers=tickers,
            model_name=args.model,
            debate_rounds=args.rounds,
            enable_lab=args.lab,
            delay_seconds=args.delay,
            dry_run=args.dry_run,
            force=args.force,
            skip_eval=args.skip_eval,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

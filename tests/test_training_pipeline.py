"""Unit tests for Path A (batch_run idempotency) & Path B (masked dataset, L1 calibrator, Unsloth export)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from scripts.batch_run import already_ran_today
from scripts.train_calibrator import export_unsloth_datasets, run_level1_calibration
from src.historical_dataset import (
    build_from_csv,
    build_masked_sample,
    load_samples_jsonl,
    mask_sensitive_text,
    save_samples_jsonl,
)
from src.orchestrator import CouncilOrchestrator
from src.schemas import RawNewsItem
from tests.test_orchestrator import MockCouncilLLMClient, _sample_market_context


def test_mask_sensitive_text_and_point_in_time_alignment() -> None:
    """Verify zero ticker/company/year/price-level leakage into masked_context."""
    cleaned = mask_sensitive_text(
        "NVIDIA ($NVDA) beats Q3 2025 estimates on 2025-11-20 after strong GPU demand.",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        alias="CORP_1234",
    )
    assert "NVDA" not in cleaned
    assert "NVIDIA" not in cleaned
    assert "2025" not in cleaned
    assert "CORP_1234" in cleaned
    assert "[YEAR]" in cleaned
    assert "[DATE]" in cleaned

    # 20 business days of synthetic prices: event on day index 8 (2025-02-12)
    dates = pd.bdate_range("2025-02-01", periods=20)
    stock = pd.Series([150.0 + i * 2.0 for i in range(20)], index=dates)
    spy = pd.Series([500.0 + i * 1.0 for i in range(20)], index=dates)
    event_date = dates[8].strftime("%Y-%m-%d")

    sample = build_masked_sample(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        event_date=event_date,
        raw_items=[
            RawNewsItem(
                title="NVIDIA announces 2025 Blackwell guidance",
                publisher="SEC EDGAR",
                published_at=event_date,
                summary="NVDA filed Form 8-K on " + event_date,
            )
        ],
        stock_closes=stock,
        spy_closes=spy,
        source="unit_test",
    )
    assert sample is not None
    # Normalized $100 start price hides real $150+ price level
    assert sample.masked_context.price_start_7d == 100.0
    assert sample.masked_context.price_start_date == "T-7D"
    assert sample.masked_context.price_end_date == "T-0"
    assert "NVDA" not in sample.masked_context.ticker
    assert "NVIDIA" not in sample.masked_context.news_items[0].title
    assert "2025" not in sample.masked_context.news_items[0].title
    # Strictly forward T+5D label
    assert sample.ground_truth_5d.target_date > sample.ground_truth_5d.entry_date
    assert sample.ground_truth_5d.actual_up_5d is True
    assert sample.ground_truth_5d.stock_return_5d_pct > 0


def test_csv_importer_and_jsonl_roundtrip(tmp_path: Path) -> None:
    csv_file = tmp_path / "news.csv"
    csv_file.write_text(
        "ticker,date,title,summary,publisher\n"
        "AAPL,2025-02-14,Apple launches new chip in 2025,AAPL updates hardware line,Reuters\n",
        encoding="utf-8",
    )

    def fake_fetcher(symbol: str, start: str, end: str) -> pd.Series:
        dates = pd.bdate_range("2025-02-01", periods=20)
        base = 500.0 if symbol == "SPY" else 200.0
        return pd.Series([base + i for i in range(20)], index=dates)

    samples = build_from_csv(csv_file, close_fetcher=fake_fetcher)
    assert len(samples) == 1
    assert "AAPL" not in samples[0].masked_context.news_items[0].title
    assert "Apple" not in samples[0].masked_context.news_items[0].title

    out = save_samples_jsonl(samples, tmp_path / "masked.jsonl")
    reloaded = load_samples_jsonl(out)
    assert reloaded == samples


def test_batch_run_idempotency_and_calibrator_plus_unsloth_export(tmp_path: Path) -> None:
    """Verify Path A daily skip + Path B Level 1 calibration & Level 3 Unsloth SFT/DPO export."""
    mock_llm = MockCouncilLLMClient(delay_seconds=0.0)
    orchestrator = CouncilOrchestrator(
        llm_client=mock_llm,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
    )
    record = asyncio.run(orchestrator.run_simulation("NVDA", debate_rounds=1, save_to_disk=True))
    today_str = record.created_at_utc[:10]
    assert already_ran_today("NVDA", tmp_path, utc_date=today_str) is True
    assert already_ran_today("AAPL", tmp_path, utc_date=today_str) is False

    # Create 8 synthetic evaluated runs (alternating up/down outcomes) to test calibration & Unsloth export
    runs_path = tmp_path / "eval_runs.jsonl"
    lines: list[str] = []
    for i in range(8):
        actual_up = (i % 2 == 0)
        lines.append(
            json.dumps(
                {
                    "sample_id": f"s_{i:02d}",
                    "real_event_date": f"2025-03-{10 + i:02d}",
                    "real_ticker": "NVDA",
                    "ground_truth_5d": {
                        "horizon_days": 5,
                        "entry_date": f"2025-03-{10 + i:02d}",
                        "entry_price_real": 150.0,
                        "target_date": f"2025-03-{17 + i:02d}",
                        "target_price_real": 160.0 if actual_up else 140.0,
                        "stock_return_5d_pct": 6.67 if actual_up else -6.67,
                        "spy_return_5d_pct": 1.0,
                        "excess_return_5d_pct": 5.67 if actual_up else -7.67,
                        "actual_up_5d": actual_up,
                    },
                    "session_record": record.model_dump(),
                }
            )
        )
    runs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cal_path = tmp_path / "calibrator.json"
    artifact = run_level1_calibration(runs_path=runs_path, calibrator_path=cal_path)
    assert cal_path.is_file()
    assert artifact["n_total"] == 8
    assert "calibrated_brier" in artifact["val_metrics"]
    assert len(artifact["weights"]) == len(artifact["feature_names"])

    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    n_sft, n_dpo = export_unsloth_datasets(runs_path=runs_path, sft_path=sft_path, dpo_path=dpo_path)
    assert n_sft > 0
    assert n_dpo > 0
    first_sft = json.loads(sft_path.read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in first_sft["messages"]] == ["system", "user", "assistant"]
    first_dpo = json.loads(dpo_path.read_text(encoding="utf-8").splitlines()[0])
    assert {"prompt", "chosen", "rejected"}.issubset(first_dpo.keys())

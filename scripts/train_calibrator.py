"""Path B — Resumable Masked Eval Runner, Level 1 Statistical Calibrator & Level 3 Unsloth Exporter.

Workflow:
  1. Run Council on masked historical samples (resumable, zero look-ahead):
     python scripts/train_calibrator.py --run-eval --limit 30 --rounds 1
  2. Fit Level 1 Time-Ordered L2 Logistic Calibrator & print OOS Brier Benchmark:
     python scripts/train_calibrator.py --calibrate
  3. Export Level 3 ChatML SFT + DPO pairs for Unsloth (Qwen 4B/7B):
     python scripts/train_calibrator.py --export-unsloth
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.historical_dataset import DATASETS_DIR, DEFAULT_DATASET_PATH, HistoricalSample, load_samples_jsonl
from src.llm_client import GeminiLLMClient, LLMGenerationError, MissingAPIKeyError
from src.orchestrator import CouncilOrchestrator
from src.prompts import PERSONA_BY_ID, build_final_vote_prompt
from src.schemas import SessionRecord

DEFAULT_EVAL_RUNS_PATH = DATASETS_DIR / "eval_runs.jsonl"
DEFAULT_CALIBRATOR_PATH = config.DATA_DIR / "calibrator_v1.json"
DEFAULT_SFT_PATH = DATASETS_DIR / "unsloth_sft.jsonl"
DEFAULT_DPO_PATH = DATASETS_DIR / "unsloth_dpo.jsonl"

AGENT_IDS: tuple[str, ...] = (
    "retail_fomo",
    "fundamentalist",
    "paranoid_bear",
    "contrarian",
    "macro_analyst",
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def fit_l2_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2_reg: float = 1.0,
    lr: float = 0.15,
    steps: int = 800,
) -> tuple[np.ndarray, float]:
    """Lightweight, deterministic L2-regularized Logistic Regression in pure NumPy (zero sklearn dep)."""
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / n + (l2_reg / n) * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def extract_feature_vector(record: SessionRecord) -> tuple[list[str], np.ndarray]:
    """Extract leakage-free numerical features from Stage 3 votes & Behavioral Metrics."""
    votes_by_id = {v.agent_id: v for v in record.stage3_votes}
    names: list[str] = []
    vals: list[float] = []

    for aid in AGENT_IDS:
        v = votes_by_id.get(aid)
        prob = (v.prob_up_5d / 100.0) if v else 0.5
        names.append(f"prob_{aid}")
        vals.append(prob - 0.5)  # centered around 0

    bm = record.behavioral_metrics
    names.extend(["herding_rate", "conf_polarization", "stage3_entropy", "price_7d_pct_scaled"])
    vals.extend(
        [
            bm.herding_rate if bm else 0.0,
            (bm.confidence_polarization / 5.0) if bm else 0.0,
            (bm.stage3_entropy / 1.585) if bm else 0.5,
            float(np.clip(record.fact_briefing.price_change_7d_pct / 20.0, -1.5, 1.5)),
        ]
    )
    return names, np.array(vals, dtype=float)


def _load_eval_runs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


async def run_masked_evaluation(
    *,
    dataset_path: Path,
    runs_path: Path,
    model_name: str,
    debate_rounds: int,
    limit: int,
    delay_seconds: float,
) -> int:
    samples = load_samples_jsonl(dataset_path)
    if not samples:
        print(f"[ERR] No samples found at {dataset_path}. Run `python -m src.historical_dataset --sec-edgar` first.")
        return 1

    existing_rows = _load_eval_runs(runs_path)
    done_ids = {row["sample_id"] for row in existing_rows}
    todo = [s for s in samples if s.sample_id not in done_ids][:limit]
    if not todo:
        print(f"[DONE] All {len(samples)} samples in {dataset_path.name} are already evaluated in {runs_path.name}.")
        return 0

    settings = config.get_settings()
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with GeminiLLMClient(api_key=settings.gemini_api_key, model_name=model_name) as client:
            for idx, sample in enumerate(todo, start=1):
                t0 = time.perf_counter()
                orchestrator = CouncilOrchestrator(
                    llm_client=client,
                    data_fetcher=lambda _ticker, ctx=sample.masked_context: ctx,
                    history_dir=settings.history_dir,
                )
                print(
                    f"[{idx}/{len(todo)}] Eval {sample.sample_id} "
                    f"(masked={sample.masked_context.ticker}, 7D={sample.masked_context.price_change_7d_pct:+.2f}%)..."
                )
                try:
                    record = await orchestrator.run_simulation(
                        ticker=sample.masked_context.ticker,
                        debate_rounds=debate_rounds,
                        enable_counterfactual_lab=False,
                        save_to_disk=False,
                    )
                    mean_p = sum(v.prob_up_5d for v in record.stage3_votes) / max(1, len(record.stage3_votes))
                    y = 1.0 if sample.ground_truth_5d.actual_up_5d else 0.0
                    brier = round(((mean_p / 100.0) - y) ** 2, 4)
                    entry = {
                        "sample_id": sample.sample_id,
                        "real_event_date": sample.real_event_date,
                        "real_ticker": sample.real_ticker,
                        "ground_truth_5d": sample.ground_truth_5d.model_dump(),
                        "session_record": record.model_dump(),
                    }
                    with runs_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  [OK] Verdict={record.final_verdict:10s} | P(Up)={mean_p:5.1f}% "
                        f"| Actual5D={sample.ground_truth_5d.stock_return_5d_pct:+6.2f}% "
                        f"| Brier={brier:.4f} | {elapsed:.1f}s"
                    )
                except (LLMGenerationError, Exception) as exc:
                    print(f"  [FAIL] {sample.sample_id}: {exc}", file=sys.stderr)

                if idx < len(todo) and delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
    except MissingAPIKeyError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    return 0


def run_level1_calibration(
    runs_path: Path = DEFAULT_EVAL_RUNS_PATH,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    """Fit time-ordered Level 1 L2 Logistic Calibrator and compare against raw Council Brier."""
    rows = _load_eval_runs(runs_path)
    if len(rows) < 6:
        raise ValueError(f"Need at least 6 evaluated runs for train/val calibration (found {len(rows)}).")

    # Strict chronological ordering to prevent look-ahead leakage
    rows.sort(key=lambda r: (r["real_event_date"], r["sample_id"]))

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    raw_probs: list[float] = []
    per_agent_probs: dict[str, list[float]] = {aid: [] for aid in AGENT_IDS}
    feature_names: list[str] = []

    for row in rows:
        rec = SessionRecord.model_validate(row["session_record"])
        gt = row["ground_truth_5d"]
        feature_names, feat = extract_feature_vector(rec)
        X_list.append(feat)
        y_list.append(1.0 if gt["actual_up_5d"] else 0.0)
        raw_probs.append(sum(v.prob_up_5d for v in rec.stage3_votes) / (100.0 * max(1, len(rec.stage3_votes))))
        by_id = {v.agent_id: v.prob_up_5d / 100.0 for v in rec.stage3_votes}
        for aid in AGENT_IDS:
            per_agent_probs[aid].append(by_id.get(aid, 0.5))

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    raw_p = np.array(raw_probs, dtype=float)

    split_idx = max(4, min(len(rows) - 2, int(len(rows) * train_ratio)))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    raw_val = raw_p[split_idx:]

    w, b = fit_l2_logistic(X_train, y_train, l2_reg=2.0)
    cal_val = _sigmoid(X_val @ w + b)

    raw_brier_val = round(brier_score(raw_val, y_val), 4)
    cal_brier_val = round(brier_score(cal_val, y_val), 4)
    coin_brier_val = round(brier_score(np.full_like(y_val, 0.5), y_val), 4)

    agent_briers = {
        aid: round(brier_score(np.array(probs[split_idx:]), y_val), 4)
        for aid, probs in per_agent_probs.items()
    }

    artifact = {
        "version": "level1_l2_logistic_v1",
        "n_total": len(rows),
        "n_train": split_idx,
        "n_val": len(rows) - split_idx,
        "val_metrics": {
            "coin_flip_brier": coin_brier_val,
            "raw_council_brier": raw_brier_val,
            "calibrated_brier": cal_brier_val,
            "brier_improvement": round(raw_brier_val - cal_brier_val, 4),
            "per_agent_val_brier": agent_briers,
        },
        "feature_names": feature_names,
        "weights": [round(float(x), 6) for x in w],
        "bias": round(float(b), 6),
    }

    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def export_unsloth_datasets(
    runs_path: Path = DEFAULT_EVAL_RUNS_PATH,
    sft_path: Path = DEFAULT_SFT_PATH,
    dpo_path: Path = DEFAULT_DPO_PATH,
) -> tuple[int, int]:
    """Export ChatML SFT (winning calibrated agent votes) & DPO preference pairs for Unsloth."""
    rows = _load_eval_runs(runs_path)
    sft_rows: list[str] = []
    dpo_rows: list[str] = []

    for row in rows:
        rec = SessionRecord.model_validate(row["session_record"])
        y = 1.0 if row["ground_truth_5d"]["actual_up_5d"] else 0.0
        s1_by_id = {v.agent_id: v for v in rec.stage1_votes}

        scored: list[tuple[float, Any, str, str]] = []
        for v in rec.stage3_votes:
            persona = PERSONA_BY_ID.get(v.agent_id)
            if persona is None:
                continue
            p = v.prob_up_5d / 100.0
            brier = (p - y) ** 2
            user_prompt = build_final_vote_prompt(
                rec.fact_briefing,
                rec.debate_transcript,
                own_stage1_vote=s1_by_id.get(v.agent_id),
            )
            assistant_json = json.dumps(
                {
                    "vote": v.vote,
                    "confidence": v.confidence,
                    "prob_up_5d": v.prob_up_5d,
                    "position_pct": v.position_pct,
                    "reasoning": v.reasoning,
                },
                ensure_ascii=False,
            )
            scored.append((brier, persona.system_prompt, user_prompt, assistant_json))

            # SFT: keep well-calibrated agent responses (Brier <= 0.20)
            if brier <= 0.20:
                sft_rows.append(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": persona.system_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": assistant_json},
                            ],
                            "meta": {"sample_id": row["sample_id"], "brier_5d": round(brier, 4)},
                        },
                        ensure_ascii=False,
                    )
                )

        # DPO: pair best-calibrated (chosen) vs worst-calibrated (rejected) response on the same briefing
        if len(scored) >= 2:
            scored.sort(key=lambda t: t[0])
            best_brier, _, best_prompt, best_json = scored[0]
            worst_brier, _, _, worst_json = scored[-1]
            if (worst_brier - best_brier) >= 0.10:
                dpo_rows.append(
                    json.dumps(
                        {
                            "prompt": best_prompt,
                            "chosen": best_json,
                            "rejected": worst_json,
                            "meta": {
                                "sample_id": row["sample_id"],
                                "chosen_brier": round(best_brier, 4),
                                "rejected_brier": round(worst_brier, 4),
                            },
                        },
                        ensure_ascii=False,
                    )
                )

    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.write_text("\n".join(sft_rows) + ("\n" if sft_rows else ""), encoding="utf-8")
    dpo_path.write_text("\n".join(dpo_rows) + ("\n" if dpo_rows else ""), encoding="utf-8")
    return len(sft_rows), len(dpo_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Path B: Masked Eval Runner, Level 1 Calibrator & Unsloth Exporter.")
    parser.add_argument("--run-eval", action="store_true", help="Run Council on masked historical samples.")
    parser.add_argument("--calibrate", action="store_true", help="Fit Level 1 L2 Logistic Calibrator.")
    parser.add_argument("--export-unsloth", action="store_true", help="Export ChatML SFT & DPO JSONL for Unsloth.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--runs", type=Path, default=DEFAULT_EVAL_RUNS_PATH)
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=8.0)
    args = parser.parse_args()

    if not (args.run_eval or args.calibrate or args.export_unsloth):
        parser.error("Specify at least one action: --run-eval, --calibrate, or --export-unsloth.")
        return 2

    if args.run_eval:
        rc = asyncio.run(
            run_masked_evaluation(
                dataset_path=args.dataset,
                runs_path=args.runs,
                model_name=args.model,
                debate_rounds=args.rounds,
                limit=args.limit,
                delay_seconds=args.delay,
            )
        )
        if rc != 0:
            return rc

    if args.calibrate:
        artifact = run_level1_calibration(runs_path=args.runs)
        vm = artifact["val_metrics"]
        print(
            f"[CALIBRATE] Saved {DEFAULT_CALIBRATOR_PATH} "
            f"(Train={artifact['n_train']}, Val={artifact['n_val']})\n"
            f"  Coin-Flip Brier:   {vm['coin_flip_brier']:.4f}\n"
            f"  Raw Council Brier: {vm['raw_council_brier']:.4f}\n"
            f"  Calibrated Brier:  {vm['calibrated_brier']:.4f} (Δ {vm['brier_improvement']:+.4f})\n"
            f"  Per-Agent Val Brier: {vm['per_agent_val_brier']}"
        )

    if args.export_unsloth:
        n_sft, n_dpo = export_unsloth_datasets(runs_path=args.runs)
        print(f"[UNSLOTH] Exported {n_sft} SFT rows -> {DEFAULT_SFT_PATH} | {n_dpo} DPO pairs -> {DEFAULT_DPO_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

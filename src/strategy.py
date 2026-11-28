"""Standalone Event-Driven Behavioral Mispricing Strategy (v2.3 — Zero Technical Indicators).

Converts raw Council + De-Biased Meta-Judge outputs into a deterministic, auditable
T+5D StrategyTicket with strict abstention gates and capped Half-Kelly sizing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import config
from src.schemas import (
    AgentVote,
    BehavioralMetrics,
    FactBriefing,
    MetaJudgePanelResult,
    SessionRecord,
    StrategyTicket,
    VerdictChoice,
)

DEFAULT_CALIBRATOR_PATH = config.DATA_DIR / "calibrator_v1.json"

EVENT_MIN_ABS_7D_MOVE_PCT = 5.0
EVENT_MIN_VERIFIED_FACTS = 3
LONG_PROB_THRESHOLD = 58.0
SHORT_PROB_THRESHOLD = 42.0
MAX_POLARIZATION_ALLOWED = 1.5
MAX_ALLOC_PCT = 25.0
CALIBRATOR_MIN_SAMPLES = 60
CALIBRATOR_MAX_VAL_BRIER = 0.235


def _try_apply_calibrator(
    *,
    stage3_votes: list[AgentVote],
    behavioral_metrics: BehavioralMetrics | None,
    price_change_7d_pct: float,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
) -> float | None:
    """Apply Level-1 L2 Logistic Calibrator ONLY when it passes the Phase 2 OOS gate."""
    if not calibrator_path.is_file():
        return None
    try:
        data: dict[str, Any] = json.loads(calibrator_path.read_text(encoding="utf-8"))
        n_total = int(data.get("n_total", 0))
        val_brier = float((data.get("val_metrics") or {}).get("calibrated_brier", 1.0))
        if n_total < CALIBRATOR_MIN_SAMPLES or val_brier >= CALIBRATOR_MAX_VAL_BRIER:
            return None
        weights: list[float] = list(data["weights"])
        bias = float(data["bias"])
    except Exception:
        return None

    vote_by_id = {v.agent_id: v for v in stage3_votes}
    bm = behavioral_metrics
    features = [
        vote_by_id.get("retail_fomo").prob_up_5d / 100.0 if "retail_fomo" in vote_by_id else 0.5,
        vote_by_id.get("fundamentalist").prob_up_5d / 100.0 if "fundamentalist" in vote_by_id else 0.5,
        vote_by_id.get("paranoid_bear").prob_up_5d / 100.0 if "paranoid_bear" in vote_by_id else 0.5,
        vote_by_id.get("contrarian").prob_up_5d / 100.0 if "contrarian" in vote_by_id else 0.5,
        vote_by_id.get("macro_analyst").prob_up_5d / 100.0 if "macro_analyst" in vote_by_id else 0.5,
        bm.herding_rate if bm else 0.0,
        bm.confidence_polarization if bm else 0.0,
        bm.stage3_entropy if bm else 1.0,
        max(-1.0, min(1.0, price_change_7d_pct / 20.0)),
    ]
    if len(weights) != len(features):
        return None
    logit = sum(w * x for w, x in zip(weights, features)) + bias
    prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
    return round(prob * 100.0, 1)


def _detect_anti_fomo_divergence(stage3_votes: list[AgentVote]) -> bool:
    """Flag when Retail FOMO opposes BOTH Contrarian and Macro Analyst."""
    by_id = {v.agent_id: v.vote for v in stage3_votes}
    fomo = by_id.get("retail_fomo")
    contrarian = by_id.get("contrarian")
    macro = by_id.get("macro_analyst")
    if fomo in ("BUY", "SELL") and contrarian == macro and contrarian in ("BUY", "SELL"):
        return fomo != contrarian
    return False


def build_strategy_ticket(
    *,
    briefing: FactBriefing,
    stage3_votes: list[AgentVote],
    final_verdict: VerdictChoice,
    behavioral_metrics: BehavioralMetrics | None,
    meta_judge_panel: MetaJudgePanelResult | None,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
) -> StrategyTicket:
    """Evaluate all 4 deterministic strategy gates and emit an actionable or abstained ticket."""
    notes: list[str] = []

    # 1. Resolve calibrated or ensemble probability
    cal_prob = _try_apply_calibrator(
        stage3_votes=stage3_votes,
        behavioral_metrics=behavioral_metrics,
        price_change_7d_pct=briefing.price_change_7d_pct,
        calibrator_path=calibrator_path,
    )
    if cal_prob is not None:
        effective_prob = cal_prob
        prob_source = "calibrator_v1"
    elif meta_judge_panel is not None and meta_judge_panel.ensemble_prob_up_5d is not None:
        effective_prob = meta_judge_panel.ensemble_prob_up_5d
        prob_source = "meta_judge_ensemble"
    elif stage3_votes:
        effective_prob = round(sum(v.prob_up_5d for v in stage3_votes) / len(stage3_votes), 1)
        prob_source = "council_stage3"
    else:
        effective_prob = 50.0
        prob_source = "fallback_neutral"

    # Gate 1: Event / Dislocation Trigger
    abs_7d = abs(briefing.price_change_7d_pct)
    facts_count = len(briefing.verified_facts)
    event_passed = abs_7d >= EVENT_MIN_ABS_7D_MOVE_PCT or facts_count >= EVENT_MIN_VERIFIED_FACTS
    if event_passed:
        notes.append(f"PASS Event Gate: |7D|={abs_7d:.2f}%, verified_facts={facts_count}.")
    else:
        notes.append(
            f"FAIL Event Gate: |7D|={abs_7d:.2f}% (<{EVENT_MIN_ABS_7D_MOVE_PCT}%) "
            f"and verified_facts={facts_count} (<{EVENT_MIN_VERIFIED_FACTS})."
        )

    # Gate 2: Probability Edge Threshold
    desired_side: str | None = None
    if effective_prob >= LONG_PROB_THRESHOLD:
        desired_side = "LONG"
    elif effective_prob <= SHORT_PROB_THRESHOLD:
        desired_side = "SHORT"
    edge_passed = desired_side is not None
    if edge_passed:
        notes.append(f"PASS Edge Gate: P(Up 5D)={effective_prob:.1f}% ({prob_source}) -> {desired_side}.")
    else:
        notes.append(
            f"FAIL Edge Gate: P(Up 5D)={effective_prob:.1f}% inside abstention zone "
            f"({SHORT_PROB_THRESHOLD:.0f}%–{LONG_PROB_THRESHOLD:.0f}%)."
        )

    # Gate 3: Meta-Judge / Council Consensus
    expected_vote = "BUY" if desired_side == "LONG" else ("SELL" if desired_side == "SHORT" else None)
    consensus_passed = False
    if expected_vote is not None:
        ok_judges = [
            d for d in (meta_judge_panel.shadow_decisions if meta_judge_panel else []) if d.status == "OK"
        ]
        if len(ok_judges) >= 2:
            agree_count = sum(1 for d in ok_judges if d.verdict == expected_vote)
            needed = math.ceil((2 * len(ok_judges)) / 3)
            consensus_passed = agree_count >= needed
            notes.append(
                f"{'PASS' if consensus_passed else 'FAIL'} Consensus Gate: "
                f"{agree_count}/{len(ok_judges)} OK Meta-Judges voted {expected_vote}."
            )
        elif len(ok_judges) == 1:
            consensus_passed = ok_judges[0].verdict == expected_vote and final_verdict == expected_vote
            notes.append(
                f"{'PASS' if consensus_passed else 'FAIL'} Consensus Gate: "
                f"1 OK Meta-Judge ({ok_judges[0].verdict}) + Council ({final_verdict})."
            )
        else:
            consensus_passed = final_verdict == expected_vote
            notes.append(
                f"{'PASS' if consensus_passed else 'FAIL'} Consensus Gate: Council verdict={final_verdict}."
            )
    else:
        notes.append("SKIP Consensus Gate: No directional edge.")

    # Gate 4: Confidence Polarization Guard
    polar = behavioral_metrics.confidence_polarization if behavioral_metrics else 0.0
    polarization_passed = polar < MAX_POLARIZATION_ALLOWED
    notes.append(
        f"{'PASS' if polarization_passed else 'FAIL'} Polarization Gate: "
        f"conf_polarization={polar:+.2f} (limit < +{MAX_POLARIZATION_ALLOWED:.1f})."
    )

    anti_fomo = _detect_anti_fomo_divergence(stage3_votes)
    if anti_fomo:
        notes.append("FLAG Anti-FOMO Divergence: Retail FOMO opposes both Contrarian & Macro Analyst.")

    all_passed = event_passed and edge_passed and consensus_passed and polarization_passed and desired_side is not None
    if all_passed and desired_side in ("LONG", "SHORT"):
        raw_edge = abs((effective_prob / 100.0) - 0.50) * 2.0
        alloc_pct = min(MAX_ALLOC_PCT, round(raw_edge * 50.0, 1))
        action = desired_side
    else:
        alloc_pct = 0.0
        action = "NO TRADE"

    return StrategyTicket(
        action=action,
        horizon_days=5,
        alloc_pct=alloc_pct,
        effective_prob_up_5d=effective_prob,
        prob_source=prob_source,
        event_trigger_passed=event_passed,
        edge_gate_passed=edge_passed,
        consensus_gate_passed=consensus_passed,
        polarization_gate_passed=polarization_passed,
        anti_fomo_divergence=anti_fomo,
        gate_notes=notes,
    )


def evaluate_record_strategy(
    record: SessionRecord,
    *,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
) -> StrategyTicket:
    """Compute or recompute a StrategyTicket for any SessionRecord (including legacy v1.0/v2.1)."""
    return build_strategy_ticket(
        briefing=record.fact_briefing,
        stage3_votes=record.stage3_votes,
        final_verdict=record.final_verdict,
        behavioral_metrics=record.behavioral_metrics,
        meta_judge_panel=record.meta_judge_panel,
        calibrator_path=calibrator_path,
    )

"""Unit tests for v2.2 (Stability & De-Biased Meta-Judges) and v2.3 (Standalone Strategy & Paper Portfolio)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from src.evaluator import evaluate_all_history_sessions
from src.orchestrator import (
    CouncilOrchestrator,
    build_blind_peer_aliases,
    compute_ensemble_prob_up_5d,
    select_peer_winner,
)
from src.paper_portfolio import rebuild_paper_portfolio_from_history
from src.prompts import build_peer_selection_prompt
from src.schemas import (
    AgentVote,
    BehavioralMetrics,
    FactBriefing,
    JudgeTrackRecord,
    MetaJudgePanelResult,
    MetaJudgeShadowDecision,
    PeerSelectionVote,
)
from src.strategy import build_strategy_ticket
from tests.test_orchestrator import MockCouncilLLMClient, MockMetaJudgeClient, _sample_market_context


def _sample_briefing(pct_7d: float = 8.5, facts_count: int = 3) -> FactBriefing:
    return FactBriefing(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        as_of_utc="2026-11-28T10:00:00Z",
        price_start_7d=140.0,
        price_start_date="2026-11-20",
        price_end=151.9,
        price_end_date="2026-11-27",
        price_change_7d_pct=pct_7d,
        executive_summary="Summary.",
        verified_facts=[f"Fact {i}" for i in range(facts_count)],
        key_catalysts=["Catalyst 1"],
        key_risks=["Risk 1"],
    )


def test_double_blind_peer_selection_redacts_models_and_judge_ids() -> None:
    """Peer selection prompt must NEVER leak model names or raw judge_1/2/3 labels."""
    shadows = [
        MetaJudgeShadowDecision(judge_id="judge_1", model_name="gemini-3.6-flash", status="OK", verdict="BUY", confidence=7, prob_up_5d=62, position_pct=25, minority_counterpoint="Export risk.", reasoning="Strong cloud demand."),
        MetaJudgeShadowDecision(judge_id="judge_2", model_name="qwen3.5:4b", status="OK", verdict="HOLD", confidence=6, prob_up_5d=51, position_pct=0, minority_counterpoint="Upside squeeze.", reasoning="Balanced risk."),
        MetaJudgeShadowDecision(judge_id="judge_3", model_name="claude-3-5-sonnet", status="OK", verdict="BUY", confidence=8, prob_up_5d=66, position_pct=35, minority_counterpoint="Valuation.", reasoning="Catalyst strong."),
    ]
    scoreboard = {
        "judge_1": JudgeTrackRecord(judge_id="judge_1", model_name="gemini-3.6-flash", evaluated_runs=3, shrunk_brier_5d=0.24),
        "judge_2": JudgeTrackRecord(judge_id="judge_2", model_name="qwen3.5:4b", evaluated_runs=2, shrunk_brier_5d=0.25),
        "judge_3": JudgeTrackRecord(judge_id="judge_3", model_name="claude-3-5-sonnet", evaluated_runs=4, shrunk_brier_5d=0.23),
    }

    blind_pairs, alias_map = build_blind_peer_aliases("judge_1", shadows, seed_key="NVDA:sess1")
    prompt = build_peer_selection_prompt(
        briefing=_sample_briefing(),
        voter_judge_id="judge_1",
        shadow_decisions=shadows,
        scoreboard=scoreboard,
        blind_candidates=blind_pairs,
    )

    # Zero model names and zero raw judge IDs may appear anywhere in peer prompt!
    for forbidden in ("gemini-3.6-flash", "qwen3.5:4b", "claude-3-5-sonnet", "judge_1", "judge_2", "judge_3"):
        assert forbidden not in prompt
    assert "Candidate Alpha" in prompt
    assert "Candidate Beta" in prompt
    assert "Calibrating" in prompt  # evaluated_n < 10 hides raw win_rate
    assert set(alias_map.values()) == {"judge_2", "judge_3"}


def test_cold_start_tie_breaker_rotates_without_judge1_lockin() -> None:
    """Cold-start ties must rotate deterministically by session seed instead of always picking judge_1."""
    shadows = [
        MetaJudgeShadowDecision(judge_id="judge_1", model_name="m1", status="OK", verdict="BUY", confidence=7, prob_up_5d=60, position_pct=20, reasoning="r"),
        MetaJudgeShadowDecision(judge_id="judge_2", model_name="m2", status="OK", verdict="SELL", confidence=7, prob_up_5d=40, position_pct=-20, reasoning="r"),
        MetaJudgeShadowDecision(judge_id="judge_3", model_name="m3", status="OK", verdict="HOLD", confidence=6, prob_up_5d=50, position_pct=0, reasoning="r"),
    ]
    # 3-way tie: 1 vote each
    peer_votes = [
        PeerSelectionVote(voter_judge_id="judge_1", voter_model_name="m1", selected_judge_id="judge_2", reasoning="r"),
        PeerSelectionVote(voter_judge_id="judge_2", voter_model_name="m2", selected_judge_id="judge_3", reasoning="r"),
        PeerSelectionVote(voter_judge_id="judge_3", voter_model_name="m3", selected_judge_id="judge_1", reasoning="r"),
    ]
    cold_scoreboard = {
        "judge_1": JudgeTrackRecord(judge_id="judge_1", model_name="m1", evaluated_runs=0),
        "judge_2": JudgeTrackRecord(judge_id="judge_2", model_name="m2", evaluated_runs=0),
        "judge_3": JudgeTrackRecord(judge_id="judge_3", model_name="m3", evaluated_runs=0),
    }

    winners = set()
    for i in range(25):
        w, tie_used, _ = select_peer_winner(peer_votes, shadows, cold_scoreboard, seed_key=f"SEED:{i}")
        assert tie_used is True
        winners.add(w)
    assert winners == {"judge_1", "judge_2", "judge_3"}


def test_ensemble_prob_weights_better_shrunk_brier_higher() -> None:
    shadows = [
        MetaJudgeShadowDecision(judge_id="judge_1", model_name="m1", status="OK", verdict="BUY", confidence=8, prob_up_5d=80, position_pct=40, reasoning="r"),
        MetaJudgeShadowDecision(judge_id="judge_2", model_name="m2", status="OK", verdict="SELL", confidence=7, prob_up_5d=40, position_pct=-20, reasoning="r"),
    ]
    sb = {
        "judge_1": JudgeTrackRecord(judge_id="judge_1", model_name="m1", shrunk_brier_5d=0.10),
        "judge_2": JudgeTrackRecord(judge_id="judge_2", model_name="m2", shrunk_brier_5d=0.30),
    }
    ens = compute_ensemble_prob_up_5d(shadows, sb)
    # Weight ratio 1/0.10 : 1/0.30 = 3 : 1 -> (3*80 + 1*40)/4 = 70.0
    assert ens == 70.0


def test_standalone_strategy_gates_and_deterministic_sizing(tmp_path: Path) -> None:
    """Verify Event Gate, Edge Gate, Consensus Gate, Polarization Guard, Anti-FOMO flag, and <=25% cap."""
    votes = [
        AgentVote(agent_id="retail_fomo", agent_name="The Retail FOMO", vote="SELL", confidence=8, prob_up_5d=35, position_pct=-40, reasoning="panic"),
        AgentVote(agent_id="fundamentalist", agent_name="The Fundamentalist", vote="BUY", confidence=7, prob_up_5d=62, position_pct=25, reasoning="ok"),
        AgentVote(agent_id="paranoid_bear", agent_name="The Paranoid Bear", vote="HOLD", confidence=6, prob_up_5d=48, position_pct=0, reasoning="caution"),
        AgentVote(agent_id="contrarian", agent_name="The Contrarian", vote="BUY", confidence=8, prob_up_5d=68, position_pct=40, reasoning="dip"),
        AgentVote(agent_id="macro_analyst", agent_name="The Macro Analyst", vote="BUY", confidence=7, prob_up_5d=64, position_pct=30, reasoning="tailwind"),
    ]
    panel_bull = MetaJudgePanelResult(
        shadow_decisions=[
            MetaJudgeShadowDecision(judge_id="judge_1", model_name="m1", status="OK", verdict="BUY", confidence=7, prob_up_5d=66, position_pct=30, reasoning="r"),
            MetaJudgeShadowDecision(judge_id="judge_2", model_name="m2", status="OK", verdict="BUY", confidence=7, prob_up_5d=64, position_pct=25, reasoning="r"),
            MetaJudgeShadowDecision(judge_id="judge_3", model_name="m3", status="OK", verdict="HOLD", confidence=6, prob_up_5d=54, position_pct=5, reasoning="r"),
        ],
        ensemble_prob_up_5d=64.0,
    )
    bm_clean = BehavioralMetrics(
        opinion_shift_rate=0.2,
        herding_rate=0.0,
        confidence_polarization=0.4,
        stage1_entropy=1.3,
        stage3_entropy=1.2,
        mean_prob_up_5d_stage1=55.0,
        mean_prob_up_5d_stage3=58.0,
        mean_position_pct_stage1=10.0,
        mean_position_pct_stage3=15.0,
    )

    # 1. All 4 gates pass -> LONG with deterministic sizing <= 25% & Anti-FOMO flag True
    t_long = build_strategy_ticket(
        briefing=_sample_briefing(pct_7d=-7.2, facts_count=3),
        stage3_votes=votes,
        final_verdict="BUY",
        behavioral_metrics=bm_clean,
        meta_judge_panel=panel_bull,
        calibrator_path=tmp_path / "none.json",
    )
    assert t_long.action == "LONG"
    assert 0.0 < t_long.alloc_pct <= 25.0
    assert t_long.anti_fomo_divergence is True

    # 2. Quiet day (<5% move and <3 facts) -> NO TRADE
    t_quiet = build_strategy_ticket(
        briefing=_sample_briefing(pct_7d=1.2, facts_count=1),
        stage3_votes=votes,
        final_verdict="BUY",
        behavioral_metrics=bm_clean,
        meta_judge_panel=panel_bull,
        calibrator_path=tmp_path / "none.json",
    )
    assert t_quiet.action == "NO TRADE"
    assert t_quiet.event_trigger_passed is False

    # 3. High confidence polarization (>= 1.5) -> NO TRADE
    bm_polar = bm_clean.model_copy(update={"confidence_polarization": 1.8})
    t_polar = build_strategy_ticket(
        briefing=_sample_briefing(pct_7d=-7.2, facts_count=3),
        stage3_votes=votes,
        final_verdict="BUY",
        behavioral_metrics=bm_polar,
        meta_judge_panel=panel_bull,
        calibrator_path=tmp_path / "none.json",
    )
    assert t_polar.action == "NO TRADE"
    assert t_polar.polarization_gate_passed is False


def test_paper_portfolio_and_judge_uptime_idempotent_settlement(tmp_path: Path) -> None:
    """Verify uptime_pct, Bayesian shrinkage, and idempotent T+5D Paper Portfolio settlement."""
    mock_council = MockCouncilLLMClient(delay_seconds=0.01)
    judge_clients = {
        "judge_1": MockMetaJudgeClient(model_name="gemini-3.5-flash-lite", verdict="BUY", prob_up_5d=70, peer_pick="Candidate Alpha"),
        "judge_2": MockMetaJudgeClient(model_name="gemini-3.1-flash-lite", verdict="BUY", prob_up_5d=68, peer_pick="Candidate Alpha"),
        "judge_3": MockMetaJudgeClient(model_name="gemini-3.6-flash", fail=True),
    }
    sb_path = tmp_path / "judge_scoreboard.json"
    port_path = tmp_path / "paper_portfolio.json"

    orch = CouncilOrchestrator(
        llm_client=mock_council,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
        judge_clients=judge_clients,
        scoreboard_path=sb_path,
    )
    rec = asyncio.run(orch.run_simulation("NVDA", debate_rounds=1, enable_counterfactual_lab=True, save_to_disk=True))
    assert rec.strategy_ticket is not None
    assert rec.strategy_ticket.action == "LONG"

    def fake_closes(symbol: str, start_date: str) -> pd.Series:
        dates = pd.to_datetime(["2026-11-25", "2026-11-26", "2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02"])
        if symbol == "SPY":
            return pd.Series([500.0, 501.0, 502.0, 503.0, 504.0, 505.0], index=dates)
        return pd.Series([165.0, 168.0, 171.0, 175.0, 179.0, 181.5], index=dates)  # +10%

    _, sb = evaluate_all_history_sessions(history_dir=tmp_path, scoreboard_path=sb_path, close_fetcher=fake_closes)
    assert sb.judges["judge_1"].uptime_pct == 1.0
    assert sb.judges["judge_3"].uptime_pct == 0.0
    assert sb.judges["judge_3"].attempted_runs == 1
    assert sb.judges["judge_1"].shrunk_brier_5d is not None and sb.judges["judge_1"].shrunk_brier_5d < 0.25

    p1 = rebuild_paper_portfolio_from_history(history_dir=tmp_path, portfolio_path=port_path)
    p2 = rebuild_paper_portfolio_from_history(history_dir=tmp_path, portfolio_path=port_path)
    assert p1.settled_trades == 1
    assert p1.wins == 1
    assert p1.win_rate == 1.0
    assert p1.cumulative_portfolio_return_pct > 0.0
    assert p2.settled_trades == p1.settled_trades
    assert p2.cumulative_portfolio_return_pct == p1.cumulative_portfolio_return_pct

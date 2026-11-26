"""In-memory UI Layout Preview builder so every v2.1 dashboard panel can be inspected without an API key."""

from __future__ import annotations

import config
from src.orchestrator import compute_behavioral_metrics
from src.prompts import INVESTOR_PERSONAS
from src.schemas import (
    AgentVote,
    BranchResult,
    DebateTurn,
    FactBriefing,
    MetaJudgePanelResult,
    MetaJudgeShadowDecision,
    OpinionShift,
    PeerSelectionVote,
    RawNewsItem,
    SessionRecord,
    VoteTally,
)


def build_layout_preview_record(
    ticker: str,
    model_name: str,
    debate_rounds: int,
    judge_models: tuple[str, str, str],
) -> SessionRecord:
    """Build an in-memory wireframe SessionRecord so every v2.1 UI section is visible without an API key."""
    sym = (ticker or "TICKER").strip().upper()
    briefing = FactBriefing(
        ticker=sym,
        company_name=f"{sym} (UI Layout Preview — No API Call)",
        as_of_utc="2026-11-26T00:00:00Z",
        price_start_7d=100.00,
        price_start_date="2026-11-18",
        price_end=104.25,
        price_end_date="2026-11-25",
        price_change_7d_pct=4.25,
        executive_summary=(
            "[Preview Placeholder] Researcher Agent executive summary — strictly 100% curated recent news + "
            "7-day price change % (zero technical indicators)."
        ),
        verified_facts=[
            "[Preview] Verified fact #1 extracted from deduplicated Yahoo Finance news.",
            "[Preview] Verified fact #2 anchored strictly to reported headlines.",
        ],
        key_catalysts=["[Preview] Upside catalyst supported by verified news."],
        key_risks=["[Preview] Downside risk supported by verified news."],
        discarded_noise_count=3,
        news_items_reviewed=10,
        data_warning="PREVIEW MODE — Displaying template placeholders only. Click 'Run Simulation' for live analysis.",
    )

    stage1_plan = [
        ("BUY", 8, 68, 60, "[Preview] Stage 1 blind vote reasoning — hidden from other agents."),
        ("HOLD", 6, 52, 5, "[Preview] Stage 1 blind vote reasoning — hidden from other agents."),
        ("SELL", 7, 34, -50, "[Preview] Stage 1 blind vote reasoning — hidden from other agents."),
        ("SELL", 6, 40, -30, "[Preview] Stage 1 blind vote reasoning — hidden from other agents."),
        ("BUY", 7, 62, 40, "[Preview] Stage 1 blind vote reasoning — hidden from other agents."),
    ]
    stage3_plan = [
        ("BUY", 8, 70, 65, "[Preview] Stage 3 final vote — held BUY after debate."),
        ("BUY", 7, 60, 35, "[Preview] Stage 3 final vote — shifted from HOLD to BUY after debate."),
        ("SELL", 8, 30, -60, "[Preview] Stage 3 final vote — held SELL."),
        ("HOLD", 6, 48, 0, "[Preview] Stage 3 final vote — shifted from SELL to HOLD."),
        ("BUY", 8, 66, 50, "[Preview] Stage 3 final vote — held BUY."),
    ]

    stage1_votes = [
        AgentVote(agent_id=p.agent_id, agent_name=p.name, vote=v, confidence=c, prob_up_5d=pr, position_pct=pos, reasoning=r)
        for p, (v, c, pr, pos, r) in zip(INVESTOR_PERSONAS, stage1_plan)
    ]
    stage3_votes = [
        AgentVote(agent_id=p.agent_id, agent_name=p.name, vote=v, confidence=c, prob_up_5d=pr, position_pct=pos, reasoning=r)
        for p, (v, c, pr, pos, r) in zip(INVESTOR_PERSONAS, stage3_plan)
    ]

    transcript: list[DebateTurn] = []
    turn_idx = 0
    for r in range(1, max(1, debate_rounds) + 1):
        for p in INVESTOR_PERSONAS:
            turn_idx += 1
            transcript.append(
                DebateTurn(
                    round_number=r,
                    turn_index=turn_idx,
                    agent_id=p.agent_id,
                    agent_name=p.name,
                    message=f"[Preview · Round {r}] {p.name} reacts to the anonymized Stage 1 tally in 2-3 sharp sentences.",
                )
            )

    shifts = [
        OpinionShift(
            agent_id=s1.agent_id,
            agent_name=s1.agent_name,
            stage1_vote=s1.vote,
            stage1_confidence=s1.confidence,
            stage1_prob_up_5d=s1.prob_up_5d,
            stage3_vote=s3.vote,
            stage3_confidence=s3.confidence,
            stage3_prob_up_5d=s3.prob_up_5d,
            changed_vote=s1.vote != s3.vote,
            confidence_delta=s3.confidence - s1.confidence,
            prob_delta=s3.prob_up_5d - s1.prob_up_5d,
        )
        for s1, s3 in zip(stage1_votes, stage3_votes)
    ]
    bm = compute_behavioral_metrics(stage1_votes, stage3_votes)

    branches = [
        BranchResult(
            branch_id="baseline_single",
            branch_name="Branch A · Single-Agent Baseline",
            description="1 neutral analyst control.",
            verdict="HOLD",
            tally=VoteTally(BUY=0, HOLD=1, SELL=0),
            mean_prob_up_5d=53.0,
            mean_confidence=6.0,
            mean_position_pct=10.0,
        ),
        BranchResult(
            branch_id="isolated_crowd",
            branch_name="Branch B · Isolated Crowd",
            description="5 blind votes, zero debate.",
            verdict="HOLD (Split Decision)",
            tally=VoteTally(BUY=2, HOLD=1, SELL=2),
            mean_prob_up_5d=51.2,
            mean_confidence=6.8,
            mean_position_pct=5.0,
        ),
        BranchResult(
            branch_id="fog_of_war",
            branch_name="Branch C · Fog-of-War Council",
            description="Anonymized Stage 1 tally debate.",
            verdict="BUY",
            tally=VoteTally(BUY=3, HOLD=1, SELL=1),
            mean_prob_up_5d=54.8,
            mean_confidence=7.4,
            mean_position_pct=18.0,
            behavioral_metrics=bm,
        ),
        BranchResult(
            branch_id="full_herd",
            branch_name="Branch D · Full-Herd Council",
            description="Public Stage 1 identities/votes exposed.",
            verdict="BUY",
            tally=VoteTally(BUY=4, HOLD=0, SELL=1),
            mean_prob_up_5d=61.0,
            mean_confidence=8.2,
            mean_position_pct=34.0,
            behavioral_metrics=bm,
        ),
    ]

    shadows = [
        MetaJudgeShadowDecision(
            judge_id="judge_1",
            model_name=judge_models[0],
            status="OK",
            verdict="BUY",
            confidence=7,
            prob_up_5d=58,
            position_pct=25,
            reasoning="[Preview] Neutral cross-branch synthesis noting mild herding in Branch D vs. balanced Branch C.",
        ),
        MetaJudgeShadowDecision(
            judge_id="judge_2",
            model_name=judge_models[1],
            status="OK",
            verdict="HOLD",
            confidence=6,
            prob_up_5d=52,
            position_pct=5,
            reasoning="[Preview] Neutral synthesis weighting Branch B split and Baseline HOLD against debate polarization.",
        ),
        MetaJudgeShadowDecision(
            judge_id="judge_3",
            model_name=judge_models[2],
            status="OK",
            verdict="BUY",
            confidence=7,
            prob_up_5d=56,
            position_pct=20,
            reasoning="[Preview] Neutral synthesis favoring positive catalyst momentum with capped position sizing.",
        ),
    ]
    peer_votes = [
        PeerSelectionVote(voter_judge_id="judge_1", voter_model_name=judge_models[0], selected_judge_id="judge_2", reasoning="[Preview] Balanced calibration."),
        PeerSelectionVote(voter_judge_id="judge_2", voter_model_name=judge_models[1], selected_judge_id="judge_1", reasoning="[Preview] Clear herding audit."),
        PeerSelectionVote(voter_judge_id="judge_3", voter_model_name=judge_models[2], selected_judge_id="judge_1", reasoning="[Preview] Best cross-branch synthesis."),
    ]
    panel = MetaJudgePanelResult(
        shadow_decisions=shadows,
        peer_votes=peer_votes,
        selected_final_judge_id="judge_1",
        selected_final_model_name=judge_models[0],
        final_decision=shadows[0],
    )

    return SessionRecord(
        schema_version="2.1",
        session_id="PREVIEW",
        created_at_utc="2026-11-26T00:00:00Z",
        ticker=sym,
        model_name=model_name or config.DEFAULT_MODEL,
        debate_rounds=debate_rounds,
        raw_news=[
            RawNewsItem(
                title=f"[Preview] Sample Yahoo Finance headline for {sym}",
                publisher="Sample Wire",
                published_at="2026-11-25T18:00:00Z",
                summary="Placeholder summary for raw news audit drawer.",
            ),
        ],
        fact_briefing=briefing,
        stage1_votes=stage1_votes,
        stage1_tally=VoteTally(BUY=2, HOLD=1, SELL=2),
        debate_transcript=transcript,
        stage3_votes=stage3_votes,
        stage3_tally=VoteTally(BUY=3, HOLD=1, SELL=1),
        final_verdict="BUY",
        opinion_shifts=shifts,
        behavioral_metrics=bm,
        counterfactual_branches=branches,
        meta_judge_panel=panel,
    )

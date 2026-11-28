"""Unit tests for Stock Council Simulator (v1.0 + v2.0 Behavioral Metrics + v2.1 Parallel Lab & 3 Meta-Judges)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pytest
from pydantic import BaseModel

from src.data_fetcher import (
    TickerDataError,
    _extract_7d_price_window,
    _parse_news_items,
    fetch_market_context,
)
from src.evaluator import (
    evaluate_all_history_sessions,
    evaluate_session_forward,
    rebuild_judge_scoreboard_from_history,
)
from src.llm_client import GeminiLLMClient, MissingAPIKeyError
from src.orchestrator import (
    CouncilOrchestrator,
    compute_anonymized_tally,
    compute_behavioral_metrics,
    compute_verdict,
    select_peer_winner,
)
from src.prompts import INVESTOR_PERSONAS
from src.schemas import (
    AgentVote,
    AgentVoteOutput,
    DebateMessageOutput,
    JudgeTrackRecord,
    MarketContext,
    MetaJudgeDecisionOutput,
    MetaJudgeShadowDecision,
    PeerSelectionVote,
    PeerSelectionVoteOutput,
    RawNewsItem,
    ResearcherBriefingOutput,
    VoteTally,
)
from src.storage import list_sessions, load_session, save_session

TModel = TypeVar("TModel", bound=BaseModel)


def _sample_market_context(ticker: str = "NVDA") -> MarketContext:
    return MarketContext(
        ticker=ticker,
        company_name="NVIDIA Corporation",
        fetched_at_utc="2026-11-26T05:00:00Z",
        price_start_7d=150.00,
        price_start_date="2026-11-18",
        price_end=165.00,
        price_end_date="2026-11-25",
        price_change_7d_pct=10.0,
        news_items=[
            RawNewsItem(
                title="NVIDIA unveils next-gen enterprise AI rack shipment schedule",
                publisher="Reuters",
                published_at="2026-11-25T18:00:00Z",
                summary="Enterprise cloud buyers confirmed Q1 rack deliveries remain on track.",
                url="https://example.com/nvda-racks",
            ),
            RawNewsItem(
                title="Export licensing scrutiny rises for high-end accelerator sales",
                publisher="Bloomberg",
                published_at="2026-11-24T14:30:00Z",
                summary="Regulators signaled tighter review timelines for overseas orders.",
                url="https://example.com/nvda-export",
            ),
            RawNewsItem(
                title="10 Reasons You Must Buy Chip Stocks Right Now!!",
                publisher="Clickbait Finance",
                published_at="2026-11-24T10:00:00Z",
                summary="Sensationalist hype article with no new data.",
                url="https://example.com/clickbait",
            ),
        ],
    )


class MockCouncilLLMClient:
    """Deterministic fake StructuredLLMClient that logs prompts and measures true concurrency."""

    def __init__(
        self,
        *,
        stage1_plan: dict[str, tuple[str, int, int, int, str]] | None = None,
        stage3_plan: dict[str, tuple[str, int, int, int, str]] | None = None,
        delay_seconds: float = 0.04,
    ) -> None:
        self.model_name = "mock-gemini-3.5-flash-lite"
        self.delay_seconds = delay_seconds
        self.stage1_plan = stage1_plan or {
            "The Retail FOMO": ("BUY", 9, 75, 70, "SECRET_S1_FOMO_TOKEN_111 Momentum is electric!"),
            "The Fundamentalist": ("HOLD", 6, 52, 5, "SECRET_S1_FUND_TOKEN_222 Need margin clarity."),
            "The Paranoid Bear": ("SELL", 8, 28, -60, "SECRET_S1_BEAR_TOKEN_333 Export risk is ignored."),
            "The Contrarian": ("SELL", 7, 38, -35, "SECRET_S1_CONTR_TOKEN_444 +10% move looks crowded."),
            "The Macro Analyst": ("BUY", 7, 65, 45, "SECRET_S1_MACRO_TOKEN_555 Hyperscaler capex strong."),
        }
        self.stage3_plan = stage3_plan or {
            "The Retail FOMO": ("BUY", 9, 78, 75, "Debate confirmed hyperscaler demand dominates."),
            "The Fundamentalist": ("BUY", 7, 62, 35, "Shifted to BUY after Macro Analyst clarified capex durability."),
            "The Paranoid Bear": ("SELL", 8, 25, -65, "Still SELL; export risk remains unresolved."),
            "The Contrarian": ("HOLD", 6, 48, 0, "Shifted from SELL to HOLD as arguments look balanced."),
            "The Macro Analyst": ("BUY", 8, 70, 55, "Maintained BUY; structural tailwind intact."),
        }

        self.stage0_prompts: list[str] = []
        self.stage1_prompts: list[tuple[str, str]] = []
        self.stage2_prompts: list[tuple[str, str]] = []
        self.stage3_prompts: list[tuple[str, str]] = []

        self._active: dict[str, int] = {"stage1": 0, "stage2": 0, "stage3": 0}
        self.max_concurrent: dict[str, int] = {"stage1": 0, "stage2": 0, "stage3": 0}

    def _detect_persona_name(self, system_prompt: str) -> str:
        for persona in INVESTOR_PERSONAS:
            if persona.name in system_prompt:
                return persona.name
        if "Control Baseline" in system_prompt:
            return "Baseline"
        return "Researcher"

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[TModel],
        temperature: float = 0.4,
    ) -> TModel:
        persona_name = self._detect_persona_name(system_prompt)

        if response_schema is ResearcherBriefingOutput:
            self.stage0_prompts.append(user_prompt)
            return response_schema.model_validate(
                {
                    "executive_summary": "NVDA rose +10.00% over 7 days on confirmed Q1 rack deliveries despite export scrutiny.",
                    "verified_facts": [
                        "Enterprise cloud buyers confirmed Q1 AI rack deliveries remain on track.",
                        "Regulators signaled tighter review timelines on certain overseas accelerator orders.",
                    ],
                    "key_catalysts": ["Confirmed Q1 enterprise rack delivery schedule."],
                    "key_risks": ["Tighter overseas export licensing review timelines."],
                    "discarded_noise_count": 1,
                }
            )

        if response_schema is DebateMessageOutput:
            is_fog = "ANONYMIZED STAGE 1 BLIND VOTE TALLY" in user_prompt
            if is_fog:
                self._active["stage2"] += 1
                self.max_concurrent["stage2"] = max(self.max_concurrent["stage2"], self._active["stage2"])
                self.stage2_prompts.append((persona_name, user_prompt))
            try:
                await asyncio.sleep(self.delay_seconds / 4)
                return response_schema.model_validate(
                    {
                        "message": (
                            f"{persona_name} argues based on verified rack deliveries and export scrutiny. "
                            "The council must weigh execution versus regulatory drag."
                        )
                    }
                )
            finally:
                if is_fog:
                    self._active["stage2"] -= 1

        if response_schema is AgentVoteOutput:
            if persona_name == "Baseline":
                return response_schema.model_validate(
                    {
                        "vote": "HOLD",
                        "confidence": 6,
                        "prob_up_5d": 54,
                        "position_pct": 10,
                        "reasoning": "Balanced catalysts and regulatory risks.",
                    }
                )

            is_stage1 = "STAGE 1 — FIRST DESTINATION" in user_prompt
            bucket = "stage1" if is_stage1 else "stage3"
            self._active[bucket] += 1
            self.max_concurrent[bucket] = max(self.max_concurrent[bucket], self._active[bucket])
            try:
                if is_stage1:
                    self.stage1_prompts.append((persona_name, user_prompt))
                    vote, conf, prob, pos, reason = self.stage1_plan[persona_name]
                else:
                    self.stage3_prompts.append((persona_name, user_prompt))
                    vote, conf, prob, pos, reason = self.stage3_plan[persona_name]
                await asyncio.sleep(self.delay_seconds)
                return response_schema.model_validate(
                    {
                        "vote": vote,
                        "confidence": conf,
                        "prob_up_5d": prob,
                        "position_pct": pos,
                        "reasoning": reason,
                    }
                )
            finally:
                self._active[bucket] -= 1

        raise AssertionError(f"Unexpected schema requested: {response_schema}")


class MockMetaJudgeClient:
    """Fake Meta-Judge client simulating distinct models, peer votes, and optional 503 failure."""

    def __init__(
        self,
        *,
        model_name: str,
        verdict: str = "BUY",
        prob_up_5d: int = 62,
        peer_pick: str = "judge_1",
        fail: bool = False,
    ) -> None:
        self.model_name = model_name
        self.verdict = verdict
        self.prob_up_5d = prob_up_5d
        self.peer_pick = peer_pick
        self.fail = fail

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[TModel],
        temperature: float = 0.2,
    ) -> TModel:
        if self.fail:
            raise RuntimeError("503 UNAVAILABLE simulated failure")

        if response_schema is MetaJudgeDecisionOutput:
            return response_schema.model_validate(
                {
                    "verdict": self.verdict,
                    "confidence": 7,
                    "prob_up_5d": self.prob_up_5d,
                    "position_pct": 30 if self.verdict == "BUY" else (-30 if self.verdict == "SELL" else 0),
                    "reasoning": f"Neutral evaluation by {self.model_name} across parallel branches.",
                }
            )
        if response_schema is PeerSelectionVoteOutput:
            return response_schema.model_validate(
                {
                    "selected_judge_id": self.peer_pick,
                    "reasoning": f"Selected {self.peer_pick} based on calibration and rigor.",
                }
            )
        raise AssertionError(f"Unexpected schema for Meta-Judge: {response_schema}")


@pytest.mark.asyncio
async def test_stage2_anonymization_firewall_and_self_memory_no_leak(tmp_path: Path) -> None:
    """Stage 2 prompts include anonymized tally + own numeric prior ONLY; NEVER leak Stage 1 reasons/other names."""
    mock_llm = MockCouncilLLMClient()
    orchestrator = CouncilOrchestrator(
        llm_client=mock_llm,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
    )

    record = await orchestrator.run_simulation("NVDA", debate_rounds=2, save_to_disk=False)

    assert record.stage1_tally == VoteTally(BUY=2, HOLD=1, SELL=2)
    assert len(mock_llm.stage2_prompts) == 10

    secret_stage1_tokens = [
        "SECRET_S1_FOMO_TOKEN_111",
        "SECRET_S1_FUND_TOKEN_222",
        "SECRET_S1_BEAR_TOKEN_333",
        "SECRET_S1_CONTR_TOKEN_444",
        "SECRET_S1_MACRO_TOKEN_555",
    ]
    expected_tally_line = "BUY: 2 | HOLD: 1 | SELL: 2"

    # Turn 1 of Round 1: ZERO individual agent names appear, OWN Stage 1 numeric stance IS present (anti-amnesia).
    first_speaker, first_stage2_prompt = mock_llm.stage2_prompts[0]
    assert first_speaker == "The Retail FOMO"
    assert expected_tally_line in first_stage2_prompt
    assert "YOUR OWN PRIVATE STAGE 1 PRIOR" in first_stage2_prompt
    assert "Your Stage 1 Vote: BUY" in first_stage2_prompt
    for persona in INVESTOR_PERSONAS:
        assert persona.name not in first_stage2_prompt

    # Across ALL Stage 2 & Stage 3 turns: ZERO Stage 1 secret reasoning tokens ever leak.
    for _, prompt in mock_llm.stage2_prompts:
        assert expected_tally_line in prompt
        for token in secret_stage1_tokens:
            assert token not in prompt

    for _, prompt in mock_llm.stage3_prompts:
        assert "YOUR OWN PRIVATE STAGE 1 PRIOR" in prompt
        for token in secret_stage1_tokens:
            assert token not in prompt


@pytest.mark.asyncio
async def test_stage1_and_stage3_execute_concurrently_while_stage2_is_sequential(tmp_path: Path) -> None:
    """Verify Stage 1 and Stage 3 fan out across all 5 agents concurrently via asyncio.gather."""
    mock_llm = MockCouncilLLMClient(delay_seconds=0.05)
    orchestrator = CouncilOrchestrator(
        llm_client=mock_llm,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
    )

    await orchestrator.run_simulation("NVDA", debate_rounds=1, save_to_disk=False)

    assert mock_llm.max_concurrent["stage1"] == 5
    assert mock_llm.max_concurrent["stage2"] == 1
    assert mock_llm.max_concurrent["stage3"] == 5


@pytest.mark.asyncio
async def test_full_pipeline_verdict_shifts_metrics_and_storage_roundtrip(tmp_path: Path) -> None:
    """Verify 4-stage pipeline, behavioral metrics, opinion shifts, and JSON roundtrip."""
    mock_llm = MockCouncilLLMClient()
    progress_events: list[tuple[str, str]] = []
    orchestrator = CouncilOrchestrator(
        llm_client=mock_llm,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
    )

    record = await orchestrator.run_simulation(
        "nvda",
        debate_rounds=2,
        on_progress=lambda stage, detail: progress_events.append((stage, detail)),
        save_to_disk=True,
    )

    assert record.ticker == "NVDA"
    assert record.fact_briefing.price_change_7d_pct == 10.0
    assert record.stage1_tally == VoteTally(BUY=2, HOLD=1, SELL=2)
    assert record.stage3_tally == VoteTally(BUY=3, HOLD=1, SELL=1)
    assert record.final_verdict == "BUY"
    assert record.behavioral_metrics is not None
    assert record.behavioral_metrics.opinion_shift_rate == 0.4
    assert record.behavioral_metrics.confidence_polarization == 0.2

    saved_summaries = list_sessions(tmp_path)
    assert len(saved_summaries) == 1
    reloaded = load_session(saved_summaries[0].filename, history_dir=tmp_path)
    assert reloaded == record

    (tmp_path / "corrupt_session.json").write_text("{not valid json", encoding="utf-8")
    assert len(list_sessions(tmp_path)) == 1


@pytest.mark.asyncio
async def test_v21_parallel_lab_3_meta_judges_fault_tolerance_and_forward_eval(tmp_path: Path) -> None:
    """Verify v2.1 Parallel Branches, 3-Model Meta-Judges (incl. 1 model 503 failure), Peer Selection & Forward Eval."""
    mock_council = MockCouncilLLMClient(delay_seconds=0.01)
    judge_clients = {
        "judge_1": MockMetaJudgeClient(model_name="gemini-3.6-flash", verdict="BUY", prob_up_5d=70, peer_pick="judge_2"),
        "judge_2": MockMetaJudgeClient(model_name="gemini-3.5-flash", verdict="SELL", prob_up_5d=30, peer_pick="judge_1"),
        # Simulate judge_3 hitting 503 UNAVAILABLE — must NOT crash the pipeline!
        "judge_3": MockMetaJudgeClient(model_name="gemini-3.1-flash-lite", fail=True),
    }
    scoreboard_path = tmp_path / "judge_scoreboard.json"
    orchestrator = CouncilOrchestrator(
        llm_client=mock_council,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
        judge_clients=judge_clients,
        scoreboard_path=scoreboard_path,
    )

    record = await orchestrator.run_simulation(
        "NVDA",
        debate_rounds=1,
        enable_counterfactual_lab=True,
        save_to_disk=True,
    )

    assert len(record.counterfactual_branches) == 4
    branch_ids = [b.branch_id for b in record.counterfactual_branches]
    assert branch_ids == ["baseline_single", "isolated_crowd", "fog_of_war", "full_herd"]

    panel = record.meta_judge_panel
    assert panel is not None
    assert len(panel.shadow_decisions) == 3
    status_by_id = {d.judge_id: d.status for d in panel.shadow_decisions}
    assert status_by_id == {"judge_1": "OK", "judge_2": "OK", "judge_3": "UNAVAILABLE"}
    # Both judge_1 and judge_2 voted for each other -> unbiased cold-start tie-breaker picks one of them.
    assert panel.selected_final_judge_id in ("judge_1", "judge_2")
    assert panel.tie_breaker_used is True
    expected_verdict = "BUY" if panel.selected_final_judge_id == "judge_1" else "SELL"
    assert panel.final_decision is not None and panel.final_decision.verdict == expected_verdict

    # Forward-Run Evaluation with deterministic synthetic future closes (stock UP +10% in 5D, SPY +2%)
    def fake_close_fetcher(symbol: str, start_date: str) -> pd.Series:
        dates = pd.to_datetime(
            ["2026-11-25", "2026-11-26", "2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02"]
        )
        if symbol == "SPY":
            return pd.Series([500.0, 502.0, 504.0, 506.0, 508.0, 510.0], index=dates)
        return pd.Series([165.0, 168.0, 170.0, 174.0, 178.0, 181.5], index=dates)

    updated_count, scoreboard = evaluate_all_history_sessions(
        history_dir=tmp_path,
        scoreboard_path=scoreboard_path,
        close_fetcher=fake_close_fetcher,
    )
    assert updated_count == 1
    assert scoreboard.judges["judge_1"].wins == 1
    assert scoreboard.judges["judge_1"].win_rate == 1.0
    assert scoreboard.judges["judge_1"].avg_brier_5d == 0.09  # (0.7 - 1.0)^2
    assert scoreboard.judges["judge_2"].wins == 0
    assert scoreboard.judges["judge_2"].win_rate == 0.0
    assert scoreboard.judges["judge_2"].avg_brier_5d == 0.49  # (0.3 - 1.0)^2

    # Idempotency check: re-evaluating must NOT double-count wins!
    _, scoreboard2 = evaluate_all_history_sessions(
        history_dir=tmp_path,
        scoreboard_path=scoreboard_path,
        close_fetcher=fake_close_fetcher,
    )
    assert scoreboard2.judges["judge_1"].wins == 1
    assert scoreboard2.judges["judge_1"].evaluated_runs == 1


def test_peer_selection_enforces_anti_self_vote_and_brier_tie_breaker() -> None:
    shadows = [
        MetaJudgeShadowDecision(judge_id="judge_1", model_name="m1", status="OK", verdict="BUY", confidence=7, prob_up_5d=60, position_pct=20, reasoning="r"),
        MetaJudgeShadowDecision(judge_id="judge_2", model_name="m2", status="OK", verdict="SELL", confidence=7, prob_up_5d=40, position_pct=-20, reasoning="r"),
        MetaJudgeShadowDecision(judge_id="judge_3", model_name="m3", status="OK", verdict="HOLD", confidence=6, prob_up_5d=50, position_pct=0, reasoning="r"),
    ]
    # judge_1 illegally tries to self-vote -> MUST be ignored; judge_2 & judge_3 tie -> Brier tie-breaker picks judge_2.
    peer_votes = [
        PeerSelectionVote(voter_judge_id="judge_1", voter_model_name="m1", selected_judge_id="judge_1", reasoning="me!"),
        PeerSelectionVote(voter_judge_id="judge_2", voter_model_name="m2", selected_judge_id="judge_3", reasoning="r"),
        PeerSelectionVote(voter_judge_id="judge_3", voter_model_name="m3", selected_judge_id="judge_2", reasoning="r"),
    ]
    scoreboard = {
        "judge_1": JudgeTrackRecord(judge_id="judge_1", model_name="m1", avg_brier_5d=0.30, win_rate=0.5),
        "judge_2": JudgeTrackRecord(judge_id="judge_2", model_name="m2", avg_brier_5d=0.14, win_rate=0.7),
        "judge_3": JudgeTrackRecord(judge_id="judge_3", model_name="m3", avg_brier_5d=0.22, win_rate=0.6),
    }
    winner, tie_used, reason = select_peer_winner(peer_votes, shadows, scoreboard)
    assert winner == "judge_2"
    assert tie_used is True
    assert reason is not None and "0.140" in reason


def test_compute_verdict_and_herding_metrics() -> None:
    assert compute_verdict(VoteTally(BUY=3, HOLD=1, SELL=1)) == "BUY"
    assert compute_verdict(VoteTally(BUY=2, HOLD=1, SELL=2)) == "HOLD (Split Decision)"

    s1 = [
        AgentVote(agent_id="a1", agent_name="A1", vote="BUY", confidence=6, prob_up_5d=60, position_pct=30, reasoning="r"),
        AgentVote(agent_id="a2", agent_name="A2", vote="BUY", confidence=6, prob_up_5d=60, position_pct=30, reasoning="r"),
        AgentVote(agent_id="a3", agent_name="A3", vote="BUY", confidence=6, prob_up_5d=60, position_pct=30, reasoning="r"),
        AgentVote(agent_id="a4", agent_name="A4", vote="SELL", confidence=6, prob_up_5d=40, position_pct=-20, reasoning="r"),
        AgentVote(agent_id="a5", agent_name="A5", vote="HOLD", confidence=6, prob_up_5d=50, position_pct=0, reasoning="r"),
    ]
    # Both non-majority voters (a4, a5) herd into BUY -> herding_rate == 1.0
    s3 = [
        v.model_copy(update={"vote": "BUY", "confidence": 8, "prob_up_5d": 75, "position_pct": 50})
        for v in s1
    ]
    bm = compute_behavioral_metrics(s1, s3)
    assert bm.herding_rate == 1.0
    assert bm.confidence_polarization == 2.0
    assert bm.stage3_entropy == 0.0


def test_data_fetcher_parses_nested_and_legacy_news_and_7d_window() -> None:
    raw_items = [
        {
            "id": "1",
            "content": {
                "title": "NVDA announces major cloud partnership",
                "summary": "<p>Multi-year <b>GPU</b> contract signed.</p>",
                "pubDate": "2026-11-25T20:00:00Z",
                "provider": {"displayName": "Yahoo Finance"},
                "canonicalUrl": {"url": "https://finance.yahoo.com/news/nvda-1"},
            },
        },
        {
            "id": "2",
            "content": {"title": "nvda announces major cloud partnership", "summary": "Duplicate copy."},
        },
        {
            "title": "Analysts debate AI server margins",
            "publisher": "Reuters",
            "providerPublishTime": 1764000000,
            "link": "https://reuters.com/nvda-margins",
        },
    ]
    parsed = _parse_news_items(raw_items, max_items=10)
    assert len(parsed) == 2
    assert parsed[0].summary == "Multi-year GPU contract signed."

    dates = pd.to_datetime(
        ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-20", "2026-11-24", "2026-11-25"]
    )
    hist = pd.DataFrame({"Close": [100.0, 102.0, 120.0, 125.0, 130.0, 132.0]}, index=dates)
    p_start, d_start, p_end, d_end, pct = _extract_7d_price_window(hist)
    assert (p_start, d_start, p_end, d_end, pct) == (120.0, "2026-11-18", 132.0, "2026-11-25", 10.0)

    with pytest.raises(TickerDataError):
        fetch_market_context("   ")


def test_storage_and_llm_key_guards(tmp_path: Path) -> None:
    with pytest.raises(MissingAPIKeyError):
        GeminiLLMClient(api_key="")
    with pytest.raises(FileNotFoundError):
        load_session("../../nonexistent.json", history_dir=tmp_path)


def test_streamlit_app_boots_replays_and_previews_v21_lab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify app.py boots cleanly, replays v2.1 Lab sessions, and renders Preview mode."""
    from streamlit.testing.v1 import AppTest
    import config as app_config

    mock_council = MockCouncilLLMClient(delay_seconds=0.01)
    judge_clients = {
        "judge_1": MockMetaJudgeClient(model_name="gemini-3.6-flash", verdict="BUY", peer_pick="judge_2"),
        "judge_2": MockMetaJudgeClient(model_name="gemini-3.5-flash", verdict="HOLD", peer_pick="judge_1"),
        "judge_3": MockMetaJudgeClient(model_name="gemini-3.1-flash-lite", verdict="BUY", peer_pick="judge_1"),
    }
    sb_path = tmp_path / "judge_scoreboard.json"
    orchestrator = CouncilOrchestrator(
        llm_client=mock_council,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
        judge_clients=judge_clients,
        scoreboard_path=sb_path,
    )
    asyncio.run(orchestrator.run_simulation("NVDA", debate_rounds=1, enable_counterfactual_lab=True, save_to_disk=True))

    monkeypatch.setattr(app_config, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(app_config, "JUDGE_SCOREBOARD_PATH", sb_path)

    at = AppTest.from_file(str(app_config.PROJECT_ROOT / "app.py"), default_timeout=15).run()
    assert not at.exception

    # Verify Tier Slider Slot switches providers cleanly
    assert len(at.sidebar.select_slider) == 1
    at.sidebar.select_slider[0].set_value("Ollama · Qwen 4B").run()
    assert not at.exception
    at.sidebar.select_slider[0].set_value("Claude · 3.5 Sonnet").run()
    assert not at.exception

    replay_buttons = [
        btn for btn in at.sidebar.button if btn.key and btn.key.startswith("hist_") and "NVDA" in btn.label
    ]
    assert len(replay_buttons) == 1
    replay_buttons[0].click().run()
    assert not at.exception

    subheaders = [sub.value for sub in at.subheader]
    assert "Market Brief" in subheaders
    assert "Initial Votes" in subheaders
    assert "Council Debate" in subheaders
    assert "Final Verdict" in subheaders
    assert "Experiment Lab" in subheaders
    assert "Meta-Judges" in subheaders
    assert len(at.dataframe) >= 3


def test_hardening_timeout_formatting_config_float_and_eval_cache(tmp_path: Path) -> None:
    """Verify non-empty TimeoutError formatting, safe float parsing, and evaluator caching/skipping."""
    import config as app_config
    from src.llm_client import _format_exc

    assert _format_exc(asyncio.TimeoutError(), 40.0) == "Timed out after 40s"
    assert _format_exc(RuntimeError(""), 40.0) == "RuntimeError"
    assert app_config._parse_float("", 60.0, 5.0, 300.0) == 60.0
    assert app_config._parse_float("not-a-number", 60.0, 5.0, 300.0) == 60.0
    assert app_config._parse_float("999", 60.0, 5.0, 300.0) == 300.0

    # Save 2 sessions on the same entry_date -> SPY should only be fetched ONCE in pass 1,
    # and once all 3 horizons (1D/5D/10D) are EVALUATED, pass 2 should make 0 fetches.
    mock_council = MockCouncilLLMClient(delay_seconds=0.0)
    orchestrator = CouncilOrchestrator(
        llm_client=mock_council,
        data_fetcher=lambda ticker: _sample_market_context(ticker),
        history_dir=tmp_path,
    )
    asyncio.run(orchestrator.run_simulation("NVDA", debate_rounds=1, save_to_disk=True))
    asyncio.run(orchestrator.run_simulation("AAPL", debate_rounds=1, save_to_disk=True))

    calls: list[tuple[str, str]] = []

    def counting_fetcher(symbol: str, start_date: str) -> pd.Series:
        calls.append((symbol, start_date))
        dates = pd.bdate_range("2026-11-25", periods=12)
        return pd.Series([100.0 + i for i in range(12)], index=dates)

    sb_path = tmp_path / "sb.json"
    updated1, _ = evaluate_all_history_sessions(
        history_dir=tmp_path, scoreboard_path=sb_path, close_fetcher=counting_fetcher
    )
    assert updated1 == 2
    assert sum(1 for sym, _ in calls if sym == "SPY") == 1  # cached across both sessions

    calls.clear()
    updated2, _ = evaluate_all_history_sessions(
        history_dir=tmp_path, scoreboard_path=sb_path, close_fetcher=counting_fetcher
    )
    assert updated2 == 0
    assert calls == []  # fully evaluated sessions skipped completely

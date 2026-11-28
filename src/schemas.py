"""Strict Pydantic schemas for the Stock Council Simulator (v1.0 -> v2.1 Lab).

All v2.0/v2.1 fields carry backward-compatible defaults so existing v1.0 JSON
session files in `data/history/` continue to load and replay cleanly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

VoteChoice = Literal["BUY", "HOLD", "SELL"]
VerdictChoice = Literal["BUY", "HOLD", "SELL", "HOLD (Split Decision)"]
BranchId = Literal["baseline_single", "isolated_crowd", "fog_of_war", "full_herd"]


class RawNewsItem(BaseModel):
    """Normalized news article fetched from yfinance (no technical indicators)."""

    title: str
    publisher: str = "Unknown"
    published_at: str = ""
    summary: str = ""
    url: str = ""


class MarketContext(BaseModel):
    """Raw Stage 0 input fetched from yfinance: strictly news + 7-day price change %."""

    ticker: str
    company_name: str
    fetched_at_utc: str
    price_start_7d: float
    price_start_date: str
    price_end: float
    price_end_date: str
    price_change_7d_pct: float
    news_items: list[RawNewsItem] = Field(default_factory=list)
    data_warning: str | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class ResearcherBriefingOutput(BaseModel):
    """Structured LLM output contract for the non-voting Researcher Agent."""

    executive_summary: str = Field(
        description="2-3 sentence factual summary of the stock's recent news flow and 7-day price move."
    )
    verified_facts: list[str] = Field(
        default_factory=list,
        description="Deduplicated, verifiable factual statements extracted from the news articles.",
    )
    key_catalysts: list[str] = Field(
        default_factory=list,
        description="Concrete upside catalysts supported by the provided news.",
    )
    key_risks: list[str] = Field(
        default_factory=list,
        description="Concrete downside risks or uncertainties supported by the provided news.",
    )
    discarded_noise_count: int = Field(
        default=0,
        ge=0,
        description="Count of clickbait, duplicate, or unverified rumor items filtered out.",
    )


class FactBriefing(BaseModel):
    """Authoritative Stage 0 output distributed to all council agents."""

    ticker: str
    company_name: str
    as_of_utc: str
    price_start_7d: float
    price_start_date: str
    price_end: float
    price_end_date: str
    price_change_7d_pct: float
    executive_summary: str
    verified_facts: list[str] = Field(default_factory=list)
    key_catalysts: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    discarded_noise_count: int = Field(default=0, ge=0)
    news_items_reviewed: int = Field(default=0, ge=0)
    data_warning: str | None = None


class AgentVoteOutput(BaseModel):
    """Structured LLM output contract for Stage 1, Stage 3, and Baseline votes."""

    vote: VoteChoice
    confidence: int = Field(ge=1, le=10)
    prob_up_5d: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Estimated probability (0-100%) that the stock closes higher in 5 trading days.",
    )
    position_pct: int = Field(
        default=0,
        ge=-100,
        le=100,
        description="Conviction-weighted directional allocation (-100 max short, 0 flat/hold, +100 max long).",
    )
    reasoning: str = Field(min_length=1, description="Concise justification for the vote.")

    @field_validator("reasoning")
    @classmethod
    def _strip_reasoning(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reasoning cannot be blank")
        return cleaned


class AgentVote(BaseModel):
    """Full recorded vote bound deterministically to an agent persona."""

    agent_id: str
    agent_name: str
    vote: VoteChoice
    confidence: int = Field(ge=1, le=10)
    prob_up_5d: int = Field(default=50, ge=0, le=100)
    position_pct: int = Field(default=0, ge=-100, le=100)
    reasoning: str


class VoteTally(BaseModel):
    """Anonymized aggregate vote counts. Contains zero agent identities or reasoning."""

    BUY: int = Field(default=0, ge=0)
    HOLD: int = Field(default=0, ge=0)
    SELL: int = Field(default=0, ge=0)

    def as_dict(self) -> dict[str, int]:
        return {"BUY": self.BUY, "HOLD": self.HOLD, "SELL": self.SELL}

    def total(self) -> int:
        return self.BUY + self.HOLD + self.SELL

    def format_anonymous_line(self) -> str:
        return f"BUY: {self.BUY} | HOLD: {self.HOLD} | SELL: {self.SELL}"


class DebateMessageOutput(BaseModel):
    """Structured LLM output contract for a single Stage 2 debate turn."""

    message: str = Field(
        min_length=1,
        description="Sharp argument limited strictly to 2-3 sentences.",
    )

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be blank")
        return cleaned


class DebateTurn(BaseModel):
    """Recorded turn in the Stage 2 open discussion transcript."""

    round_number: int = Field(ge=1)
    turn_index: int = Field(ge=1)
    agent_id: str
    agent_name: str
    message: str


class OpinionShift(BaseModel):
    """Stage 1 vs Stage 3 comparison record for one agent."""

    agent_id: str
    agent_name: str
    stage1_vote: VoteChoice
    stage1_confidence: int = Field(ge=1, le=10)
    stage1_prob_up_5d: int = Field(default=50, ge=0, le=100)
    stage3_vote: VoteChoice
    stage3_confidence: int = Field(ge=1, le=10)
    stage3_prob_up_5d: int = Field(default=50, ge=0, le=100)
    changed_vote: bool
    confidence_delta: int
    prob_delta: int = 0


class BehavioralMetrics(BaseModel):
    """Quantitative behavioral-finance telemetry measuring Stage 1 -> Stage 3 dynamics."""

    opinion_shift_rate: float = Field(ge=0.0, le=1.0)
    herding_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of non-majority Stage 1 voters who shifted toward the Stage 1 leader.",
    )
    confidence_polarization: float = Field(
        description="Mean (Stage 3 confidence - Stage 1 confidence) with zero new news added."
    )
    stage1_entropy: float = Field(ge=0.0)
    stage3_entropy: float = Field(ge=0.0)
    mean_prob_up_5d_stage1: float = Field(ge=0.0, le=100.0)
    mean_prob_up_5d_stage3: float = Field(ge=0.0, le=100.0)
    mean_position_pct_stage1: float = Field(ge=-100.0, le=100.0)
    mean_position_pct_stage3: float = Field(ge=-100.0, le=100.0)


class BranchResult(BaseModel):
    """Outcome of one parallel counterfactual branch in the v2.1 experiment lab."""

    branch_id: BranchId
    branch_name: str
    description: str
    verdict: VerdictChoice
    tally: VoteTally
    mean_prob_up_5d: float = Field(ge=0.0, le=100.0)
    mean_confidence: float = Field(ge=1.0, le=10.0)
    mean_position_pct: float = Field(ge=-100.0, le=100.0)
    behavioral_metrics: BehavioralMetrics | None = None
    votes: list[AgentVote] = Field(default_factory=list)
    debate_transcript: list[DebateTurn] = Field(default_factory=list)


# --- v2.1 Option B: 3 Blank Meta-Judges (Different Models) + Peer Selection + Scoreboard ---


class MetaJudgeDecisionOutput(BaseModel):
    """Structured output for a neutral, character-free Meta-Judge decision."""

    verdict: VoteChoice
    confidence: int = Field(ge=1, le=10)
    prob_up_5d: int = Field(ge=0, le=100)
    position_pct: int = Field(ge=-100, le=100)
    minority_counterpoint: str = Field(
        default="",
        description="Strongest counter-argument against the majority branch before locking your verdict.",
    )
    reasoning: str = Field(min_length=1)

    @field_validator("reasoning")
    @classmethod
    def _strip_reasoning(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reasoning cannot be blank")
        return cleaned


class MetaJudgeShadowDecision(BaseModel):
    """Independent shadow verdict from one Meta-Judge (scored equally in Forward Eval)."""

    judge_id: str
    model_name: str
    status: Literal["OK", "UNAVAILABLE"] = "OK"
    error: str | None = None
    verdict: VoteChoice | None = None
    confidence: int | None = Field(default=None, ge=1, le=10)
    prob_up_5d: int | None = Field(default=None, ge=0, le=100)
    position_pct: int | None = Field(default=None, ge=-100, le=100)
    minority_counterpoint: str = ""
    reasoning: str = ""


class PeerSelectionVoteOutput(BaseModel):
    """Structured output for anti-self-vote peer selection among the 3 Meta-Judges."""

    selected_judge_id: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)


class PeerSelectionVote(BaseModel):
    """Recorded peer-selection vote cast by one judge for another judge."""

    voter_judge_id: str
    voter_model_name: str
    selected_judge_id: str
    reasoning: str
    status: Literal["OK", "UNAVAILABLE"] = "OK"
    error: str | None = None


class JudgeTrackRecord(BaseModel):
    """Persistent forward-run empirical performance bound to each Meta-Judge slot/model."""

    judge_id: str
    model_name: str
    attempted_runs: int = 0
    total_runs: int = 0
    uptime_pct: float | None = None
    evaluated_runs: int = 0
    directional_calls: int = 0
    wins: int = 0
    win_rate: float | None = None
    shrunk_win_rate: float | None = None
    avg_brier_5d: float | None = None
    shrunk_brier_5d: float | None = None
    avg_excess_return_5d: float | None = None
    times_selected_final_judge: int = 0
    selected_directional_calls: int = 0
    selected_wins: int = 0
    selected_win_rate: float | None = None


class JudgeScoreboard(BaseModel):
    """Persistent scoreboard stored in `data/judge_scoreboard.json`."""

    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    evaluated_sessions: list[str] = Field(default_factory=list)
    judges: dict[str, JudgeTrackRecord] = Field(default_factory=dict)


class MetaJudgePanelResult(BaseModel):
    """Complete v2.2 de-biased Meta-Judge panel outcome: Shadow -> Double-Blind Peer Select -> Final & Ensemble."""

    shadow_decisions: list[MetaJudgeShadowDecision] = Field(default_factory=list)
    peer_votes: list[PeerSelectionVote] = Field(default_factory=list)
    peer_blind_aliases: dict[str, dict[str, str]] = Field(default_factory=dict)
    selected_final_judge_id: str | None = None
    selected_final_model_name: str | None = None
    tie_breaker_used: bool = False
    tie_breaker_reason: str | None = None
    final_decision: MetaJudgeShadowDecision | None = None
    ensemble_prob_up_5d: float | None = None
    scoreboard_snapshot: dict[str, JudgeTrackRecord] = Field(default_factory=dict)


class StrategyTicket(BaseModel):
    """Standalone Event-Driven Behavioral Mispricing trade plan (v2.3)."""

    action: Literal["LONG", "SHORT", "NO TRADE"] = "NO TRADE"
    horizon_days: int = Field(default=5, ge=1)
    alloc_pct: float = Field(default=0.0, ge=0.0, le=25.0)
    effective_prob_up_5d: float = Field(default=50.0, ge=0.0, le=100.0)
    prob_source: str = "council_stage3"
    event_trigger_passed: bool = False
    edge_gate_passed: bool = False
    consensus_gate_passed: bool = False
    polarization_gate_passed: bool = True
    anti_fomo_divergence: bool = False
    gate_notes: list[str] = Field(default_factory=list)


# --- v2.0 Forward-Run Paper Evaluation (Zero Look-Ahead Bias) ---


class ForwardHorizonResult(BaseModel):
    """Realized post-run price performance for one forward horizon (1D / 5D / 10D)."""

    horizon_days: int = Field(ge=1)
    target_date: str | None = None
    target_price: float | None = None
    stock_return_pct: float | None = None
    spy_return_pct: float | None = None
    excess_return_pct: float | None = None
    actual_up: bool | None = None
    status: Literal["PENDING", "EVALUATED", "UNAVAILABLE"] = "PENDING"


class EntityEvalScore(BaseModel):
    """5-day directional hit & probabilistic Brier score for a branch or Meta-Judge."""

    verdict: str | None = None
    prob_up_5d: float | None = None
    is_directional: bool = False
    correct_5d: bool | None = None
    brier_5d: float | None = None
    excess_return_5d: float | None = None


class ForwardEvaluation(BaseModel):
    """Realized forward-run evaluation attached to a SessionRecord once market days elapse."""

    evaluated_at_utc: str
    entry_date: str
    entry_price: float
    horizons: dict[str, ForwardHorizonResult] = Field(default_factory=dict)
    council_score_5d: EntityEvalScore | None = None
    branch_scores_5d: dict[str, EntityEvalScore] = Field(default_factory=dict)
    judge_scores_5d: dict[str, EntityEvalScore] = Field(default_factory=dict)


class SessionRecord(BaseModel):
    """Complete God View record saved to data/history/ for deterministic replay."""

    schema_version: str = "2.1"
    session_id: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    ticker: str
    model_name: str
    debate_rounds: int = Field(ge=1)
    raw_news: list[RawNewsItem] = Field(default_factory=list)
    fact_briefing: FactBriefing
    stage1_votes: list[AgentVote]
    stage1_tally: VoteTally
    debate_transcript: list[DebateTurn]
    stage3_votes: list[AgentVote]
    stage3_tally: VoteTally
    final_verdict: VerdictChoice
    opinion_shifts: list[OpinionShift] = Field(default_factory=list)
    behavioral_metrics: BehavioralMetrics | None = None
    counterfactual_branches: list[BranchResult] = Field(default_factory=list)
    meta_judge_panel: MetaJudgePanelResult | None = None
    strategy_ticket: StrategyTicket | None = None
    forward_eval: ForwardEvaluation | None = None


class SessionSummary(BaseModel):
    """Lightweight index entry for the Streamlit History Replay sidebar."""

    session_id: str
    filename: str
    created_at_utc: str
    ticker: str
    final_verdict: VerdictChoice
    price_change_7d_pct: float
    model_name: str
    debate_rounds: int
    shifts_count: int = 0
    has_lab_branches: bool = False
    meta_verdict: VerdictChoice | None = None
    forward_5d_status: str = "PENDING"

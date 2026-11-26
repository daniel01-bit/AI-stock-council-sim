"""System prompts, agent roster, and strictly scoped stage prompt builders.

Information Asymmetry Guarantee (Fog of War):
- `build_blind_vote_prompt` accepts ONLY `FactBriefing`.
- `build_debate_prompt` accepts ONLY `FactBriefing`, anonymized `VoteTally`, `DebateTurn` history,
  and optionally the speaker's OWN Stage 1 numeric stance (`own_stage1_vote`) to prevent self-amnesia
  WITHOUT exposing any other agent's identity, vote, or Stage 1 reasoning.
- `build_full_herd_debate_prompt` is used ONLY in the `full_herd` counterfactual control branch
  to explicitly measure social/anchoring pressure vs. `fog_of_war`.
- `META_JUDGE_BLANK_SYSTEM_PROMPT` is strictly neutral and character-free (Option B: model diversity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.schemas import (
    AgentVote,
    BranchResult,
    DebateTurn,
    FactBriefing,
    JudgeTrackRecord,
    MarketContext,
    MetaJudgeShadowDecision,
    VoteTally,
)


@dataclass(frozen=True)
class InvestorPersona:
    """Immutable definition of one voting council member."""

    agent_id: str
    name: str
    short_label: str
    avatar: str
    archetype: str
    system_prompt: str


RESEARCHER_SYSTEM_PROMPT = """\
You are the Non-Voting Researcher Agent for the Stock Council Simulator.
Your sole job is to curate an objective, fact-checked briefing from raw recent news articles \
and the stock's 7-day price change percentage.

STRICT RULES:
1. Base EVERY fact, catalyst, and risk strictly on the provided news items and 7-day price context. \
NEVER invent news events, financial numbers, or technical indicators (no RSI, MACD, moving averages, or chart patterns).
2. Aggressively filter out clickbait, sensational opinion fluff, promotional filler, duplicates, unverified rumors, and off-topic articles primarily about unrelated companies/tickers. \
Count how many raw articles were discarded as noise/duplicates/off-topic in `discarded_noise_count`.
3. Keep `verified_facts` concise, concrete, and deduplicated (3 to 8 bullet points when news is available).
4. Separate upside `key_catalysts` and downside `key_risks` neutrally without taking a BUY/HOLD/SELL stance.
5. If no news items were available, explicitly state that no recent verified news articles were found and keep lists empty rather than fabricating facts.
"""


BASELINE_SINGLE_SYSTEM_PROMPT = """\
You are a neutral, disciplined equity analyst serving as the Single-Agent Control Baseline.
You have NO persona bias (neither bullish nor bearish).
Base your assessment strictly on the provided Fact Briefing (recent verified news + 7-day price change % only; no technical indicators).
Provide calibrated probabilities (`prob_up_5d`) and conviction-weighted `position_pct` (-100 to +100).
"""


META_JUDGE_BLANK_SYSTEM_PROMPT = """\
You are an objective, character-free quantitative decision evaluator.
You have NO investor persona, NO emotional bias, and NO stylistic roleplay.
Evaluate the provided Fact Briefing, parallel behavioral experiment branches, behavioral bias telemetry \
(herding rate, confidence polarization, entropy), and historical forward-run track records strictly on evidence and calibration.
Do NOT invent facts or technical indicators outside the provided context.
"""


_COMMON_INVESTOR_RULES = """\
STRICT BEHAVIORAL & DATA CONSTRAINTS:
- You have ONLY the provided Fact Briefing (verified recent news + 7-day price change %).
- Do NOT invent or cite technical indicators (no RSI, MACD, moving averages, support/resistance, chart patterns).
- Stay relentlessly faithful to your investor persona — disagree sharply with other personas when their thesis violates yours.
- Never fabricate facts outside the Fact Briefing.
- Ensure `prob_up_5d` (0-100%) and `position_pct` (-100 to +100) are directionally consistent with your vote \
(BUY -> position_pct > 0; SELL -> position_pct < 0; HOLD -> position_pct near 0).
"""


INVESTOR_PERSONAS: tuple[InvestorPersona, ...] = (
    InvestorPersona(
        agent_id="retail_fomo",
        name="The Retail FOMO",
        short_label="Retail FOMO",
        avatar="🚀",
        archetype="Momentum & Narrative Chaser",
        system_prompt=(
            "You are **The Retail FOMO**, an impulsive momentum-chasing retail trader on the Stock Council.\n"
            "CORE PERSONALITY:\n"
            "- Driven by viral narratives, headline excitement, product buzz, and short-term 7-day price momentum.\n"
            "- Quick to chase rallies out of fear of missing out, and equally quick to panic-sell on scary negative headlines.\n"
            "- Impatient with slow fundamentalist lectures — you believe narrative and crowd flow move stocks first.\n"
            "- In debate, hype compelling catalysts aggressively or push back when skeptics ignore obvious market momentum.\n\n"
            + _COMMON_INVESTOR_RULES
        ),
    ),
    InvestorPersona(
        agent_id="fundamentalist",
        name="The Fundamentalist",
        short_label="Fundamentalist",
        avatar="🏛️",
        archetype="Business Quality & Moat",
        system_prompt=(
            "You are **The Fundamentalist**, a disciplined long-term business analyst on the Stock Council.\n"
            "CORE PERSONALITY:\n"
            "- Ignores short-term 7-day noise and headline hype unless it materially alters revenue, margins, cash flows, unit economics, or durable competitive moat.\n"
            "- Demands concrete business substance from the facts; defaults to HOLD when news is loud but financially immaterial.\n"
            "- In debate, calmly dismantle emotional hype or knee-jerk panic by asking what actually changes intrinsic earning power over 3-5 years.\n\n"
            + _COMMON_INVESTOR_RULES
        ),
    ),
    InvestorPersona(
        agent_id="paranoid_bear",
        name="The Paranoid Bear",
        short_label="Paranoid Bear",
        avatar="🐻",
        archetype="Skeptical Risk Manager",
        system_prompt=(
            "You are **The Paranoid Bear**, a ruthless risk manager and skeptical short-biased analyst on the Stock Council.\n"
            "CORE PERSONALITY:\n"
            "- Assumes good news is already priced in, management spins narratives, and hidden landmines lurk beneath every headline.\n"
            "- Hunts aggressively for execution risks, regulatory/legal threats, margin pressure, competitive erosion, and valuation traps.\n"
            "- In debate, interrogate other speakers directly, expose blind spots in bullish arguments, and stress-test worst-case downside scenarios.\n\n"
            + _COMMON_INVESTOR_RULES
        ),
    ),
    InvestorPersona(
        agent_id="contrarian",
        name="The Contrarian",
        short_label="Contrarian",
        avatar="🧭",
        archetype="Consensus Challenger",
        system_prompt=(
            "You are **The Contrarian**, an independent second-level thinker on the Stock Council.\n"
            "CORE PERSONALITY:\n"
            "- Actively challenges crowd consensus and suspects market overreaction whenever news and the 7-day move look overly euphoric or overly pessimistic.\n"
            "- Looks for asymmetric setups where the crowd is leaning too far to one side.\n"
            "- In debate, call out herd mentality directly (including the anonymous Stage 1 vote skew) and argue the overlooked side of the trade.\n\n"
            + _COMMON_INVESTOR_RULES
        ),
    ),
    InvestorPersona(
        agent_id="macro_analyst",
        name="The Macro Analyst",
        short_label="Macro Analyst",
        avatar="🌐",
        archetype="Sector, Supply Chain & Macro",
        system_prompt=(
            "You are **The Macro Analyst**, a top-down strategist on the Stock Council.\n"
            "CORE PERSONALITY:\n"
            "- Connects company-specific news to broader industry cycles, supply chains, competitor dynamics, regulation, tariffs, capital expenditure waves, and macro conditions.\n"
            "- Evaluates whether the company is riding a structural tailwind or fighting a sector-wide headwind.\n"
            "- In debate, zoom out beyond isolated headlines to challenge narrow single-company thinking with broader ecosystem consequences.\n\n"
            + _COMMON_INVESTOR_RULES
        ),
    ),
)

PERSONA_BY_ID: dict[str, InvestorPersona] = {p.agent_id: p for p in INVESTOR_PERSONAS}


def format_fact_briefing(briefing: FactBriefing) -> str:
    """Render FactBriefing into a deterministic, token-efficient prompt block."""
    facts = "\n".join(f"- {fact}" for fact in briefing.verified_facts) or "- None available."
    catalysts = "\n".join(f"- {item}" for item in briefing.key_catalysts) or "- None identified."
    risks = "\n".join(f"- {item}" for item in briefing.key_risks) or "- None identified."
    warning_block = f"\nDATA WARNING: {briefing.data_warning}\n" if briefing.data_warning else ""

    return (
        f"=== FACT BRIEFING ({briefing.ticker} — {briefing.company_name}) ===\n"
        f"As of (UTC): {briefing.as_of_utc}\n"
        f"7-Day Price Context: {briefing.price_change_7d_pct:+.2f}% "
        f"(${briefing.price_start_7d:.2f} on {briefing.price_start_date} -> "
        f"${briefing.price_end:.2f} on {briefing.price_end_date})\n"
        f"Articles Reviewed: {briefing.news_items_reviewed} "
        f"(Noise Filtered: {briefing.discarded_noise_count})\n"
        f"{warning_block}\n"
        f"Executive Summary:\n{briefing.executive_summary}\n\n"
        f"Verified Facts:\n{facts}\n\n"
        f"Key Catalysts:\n{catalysts}\n\n"
        f"Key Risks:\n{risks}\n"
        "=== END FACT BRIEFING ==="
    )


def _format_own_prior_block(own_stage1_vote: AgentVote | None) -> str:
    """Format ONLY the speaker's own Stage 1 numeric stance (no reasoning text, no other agents)."""
    if own_stage1_vote is None:
        return ""
    return (
        "YOUR OWN PRIVATE STAGE 1 PRIOR (visible ONLY to you; never shown to other members):\n"
        f"- Your Stage 1 Vote: {own_stage1_vote.vote} | Confidence: {own_stage1_vote.confidence}/10 | "
        f"P(Up in 5D): {own_stage1_vote.prob_up_5d}% | Position: {own_stage1_vote.position_pct:+d}%\n\n"
    )


def build_researcher_prompt(context: MarketContext) -> str:
    """Build Stage 0 prompt for the Researcher Agent from raw yfinance data."""
    if context.news_items:
        article_blocks = []
        for idx, item in enumerate(context.news_items, start=1):
            summary_line = f"  Summary: {item.summary}" if item.summary else "  Summary: (none)"
            article_blocks.append(
                f"[{idx}] Title: {item.title}\n"
                f"  Publisher: {item.publisher} | Published: {item.published_at or 'unknown'}\n"
                f"{summary_line}"
            )
        articles_text = "\n\n".join(article_blocks)
    else:
        articles_text = "(No recent news articles returned by Yahoo Finance for this ticker.)"

    warning_line = f"\nData Warning: {context.data_warning}" if context.data_warning else ""

    return (
        f"Prepare a FactBriefing for {context.ticker} ({context.company_name}).\n"
        f"7-Day Price Change: {context.price_change_7d_pct:+.2f}% "
        f"(${context.price_start_7d:.2f} on {context.price_start_date} -> "
        f"${context.price_end:.2f} on {context.price_end_date}){warning_line}\n\n"
        f"RAW NEWS ARTICLES ({len(context.news_items)} total):\n"
        f"{articles_text}\n\n"
        "Filter out clickbait, duplicates, and unverified rumors, then return the structured briefing."
    )


def build_baseline_single_prompt(briefing: FactBriefing) -> str:
    """Control Branch A: Single neutral agent with zero debate or persona roleplay."""
    return (
        f"{format_fact_briefing(briefing)}\n\n"
        "SINGLE-AGENT CONTROL BASELINE:\n"
        "Provide an objective, calibrated investment decision for the next 5 trading days: "
        "vote (BUY, HOLD, or SELL), confidence (1-10), prob_up_5d (0-100%), position_pct (-100 to +100), "
        "and a concise 2-3 sentence rationale."
    )


def build_blind_vote_prompt(briefing: FactBriefing) -> str:
    """Stage 1 prompt: strictly FactBriefing ONLY (complete blind isolation)."""
    return (
        f"{format_fact_briefing(briefing)}\n\n"
        "STAGE 1 — FIRST DESTINATION (BLIND VOTE):\n"
        "You are voting independently in complete isolation. You do NOT know how any other council member is voting.\n"
        "Based strictly on the Fact Briefing and your persona, cast your vote (BUY, HOLD, or SELL), "
        "rate your confidence (1-10), estimate prob_up_5d (0-100% probability the stock closes higher in 5 trading days), "
        "choose position_pct (-100 to +100), and give a concise 2-3 sentence reasoning."
    )


def build_debate_prompt(
    briefing: FactBriefing,
    anonymized_tally: VoteTally,
    transcript: Sequence[DebateTurn],
    round_number: int,
    total_rounds: int,
    own_stage1_vote: AgentVote | None = None,
) -> str:
    """Stage 2 Fog-of-War prompt: FactBriefing + Anonymized Stage 1 Tally ONLY + Debate transcript.

    CRITICAL FIREWALL: Other agents' Stage 1 identities, votes, and reasoning are structurally excluded.
    Optional `own_stage1_vote` injects ONLY the speaker's own numeric prior stance to prevent self-amnesia.
    """
    if transcript:
        lines = [
            f"[Round {turn.round_number}] {turn.agent_name}: {turn.message}"
            for turn in transcript
        ]
        transcript_block = "\n".join(lines)
    else:
        transcript_block = "(You are opening the floor. No council member has spoken yet.)"

    own_block = _format_own_prior_block(own_stage1_vote)

    return (
        f"{format_fact_briefing(briefing)}\n\n"
        f"{own_block}"
        "ANONYMIZED STAGE 1 BLIND VOTE TALLY (identities and individual reasons are hidden):\n"
        f"{anonymized_tally.format_anonymous_line()}\n\n"
        f"OPEN DISCUSSION TRANSCRIPT SO FAR (Round {round_number} of {total_rounds}):\n"
        f"{transcript_block}\n\n"
        f"STAGE 2 INSTRUCTIONS (Round {round_number}/{total_rounds}):\n"
        "- Deliver your argument to the council in strictly 2 to 3 sharp, high-impact sentences.\n"
        "- Remain consistent with your own Stage 1 prior unless a prior speaker's argument genuinely changed your view (if so, say what changed your mind).\n"
        "- React to the anonymized Stage 1 tally and directly challenge, question, or build on prior speakers by name when relevant.\n"
        "- Stay strictly grounded in the Fact Briefing and your persona. No fluff or filler."
    )


def build_full_herd_debate_prompt(
    briefing: FactBriefing,
    stage1_votes: Sequence[AgentVote],
    transcript: Sequence[DebateTurn],
    round_number: int,
    total_rounds: int,
    own_stage1_vote: AgentVote | None = None,
) -> str:
    """Counterfactual Branch D (Full-Herd Social Pressure): exposes ALL Stage 1 identities & votes."""
    exposed_lines = [
        f"- {v.agent_name}: {v.vote} (Conf {v.confidence}/10, P(Up 5D)={v.prob_up_5d}%, Pos={v.position_pct:+d}%)"
        for v in stage1_votes
    ]
    exposed_block = "\n".join(exposed_lines)

    if transcript:
        transcript_block = "\n".join(
            f"[Round {t.round_number}] {t.agent_name}: {t.message}" for t in transcript
        )
    else:
        transcript_block = "(You are opening the floor. No council member has spoken yet.)"

    own_block = _format_own_prior_block(own_stage1_vote)

    return (
        f"{format_fact_briefing(briefing)}\n\n"
        f"{own_block}"
        "FULL PUBLIC STAGE 1 VOTES (all council identities and votes are publicly visible):\n"
        f"{exposed_block}\n\n"
        f"OPEN DISCUSSION TRANSCRIPT SO FAR (Round {round_number} of {total_rounds}):\n"
        f"{transcript_block}\n\n"
        f"FULL-HERD DEBATE INSTRUCTIONS (Round {round_number}/{total_rounds}):\n"
        "- Deliver your argument in strictly 2 to 3 sharp sentences.\n"
        "- Explicitly react to who voted what in Stage 1 and respond to prior speakers by name."
    )


def build_final_vote_prompt(
    briefing: FactBriefing,
    transcript: Sequence[DebateTurn],
    own_stage1_vote: AgentVote | None = None,
) -> str:
    """Stage 3 prompt: FactBriefing + Full Debate Transcript + own Stage 1 prior (simultaneous secret vote)."""
    if transcript:
        lines = [
            f"[Round {turn.round_number}] {turn.agent_name}: {turn.message}"
            for turn in transcript
        ]
        transcript_block = "\n".join(lines)
    else:
        transcript_block = "(No debate transcript recorded.)"

    own_block = _format_own_prior_block(own_stage1_vote)

    return (
        f"{format_fact_briefing(briefing)}\n\n"
        f"{own_block}"
        "FULL STAGE 2 COUNCIL DEBATE TRANSCRIPT:\n"
        f"{transcript_block}\n\n"
        "STAGE 3 — FINAL DESTINATION (SIMULTANEOUS SECRET VOTE):\n"
        "All council members are casting their final votes simultaneously in secret.\n"
        "Based on the Fact Briefing and the full debate above, cast your final vote (BUY, HOLD, or SELL), "
        "confidence (1-10), prob_up_5d (0-100%), position_pct (-100 to +100), and concise 2-3 sentence reasoning "
        "explicitly comparing against your own Stage 1 prior (state clearly if you HELD your Stage 1 vote or SHIFTED and why)."
    )


# --- v2.1 Meta-Judge Prompt Builders (Blank / Character-Free) ---


def format_judge_scoreboard_block(scoreboard: dict[str, JudgeTrackRecord]) -> str:
    """Format empirical forward-run track records with explicit sample-size & cold-start context."""
    if not scoreboard:
        return "(No historical forward-run evaluations recorded yet — cold-start regime, n=0.)"

    lines = []
    max_n = 0
    for jid in sorted(scoreboard.keys()):
        rec = scoreboard[jid]
        max_n = max(max_n, rec.evaluated_runs)
        wr = f"{rec.win_rate * 100:.1f}%" if rec.win_rate is not None else "N/A"
        brier = f"{rec.avg_brier_5d:.3f}" if rec.avg_brier_5d is not None else "N/A"
        excess = f"{rec.avg_excess_return_5d:+.2f}%" if rec.avg_excess_return_5d is not None else "N/A"
        lines.append(
            f"- {rec.judge_id} (model={rec.model_name}): evaluated_n={rec.evaluated_runs}, "
            f"directional_n={rec.directional_calls}, win_rate={wr}, avg_brier_5d={brier} (lower is better), "
            f"avg_excess_5d_vs_SPY={excess}, times_picked_final={rec.times_selected_final_judge}"
        )

    cold_note = (
        "\nCOLD-START WARNING: evaluated_n < 10. Small-sample win rates have high variance; "
        "weigh reasoning quality and branch evidence heavily alongside track record."
        if max_n < 10
        else ""
    )
    return "\n".join(lines) + cold_note


def format_branches_summary(branches: Sequence[BranchResult]) -> str:
    """Render all parallel counterfactual branches and behavioral telemetry cleanly."""
    blocks = []
    for br in branches:
        bm_line = ""
        if br.behavioral_metrics is not None:
            bm = br.behavioral_metrics
            bm_line = (
                f"\n  Behavioral Telemetry: shift_rate={bm.opinion_shift_rate:.0%}, "
                f"herding_rate={bm.herding_rate:.0%}, "
                f"conf_polarization={bm.confidence_polarization:+.2f}, "
                f"entropy={bm.stage1_entropy:.2f}->{bm.stage3_entropy:.2f}"
            )
        blocks.append(
            f"[{br.branch_id}] {br.branch_name}\n"
            f"  Verdict: {br.verdict} | Tally: {br.tally.format_anonymous_line()} | "
            f"Mean P(Up 5D): {br.mean_prob_up_5d:.1f}% | Mean Conf: {br.mean_confidence:.1f}/10 | "
            f"Mean Pos: {br.mean_position_pct:+.1f}%{bm_line}"
        )
    return "\n\n".join(blocks)


def build_meta_judge_shadow_prompt(
    briefing: FactBriefing,
    branches: Sequence[BranchResult],
    scoreboard: dict[str, JudgeTrackRecord],
    own_judge_id: str,
) -> str:
    """Identical blank neutral prompt for all 3 Meta-Judges during parallel Shadow Scoring."""
    return (
        f"{format_fact_briefing(briefing)}\n\n"
        "=== PARALLEL COUNTERFACTUAL EXPERIMENT BRANCHES ===\n"
        f"{format_branches_summary(branches)}\n\n"
        "=== HISTORICAL FORWARD-RUN META-JUDGE TRACK RECORD ===\n"
        f"{format_judge_scoreboard_block(scoreboard)}\n\n"
        f"TASK (You are neutral evaluator `{own_judge_id}` — independent shadow decision):\n"
        "Synthesize the Fact Briefing and the behavioral divergence across the parallel branches above "
        "(e.g., compare Baseline vs. Isolated Crowd vs. Fog-of-War vs. Full-Herd to detect herding or polarization distortion).\n"
        "Output your independent verdict (BUY, HOLD, or SELL), confidence (1-10), prob_up_5d (0-100%), "
        "position_pct (-100 to +100), and concise 2-3 sentence analytical reasoning."
    )


def build_peer_selection_prompt(
    briefing: FactBriefing,
    voter_judge_id: str,
    shadow_decisions: Sequence[MetaJudgeShadowDecision],
    scoreboard: dict[str, JudgeTrackRecord],
) -> str:
    """Anti-self-vote peer selection prompt: each judge selects which OTHER judge should make the final call."""
    eligible_ids = [
        d.judge_id
        for d in shadow_decisions
        if d.status == "OK" and d.judge_id != voter_judge_id
    ]
    decision_blocks = []
    for d in shadow_decisions:
        if d.status != "OK":
            decision_blocks.append(f"- {d.judge_id}: UNAVAILABLE ({d.error or 'model error'})")
            continue
        self_tag = " [YOU — NOT ELIGIBLE FOR SELF-VOTE]" if d.judge_id == voter_judge_id else " [ELIGIBLE]"
        decision_blocks.append(
            f"- {d.judge_id}{self_tag}: Verdict={d.verdict}, Conf={d.confidence}/10, "
            f"P(Up 5D)={d.prob_up_5d}%, Pos={d.position_pct:+d}%\n"
            f"  Reasoning: {d.reasoning}"
        )

    return (
        f"Ticker: {briefing.ticker} ({briefing.company_name}) | 7D Move: {briefing.price_change_7d_pct:+.2f}%\n\n"
        "=== HISTORICAL TRACK RECORDS ===\n"
        f"{format_judge_scoreboard_block(scoreboard)}\n\n"
        "=== SHADOW DECISIONS FROM ALL META-JUDGES ===\n"
        + "\n".join(decision_blocks)
        + "\n\n"
        f"PEER SELECTION TASK (You are `{voter_judge_id}`):\n"
        f"Select EXACTLY ONE judge ID from the eligible list {eligible_ids} to act as the Final Judge for this ticker.\n"
        "RULES:\n"
        f"- You MUST NOT select yourself (`{voter_judge_id}`).\n"
        "- Base your choice on BOTH empirical historical calibration (Brier score, win rate, sample size) "
        "AND the analytical rigor of their shadow reasoning above.\n"
        "- Return `selected_judge_id` matching one of the eligible IDs verbatim and a 1-2 sentence rationale."
    )


def build_final_judge_synthesis_prompt(
    briefing: FactBriefing,
    branches: Sequence[BranchResult],
    selected_judge_id: str,
    shadow_decisions: Sequence[MetaJudgeShadowDecision],
    scoreboard: dict[str, JudgeTrackRecord],
) -> str:
    """Final synthesis prompt executed ONLY by the peer-selected Final Judge."""
    shadow_block = "\n".join(
        f"- {d.judge_id} ({d.model_name}): "
        + (
            f"Verdict={d.verdict}, Conf={d.confidence}/10, P(Up 5D)={d.prob_up_5d}%, Pos={d.position_pct:+d}% | {d.reasoning}"
            if d.status == "OK"
            else f"UNAVAILABLE ({d.error})"
        )
        for d in shadow_decisions
    )

    return (
        f"{format_fact_briefing(briefing)}\n\n"
        "=== PARALLEL BRANCH SUMMARY ===\n"
        f"{format_branches_summary(branches)}\n\n"
        "=== HISTORICAL TRACK RECORDS ===\n"
        f"{format_judge_scoreboard_block(scoreboard)}\n\n"
        "=== PEER SHADOW DECISIONS ===\n"
        f"{shadow_block}\n\n"
        f"FINAL JUDGE MANDATE (You are `{selected_judge_id}`, selected by your peers):\n"
        "You hold final decision authority for this session. Weigh your own shadow thesis against your peers' "
        "arguments and the parallel branch behavioral signals, then issue the definitive Final Verdict "
        "(BUY, HOLD, or SELL), confidence (1-10), prob_up_5d (0-100%), position_pct (-100 to +100), and concise 2-3 sentence rationale."
    )

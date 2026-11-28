"""4-Stage Stock Council Orchestrator + v2.1 Parallel Counterfactual Lab & 3-Model Meta-Judges."""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import (
    DEBATE_ROUNDS,
    HISTORY_DIR,
    JUDGE_SCOREBOARD_PATH,
    MAX_DEBATE_ROUNDS,
    META_JUDGE_MODELS,
    MIN_DEBATE_ROUNDS,
)
from src.data_fetcher import fetch_market_context
from src.evaluator import load_judge_scoreboard, rebuild_judge_scoreboard_from_history
from src.llm_client import StructuredLLMClient
from src.prompts import (
    BASELINE_SINGLE_SYSTEM_PROMPT,
    INVESTOR_PERSONAS,
    META_JUDGE_BLANK_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    InvestorPersona,
    build_baseline_single_prompt,
    build_blind_vote_prompt,
    build_debate_prompt,
    build_final_judge_synthesis_prompt,
    build_final_vote_prompt,
    build_full_herd_debate_prompt,
    build_meta_judge_shadow_prompt,
    build_peer_selection_prompt,
    build_researcher_prompt,
)
from src.schemas import (
    AgentVote,
    AgentVoteOutput,
    BehavioralMetrics,
    BranchResult,
    DebateMessageOutput,
    DebateTurn,
    FactBriefing,
    JudgeTrackRecord,
    MarketContext,
    MetaJudgeDecisionOutput,
    MetaJudgePanelResult,
    MetaJudgeShadowDecision,
    OpinionShift,
    PeerSelectionVote,
    PeerSelectionVoteOutput,
    ResearcherBriefingOutput,
    SessionRecord,
    VerdictChoice,
    VoteTally,
)
from src.storage import save_session
from src.strategy import build_strategy_ticket

ProgressCallback = Callable[[str, str], None]
DataFetcherFn = Callable[[str], MarketContext]


def compute_anonymized_tally(votes: Sequence[AgentVote]) -> VoteTally:
    """Strip all agent identities and reasoning, leaving ONLY aggregate BUY/HOLD/SELL counts."""
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for item in votes:
        counts[item.vote] += 1
    return VoteTally(**counts)


def compute_verdict(tally: VoteTally) -> VerdictChoice:
    """Return majority vote, or 'HOLD (Split Decision)' whenever the top count is tied."""
    scores: list[tuple[VerdictChoice, int]] = [
        ("BUY", tally.BUY),
        ("HOLD", tally.HOLD),
        ("SELL", tally.SELL),
    ]
    max_votes = max(count for _, count in scores)
    if max_votes == 0:
        return "HOLD (Split Decision)"

    leaders = [label for label, count in scores if count == max_votes]
    if len(leaders) != 1:
        return "HOLD (Split Decision)"
    return leaders[0]


def compute_vote_entropy(tally: VoteTally) -> float:
    """Compute Shannon entropy (bits) of the vote distribution (0.0 = unanimous, ~1.585 = max split)."""
    total = tally.total()
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in (tally.BUY, tally.HOLD, tally.SELL):
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def compute_opinion_shifts(
    stage1_votes: Sequence[AgentVote],
    stage3_votes: Sequence[AgentVote],
) -> list[OpinionShift]:
    """Compare each agent's Stage 1 blind vote against their Stage 3 final vote."""
    stage3_by_id = {vote.agent_id: vote for vote in stage3_votes}
    shifts: list[OpinionShift] = []
    for s1 in stage1_votes:
        s3 = stage3_by_id[s1.agent_id]
        shifts.append(
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
        )
    return shifts


def compute_behavioral_metrics(
    stage1_votes: Sequence[AgentVote],
    stage3_votes: Sequence[AgentVote],
) -> BehavioralMetrics:
    """Compute quantitative behavioral metrics comparing Stage 1 priors vs. Stage 3 posteriors."""
    n = len(stage1_votes)
    if n == 0:
        return BehavioralMetrics(
            opinion_shift_rate=0.0,
            herding_rate=0.0,
            confidence_polarization=0.0,
            stage1_entropy=0.0,
            stage3_entropy=0.0,
            mean_prob_up_5d_stage1=50.0,
            mean_prob_up_5d_stage3=50.0,
            mean_position_pct_stage1=0.0,
            mean_position_pct_stage3=0.0,
        )

    s1_tally = compute_anonymized_tally(stage1_votes)
    s3_tally = compute_anonymized_tally(stage3_votes)
    shifts = compute_opinion_shifts(stage1_votes, stage3_votes)

    shift_rate = round(sum(1 for sh in shifts if sh.changed_vote) / n, 4)
    conf_polarization = round(sum(sh.confidence_delta for sh in shifts) / n, 2)

    # Herding rate: among agents NOT voting with the unique Stage 1 leader, how many moved TO that leader?
    s1_leader = compute_verdict(s1_tally)
    herding_rate = 0.0
    if s1_leader in ("BUY", "HOLD", "SELL"):
        non_leader = [sh for sh in shifts if sh.stage1_vote != s1_leader]
        if non_leader:
            moved_to_leader = sum(1 for sh in non_leader if sh.stage3_vote == s1_leader)
            herding_rate = round(moved_to_leader / len(non_leader), 4)

    return BehavioralMetrics(
        opinion_shift_rate=shift_rate,
        herding_rate=herding_rate,
        confidence_polarization=conf_polarization,
        stage1_entropy=compute_vote_entropy(s1_tally),
        stage3_entropy=compute_vote_entropy(s3_tally),
        mean_prob_up_5d_stage1=round(sum(v.prob_up_5d for v in stage1_votes) / n, 1),
        mean_prob_up_5d_stage3=round(sum(v.prob_up_5d for v in stage3_votes) / n, 1),
        mean_position_pct_stage1=round(sum(v.position_pct for v in stage1_votes) / n, 1),
        mean_position_pct_stage3=round(sum(v.position_pct for v in stage3_votes) / n, 1),
    )


BLIND_CANDIDATE_ALIASES: tuple[str, ...] = (
    "Candidate Alpha",
    "Candidate Beta",
    "Candidate Gamma",
    "Candidate Delta",
)


def build_blind_peer_aliases(
    voter_judge_id: str,
    ok_shadows: Sequence[MetaJudgeShadowDecision],
    seed_key: str,
) -> tuple[list[tuple[str, MetaJudgeShadowDecision]], dict[str, str]]:
    """Deterministically permute eligible peers into blind aliases per voter to eliminate order/label bias."""
    eligible = [d for d in ok_shadows if d.judge_id != voter_judge_id]
    digest = hashlib.sha256(f"{seed_key}:{voter_judge_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    pairs = [(BLIND_CANDIDATE_ALIASES[idx], item) for idx, item in enumerate(shuffled)]
    alias_to_jid = {alias: item.judge_id for alias, item in pairs}
    return pairs, alias_to_jid


def compute_ensemble_prob_up_5d(
    ok_shadows: Sequence[MetaJudgeShadowDecision],
    scoreboard: dict[str, JudgeTrackRecord],
) -> float | None:
    """Compute inverse shrunk-Brier weighted 5D probability across all OK Meta-Judges."""
    valid = [d for d in ok_shadows if d.prob_up_5d is not None]
    if not valid:
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for d in valid:
        rec = scoreboard.get(d.judge_id)
        brier = rec.shrunk_brier_5d if (rec and rec.shrunk_brier_5d is not None) else 0.25
        weight = 1.0 / max(0.05, brier)
        weighted_sum += weight * float(d.prob_up_5d)
        total_weight += weight
    return round(weighted_sum / total_weight, 1) if total_weight > 0 else None


def select_peer_winner(
    peer_votes: Sequence[PeerSelectionVote],
    shadow_decisions: Sequence[MetaJudgeShadowDecision],
    scoreboard: dict[str, JudgeTrackRecord],
    *,
    seed_key: str = "",
) -> tuple[str | None, bool, str | None]:
    """Resolve peer selection winner with anti-self-vote, shrunk-Brier ranking, and unbiased cold-start rotation."""
    available_ids = [d.judge_id for d in shadow_decisions if d.status == "OK"]
    if not available_ids:
        return None, False, "No Meta-Judges available."
    if len(available_ids) == 1:
        return available_ids[0], True, "Only one Meta-Judge model responded; selected automatically."

    valid_counts: Counter[str] = Counter()
    for pv in peer_votes:
        if pv.status != "OK":
            continue
        if pv.selected_judge_id == pv.voter_judge_id:
            continue
        if pv.selected_judge_id in available_ids:
            valid_counts[pv.selected_judge_id] += 1

    if valid_counts:
        max_votes = max(valid_counts.values())
        leaders = [jid for jid, cnt in valid_counts.items() if cnt == max_votes]
        if len(leaders) == 1:
            return leaders[0], False, None
        candidates = leaders
        tie_reason = f"Peer vote tied ({max_votes} each among {', '.join(sorted(leaders))})"
    else:
        candidates = available_ids
        tie_reason = "No valid non-self peer votes cast"

    # Check if we have real Brier track records to break ties empirically; otherwise use unbiased session-seeded rotation.
    has_brier = any(
        scoreboard.get(jid) is not None
        and (scoreboard[jid].avg_brier_5d is not None or scoreboard[jid].evaluated_runs >= 10)
        for jid in candidates
    )

    if has_brier:
        def _rank_key(jid: str) -> tuple[float, float, str]:
            rec = scoreboard.get(jid)
            if rec is None:
                return (9.999, 1.0, jid)
            brier = rec.shrunk_brier_5d if rec.shrunk_brier_5d is not None else (rec.avg_brier_5d if rec.avg_brier_5d is not None else 9.999)
            wr = rec.shrunk_win_rate if rec.shrunk_win_rate is not None else (rec.win_rate if rec.win_rate is not None else 0.0)
            return (brier, -wr, jid)

        winner = min(candidates, key=_rank_key)
        rec = scoreboard.get(winner)
        b_val = rec.shrunk_brier_5d if (rec and rec.shrunk_brier_5d is not None) else (rec.avg_brier_5d if rec else None)
        metric_note = f"best historical Brier={b_val:.3f}" if b_val is not None else "empirical rank"
        return winner, True, f"{tie_reason}; resolved via {metric_note}."

    # Cold-start unbiased tie-breaker: deterministic session-seeded rotation (never hardcoded to judge_1).
    sorted_cands = sorted(candidates)
    if seed_key:
        digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % len(sorted_cands)
        winner = sorted_cands[idx]
        return winner, True, f"{tie_reason}; resolved via unbiased cold-start rotation."
    winner = sorted_cands[0]
    return winner, True, f"{tie_reason}; resolved via cold-start deterministic order."


def build_authoritative_briefing(
    context: MarketContext,
    researcher_output: ResearcherBriefingOutput,
) -> FactBriefing:
    """Bind LLM-curated facts to deterministic market numbers so price math can never drift."""
    return FactBriefing(
        ticker=context.ticker,
        company_name=context.company_name,
        as_of_utc=context.fetched_at_utc,
        price_start_7d=context.price_start_7d,
        price_start_date=context.price_start_date,
        price_end=context.price_end,
        price_end_date=context.price_end_date,
        price_change_7d_pct=context.price_change_7d_pct,
        executive_summary=researcher_output.executive_summary.strip(),
        verified_facts=[item.strip() for item in researcher_output.verified_facts if item.strip()],
        key_catalysts=[item.strip() for item in researcher_output.key_catalysts if item.strip()],
        key_risks=[item.strip() for item in researcher_output.key_risks if item.strip()],
        discarded_noise_count=min(researcher_output.discarded_noise_count, len(context.news_items)),
        news_items_reviewed=len(context.news_items),
        data_warning=context.data_warning,
    )


def _summarize_branch(
    *,
    branch_id: str,
    branch_name: str,
    description: str,
    votes: Sequence[AgentVote],
    behavioral_metrics: BehavioralMetrics | None = None,
    transcript: Sequence[DebateTurn] = (),
) -> BranchResult:
    tally = compute_anonymized_tally(votes)
    n = len(votes) or 1
    return BranchResult(
        branch_id=branch_id,
        branch_name=branch_name,
        description=description,
        verdict=compute_verdict(tally),
        tally=tally,
        mean_prob_up_5d=round(sum(v.prob_up_5d for v in votes) / n, 1),
        mean_confidence=round(sum(v.confidence for v in votes) / n, 2),
        mean_position_pct=round(sum(v.position_pct for v in votes) / n, 1),
        behavioral_metrics=behavioral_metrics,
        votes=list(votes),
        debate_transcript=list(transcript),
    )


class CouncilOrchestrator:
    """Runs the 4-stage Stock Council pipeline + optional v2.1 Parallel Counterfactual Lab."""

    def __init__(
        self,
        llm_client: StructuredLLMClient,
        data_fetcher: DataFetcherFn = fetch_market_context,
        history_dir: Path = HISTORY_DIR,
        personas: Sequence[InvestorPersona] = INVESTOR_PERSONAS,
        judge_clients: dict[str, StructuredLLMClient] | None = None,
        scoreboard_path: Path = JUDGE_SCOREBOARD_PATH,
    ) -> None:
        self.llm_client = llm_client
        self.data_fetcher = data_fetcher
        self.history_dir = history_dir
        self.personas = tuple(personas)
        self.judge_clients = judge_clients or {}
        self.scoreboard_path = scoreboard_path

    def _notify(self, callback: ProgressCallback | None, stage: str, detail: str) -> None:
        if callback is not None:
            callback(stage, detail)

    async def _run_stage0_research(
        self,
        ticker: str,
        on_progress: ProgressCallback | None,
    ) -> tuple[MarketContext, FactBriefing]:
        self._notify(on_progress, "Stage 0", f"Fetching recent news & 7-day price change for {ticker}...")
        context = await asyncio.to_thread(self.data_fetcher, ticker)

        self._notify(
            on_progress,
            "Stage 0",
            f"Researcher Agent curating {len(context.news_items)} articles ({context.price_change_7d_pct:+.2f}% 7D move)...",
        )
        researcher_output = await self.llm_client.generate_structured(
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            user_prompt=build_researcher_prompt(context),
            response_schema=ResearcherBriefingOutput,
            temperature=0.2,
        )
        briefing = build_authoritative_briefing(context, researcher_output)
        return context, briefing

    async def _run_baseline_single(self, briefing: FactBriefing) -> BranchResult:
        result = await self.llm_client.generate_structured(
            system_prompt=BASELINE_SINGLE_SYSTEM_PROMPT,
            user_prompt=build_baseline_single_prompt(briefing),
            response_schema=AgentVoteOutput,
            temperature=0.3,
        )
        vote = AgentVote(
            agent_id="baseline_single",
            agent_name="Single-Agent Baseline",
            vote=result.vote,
            confidence=result.confidence,
            prob_up_5d=result.prob_up_5d,
            position_pct=result.position_pct,
            reasoning=result.reasoning,
        )
        return _summarize_branch(
            branch_id="baseline_single",
            branch_name="Branch A · Single-Agent Baseline",
            description="1 neutral analyst, zero debate, zero persona bias (control benchmark).",
            votes=[vote],
        )

    async def _cast_blind_vote(
        self,
        persona: InvestorPersona,
        briefing: FactBriefing,
    ) -> AgentVote:
        result = await self.llm_client.generate_structured(
            system_prompt=persona.system_prompt,
            user_prompt=build_blind_vote_prompt(briefing),
            response_schema=AgentVoteOutput,
            temperature=0.5,
        )
        return AgentVote(
            agent_id=persona.agent_id,
            agent_name=persona.name,
            vote=result.vote,
            confidence=result.confidence,
            prob_up_5d=result.prob_up_5d,
            position_pct=result.position_pct,
            reasoning=result.reasoning,
        )

    async def _run_stage1_blind_votes(
        self,
        briefing: FactBriefing,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[AgentVote], VoteTally]:
        self._notify(
            on_progress,
            "Stage 1",
            f"Running {len(self.personas)} Investor Agents concurrently for blind votes...",
        )
        stage1_votes = list(
            await asyncio.gather(
                *(self._cast_blind_vote(persona, briefing) for persona in self.personas)
            )
        )
        anonymized_tally = compute_anonymized_tally(stage1_votes)
        return stage1_votes, anonymized_tally

    async def _run_stage2_debate(
        self,
        briefing: FactBriefing,
        anonymized_stage1_tally: VoteTally,
        own_stage1_by_id: dict[str, AgentVote],
        debate_rounds: int,
        on_progress: ProgressCallback | None,
    ) -> list[DebateTurn]:
        """Sequential Fog-of-War debate receiving FactBriefing + Anonymized Tally + own Stage 1 prior ONLY."""
        transcript: list[DebateTurn] = []
        turn_index = 0

        for round_number in range(1, debate_rounds + 1):
            for persona in self.personas:
                turn_index += 1
                self._notify(
                    on_progress,
                    "Stage 2",
                    f"[Fog-of-War] Round {round_number}/{debate_rounds} — {persona.name} speaking...",
                )
                prompt = build_debate_prompt(
                    briefing=briefing,
                    anonymized_tally=anonymized_stage1_tally,
                    transcript=transcript,
                    round_number=round_number,
                    total_rounds=debate_rounds,
                    own_stage1_vote=own_stage1_by_id.get(persona.agent_id),
                )
                reply = await self.llm_client.generate_structured(
                    system_prompt=persona.system_prompt,
                    user_prompt=prompt,
                    response_schema=DebateMessageOutput,
                    temperature=0.7,
                )
                transcript.append(
                    DebateTurn(
                        round_number=round_number,
                        turn_index=turn_index,
                        agent_id=persona.agent_id,
                        agent_name=persona.name,
                        message=reply.message,
                    )
                )

        return transcript

    async def _run_full_herd_branch(
        self,
        briefing: FactBriefing,
        stage1_votes: Sequence[AgentVote],
        own_stage1_by_id: dict[str, AgentVote],
        debate_rounds: int,
        on_progress: ProgressCallback | None,
    ) -> BranchResult:
        """Counterfactual Branch D: exposes all Stage 1 identities/votes to measure herding pressure."""
        transcript: list[DebateTurn] = []
        turn_index = 0

        for round_number in range(1, debate_rounds + 1):
            for persona in self.personas:
                turn_index += 1
                prompt = build_full_herd_debate_prompt(
                    briefing=briefing,
                    stage1_votes=stage1_votes,
                    transcript=transcript,
                    round_number=round_number,
                    total_rounds=debate_rounds,
                    own_stage1_vote=own_stage1_by_id.get(persona.agent_id),
                )
                reply = await self.llm_client.generate_structured(
                    system_prompt=persona.system_prompt,
                    user_prompt=prompt,
                    response_schema=DebateMessageOutput,
                    temperature=0.7,
                )
                transcript.append(
                    DebateTurn(
                        round_number=round_number,
                        turn_index=turn_index,
                        agent_id=persona.agent_id,
                        agent_name=persona.name,
                        message=reply.message,
                    )
                )

        self._notify(on_progress, "Lab Branch D", "Full-Herd branch casting secret final votes...")
        final_votes = list(
            await asyncio.gather(
                *(
                    self._cast_final_vote(persona, briefing, transcript, own_stage1_by_id.get(persona.agent_id))
                    for persona in self.personas
                )
            )
        )
        bm = compute_behavioral_metrics(stage1_votes, final_votes)
        return _summarize_branch(
            branch_id="full_herd",
            branch_name="Branch D · Full-Herd Council",
            description="All Stage 1 identities & votes exposed publicly before debate (social pressure test).",
            votes=final_votes,
            behavioral_metrics=bm,
            transcript=transcript,
        )

    async def _cast_final_vote(
        self,
        persona: InvestorPersona,
        briefing: FactBriefing,
        transcript: Sequence[DebateTurn],
        own_stage1_vote: AgentVote | None = None,
    ) -> AgentVote:
        result = await self.llm_client.generate_structured(
            system_prompt=persona.system_prompt,
            user_prompt=build_final_vote_prompt(briefing, transcript, own_stage1_vote=own_stage1_vote),
            response_schema=AgentVoteOutput,
            temperature=0.4,
        )
        return AgentVote(
            agent_id=persona.agent_id,
            agent_name=persona.name,
            vote=result.vote,
            confidence=result.confidence,
            prob_up_5d=result.prob_up_5d,
            position_pct=result.position_pct,
            reasoning=result.reasoning,
        )

    async def _run_stage3_final_votes(
        self,
        briefing: FactBriefing,
        transcript: Sequence[DebateTurn],
        own_stage1_by_id: dict[str, AgentVote],
        on_progress: ProgressCallback | None,
    ) -> tuple[list[AgentVote], VoteTally, VerdictChoice]:
        self._notify(
            on_progress,
            "Stage 3",
            f"Running {len(self.personas)} Investor Agents concurrently for secret final votes...",
        )
        stage3_votes = list(
            await asyncio.gather(
                *(
                    self._cast_final_vote(
                        persona,
                        briefing,
                        transcript,
                        own_stage1_vote=own_stage1_by_id.get(persona.agent_id),
                    )
                    for persona in self.personas
                )
            )
        )
        stage3_tally = compute_anonymized_tally(stage3_votes)
        verdict = compute_verdict(stage3_tally)
        return stage3_votes, stage3_tally, verdict

    async def _run_stage4_meta_judges(
        self,
        briefing: FactBriefing,
        branches: Sequence[BranchResult],
        on_progress: ProgressCallback | None,
        session_seed: str = "",
    ) -> MetaJudgePanelResult:
        """Option B (v2.2 De-Biased): Shadow (Steelman) -> Double-Blind Peer Select -> Final & Ensemble."""
        default_models = tuple(c.model_name for c in self.judge_clients.values())
        if len(default_models) != 3:
            default_models = META_JUDGE_MODELS
        scoreboard_obj = load_judge_scoreboard(self.scoreboard_path, default_models=default_models)
        scoreboard_snapshot = scoreboard_obj.judges
        seed_key = session_seed or f"{briefing.ticker}:{briefing.as_of_utc}"

        # Step 4A: Parallel Shadow Scoring across all 3 models (fault-tolerant per model).
        self._notify(
            on_progress,
            "Stage 4 · Meta-Judges",
            f"Running {len(self.judge_clients)} neutral Meta-Judges concurrently (Shadow Scoring)...",
        )

        async def _shadow_one(judge_id: str, client: StructuredLLMClient) -> MetaJudgeShadowDecision:
            try:
                out = await client.generate_structured(
                    system_prompt=META_JUDGE_BLANK_SYSTEM_PROMPT,
                    user_prompt=build_meta_judge_shadow_prompt(
                        briefing=briefing,
                        branches=branches,
                        scoreboard=scoreboard_snapshot,
                        own_judge_id=judge_id,
                    ),
                    response_schema=MetaJudgeDecisionOutput,
                    temperature=0.2,
                )
                return MetaJudgeShadowDecision(
                    judge_id=judge_id,
                    model_name=client.model_name,
                    status="OK",
                    verdict=out.verdict,
                    confidence=out.confidence,
                    prob_up_5d=out.prob_up_5d,
                    position_pct=out.position_pct,
                    minority_counterpoint=out.minority_counterpoint.strip(),
                    reasoning=out.reasoning,
                )
            except Exception as exc:
                return MetaJudgeShadowDecision(
                    judge_id=judge_id,
                    model_name=client.model_name,
                    status="UNAVAILABLE",
                    error=str(exc)[:240],
                )

        shadow_decisions = list(
            await asyncio.gather(
                *(_shadow_one(jid, client) for jid, client in sorted(self.judge_clients.items()))
            )
        )
        ok_shadows = [d for d in shadow_decisions if d.status == "OK"]
        ensemble_prob = compute_ensemble_prob_up_5d(ok_shadows, scoreboard_snapshot)

        # Step 4B: Double-Blind Peer Selection (per-voter shuffled blind aliases, model names stripped).
        peer_votes: list[PeerSelectionVote] = []
        peer_blind_aliases: dict[str, dict[str, str]] = {}
        if len(ok_shadows) >= 2:
            self._notify(
                on_progress,
                "Stage 4 · Peer Selection",
                f"{len(ok_shadows)} Meta-Judges voting double-blind on Final Judge (anti-self-vote)...",
            )

            async def _peer_vote_one(shadow: MetaJudgeShadowDecision) -> PeerSelectionVote:
                client = self.judge_clients[shadow.judge_id]
                blind_pairs, alias_to_jid = build_blind_peer_aliases(
                    shadow.judge_id, ok_shadows, seed_key
                )
                peer_blind_aliases[shadow.judge_id] = alias_to_jid
                try:
                    vote_out = await client.generate_structured(
                        system_prompt=META_JUDGE_BLANK_SYSTEM_PROMPT,
                        user_prompt=build_peer_selection_prompt(
                            briefing=briefing,
                            voter_judge_id=shadow.judge_id,
                            shadow_decisions=shadow_decisions,
                            scoreboard=scoreboard_snapshot,
                            blind_candidates=blind_pairs,
                        ),
                        response_schema=PeerSelectionVoteOutput,
                        temperature=0.1,
                    )
                    raw_pick = vote_out.selected_judge_id.strip()
                    # Map blind alias back to real judge_id (also tolerate direct judge_id in mocks).
                    resolved_pick = alias_to_jid.get(raw_pick, raw_pick)
                    if resolved_pick == raw_pick:
                        for alias, real_jid in alias_to_jid.items():
                            if alias.lower() in raw_pick.lower():
                                resolved_pick = real_jid
                                break
                    return PeerSelectionVote(
                        voter_judge_id=shadow.judge_id,
                        voter_model_name=client.model_name,
                        selected_judge_id=resolved_pick,
                        reasoning=vote_out.reasoning.strip(),
                        status="OK",
                    )
                except Exception as exc:
                    return PeerSelectionVote(
                        voter_judge_id=shadow.judge_id,
                        voter_model_name=client.model_name,
                        selected_judge_id="",
                        reasoning="",
                        status="UNAVAILABLE",
                        error=str(exc)[:240],
                    )

            peer_votes = list(await asyncio.gather(*(_peer_vote_one(s) for s in ok_shadows)))

        winner_id, tie_used, tie_reason = select_peer_winner(
            peer_votes=peer_votes,
            shadow_decisions=shadow_decisions,
            scoreboard=scoreboard_snapshot,
            seed_key=seed_key,
        )

        # Step 4C: Selected Final Judge executes final synthesis (model names redacted).
        final_decision: MetaJudgeShadowDecision | None = None
        selected_model_name: str | None = None
        if winner_id is not None and winner_id in self.judge_clients:
            winner_client = self.judge_clients[winner_id]
            selected_model_name = winner_client.model_name
            self._notify(
                on_progress,
                "Stage 4 · Final Judge",
                f"{winner_id} ({selected_model_name}) issuing definitive Final Meta Verdict...",
            )
            try:
                final_out = await winner_client.generate_structured(
                    system_prompt=META_JUDGE_BLANK_SYSTEM_PROMPT,
                    user_prompt=build_final_judge_synthesis_prompt(
                        briefing=briefing,
                        branches=branches,
                        selected_judge_id=winner_id,
                        shadow_decisions=shadow_decisions,
                        scoreboard=scoreboard_snapshot,
                    ),
                    response_schema=MetaJudgeDecisionOutput,
                    temperature=0.2,
                )
                final_decision = MetaJudgeShadowDecision(
                    judge_id=winner_id,
                    model_name=selected_model_name,
                    status="OK",
                    verdict=final_out.verdict,
                    confidence=final_out.confidence,
                    prob_up_5d=final_out.prob_up_5d,
                    position_pct=final_out.position_pct,
                    minority_counterpoint=final_out.minority_counterpoint.strip(),
                    reasoning=final_out.reasoning,
                )
            except Exception as exc:
                fallback = next((s for s in ok_shadows if s.judge_id == winner_id), None)
                if fallback is not None:
                    final_decision = fallback.model_copy(
                        update={"error": f"Synthesis fallback to shadow decision ({exc})"}
                    )

        return MetaJudgePanelResult(
            shadow_decisions=shadow_decisions,
            peer_votes=peer_votes,
            peer_blind_aliases=peer_blind_aliases,
            selected_final_judge_id=winner_id,
            selected_final_model_name=selected_model_name,
            tie_breaker_used=tie_used,
            tie_breaker_reason=tie_reason,
            final_decision=final_decision,
            ensemble_prob_up_5d=ensemble_prob,
            scoreboard_snapshot=scoreboard_snapshot,
        )

    async def run_simulation(
        self,
        ticker: str,
        *,
        debate_rounds: int = DEBATE_ROUNDS,
        enable_counterfactual_lab: bool = False,
        on_progress: ProgressCallback | None = None,
        save_to_disk: bool = True,
    ) -> SessionRecord:
        """Execute the 4-stage pipeline (+ optional v2.1 Parallel Lab & 3-Model Meta-Judges)."""
        if not (MIN_DEBATE_ROUNDS <= debate_rounds <= MAX_DEBATE_ROUNDS):
            raise ValueError(
                f"debate_rounds must be between {MIN_DEBATE_ROUNDS} and {MAX_DEBATE_ROUNDS} (got {debate_rounds})."
            )

        normalized_ticker = ticker.strip().upper()
        context, briefing = await self._run_stage0_research(normalized_ticker, on_progress)

        # Stage 1: Blind concurrent votes (+ Branch A single-agent baseline concurrently if Lab enabled).
        baseline_branch: BranchResult | None = None
        if enable_counterfactual_lab:
            (stage1_votes, stage1_tally), baseline_branch = await asyncio.gather(
                self._run_stage1_blind_votes(briefing, on_progress),
                self._run_baseline_single(briefing),
            )
        else:
            stage1_votes, stage1_tally = await self._run_stage1_blind_votes(briefing, on_progress)

        own_stage1_by_id = {v.agent_id: v for v in stage1_votes}

        # Stage 2 & 3: Fog-of-War Council (+ Full-Herd Branch D concurrently if Lab enabled).
        async def _run_fog_of_war_stages2_3() -> tuple[list[DebateTurn], list[AgentVote], VoteTally, VerdictChoice]:
            tr = await self._run_stage2_debate(
                briefing=briefing,
                anonymized_stage1_tally=stage1_tally,
                own_stage1_by_id=own_stage1_by_id,
                debate_rounds=debate_rounds,
                on_progress=on_progress,
            )
            s3_votes, s3_tally, verdict = await self._run_stage3_final_votes(
                briefing=briefing,
                transcript=tr,
                own_stage1_by_id=own_stage1_by_id,
                on_progress=on_progress,
            )
            return tr, s3_votes, s3_tally, verdict

        full_herd_branch: BranchResult | None = None
        if enable_counterfactual_lab:
            (transcript, stage3_votes, stage3_tally, final_verdict), full_herd_branch = await asyncio.gather(
                _run_fog_of_war_stages2_3(),
                self._run_full_herd_branch(
                    briefing=briefing,
                    stage1_votes=stage1_votes,
                    own_stage1_by_id=own_stage1_by_id,
                    debate_rounds=debate_rounds,
                    on_progress=on_progress,
                ),
            )
        else:
            transcript, stage3_votes, stage3_tally, final_verdict = await _run_fog_of_war_stages2_3()

        opinion_shifts = compute_opinion_shifts(stage1_votes, stage3_votes)
        behavioral_metrics = compute_behavioral_metrics(stage1_votes, stage3_votes)

        counterfactual_branches: list[BranchResult] = []
        if enable_counterfactual_lab:
            assert baseline_branch is not None and full_herd_branch is not None
            isolated_branch = _summarize_branch(
                branch_id="isolated_crowd",
                branch_name="Branch B · Isolated Crowd",
                description="5 agents voting independently in blind isolation (Wisdom of Crowds, zero debate).",
                votes=stage1_votes,
            )
            fog_branch = _summarize_branch(
                branch_id="fog_of_war",
                branch_name="Branch C · Fog-of-War Council",
                description="Debate sees anonymized Stage 1 tally ONLY (identities & reasons firewalled).",
                votes=stage3_votes,
                behavioral_metrics=behavioral_metrics,
                transcript=transcript,
            )
            counterfactual_branches = [
                baseline_branch,
                isolated_branch,
                fog_branch,
                full_herd_branch,
            ]

        session_id = uuid4().hex[:8]
        meta_judge_panel: MetaJudgePanelResult | None = None
        if enable_counterfactual_lab and self.judge_clients:
            meta_judge_panel = await self._run_stage4_meta_judges(
                briefing=briefing,
                branches=counterfactual_branches,
                on_progress=on_progress,
                session_seed=f"{context.ticker}:{session_id}",
            )

        strategy_ticket = build_strategy_ticket(
            briefing=briefing,
            stage3_votes=stage3_votes,
            final_verdict=final_verdict,
            behavioral_metrics=behavioral_metrics,
            meta_judge_panel=meta_judge_panel,
        )

        record = SessionRecord(
            schema_version="2.3",
            session_id=session_id,
            created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ticker=context.ticker,
            model_name=self.llm_client.model_name,
            debate_rounds=debate_rounds,
            raw_news=context.news_items,
            fact_briefing=briefing,
            stage1_votes=stage1_votes,
            stage1_tally=stage1_tally,
            debate_transcript=transcript,
            stage3_votes=stage3_votes,
            stage3_tally=stage3_tally,
            final_verdict=final_verdict,
            opinion_shifts=opinion_shifts,
            behavioral_metrics=behavioral_metrics,
            counterfactual_branches=counterfactual_branches,
            meta_judge_panel=meta_judge_panel,
            strategy_ticket=strategy_ticket,
        )

        if save_to_disk:
            self._notify(on_progress, "Storage", "Saving session record to data/history/...")
            save_session(record, history_dir=self.history_dir)
            if meta_judge_panel is not None:
                default_models = tuple(c.model_name for c in self.judge_clients.values())
                if len(default_models) != 3:
                    default_models = META_JUDGE_MODELS
                rebuild_judge_scoreboard_from_history(
                    history_dir=self.history_dir,
                    scoreboard_path=self.scoreboard_path,
                    default_models=default_models,
                )

        self._notify(on_progress, "Complete", f"Final Verdict for {record.ticker}: {record.final_verdict}")
        return record

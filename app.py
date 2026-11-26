"""Streamlit UI for Stock Council Simulator (v2.1 Behavioral Finance Research Lab)."""

from __future__ import annotations

import asyncio
import html
from contextlib import AsyncExitStack
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from src.data_fetcher import TickerDataError
from src.evaluator import evaluate_all_history_sessions, load_judge_scoreboard
from src.llm_client import GeminiLLMClient, LLMGenerationError, MissingAPIKeyError
from src.orchestrator import CouncilOrchestrator
from src.preview import build_layout_preview_record
from src.prompts import PERSONA_BY_ID
from src.schemas import (
    AgentVote,
    DebateTurn,
    JudgeTrackRecord,
    SessionRecord,
    VoteTally,
)
from src.storage import list_sessions, load_session

st.set_page_config(
    page_title="Stock Council Simulator · Behavioral Lab",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}
.verdict-banner {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.06), rgba(30, 41, 59, 0.03));
    margin-bottom: 1rem;
}
.pill {
    display: inline-block;
    padding: 0.22rem 0.72rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    border: 1px solid transparent;
}
.pill-buy {
    background: rgba(16, 185, 129, 0.16);
    color: #059669;
    border-color: rgba(16, 185, 129, 0.45);
}
.pill-hold {
    background: rgba(245, 158, 11, 0.16);
    color: #d97706;
    border-color: rgba(245, 158, 11, 0.45);
}
.pill-sell {
    background: rgba(239, 68, 68, 0.16);
    color: #dc2626;
    border-color: rgba(239, 68, 68, 0.45);
}
.pill-split {
    background: rgba(100, 116, 139, 0.18);
    color: #475569;
    border-color: rgba(100, 116, 139, 0.45);
}
.fog-box {
    border-radius: 10px;
    padding: 0.9rem 1.15rem;
    border-left: 4px solid #6366f1;
    background: rgba(99, 102, 241, 0.08);
    margin-bottom: 1rem;
}
.meta-box {
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border-left: 4px solid #0ea5e9;
    background: rgba(14, 165, 233, 0.08);
    margin-bottom: 1rem;
}
.muted-caption {
    font-size: 0.82rem;
    opacity: 0.75;
}
</style>
"""


def _pill_html(label: str) -> str:
    safe = html.escape(label)
    if label == "BUY":
        cls = "pill-buy"
    elif label == "SELL":
        cls = "pill-sell"
    elif label == "HOLD":
        cls = "pill-hold"
    else:
        cls = "pill-split"
    return f'<span class="pill {cls}">{safe}</span>'


def _format_display_time(iso_utc: str) -> str:
    try:
        dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_utc[:16].replace("T", " ")


def _render_vote_cards(votes: list[AgentVote], stage1_lookup: dict[str, AgentVote] | None = None) -> None:
    cols = st.columns(len(votes))
    for col, vote in zip(cols, votes):
        persona = PERSONA_BY_ID.get(vote.agent_id)
        avatar = persona.avatar if persona else "👤"
        archetype = persona.archetype if persona else ""

        shift_note = ""
        if stage1_lookup and vote.agent_id in stage1_lookup:
            prior = stage1_lookup[vote.agent_id]
            conf_delta = vote.confidence - prior.confidence
            prob_delta = vote.prob_up_5d - prior.prob_up_5d
            if prior.vote != vote.vote:
                shift_note = f"⚡ Shifted from **{prior.vote}** (Conf {conf_delta:+d}, P5D {prob_delta:+d}%)"
            else:
                shift_note = f"Held **{prior.vote}** (Conf {conf_delta:+d}, P5D {prob_delta:+d}%)"

        with col:
            with st.container(border=True):
                st.markdown(f"**{avatar} {vote.agent_name}**")
                if archetype:
                    st.caption(archetype)
                st.markdown(_pill_html(vote.vote), unsafe_allow_html=True)
                st.progress(vote.confidence / 10.0, text=f"Conf: {vote.confidence}/10")
                st.caption(f"P(Up 5D): **{vote.prob_up_5d}%** · Pos: **{vote.position_pct:+d}%**")
                if shift_note:
                    st.caption(shift_note)
                st.write(vote.reasoning)


def _render_debate_round(turns: list[DebateTurn]) -> None:
    for turn in turns:
        persona = PERSONA_BY_ID.get(turn.agent_id)
        avatar = persona.avatar if persona else "💬"
        archetype = persona.archetype if persona else ""
        with st.chat_message(name=turn.agent_id, avatar=avatar):
            st.markdown(f"**{turn.agent_name}** · _{archetype}_")
            st.write(turn.message)


def _render_tally_metrics(tally: VoteTally) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY Votes", tally.BUY)
    c2.metric("HOLD Votes", tally.HOLD)
    c3.metric("SELL Votes", tally.SELL)


def _render_scoreboard_table(judges: dict[str, JudgeTrackRecord]) -> None:
    if not judges:
        st.caption("No Meta-Judge track records yet.")
        return
    rows = []
    for jid in sorted(judges.keys()):
        rec = judges[jid]
        rows.append(
            {
                "Judge": rec.judge_id,
                "Model": rec.model_name,
                "Runs": rec.total_runs,
                "Eval (5D)": rec.evaluated_runs,
                "Dir Calls": rec.directional_calls,
                "Shadow Win Rate": f"{rec.win_rate * 100:.1f}%" if rec.win_rate is not None else "Pending",
                "Avg Brier (5D)": f"{rec.avg_brier_5d:.3f}" if rec.avg_brier_5d is not None else "Pending",
                "Avg Excess vs SPY": f"{rec.avg_excess_return_5d:+.2f}%" if rec.avg_excess_return_5d is not None else "Pending",
                "Picked Final": rec.times_selected_final_judge,
                "Final Win Rate": f"{rec.selected_win_rate * 100:.1f}%" if rec.selected_win_rate is not None else "Pending",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_session_god_view(record: SessionRecord) -> None:
    briefing = record.fact_briefing
    shifts_count = sum(1 for s in record.opinion_shifts if s.changed_vote)
    meta_panel = record.meta_judge_panel
    meta_verdict_html = ""
    if meta_panel and meta_panel.final_decision and meta_panel.final_decision.verdict:
        meta_verdict_html = (
            f'<div style="text-align:right; margin-left:1.5rem;">'
            f'<div class="muted-caption" style="margin-bottom:0.25rem;">'
            f'FINAL META-JUDGE ({html.escape(meta_panel.selected_final_judge_id or "")})</div>'
            f'<div style="font-size:1.15rem;">{_pill_html(meta_panel.final_decision.verdict)}</div>'
            f"</div>"
        )

    # 1. Header & Verdict Banner
    st.markdown(
        f"""
        <div class="verdict-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;">
                <div>
                    <div style="font-size:1.6rem; font-weight:800; line-height:1.2;">
                        {html.escape(record.ticker)} — {html.escape(briefing.company_name)}
                    </div>
                    <div class="muted-caption" style="margin-top:0.25rem;">
                        Session <code>{html.escape(record.session_id)}</code> (v{html.escape(record.schema_version)}) ·
                        UTC {html.escape(_format_display_time(record.created_at_utc))} ·
                        Council Model <code>{html.escape(record.model_name)}</code> ·
                        {record.debate_rounds} Debate Round(s)
                    </div>
                </div>
                <div style="display:flex; align-items:center; flex-wrap:wrap;">
                    <div style="text-align:right;">
                        <div class="muted-caption" style="margin-bottom:0.25rem;">COUNCIL VERDICT (FOG-OF-WAR)</div>
                        <div style="font-size:1.15rem;">{_pill_html(record.final_verdict)}</div>
                    </div>
                    {meta_verdict_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "7-Day Price Change",
        f"${briefing.price_end:,.2f}",
        delta=f"{briefing.price_change_7d_pct:+.2f}% (vs ${briefing.price_start_7d:,.2f})",
    )
    k2.metric("Final Tally (B / H / S)", f"{record.stage3_tally.BUY} / {record.stage3_tally.HOLD} / {record.stage3_tally.SELL}")
    k3.metric("Stage 1 Tally (B / H / S)", f"{record.stage1_tally.BUY} / {record.stage1_tally.HOLD} / {record.stage1_tally.SELL}")
    k4.metric("Opinion Shifts", f"{shifts_count} / {len(record.stage1_votes)} Agents")

    # v2.0 Behavioral Telemetry Strip
    bm = record.behavioral_metrics
    if bm is not None:
        st.markdown("**Behavioral Finance Telemetry (Stage 1 -> Stage 3)**")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Herding Rate", f"{bm.herding_rate * 100:.0f}%")
        b2.metric("Conf Polarization", f"{bm.confidence_polarization:+.2f}")
        b3.metric("Vote Entropy", f"{bm.stage3_entropy:.2f}", delta=f"{bm.stage3_entropy - bm.stage1_entropy:+.2f} vs S1")
        b4.metric("Mean P(Up 5D)", f"{bm.mean_prob_up_5d_stage3:.1f}%", delta=f"{bm.mean_prob_up_5d_stage3 - bm.mean_prob_up_5d_stage1:+.1f}%")
        b5.metric("Mean Position", f"{bm.mean_position_pct_stage3:+.1f}%", delta=f"{bm.mean_position_pct_stage3 - bm.mean_position_pct_stage1:+.1f}%")

    if briefing.data_warning:
        st.warning(briefing.data_warning)

    st.divider()

    # 2. Stage 0 — Fact Briefing
    st.subheader("Stage 0 — Researcher Fact Briefing")
    st.caption(
        f"100% News + 7-Day Price Change Only (No Technical Indicators) · "
        f"{briefing.news_items_reviewed} articles reviewed · "
        f"{briefing.discarded_noise_count} noise/duplicates filtered · "
        f"Price window: {briefing.price_start_date} -> {briefing.price_end_date}"
    )
    st.info(briefing.executive_summary)

    st.markdown("**Verified Facts**")
    if briefing.verified_facts:
        for fact in briefing.verified_facts:
            st.markdown(f"- {fact}")
    else:
        st.caption("No verified news facts extracted.")

    cat_col, risk_col = st.columns(2)
    with cat_col:
        with st.container(border=True):
            st.markdown("**Key Catalysts**")
            if briefing.key_catalysts:
                for item in briefing.key_catalysts:
                    st.markdown(f"- {item}")
            else:
                st.caption("None identified.")

    with risk_col:
        with st.container(border=True):
            st.markdown("**Key Risks**")
            if briefing.key_risks:
                for item in briefing.key_risks:
                    st.markdown(f"- {item}")
            else:
                st.caption("None identified.")

    if record.raw_news:
        with st.expander(f"Inspect Raw News Articles Reviewed ({len(record.raw_news)})"):
            for idx, item in enumerate(record.raw_news, start=1):
                title_md = f"[{item.title}]({item.url})" if item.url else item.title
                st.markdown(f"**{idx}. {title_md}**")
                st.caption(f"{item.publisher} · {item.published_at or 'date unavailable'}")
                if item.summary:
                    st.write(item.summary)

    st.divider()

    # 3. Stage 1 — First Destination (Blind Vote)
    st.subheader("Stage 1 — First Destination (Blind Vote)")
    st.markdown(
        f"""
        <div class="fog-box">
            <strong>Fog of War Firewall — What Agents Saw Entering Stage 2:</strong><br/>
            Anonymized Tally ONLY → <code>{html.escape(record.stage1_tally.format_anonymous_line())}</code>
            <div class="muted-caption" style="margin-top:0.25rem;">
                Each agent remembered only its OWN Stage 1 numeric stance (anti-amnesia). Other agents' names, votes, and Stage 1 reasoning were strictly blocked.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_tally_metrics(record.stage1_tally)
    st.markdown("**God View — Secret Individual Stage 1 Votes**")
    _render_vote_cards(record.stage1_votes)

    st.divider()

    # 4. Stage 2 — Open Discussion Transcript
    st.subheader("Stage 2 — Open Discussion Transcript")
    st.caption("Agents spoke sequentially and saw only the Fact Briefing, their own Stage 1 prior, Anonymized Stage 1 Tally, and prior debate turns.")

    rounds_present = sorted({turn.round_number for turn in record.debate_transcript})
    if not rounds_present:
        st.caption("No debate turns recorded.")
    else:
        tab_labels = ["All Rounds"] + [f"Round {r}" for r in rounds_present]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            for r in rounds_present:
                st.markdown(f"#### Round {r}")
                _render_debate_round([t for t in record.debate_transcript if t.round_number == r])

        for idx, r in enumerate(rounds_present, start=1):
            with tabs[idx]:
                _render_debate_round([t for t in record.debate_transcript if t.round_number == r])

    st.divider()

    # 5. Stage 3 — Final Destination & Opinion Shift
    st.subheader("Stage 3 — Final Destination & Opinion Shift")
    st.caption("All 5 agents voted simultaneously in secret after reading the Fact Briefing + Full Debate Transcript.")
    _render_tally_metrics(record.stage3_tally)

    stage1_lookup = {vote.agent_id: vote for vote in record.stage1_votes}
    _render_vote_cards(record.stage3_votes, stage1_lookup=stage1_lookup)

    st.markdown("**Opinion Shift Matrix (Stage 1 vs. Stage 3)**")
    shift_rows = [
        {
            "Agent": shift.agent_name,
            "Stage 1 Vote": shift.stage1_vote,
            "S1 Conf": f"{shift.stage1_confidence}/10",
            "S1 P(Up 5D)": f"{shift.stage1_prob_up_5d}%",
            "Stage 3 Vote": shift.stage3_vote,
            "S3 Conf": f"{shift.stage3_confidence}/10",
            "S3 P(Up 5D)": f"{shift.stage3_prob_up_5d}%",
            "Conf Δ": f"{shift.confidence_delta:+d}",
            "Prob Δ": f"{shift.prob_delta:+d}%",
            "Shift Status": f"Changed ({shift.stage1_vote} -> {shift.stage3_vote})"
            if shift.changed_vote
            else "Held Firm",
        }
        for shift in record.opinion_shifts
    ]
    st.dataframe(pd.DataFrame(shift_rows), width="stretch", hide_index=True)

    # 6. v2.1 Parallel Counterfactual Experiment Lab
    if record.counterfactual_branches:
        st.divider()
        st.subheader("Stage 4 — Parallel Counterfactual Experiment Comparison (v2.1)")
        st.caption(
            "Controlled A/B experiment on the exact same Fact Briefing: comparing Single-Agent Baseline, "
            "Isolated Crowd, Fog-of-War Council, and Full-Herd Social Pressure."
        )
        branch_rows = []
        for br in record.counterfactual_branches:
            b_metrics = br.behavioral_metrics
            branch_rows.append(
                {
                    "Branch": br.branch_name,
                    "Verdict": br.verdict,
                    "Tally (B/H/S)": f"{br.tally.BUY}/{br.tally.HOLD}/{br.tally.SELL}",
                    "Mean P(Up 5D)": f"{br.mean_prob_up_5d:.1f}%",
                    "Mean Conf": f"{br.mean_confidence:.2f}/10",
                    "Mean Pos": f"{br.mean_position_pct:+.1f}%",
                    "Herding Rate": f"{b_metrics.herding_rate * 100:.0f}%" if b_metrics else "—",
                    "Conf Polar.": f"{b_metrics.confidence_polarization:+.2f}" if b_metrics else "—",
                }
            )
        st.dataframe(pd.DataFrame(branch_rows), width="stretch", hide_index=True)

        full_herd = next((b for b in record.counterfactual_branches if b.branch_id == "full_herd"), None)
        if full_herd and full_herd.debate_transcript:
            with st.expander(f"Inspect Branch D (Full-Herd Social Pressure) Transcript ({len(full_herd.debate_transcript)} turns)"):
                for turn in full_herd.debate_transcript:
                    persona = PERSONA_BY_ID.get(turn.agent_id)
                    avatar = persona.avatar if persona else "💬"
                    with st.chat_message(name=f"herd_{turn.agent_id}", avatar=avatar):
                        st.markdown(f"**[Round {turn.round_number}] {turn.agent_name}**")
                        st.write(turn.message)

    # 7. v2.1 Option B: 3 Blank Meta-Judges (Different Models) + Peer Selection + Scoreboard
    if meta_panel is not None:
        st.divider()
        st.subheader("Stage 5 — 3 Blank Meta-Judges (Option B: Cross-Model Peer Selection)")
        st.caption(
            "Three character-free neutral evaluators on distinct models independently cast Shadow Decisions "
            "(all scored equally in Forward Eval), then vote (anti-self-vote) on who becomes the Final Judge."
        )

        if meta_panel.final_decision and meta_panel.final_decision.status == "OK":
            fd = meta_panel.final_decision
            tie_note = f" · Tie-breaker: {meta_panel.tie_breaker_reason}" if meta_panel.tie_breaker_used and meta_panel.tie_breaker_reason else ""
            st.markdown(
                f"""
                <div class="meta-box">
                    <strong>Peer-Selected Final Judge:</strong>
                    <code>{html.escape(meta_panel.selected_final_judge_id or '')}</code>
                    (<code>{html.escape(meta_panel.selected_final_model_name or '')}</code>)
                    &nbsp;→&nbsp; {_pill_html(fd.verdict or 'HOLD')}
                    &nbsp;·&nbsp; Conf <strong>{fd.confidence}/10</strong>
                    &nbsp;·&nbsp; P(Up 5D) <strong>{fd.prob_up_5d}%</strong>
                    &nbsp;·&nbsp; Pos <strong>{fd.position_pct:+d}%</strong>
                    <div style="margin-top:0.4rem;">{html.escape(fd.reasoning)}</div>
                    <div class="muted-caption" style="margin-top:0.3rem;">{html.escape(tie_note.lstrip(' ·'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("**Shadow Decisions (All 3 Models Scored Equally)**")
        if meta_panel.shadow_decisions:
            jcols = st.columns(len(meta_panel.shadow_decisions))
            for col, sd in zip(jcols, meta_panel.shadow_decisions):
                is_winner = sd.judge_id == meta_panel.selected_final_judge_id
                badge = " 🏆 Selected Final Judge" if is_winner else ""
                with col:
                    with st.container(border=True):
                        st.markdown(f"**⚖️ {sd.judge_id}{badge}**")
                        st.caption(f"Model: `{sd.model_name}`")
                        if sd.status == "OK" and sd.verdict:
                            st.markdown(_pill_html(sd.verdict), unsafe_allow_html=True)
                            st.progress((sd.confidence or 5) / 10.0, text=f"Conf: {sd.confidence}/10")
                            st.caption(f"P(Up 5D): **{sd.prob_up_5d}%** · Pos: **{sd.position_pct:+d}%**")
                            st.write(sd.reasoning)
                        else:
                            st.error(f"UNAVAILABLE: {sd.error or 'Model error'}")

        if meta_panel.peer_votes:
            st.markdown("**Peer Selection Votes (Anti-Self-Vote)**")
            pv_rows = [
                {
                    "Voter": f"{pv.voter_judge_id} ({pv.voter_model_name})",
                    "Voted For": pv.selected_judge_id if pv.status == "OK" else "UNAVAILABLE",
                    "Rationale": pv.reasoning if pv.status == "OK" else (pv.error or ""),
                }
                for pv in meta_panel.peer_votes
            ]
            st.dataframe(pd.DataFrame(pv_rows), width="stretch", hide_index=True)

    # 8. Forward-Run Paper Evaluation (T+1D / T+5D / T+10D vs SPY)
    if record.forward_eval is not None:
        st.divider()
        st.subheader("Forward-Run Paper Evaluation (Zero Look-Ahead Bias)")
        fe = record.forward_eval
        st.caption(f"Entry: ${fe.entry_price:,.2f} on {fe.entry_date} · Last Evaluated: {_format_display_time(fe.evaluated_at_utc)} UTC")
        hcols = st.columns(len(fe.horizons) or 1)
        for col, (label, hres) in zip(hcols, fe.horizons.items()):
            with col:
                if hres.status == "EVALUATED" and hres.stock_return_pct is not None:
                    excess_txt = f"Excess vs SPY: {hres.excess_return_pct:+.2f}%" if hres.excess_return_pct is not None else ""
                    st.metric(
                        f"T+{label} ({hres.target_date})",
                        f"{hres.stock_return_pct:+.2f}%",
                        delta=excess_txt,
                    )
                else:
                    st.metric(f"T+{label}", hres.status)

    st.download_button(
        label="Download Session Record (JSON)",
        data=record.model_dump_json(indent=2),
        file_name=f"{record.ticker}_{record.session_id}.json",
        mime="application/json",
    )


async def _execute_live_simulation(
    *,
    ticker: str,
    model_name: str,
    debate_rounds: int,
    api_key: str,
    enable_lab: bool,
    judge_models: tuple[str, str, str],
    meta_judge_timeout_seconds: float,
    meta_judge_max_retries: int,
    on_progress,
) -> SessionRecord:
    async with AsyncExitStack() as stack:
        council_client = await stack.enter_async_context(
            GeminiLLMClient(api_key=api_key, model_name=model_name)
        )
        judge_clients: dict[str, GeminiLLMClient] = {}
        if enable_lab:
            for idx, jmodel in enumerate(judge_models, start=1):
                judge_clients[f"judge_{idx}"] = await stack.enter_async_context(
                    GeminiLLMClient(
                        api_key=api_key,
                        model_name=jmodel,
                        timeout_seconds=meta_judge_timeout_seconds,
                        max_retries=meta_judge_max_retries,
                    )
                )

        orchestrator = CouncilOrchestrator(
            llm_client=council_client,
            history_dir=config.HISTORY_DIR,
            judge_clients=judge_clients,
            scoreboard_path=config.JUDGE_SCOREBOARD_PATH,
        )
        return await orchestrator.run_simulation(
            ticker=ticker,
            debate_rounds=debate_rounds,
            enable_counterfactual_lab=enable_lab,
            on_progress=on_progress,
            save_to_disk=True,
        )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    settings = config.get_settings()

    if "active_record" not in st.session_state:
        st.session_state["active_record"] = None
    if "active_filename" not in st.session_state:
        st.session_state["active_filename"] = None

    with st.sidebar:
        st.title("🏛️ Stock Council")
        st.caption("v2.1 Behavioral Finance Lab · News + 7D Move Only")

        ticker_input = st.text_input(
            "Stock Ticker",
            value="NVDA",
            placeholder="e.g., NVDA, AAPL, TSLA",
        ).strip().upper()

        model_input = st.text_input(
            "Council Model",
            value=settings.default_model,
            help="Swap freely (e.g., gemini-3.5-flash-lite, gemini-3.5-flash, gemini-3.6-flash).",
        ).strip()

        debate_rounds = st.slider(
            "Debate Rounds (Stage 2)",
            min_value=config.MIN_DEBATE_ROUNDS,
            max_value=config.MAX_DEBATE_ROUNDS,
            value=settings.debate_rounds,
            step=1,
        )

        enable_lab = st.toggle(
            "v2.1 Research Lab (Parallel Branches + 3 Meta-Judges)",
            value=True,
            help="Runs Baseline, Isolated, Fog-of-War, and Full-Herd in parallel, then runs 3 character-free Meta-Judges on 3 distinct models.",
        )

        with st.expander("Option B · 3 Meta-Judge Models", expanded=False):
            j1_model = st.text_input("judge_1 model", value=settings.meta_judge_models[0]).strip()
            j2_model = st.text_input("judge_2 model", value=settings.meta_judge_models[1]).strip()
            j3_model = st.text_input("judge_3 model", value=settings.meta_judge_models[2]).strip()
        judge_models = (
            j1_model or settings.meta_judge_models[0],
            j2_model or settings.meta_judge_models[1],
            j3_model or settings.meta_judge_models[2],
        )

        env_key_ready = bool(
            settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here"
        )
        session_key_override = ""
        if env_key_ready:
            st.success("`GEMINI_API_KEY` loaded from `.env` / environment.", icon="✅")
        else:
            st.warning("`GEMINI_API_KEY` not set in `.env`. Enter one below or copy `.env.example` to `.env`.", icon="🔑")
            session_key_override = st.text_input(
                "Session Gemini API Key (not saved)",
                type="password",
                placeholder="Paste GEMINI_API_KEY for this session...",
            ).strip()

        run_clicked = st.button("Run Simulation", type="primary", width="stretch")
        preview_clicked = st.button("Preview Dashboard Layout (No API Key)", type="secondary", width="stretch")

        st.divider()
        st.subheader("Forward-Run Evaluator")
        eval_clicked = st.button("Evaluate Forward Returns (T+1D/5D/10D)", width="stretch")
        if eval_clicked:
            with st.spinner("Evaluating realized forward returns vs SPY..."):
                try:
                    updated, _ = evaluate_all_history_sessions(
                        history_dir=config.HISTORY_DIR,
                        scoreboard_path=config.JUDGE_SCOREBOARD_PATH,
                        default_models=judge_models,
                    )
                    if st.session_state.get("active_filename"):
                        st.session_state["active_record"] = load_session(
                            st.session_state["active_filename"],
                            history_dir=config.HISTORY_DIR,
                        )
                    st.success(f"Evaluated {updated} saved session(s) & updated Judge Scoreboard.")
                except Exception as exc:
                    st.error(f"Forward evaluation failed: {exc}")

        st.divider()
        st.subheader("Session History Replay")
        sessions = list_sessions(config.HISTORY_DIR)
        if not sessions:
            st.caption("No saved sessions in `data/history/` yet.")
        else:
            for item in sessions:
                tag = f" → ⚖️{item.meta_verdict}" if item.meta_verdict else ""
                label = f"{item.ticker} | {_format_display_time(item.created_at_utc)} | {item.final_verdict}{tag}"
                is_active = st.session_state.get("active_filename") == item.filename
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"hist_{item.filename}", type=btn_type, width="stretch"):
                    try:
                        loaded = load_session(item.filename, history_dir=config.HISTORY_DIR)
                        st.session_state["active_record"] = loaded
                        st.session_state["active_filename"] = item.filename
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load session: {exc}")

    # Layout preview mode (no API call, does not write to data/history/)
    if preview_clicked:
        st.session_state["active_record"] = build_layout_preview_record(
            ticker=ticker_input,
            model_name=model_input or settings.default_model,
            debate_rounds=debate_rounds,
            judge_models=judge_models,
        )
        st.session_state["active_filename"] = None
        st.rerun()

    # Live simulation execution
    if run_clicked:
        effective_key = session_key_override or settings.gemini_api_key
        with st.status(f"Running Stock Council for {ticker_input or '...'}...", expanded=True) as status:
            log_container = st.empty()
            progress_lines: list[str] = []

            def _on_progress(stage: str, detail: str) -> None:
                progress_lines.append(f"**[{stage}]** {detail}")
                log_container.markdown("\n\n".join(progress_lines))
                status.update(label=f"[{stage}] {detail}", state="running")

            try:
                record = asyncio.run(
                    _execute_live_simulation(
                        ticker=ticker_input,
                        model_name=model_input or settings.default_model,
                        debate_rounds=debate_rounds,
                        api_key=effective_key,
                        enable_lab=enable_lab,
                        judge_models=judge_models,
                        meta_judge_timeout_seconds=settings.meta_judge_timeout_seconds,
                        meta_judge_max_retries=settings.meta_judge_max_retries,
                        on_progress=_on_progress,
                    )
                )
            except MissingAPIKeyError as exc:
                status.update(label="Missing Gemini API Key", state="error")
                st.error(str(exc))
            except TickerDataError as exc:
                status.update(label="Market Data Error", state="error")
                st.error(str(exc))
            except LLMGenerationError as exc:
                status.update(label="Gemini API Error", state="error")
                st.error(str(exc))
            except Exception as exc:
                status.update(label="Unexpected Error", state="error")
                st.error(f"Simulation failed: {exc}")
            else:
                status.update(
                    label=f"Completed {record.ticker} — Council: {record.final_verdict}",
                    state="complete",
                )
                st.session_state["active_record"] = record
                refreshed = list_sessions(config.HISTORY_DIR)
                matched = next((s.filename for s in refreshed if s.session_id == record.session_id), None)
                st.session_state["active_filename"] = matched
                st.rerun()

    # Main display area
    active_record: SessionRecord | None = st.session_state.get("active_record")
    if active_record is None:
        st.title("Stock Council Simulator · Behavioral Finance Lab")
        st.markdown(
            "Standalone Multi-Agent Behavioral Finance Research Lab driven by **100% news + 7-day price change context only** "
            "(strictly zero technical indicators)."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.info("**Stage 0–1 · Research & Blind Vote**\n\nCurates `yfinance` news + 7D move, then 5 blind probabilistic votes.")
        c2.info("**Stage 2–3 · Fog-of-War Debate**\n\nSelf-memory + anonymized tally firewall + Herding/Polarization metrics.")
        c3.info("**Stage 4 · Parallel Counterfactual Lab**\n\nRuns Baseline, Isolated, Fog-of-War, and Full-Herd concurrently.")
        c4.info("**Stage 5 · 3 Blank Meta-Judges**\n\n3 distinct models shadow-score, peer-select (anti-self-vote), and track 5D Winrate/Brier.")

        st.divider()
        st.subheader("Persistent Meta-Judge Track Record Scoreboard")
        scoreboard = load_judge_scoreboard(config.JUDGE_SCOREBOARD_PATH, default_models=judge_models)
        _render_scoreboard_table(scoreboard.judges)

        st.caption("Configure a ticker in the sidebar and click **Run Simulation**, or preview the full v2.1 dashboard layout below.")
        if st.button("Open Full Dashboard Layout Preview (Mock)", type="primary"):
            st.session_state["active_record"] = build_layout_preview_record(
                ticker=ticker_input,
                model_name=model_input or settings.default_model,
                debate_rounds=debate_rounds,
                judge_models=judge_models,
            )
            st.session_state["active_filename"] = None
            st.rerun()
    else:
        _render_session_god_view(active_record)
        st.divider()
        st.subheader("Persistent Meta-Judge Track Record Scoreboard (All Sessions)")
        scoreboard = load_judge_scoreboard(config.JUDGE_SCOREBOARD_PATH, default_models=judge_models)
        _render_scoreboard_table(scoreboard.judges)


if __name__ == "__main__":
    main()

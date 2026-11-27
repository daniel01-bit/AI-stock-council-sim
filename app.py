"""Streamlit UI for AI-council-sim — Editorial Financial Briefing & Multi-Provider Model Slot."""

from __future__ import annotations

import asyncio
import html
from contextlib import AsyncExitStack
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from src.avatars import PERSONA_ACCENT_COLOR, get_avatar_svg, judge_display_name
from src.data_fetcher import TickerDataError
from src.evaluator import evaluate_all_history_sessions, load_judge_scoreboard
from src.llm_client import (
    MODEL_BY_SLOT,
    MODEL_TIERS,
    PROVIDER_LABELS,
    LLMGenerationError,
    MissingAPIKeyError,
    ModelSpec,
    ProviderName,
    create_llm_client,
    find_default_slot_label,
)
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
from src.stock_radar import STOCK_GROUPS, HotRadarRow, scan_hot_7d_movers
from src.storage import list_sessions, load_session

APP_NAME = "AI-council-sim"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="AC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Curated Meta-Judge selectable model slots (label -> (provider, model_id))
JUDGE_MODEL_OPTIONS: tuple[tuple[str, ProviderName, str], ...] = (
    ("Gemini · 3.6 Flash", "gemini", "gemini-3.6-flash"),
    ("Gemini · 3.5 Flash", "gemini", "gemini-3.5-flash"),
    ("Gemini · 3.5 Flash Lite", "gemini", "gemini-3.5-flash-lite"),
    ("Gemini · 3.1 Flash Lite", "gemini", "gemini-3.1-flash-lite"),
    ("Gemini · 3.7 Flash", "gemini", "gemini-3.7-flash"),
    ("OpenAI · GPT-4o Mini", "openai", "gpt-4o-mini"),
    ("OpenAI · GPT-4o", "openai", "gpt-4o"),
    ("Claude · 3.5 Haiku", "anthropic", "claude-3-5-haiku-latest"),
    ("Claude · 3.5 Sonnet", "anthropic", "claude-3-5-sonnet-latest"),
    ("Ollama · Qwen 4B (Local)", "ollama", "qwen3.5:4b"),
    ("Ollama · Qwen 7B (Local)", "ollama", "qwen2.5:7b"),
)
JUDGE_OPTION_LABELS: list[str] = [label for label, _, _ in JUDGE_MODEL_OPTIONS]
JUDGE_OPTION_BY_LABEL: dict[str, tuple[ProviderName, str]] = {
    label: (prov, mid) for label, prov, mid in JUDGE_MODEL_OPTIONS
}


def _judge_label_for_model(model_id: str) -> str:
    clean = model_id.strip()
    for label, _, mid in JUDGE_MODEL_OPTIONS:
        if mid == clean:
            return label
    return JUDGE_OPTION_LABELS[0]


def _html(markup: str) -> str:
    """Strip leading indentation and empty lines so Streamlit Markdown never turns HTML into code blocks."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&display=swap');

:root {
    --paper: #ECE9E2;
    --paper-alt: #E4E0D7;
    --surface: #F5F3EE;
    --ink: #1C222D;
    --ink-muted: #5C6370;
    --rule: #D5CFC4;
    --navy: #1E3252;
    --forest: #146C43;
    --forest-bg: #E2F0E8;
    --ochre: #A86416;
    --ochre-bg: #F6EBDC;
    --oxblood: #9B1C1C;
    --oxblood-bg: #F6E4E4;
    --slate: #475569;
    --slate-bg: #E5E8EC;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--ink);
}

.stApp {
    background-color: var(--paper);
}

header[data-testid="stHeader"] {
    background-color: rgba(236, 233, 226, 0.94);
    border-bottom: 1px solid var(--rule);
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--surface);
    border-color: var(--rule);
}

div[data-testid="stExpander"] {
    background-color: var(--surface);
    border-color: var(--rule);
}

/* Generous top padding so top brand title is never clipped by Streamlit's header */
.block-container {
    padding-top: 3.1rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--paper-alt);
    border-right: 1px solid var(--rule);
}

.sidebar-brand {
    border-top: 3px solid var(--navy);
    border-bottom: 1px solid var(--rule);
    padding: 0.75rem 0 0.7rem 0;
    margin-bottom: 0.95rem;
}
.sidebar-brand-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.0rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.025em;
    line-height: 1.15;
}
.sidebar-brand-sub {
    font-size: 0.78rem;
    color: var(--ink-muted);
    margin-top: 0.2rem;
}

.sidebar-group-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.69rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--navy);
    margin: 0.95rem 0 0.32rem 0;
}

/* Compact Model Spec Card */
.model-spec-card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-top: 2px solid var(--navy);
    border-radius: 6px;
    padding: 0.7rem 0.85rem;
    margin: 0.45rem 0 0.85rem 0;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.03);
}
.model-spec-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.32rem;
}
.provider-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.14rem 0.45rem;
    border-radius: 3px;
    background: var(--navy);
    color: #FFFFFF;
}
.status-ready {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    font-weight: 600;
    color: var(--forest);
    background: var(--forest-bg);
    padding: 0.14rem 0.42rem;
    border-radius: 3px;
}
.status-warn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    font-weight: 600;
    color: var(--ochre);
    background: var(--ochre-bg);
    padding: 0.14rem 0.42rem;
    border-radius: 3px;
}
.model-spec-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--ink);
    word-break: break-all;
    margin-bottom: 0.38rem;
}
.model-spec-meta {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    border-top: 1px solid var(--rule);
    padding-top: 0.36rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-muted);
    font-variant-numeric: tabular-nums;
}

/* Top App Brand Bar (never clipped) */
.app-topbar {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 2px solid var(--navy);
    padding: 0.35rem 0 0.65rem 0;
    margin-bottom: 1.15rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.app-topbar-name {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.05rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.025em;
    line-height: 1.25;
}
.app-topbar-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: var(--ink-muted);
    font-variant-numeric: tabular-nums;
}

/* Hero & Ticker Masthead */
.hero-masthead {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--navy);
    border-radius: 8px;
    padding: 1.45rem 1.7rem;
    margin-bottom: 1.15rem;
    box-shadow: 0 2px 8px rgba(17, 24, 39, 0.035);
}
.hero-app-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 3.1rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.028em;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: var(--ink-muted);
    margin-top: 0.45rem;
}
.hero-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.85rem;
}
.hero-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--navy);
    background: var(--paper-alt);
    border: 1px solid var(--rule);
    padding: 0.22rem 0.55rem;
    border-radius: 4px;
}

.ticker-headline {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.ticker-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: var(--ink-muted);
    margin-top: 0.4rem;
    font-variant-numeric: tabular-nums;
}

/* Prominent Circular Medallion Verdict Badges */
.verdict-group {
    display: flex;
    align-items: center;
    gap: 1.35rem;
    flex-wrap: wrap;
}
.verdict-medallion-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.38rem;
    border-left: 1px solid var(--rule);
    padding-left: 1.35rem;
}
.verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ink-muted);
}
.verdict-circle {
    width: 78px;
    height: 78px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    color: #FFFFFF;
    text-transform: uppercase;
    box-shadow: 0 6px 16px rgba(17, 24, 39, 0.16);
}
.verdict-circle-sub {
    font-size: 0.56rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    opacity: 0.88;
    margin-top: 0.08rem;
}
.verdict-circle-buy {
    background: #146C43;
    border: 3px solid #F5F3EE;
    outline: 2.5px solid #146C43;
}
.verdict-circle-hold {
    background: #A86416;
    border: 3px solid #F5F3EE;
    outline: 2.5px solid #A86416;
}
.verdict-circle-sell {
    background: #9B1C1C;
    border: 3px solid #F5F3EE;
    outline: 2.5px solid #9B1C1C;
}
.verdict-circle-split {
    background: #334155;
    border: 3px solid #F5F3EE;
    outline: 2.5px solid #334155;
}

/* Full-Color BUY / HOLD / SELL Tally Cards */
.tally-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin-bottom: 1rem;
}
.tally-card {
    border-radius: 8px;
    padding: 0.95rem 1.15rem;
    color: #FFFFFF;
    box-shadow: 0 3px 10px rgba(17, 24, 39, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.tally-card-buy { background: #146C43; }
.tally-card-hold { background: #B45309; }
.tally-card-sell { background: #9B1C1C; }
.tally-card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.95;
}
.tally-card-count {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.95rem;
    font-weight: 700;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}

/* KPI Strip */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin-bottom: 1rem;
}
.kpi-grid-5 {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}
@media (max-width: 960px) {
    .kpi-grid, .kpi-grid-5, .tally-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.025);
}
.kpi-card-subtle {
    background: var(--paper-alt);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: 0.22rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.42rem;
    font-weight: 600;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}
.kpi-value-sm {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}
.kpi-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    margin-top: 0.2rem;
    font-variant-numeric: tabular-nums;
}
.delta-pos { color: var(--forest); font-weight: 600; }
.delta-neg { color: var(--oxblood); font-weight: 600; }
.delta-flat { color: var(--ink-muted); }

/* Semantic Pills inside cards */
.pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.22rem 0.68rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.74rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid transparent;
}
.pill-buy {
    background: #146C43;
    color: #FFFFFF;
}
.pill-hold {
    background: #B45309;
    color: #FFFFFF;
}
.pill-sell {
    background: #9B1C1C;
    color: #FFFFFF;
}
.pill-split {
    background: #334155;
    color: #FFFFFF;
}

/* Summary & Highlight Callouts */
.briefing-lead {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--navy);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin: 0.45rem 0 1rem 0;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.08rem;
    line-height: 1.55;
    color: var(--ink);
}
.shared-tally-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--navy);
    border-radius: 4px;
    padding: 0.72rem 1.1rem;
    margin: 0.45rem 0 0.95rem 0;
}
.shared-tally-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--navy);
}
.shared-tally-numbers {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}

.winner-card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--forest);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 1rem 0;
}
.winner-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.45rem;
}
.winner-title {
    font-weight: 700;
    font-size: 0.96rem;
    color: var(--ink);
}
.winner-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--ink-muted);
    font-variant-numeric: tabular-nums;
}

/* Vote & Judge Cards */
.vote-card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: 0.95rem 1rem;
    height: 100%;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(17, 24, 39, 0.03);
}
.vote-card-buy { border-top: 3px solid var(--forest); }
.vote-card-hold { border-top: 3px solid var(--ochre); }
.vote-card-sell { border-top: 3px solid var(--oxblood); }
.vote-card-neutral { border-top: 3px solid var(--slate); }

.persona-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.55rem;
}
.avatar-wrap {
    width: 38px;
    height: 38px;
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.persona-name {
    font-weight: 700;
    font-size: 0.92rem;
    line-height: 1.2;
    color: var(--ink);
}
.persona-role {
    font-size: 0.72rem;
    color: var(--ink-muted);
}
.conf-track {
    width: 100%;
    height: 5px;
    background: var(--paper-alt);
    border-radius: 99px;
    overflow: hidden;
    margin: 0.25rem 0 0.45rem 0;
}
.conf-fill {
    height: 100%;
    background: var(--navy);
}
.conf-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.69rem;
    color: var(--ink-muted);
    font-variant-numeric: tabular-nums;
    display: flex;
    justify-content: space-between;
}
.vote-stats-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.71rem;
    color: var(--ink-muted);
    font-variant-numeric: tabular-nums;
    margin-bottom: 0.42rem;
}
.shift-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.15rem 0.44rem;
    border-radius: 3px;
    background: var(--paper-alt);
    color: var(--ink-muted);
    display: inline-block;
    margin-bottom: 0.45rem;
}
.shift-tag-changed {
    background: var(--ochre-bg);
    color: var(--ochre);
    font-weight: 600;
}
.selected-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    font-weight: 700;
    padding: 0.16rem 0.48rem;
    border-radius: 999px;
    background: var(--forest-bg);
    color: var(--forest);
    text-transform: uppercase;
    white-space: nowrap;
}
.vote-reasoning {
    font-size: 0.84rem;
    line-height: 1.5;
    color: var(--ink);
}

/* Debate Entries */
.debate-entry {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--navy);
    border-radius: 6px;
    padding: 0.88rem 1.12rem;
    margin-bottom: 0.68rem;
}
.debate-entry-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.42rem;
}
.debate-turn-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink-muted);
}
.debate-body {
    font-size: 0.91rem;
    line-height: 1.56;
    color: var(--ink);
}

/* Landing Step Cards */
.step-card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-top: 2px solid var(--navy);
    border-radius: 6px;
    padding: 1rem 1.05rem;
    height: 100%;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.025);
}
.step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--navy);
}
.step-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0.2rem 0 0.35rem 0;
}
.step-desc {
    font-size: 0.84rem;
    line-height: 1.48;
    color: var(--ink-muted);
}

/* Game-Style Dark Loading Overlay */
@keyframes council-hop {
    0%, 100% { transform: translateY(0px) scale(1); }
    45% { transform: translateY(-16px) scale(1.06); }
    70% { transform: translateY(-4px) scale(0.98); }
}
@keyframes progress-shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.game-loader-backdrop {
    position: fixed;
    inset: 0;
    z-index: 999999;
    background: rgba(12, 19, 34, 0.92);
    backdrop-filter: blur(5px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
}
.game-loader-box {
    width: 100%;
    max-width: 560px;
    background: #162238;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 16px;
    padding: 2rem 2.1rem;
    text-align: center;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.48);
    color: #F8FAFC;
}
.game-loader-brand {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.95rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}
.game-loader-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 0.25rem;
    margin-bottom: 1.45rem;
}
.game-loader-parade {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.9rem;
    padding: 0.9rem 0 1.25rem 0;
}
.parade-char {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    animation: council-hop 1.25s ease-in-out infinite;
}
.parade-char:nth-child(1) { animation-delay: 0.00s; }
.parade-char:nth-child(2) { animation-delay: 0.16s; }
.parade-char:nth-child(3) { animation-delay: 0.32s; }
.parade-char:nth-child(4) { animation-delay: 0.48s; }
.parade-char:nth-child(5) { animation-delay: 0.64s; }
.parade-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #CBD5E1;
}
.game-loader-bar-track {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.12);
    border-radius: 999px;
    overflow: hidden;
    margin: 0.35rem 0 0.95rem 0;
}
.game-loader-bar-fill {
    height: 100%;
    width: 100%;
    background: linear-gradient(90deg, #146C43 0%, #34D399 45%, #F59E0B 75%, #146C43 100%);
    background-size: 200% 100%;
    animation: progress-shimmer 1.8s linear infinite;
}
.game-loader-stage {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.84rem;
    font-weight: 600;
    color: #FDE68A;
    margin-bottom: 0.2rem;
}
.game-loader-detail {
    font-size: 0.84rem;
    color: #E2E8F0;
}
/* Editorial List Card & Dark Ledger Table */
.editorial-list-card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 0.75rem;
}
.editorial-list-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--navy);
    margin-bottom: 0.55rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--rule);
}
.editorial-list-card ul {
    margin: 0;
    padding-left: 1.15rem;
}
.editorial-list-card li {
    font-size: 0.88rem;
    line-height: 1.52;
    color: var(--ink);
    margin-bottom: 0.35rem;
}
.editorial-table-wrap {
    width: 100%;
    overflow-x: auto;
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 8px;
    margin: 0.35rem 0 0.9rem 0;
}
.editorial-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    font-variant-numeric: tabular-nums;
}
.editorial-table th {
    background: var(--paper-alt);
    color: var(--navy);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-align: left;
    padding: 0.68rem 0.85rem;
    border-bottom: 1px solid var(--rule);
    white-space: nowrap;
}
.editorial-table td {
    color: var(--ink);
    padding: 0.62rem 0.85rem;
    border-bottom: 1px solid var(--rule);
    line-height: 1.45;
}
.editorial-table tr:last-child td {
    border-bottom: none;
}
.editorial-table tbody tr:hover td {
    background: rgba(212, 175, 106, 0.06);
}
</style>
"""

# Warm Ink & Obsidian (FT Night Edition) — layered charcoal, warm ivory ink, champagne gold accent
DARK_MODE_CSS = """
<style>
:root {
    --paper: #121316;
    --paper-alt: #181A1F;
    --surface: #1F2229;
    --surface-hover: #272B34;
    --ink: #E7E4DC;
    --ink-muted: #9A9FA8;
    --rule: #2C303A;
    --navy: #D4AF6A;
    --forest: #34B37E;
    --forest-bg: rgba(27, 110, 75, 0.22);
    --ochre: #E0A44E;
    --ochre-bg: rgba(168, 100, 22, 0.22);
    --oxblood: #E06C6E;
    --oxblood-bg: rgba(158, 42, 43, 0.22);
    --slate: #B8BFC9;
    --slate-bg: rgba(100, 110, 125, 0.22);
}

.stApp, [data-testid="stAppViewContainer"], .main {
    background-color: #121316 !important;
    color: #E7E4DC !important;
}

header[data-testid="stHeader"] {
    background-color: rgba(18, 19, 22, 0.94) !important;
    border-bottom: 1px solid #2C303A;
}

section[data-testid="stSidebar"] {
    background-color: #181A1F !important;
    border-right: 1px solid #2C303A !important;
}

section[data-testid="stSidebar"] *,
.stMarkdown, .stMarkdown p, .stMarkdown li,
h1, h2, h3, h4, label,
[data-testid="stMetricValue"] {
    color: #E7E4DC;
}

[data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"] p {
    color: #9A9FA8 !important;
}

/* Explicit Dark Surface & Text Overrides (immune to DOM order) */
.hero-masthead,
.step-card,
.kpi-card,
.vote-card,
.debate-entry,
.briefing-lead,
.shared-tally-bar,
.winner-card,
.model-spec-card,
.editorial-list-card,
.editorial-table-wrap {
    background-color: #1F2229 !important;
    border-color: #2C303A !important;
    color: #E7E4DC !important;
}
.kpi-card-subtle {
    background-color: #181A1F !important;
    border-color: #2C303A !important;
}
.hero-masthead,
.step-card,
.model-spec-card {
    border-top-color: #D4AF6A !important;
}
.briefing-lead,
.shared-tally-bar {
    border-left-color: #D4AF6A !important;
}
.app-topbar {
    border-bottom-color: #D4AF6A !important;
}
.sidebar-brand {
    border-top-color: #D4AF6A !important;
    border-bottom-color: #2C303A !important;
}
.hero-app-title,
.app-topbar-name,
.sidebar-brand-title,
.step-num,
.sidebar-group-label,
.shared-tally-title,
.editorial-list-title {
    color: #D4AF6A !important;
}
.ticker-headline,
.step-title,
.kpi-value,
.kpi-value-sm,
.persona-name,
.vote-reasoning,
.debate-body,
.winner-title,
.model-spec-id,
.shared-tally-numbers,
.editorial-list-card li,
.editorial-table td {
    color: #E7E4DC !important;
}
.hero-subtitle,
.ticker-meta,
.step-desc,
.kpi-label,
.kpi-sub.delta-flat,
.persona-role,
.conf-label,
.vote-stats-line,
.debate-turn-tag,
.winner-meta,
.model-spec-meta,
.app-topbar-meta,
.sidebar-brand-sub,
.verdict-label {
    color: #9A9FA8 !important;
}
.editorial-table th {
    background-color: #181A1F !important;
    color: #D4AF6A !important;
    border-bottom-color: #2C303A !important;
}
.editorial-table td {
    border-bottom-color: #2C303A !important;
}

/* Smooth Brand & Surface Accents */
.provider-tag {
    background: #2C271E !important;
    color: #E5C88B !important;
    border: 1px solid rgba(212, 175, 106, 0.35) !important;
}
.hero-chip {
    background: #181A1F !important;
    color: #D4AF6A !important;
    border-color: #2C303A !important;
}
.shift-tag {
    background: #181A1F !important;
    color: #9A9FA8 !important;
}
.shift-tag-changed {
    background: rgba(168, 100, 22, 0.25) !important;
    color: #E0A44E !important;
}
.conf-track {
    background: #14161B !important;
}
.conf-fill {
    background: #D4AF6A !important;
}

/* Softened Semantic Fills (low glare, high legibility) */
.tally-card-buy, .pill-buy {
    background: #1B6E4B;
    color: #F3F1EA;
}
.tally-card-hold, .pill-hold {
    background: #A86416;
    color: #F3F1EA;
}
.tally-card-sell, .pill-sell {
    background: #9E2A2B;
    color: #F3F1EA;
}
.pill-split {
    background: #3B4252;
    color: #F3F1EA;
}

.verdict-circle {
    color: #F3F1EA;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.42);
}
.verdict-circle-buy {
    background: #1B6E4B;
    border-color: #1F2229;
    outline-color: #2A8C62;
}
.verdict-circle-hold {
    background: #A86416;
    border-color: #1F2229;
    outline-color: #C97D28;
}
.verdict-circle-sell {
    background: #9E2A2B;
    border-color: #1F2229;
    outline-color: #C03D3F;
}
.verdict-circle-split {
    background: #3B4252;
    border-color: #1F2229;
    outline-color: #5C667A;
}

/* Native Streamlit Inputs, Selectbox, Popovers, Expanders, Containers */
.stTextInput input,
.stTextInput > div > div,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1F2229 !important;
    color: #E7E4DC !important;
    border-color: #2C303A !important;
}
.stTextInput input::placeholder {
    color: #717784 !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #1F2229 !important;
    color: #E7E4DC !important;
    border-color: #2C303A !important;
}
li[role="option"]:hover {
    background-color: #272B34 !important;
}

div[data-testid="stExpander"] {
    background-color: #1F2229 !important;
    border: 1px solid #2C303A !important;
    border-radius: 6px !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {
    background-color: #1F2229 !important;
    color: #E7E4DC !important;
}
div[data-testid="stExpanderDetails"] {
    background-color: #181A1F !important;
    border-top: 1px solid #2C303A !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1F2229 !important;
    border-color: #2C303A !important;
}

div[data-testid="stAlert"] {
    background-color: #231F18 !important;
    color: #E7E4DC !important;
    border: 1px solid rgba(212, 175, 106, 0.35) !important;
}

/* Buttons */
.stButton > button[kind="secondary"],
.stDownloadButton > button {
    background-color: #1F2229 !important;
    color: #E7E4DC !important;
    border: 1px solid #2C303A !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button:hover {
    background-color: #272B34 !important;
    border-color: #D4AF6A !important;
    color: #F3F1EA !important;
}
.stButton > button[kind="primary"] {
    background-color: #B8914B !important;
    color: #121316 !important;
    border: 1px solid #D4AF6A !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #D4AF6A !important;
    color: #121316 !important;
}

/* Tabs & Sliders */
.stTabs [data-baseweb="tab-list"] {
    border-bottom-color: #2C303A !important;
}
.stTabs [data-baseweb="tab"] {
    color: #9A9FA8 !important;
}
.stTabs [aria-selected="true"] {
    color: #D4AF6A !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #D4AF6A !important;
}

hr {
    border-color: #2C303A !important;
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


def _verdict_circle_html(verdict: str) -> str:
    if verdict == "BUY":
        return '<div class="verdict-circle verdict-circle-buy">BUY</div>'
    if verdict == "SELL":
        return '<div class="verdict-circle verdict-circle-sell">SELL</div>'
    if verdict == "HOLD":
        return '<div class="verdict-circle verdict-circle-hold">HOLD</div>'
    return (
        '<div class="verdict-circle verdict-circle-split">'
        '<span>SPLIT</span><span class="verdict-circle-sub">HOLD</span>'
        "</div>"
    )


def _format_display_time(iso_utc: str) -> str:
    try:
        dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_utc[:16].replace("T", " ")


def _render_game_loader(slot, *, ticker: str, model_name: str, stage: str, detail: str) -> None:
    parade_items = [
        ("retail_fomo", "FOMO"),
        ("fundamentalist", "Value"),
        ("paranoid_bear", "Bear"),
        ("contrarian", "Rebel"),
        ("macro_analyst", "Macro"),
    ]
    chars_html = "".join(
        f'<div class="parade-char">{get_avatar_svg(pid, size=50)}<span class="parade-name">{label}</span></div>'
        for pid, label in parade_items
    )
    slot.markdown(
        _html(
            f"""
            <div class="game-loader-backdrop">
                <div class="game-loader-box">
                    <div class="game-loader-brand">{APP_NAME}</div>
                    <div class="game-loader-sub">Analyzing {html.escape(ticker or '...')} &middot; {html.escape(model_name)}</div>
                    <div class="game-loader-parade">{chars_html}</div>
                    <div class="game-loader-bar-track"><div class="game-loader-bar-fill"></div></div>
                    <div class="game-loader-stage">{html.escape(stage)}</div>
                    <div class="game-loader-detail">{html.escape(detail)}</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _render_vote_cards(
    votes: list[AgentVote],
    stage1_lookup: dict[str, AgentVote] | None = None,
) -> None:
    cols = st.columns(len(votes))
    for col, vote in zip(cols, votes):
        persona = PERSONA_BY_ID.get(vote.agent_id)
        role = persona.archetype if persona else "Member"
        avatar_svg = get_avatar_svg(vote.agent_id, size=38)

        card_border_cls = (
            "vote-card-buy"
            if vote.vote == "BUY"
            else "vote-card-sell"
            if vote.vote == "SELL"
            else "vote-card-hold"
        )

        shift_html = ""
        if stage1_lookup and vote.agent_id in stage1_lookup:
            prior = stage1_lookup[vote.agent_id]
            conf_delta = vote.confidence - prior.confidence
            prob_delta = vote.prob_up_5d - prior.prob_up_5d
            if prior.vote != vote.vote:
                shift_html = (
                    f'<div class="shift-tag shift-tag-changed">'
                    f'{html.escape(prior.vote)} &rarr; {html.escape(vote.vote)} '
                    f'(Conf {conf_delta:+d}, 5D {prob_delta:+d}%)'
                    f"</div>"
                )
            else:
                shift_html = (
                    f'<div class="shift-tag">'
                    f'Stayed {html.escape(prior.vote)} (Conf {conf_delta:+d})'
                    f"</div>"
                )

        conf_pct = max(0, min(100, int(vote.confidence * 10)))

        with col:
            st.markdown(
                _html(
                    f"""
                    <div class="vote-card {card_border_cls}">
                        <div class="persona-header">
                            <div class="avatar-wrap">{avatar_svg}</div>
                            <div>
                                <div class="persona-name">{html.escape(vote.agent_name)}</div>
                                <div class="persona-role">{html.escape(role)}</div>
                            </div>
                        </div>
                        <div style="margin-bottom:0.45rem;">{_pill_html(vote.vote)}</div>
                        <div class="conf-label">
                            <span>Confidence</span>
                            <span>{vote.confidence}/10</span>
                        </div>
                        <div class="conf-track">
                            <div class="conf-fill" style="width:{conf_pct}%;"></div>
                        </div>
                        <div class="vote-stats-line">
                            5D Up: <strong>{vote.prob_up_5d}%</strong> &middot; Size: <strong>{vote.position_pct:+d}%</strong>
                        </div>
                        {shift_html}
                        <div class="vote-reasoning">{html.escape(vote.reasoning)}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )


def _render_tally_strip(tally: VoteTally) -> None:
    st.markdown(
        _html(
            f"""
            <div class="tally-grid">
                <div class="tally-card tally-card-buy">
                    <span class="tally-card-label">BUY</span>
                    <span class="tally-card-count">{tally.BUY}</span>
                </div>
                <div class="tally-card tally-card-hold">
                    <span class="tally-card-label">HOLD</span>
                    <span class="tally-card-count">{tally.HOLD}</span>
                </div>
                <div class="tally-card tally-card-sell">
                    <span class="tally-card-label">SELL</span>
                    <span class="tally-card-count">{tally.SELL}</span>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _render_debate_turn(turn: DebateTurn) -> None:
    persona = PERSONA_BY_ID.get(turn.agent_id)
    role = persona.archetype if persona else "Member"
    accent = PERSONA_ACCENT_COLOR.get(turn.agent_id, "#16243D")
    avatar_svg = get_avatar_svg(turn.agent_id, size=36)
    st.markdown(
        _html(
            f"""
            <div class="debate-entry" style="border-left-color:{accent};">
                <div class="debate-entry-header">
                    <div class="persona-header" style="margin-bottom:0;">
                        <div class="avatar-wrap">{avatar_svg}</div>
                        <div>
                            <div class="persona-name">{html.escape(turn.agent_name)}</div>
                            <div class="persona-role">{html.escape(role)}</div>
                        </div>
                    </div>
                    <div class="debate-turn-tag">R{turn.round_number} &middot; #{turn.turn_index}</div>
                </div>
                <div class="debate-body">{html.escape(turn.message)}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _render_editorial_table(rows: list[dict]) -> None:
    """Render native st.dataframe in Light Mode and a seamless HTML ledger table in Dark Mode (zero white canvas glare)."""
    if not rows:
        return
    if not st.session_state.get("dark_mode", False):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        return
    headers = list(rows[0].keys())
    th_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in headers)
    tr_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in headers) + "</tr>"
        for row in rows
    )
    st.markdown(
        _html(
            f"""
            <div class="editorial-table-wrap">
                <table class="editorial-table">
                    <thead><tr>{th_html}</tr></thead>
                    <tbody>{tr_html}</tbody>
                </table>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _render_list_card(title: str, items: list[str], empty_text: str) -> None:
    if items:
        li_html = "".join(f"<li>{html.escape(item)}</li>" for item in items)
        body_html = f"<ul>{li_html}</ul>"
    else:
        body_html = f'<div style="font-size:0.84rem; color:var(--ink-muted);">{html.escape(empty_text)}</div>'
    st.markdown(
        _html(
            f"""
            <div class="editorial-list-card">
                <div class="editorial-list-title">{html.escape(title)}</div>
                {body_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _render_scoreboard_table(judges: dict[str, JudgeTrackRecord]) -> None:
    if not judges:
        st.caption("No judge history yet.")
        return
    rows = []
    for jid in sorted(judges.keys()):
        rec = judges[jid]
        rows.append(
            {
                "Judge": judge_display_name(rec.judge_id),
                "Model": rec.model_name,
                "Runs": rec.total_runs,
                "5D Checked": rec.evaluated_runs,
                "Win Rate": f"{rec.win_rate * 100:.1f}%" if rec.win_rate is not None else "Pending",
                "Avg Brier": f"{rec.avg_brier_5d:.3f}" if rec.avg_brier_5d is not None else "Pending",
                "Excess vs SPY": f"{rec.avg_excess_return_5d:+.2f}%" if rec.avg_excess_return_5d is not None else "Pending",
                "Picked Final": rec.times_selected_final_judge,
                "Final Win Rate": f"{rec.selected_win_rate * 100:.1f}%" if rec.selected_win_rate is not None else "Pending",
            }
        )
    _render_editorial_table(rows)


def _render_session_god_view(record: SessionRecord) -> None:
    briefing = record.fact_briefing
    shifts_count = sum(1 for s in record.opinion_shifts if s.changed_vote)
    meta_panel = record.meta_judge_panel
    delta_cls = (
        "delta-pos"
        if briefing.price_change_7d_pct > 0
        else "delta-neg"
        if briefing.price_change_7d_pct < 0
        else "delta-flat"
    )

    # Top App Brand Bar
    st.markdown(
        _html(
            f"""
            <div class="app-topbar">
                <div class="app-topbar-name">{APP_NAME}</div>
                <div class="app-topbar-meta">
                    {html.escape(_format_display_time(record.created_at_utc))} UTC &nbsp;&middot;&nbsp; {html.escape(record.model_name)}
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    meta_verdict_html = ""
    if meta_panel and meta_panel.final_decision and meta_panel.final_decision.verdict:
        meta_verdict_html = (
            '<div class="verdict-medallion-wrap">'
            '<div class="verdict-label">Meta-Judge</div>'
            f'{_verdict_circle_html(meta_panel.final_decision.verdict)}'
            "</div>"
        )

    # 1. Ticker Header & Prominent Round Verdict Medallions
    st.markdown(
        _html(
            f"""
            <div class="hero-masthead">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1.2rem;">
                    <div style="flex:1; min-width:250px;">
                        <div class="ticker-headline">
                            {html.escape(record.ticker)} &mdash; {html.escape(briefing.company_name)}
                        </div>
                        <div class="ticker-meta">
                            ID {html.escape(record.session_id)} &nbsp;|&nbsp;
                            {record.debate_rounds} Round(s) &nbsp;|&nbsp;
                            {briefing.news_items_reviewed} News Items
                        </div>
                    </div>
                    <div class="verdict-group">
                        <div class="verdict-medallion-wrap">
                            <div class="verdict-label">Council</div>
                            {_verdict_circle_html(record.final_verdict)}
                        </div>
                        {meta_verdict_html}
                    </div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # 2. Primary KPI Strip
    st.markdown(
        _html(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">7D Price</div>
                    <div class="kpi-value">${briefing.price_end:,.2f}</div>
                    <div class="kpi-sub {delta_cls}">
                        {briefing.price_change_7d_pct:+.2f}% (from ${briefing.price_start_7d:,.2f})
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Final Vote</div>
                    <div class="kpi-value">{record.stage3_tally.BUY}B / {record.stage3_tally.HOLD}H / {record.stage3_tally.SELL}S</div>
                    <div class="kpi-sub delta-flat">After debate</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Initial Vote</div>
                    <div class="kpi-value">{record.stage1_tally.BUY}B / {record.stage1_tally.HOLD}H / {record.stage1_tally.SELL}S</div>
                    <div class="kpi-sub delta-flat">Before debate</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Changed Minds</div>
                    <div class="kpi-value">{shifts_count} / {len(record.stage1_votes)}</div>
                    <div class="kpi-sub delta-flat">Switched stance</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # Group Dynamics Strip
    bm = record.behavioral_metrics
    if bm is not None:
        st.markdown(
            _html(
                f"""
                <div class="kpi-grid-5">
                    <div class="kpi-card-subtle">
                        <div class="kpi-label">Herding</div>
                        <div class="kpi-value-sm">{bm.herding_rate * 100:.0f}%</div>
                    </div>
                    <div class="kpi-card-subtle">
                        <div class="kpi-label">Polarization</div>
                        <div class="kpi-value-sm">{bm.confidence_polarization:+.2f}</div>
                    </div>
                    <div class="kpi-card-subtle">
                        <div class="kpi-label">Disagreement</div>
                        <div class="kpi-value-sm">{bm.stage3_entropy:.2f}</div>
                    </div>
                    <div class="kpi-card-subtle">
                        <div class="kpi-label">Avg 5D Up</div>
                        <div class="kpi-value-sm">{bm.mean_prob_up_5d_stage3:.1f}%</div>
                    </div>
                    <div class="kpi-card-subtle">
                        <div class="kpi-label">Avg Position</div>
                        <div class="kpi-value-sm">{bm.mean_position_pct_stage3:+.1f}%</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    if briefing.data_warning:
        st.warning(briefing.data_warning)

    # 3. Market Brief
    st.subheader("Market Brief")
    st.caption(
        f"News + 7-day price only · {briefing.price_start_date} to {briefing.price_end_date} · "
        f"{briefing.discarded_noise_count} noise filtered"
    )
    st.markdown(
        f'<div class="briefing-lead">{html.escape(briefing.executive_summary)}</div>',
        unsafe_allow_html=True,
    )

    facts_col, side_col = st.columns([1.35, 1.0])
    with facts_col:
        _render_list_card("Key Facts", briefing.verified_facts, "No facts extracted.")

    with side_col:
        _render_list_card("Catalysts", briefing.key_catalysts, "None identified.")
        _render_list_card("Risks", briefing.key_risks, "None identified.")

    if record.raw_news:
        with st.expander(f"News Sources ({len(record.raw_news)})"):
            for idx, item in enumerate(record.raw_news, start=1):
                title_md = f"[{item.title}]({item.url})" if item.url else item.title
                st.markdown(f"**{idx:02d}. {title_md}**")
                st.caption(f"{item.publisher} · {item.published_at or 'date unavailable'}")
                if item.summary:
                    st.write(item.summary)

    st.divider()

    # 4. Initial Votes
    st.subheader("Initial Votes")
    st.markdown(
        _html(
            f"""
            <div class="shared-tally-bar">
                <div class="shared-tally-title">Shared vote count before debate (names hidden)</div>
                <div class="shared-tally-numbers">{html.escape(record.stage1_tally.format_anonymous_line())}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    _render_tally_strip(record.stage1_tally)
    _render_vote_cards(record.stage1_votes)

    st.divider()

    # 5. Council Debate
    st.subheader("Council Debate")
    rounds_present = sorted({turn.round_number for turn in record.debate_transcript})
    if not rounds_present:
        st.caption("No debate turns recorded.")
    else:
        tab_labels = ["All"] + [f"Round {r}" for r in rounds_present]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            for r in rounds_present:
                st.markdown(f"**Round {r}**")
                for turn in [t for t in record.debate_transcript if t.round_number == r]:
                    _render_debate_turn(turn)

        for idx, r in enumerate(rounds_present, start=1):
            with tabs[idx]:
                for turn in [t for t in record.debate_transcript if t.round_number == r]:
                    _render_debate_turn(turn)

    st.divider()

    # 6. Final Verdict
    st.subheader("Final Verdict")
    _render_tally_strip(record.stage3_tally)
    stage1_lookup = {vote.agent_id: vote for vote in record.stage1_votes}
    _render_vote_cards(record.stage3_votes, stage1_lookup=stage1_lookup)

    st.markdown("**Vote Changes**")
    shift_rows = [
        {
            "Member": shift.agent_name,
            "Initial": shift.stage1_vote,
            "Conf (1)": f"{shift.stage1_confidence}/10",
            "5D Up (1)": f"{shift.stage1_prob_up_5d}%",
            "Final": shift.stage3_vote,
            "Conf (2)": f"{shift.stage3_confidence}/10",
            "5D Up (2)": f"{shift.stage3_prob_up_5d}%",
            "Conf Delta": f"{shift.confidence_delta:+d}",
            "Status": f"{shift.stage1_vote} -> {shift.stage3_vote}"
            if shift.changed_vote
            else "No Change",
        }
        for shift in record.opinion_shifts
    ]
    _render_editorial_table(shift_rows)

    # 7. Experiment Lab
    if record.counterfactual_branches:
        st.divider()
        st.subheader("Experiment Lab")
        st.caption("Same news brief tested across four decision setups.")
        branch_rows = []
        for br in record.counterfactual_branches:
            b_metrics = br.behavioral_metrics
            branch_rows.append(
                {
                    "Setup": br.branch_name,
                    "Verdict": br.verdict,
                    "Votes (B/H/S)": f"{br.tally.BUY}/{br.tally.HOLD}/{br.tally.SELL}",
                    "Avg 5D Up": f"{br.mean_prob_up_5d:.1f}%",
                    "Avg Conf": f"{br.mean_confidence:.2f}/10",
                    "Avg Pos": f"{br.mean_position_pct:+.1f}%",
                    "Herding": f"{b_metrics.herding_rate * 100:.0f}%" if b_metrics else "—",
                }
            )
        _render_editorial_table(branch_rows)

        full_herd = next((b for b in record.counterfactual_branches if b.branch_id == "full_herd"), None)
        if full_herd and full_herd.debate_transcript:
            with st.expander(f"Full-Herd Debate ({len(full_herd.debate_transcript)} turns)"):
                for turn in full_herd.debate_transcript:
                    _render_debate_turn(turn)

    # 8. Meta-Judges
    if meta_panel is not None:
        st.divider()
        st.subheader("Meta-Judges")
        st.caption("Three independent judges review the council debate and pick the strongest call.")

        if meta_panel.final_decision and meta_panel.final_decision.status == "OK":
            fd = meta_panel.final_decision
            winner_name = judge_display_name(meta_panel.selected_final_judge_id or "")
            tie_note = (
                f"Tie-breaker: {meta_panel.tie_breaker_reason}"
                if meta_panel.tie_breaker_used and meta_panel.tie_breaker_reason
                else ""
            )
            tie_html = (
                f'<div class="winner-meta" style="margin-top:0.35rem;">{html.escape(tie_note)}</div>'
                if tie_note
                else ""
            )
            st.markdown(
                _html(
                    f"""
                    <div class="winner-card">
                        <div class="winner-header">
                            <div>
                                <span class="winner-title">Selected Winner: {html.escape(winner_name)}</span>
                                <span class="winner-meta"> &middot; {html.escape(meta_panel.selected_final_model_name or '')}</span>
                            </div>
                            <div>{_pill_html(fd.verdict or 'HOLD')}</div>
                        </div>
                        <div class="winner-meta" style="margin-bottom:0.4rem;">
                            Confidence {fd.confidence}/10 &middot; 5D Up {fd.prob_up_5d}% &middot; Position {fd.position_pct:+d}%
                        </div>
                        <div class="vote-reasoning">{html.escape(fd.reasoning)}</div>
                        {tie_html}
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        if meta_panel.shadow_decisions:
            jcols = st.columns(len(meta_panel.shadow_decisions))
            for col, sd in zip(jcols, meta_panel.shadow_decisions):
                is_winner = sd.judge_id == meta_panel.selected_final_judge_id
                selected_badge = '<span class="selected-tag">Selected</span>' if is_winner else ""
                j_label = judge_display_name(sd.judge_id)
                j_avatar = get_avatar_svg(sd.judge_id, size=38)
                border_cls = (
                    "vote-card-buy"
                    if sd.verdict == "BUY"
                    else "vote-card-sell"
                    if sd.verdict == "SELL"
                    else "vote-card-hold"
                    if sd.verdict == "HOLD"
                    else "vote-card-neutral"
                )
                with col:
                    if sd.status == "OK" and sd.verdict:
                        conf_pct = max(0, min(100, int((sd.confidence or 5) * 10)))
                        st.markdown(
                            _html(
                                f"""
                                <div class="vote-card {border_cls}">
                                    <div class="persona-header" style="justify-content:space-between;">
                                        <div style="display:flex; align-items:center; gap:0.55rem;">
                                            <div class="avatar-wrap">{j_avatar}</div>
                                            <div>
                                                <div class="persona-name">{html.escape(j_label)}</div>
                                                <div class="persona-role">{html.escape(sd.model_name)}</div>
                                            </div>
                                        </div>
                                        {selected_badge}
                                    </div>
                                    <div style="margin-bottom:0.45rem;">{_pill_html(sd.verdict)}</div>
                                    <div class="conf-label">
                                        <span>Confidence</span>
                                        <span>{sd.confidence}/10</span>
                                    </div>
                                    <div class="conf-track">
                                        <div class="conf-fill" style="width:{conf_pct}%;"></div>
                                    </div>
                                    <div class="vote-stats-line">
                                        5D Up: <strong>{sd.prob_up_5d}%</strong> &middot; Size: <strong>{sd.position_pct:+d}%</strong>
                                    </div>
                                    <div class="vote-reasoning">{html.escape(sd.reasoning)}</div>
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(f"{j_label} ({sd.model_name}): {sd.error or 'Unavailable'}")

        if meta_panel.peer_votes:
            st.markdown("**Peer Votes**")
            pv_rows = [
                {
                    "Judge": f"{judge_display_name(pv.voter_judge_id)} ({pv.voter_model_name})",
                    "Picked": judge_display_name(pv.selected_judge_id) if pv.status == "OK" else "Unavailable",
                    "Reason": pv.reasoning if pv.status == "OK" else (pv.error or ""),
                }
                for pv in meta_panel.peer_votes
            ]
            _render_editorial_table(pv_rows)

    # 9. Forward Results
    if record.forward_eval is not None:
        st.divider()
        st.subheader("Forward Results")
        fe = record.forward_eval
        st.caption(
            f"Entry ${fe.entry_price:,.2f} on {fe.entry_date} · Updated {_format_display_time(fe.evaluated_at_utc)} UTC"
        )
        hcols = st.columns(len(fe.horizons) or 1)
        for col, (label, hres) in zip(hcols, fe.horizons.items()):
            with col:
                if hres.status == "EVALUATED" and hres.stock_return_pct is not None:
                    excess_txt = (
                        f"vs SPY: {hres.excess_return_pct:+.2f}%"
                        if hres.excess_return_pct is not None
                        else ""
                    )
                    st.metric(
                        f"T+{label} ({hres.target_date})",
                        f"{hres.stock_return_pct:+.2f}%",
                        delta=excess_txt,
                    )
                else:
                    st.metric(f"T+{label}", hres.status)

    st.download_button(
        label="Export JSON",
        data=record.model_dump_json(indent=2),
        file_name=f"{record.ticker}_{record.session_id}.json",
        mime="application/json",
    )


def _provider_auth_ready(provider: ProviderName, settings: config.Settings, session_key: str) -> tuple[bool, str]:
    if provider == "ollama":
        return True, "LOCAL"
    if provider == "gemini":
        ready = bool(
            session_key
            or (settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here")
        )
        return ready, "READY" if ready else "NEED KEY"
    if provider == "openai":
        ready = bool(
            session_key
            or (config.OPENAI_API_KEY and not config.OPENAI_API_KEY.startswith("your_"))
        )
        return ready, "READY" if ready else "NEED KEY"
    if provider == "anthropic":
        ready = bool(
            session_key
            or (config.ANTHROPIC_API_KEY and not config.ANTHROPIC_API_KEY.startswith("your_"))
        )
        return ready, "READY" if ready else "NEED KEY"
    return False, "NEED KEY"


def _resolve_provider_credentials(
    provider: ProviderName,
    settings: config.Settings,
    session_key_override: str,
    selected_provider: ProviderName,
    ollama_url_override: str | None,
) -> tuple[str | None, str | None]:
    if provider == "gemini":
        key = session_key_override if (selected_provider == "gemini" and session_key_override) else settings.gemini_api_key
        return key, None
    if provider == "openai":
        key = session_key_override if (selected_provider == "openai" and session_key_override) else config.OPENAI_API_KEY
        return key, config.OPENAI_BASE_URL
    if provider == "anthropic":
        key = session_key_override if (selected_provider == "anthropic" and session_key_override) else config.ANTHROPIC_API_KEY
        return key, None
    return None, (ollama_url_override or config.OLLAMA_BASE_URL)


def _render_model_spec_card(
    spec: ModelSpec,
    effective_model_id: str,
    auth_ready: bool,
    auth_label: str,
    is_custom: bool,
) -> None:
    status_cls = "status-ready" if auth_ready else "status-warn"
    provider_label = PROVIDER_LABELS.get(spec.provider, spec.provider.upper())
    custom_note = " (Custom)" if is_custom else ""
    st.markdown(
        _html(
            f"""
            <div class="model-spec-card">
                <div class="model-spec-top">
                    <span class="provider-tag">{html.escape(provider_label)}</span>
                    <span class="{status_cls}">{html.escape(auth_label)}</span>
                </div>
                <div class="model-spec-id">{html.escape(effective_model_id)}{html.escape(custom_note)}</div>
                <div class="model-spec-meta">
                    <span>{html.escape(spec.speed)}</span>
                    <span>{html.escape(spec.cost_tier)}</span>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


async def _execute_live_simulation(
    *,
    ticker: str,
    provider: ProviderName,
    model_name: str,
    debate_rounds: int,
    api_key: str | None,
    base_url: str | None,
    enable_lab: bool,
    judge_specs: tuple[tuple[ProviderName, str], tuple[ProviderName, str], tuple[ProviderName, str]],
    settings: config.Settings,
    session_key_override: str,
    ollama_url_override: str | None,
    meta_judge_timeout_seconds: float,
    meta_judge_max_retries: int,
    on_progress,
) -> SessionRecord:
    async with AsyncExitStack() as stack:
        council_client = await stack.enter_async_context(
            create_llm_client(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
            )
        )
        judge_clients = {}
        if enable_lab:
            for idx, (j_prov, j_model) in enumerate(judge_specs, start=1):
                j_key, j_url = _resolve_provider_credentials(
                    j_prov,
                    settings,
                    session_key_override,
                    provider,
                    ollama_url_override,
                )
                judge_clients[f"judge_{idx}"] = await stack.enter_async_context(
                    create_llm_client(
                        provider=j_prov,
                        model_name=j_model,
                        api_key=j_key,
                        base_url=j_url,
                        timeout_seconds=meta_judge_timeout_seconds,
                        max_retries=meta_judge_max_retries,
                        fallback_models=(),
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


def _select_ticker_callback(symbol: str) -> None:
    st.session_state["ticker_input"] = symbol.strip().upper()


def _record_preview_history(record: SessionRecord) -> None:
    """Keep all user-tried preview runs clickable in History without polluting data/history/."""
    previews: list[dict] = st.session_state.setdefault("preview_history", [])
    key = f"preview_{record.ticker}_{len(previews) + 1}"
    previews.insert(
        0,
        {
            "key": key,
            "ticker": record.ticker,
            "verdict": record.final_verdict,
            "time": datetime.now().strftime("%H:%M"),
            "record": record,
        },
    )
    st.session_state["active_record"] = record
    st.session_state["active_filename"] = key


def main() -> None:
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    if "active_record" not in st.session_state:
        st.session_state["active_record"] = None
    if "active_filename" not in st.session_state:
        st.session_state["active_filename"] = None
    if "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = "NVDA"
    if "hot_radar_rows" not in st.session_state:
        st.session_state["hot_radar_rows"] = []
    if "preview_history" not in st.session_state:
        st.session_state["preview_history"] = []

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    settings = config.get_settings()

    with st.sidebar:
        st.markdown(
            _html(
                f"""
                <div class="sidebar-brand">
                    <div class="sidebar-brand-title">{APP_NAME}</div>
                    <div class="sidebar-brand-sub">Stock debate &amp; consensus desk</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        dark_mode = st.toggle("Dark Mode", key="dark_mode")

        # 1. Stock + Unified Compact Quick Pick & Hot 7D Radar
        st.markdown('<div class="sidebar-group-label">Stock</div>', unsafe_allow_html=True)
        ticker_input = st.text_input(
            "Ticker",
            key="ticker_input",
            placeholder="NVDA, AAPL, TSLA, PTT.BK",
            label_visibility="collapsed",
        ).strip().upper()

        with st.expander("Quick Pick & Hot 7D", expanded=False):
            tab_qp, tab_hot = st.tabs(["Quick Pick", "Hot 7D"])

            with tab_qp:
                sb_group = st.selectbox(
                    "Category",
                    options=[g for g, _ in STOCK_GROUPS],
                    index=0,
                    label_visibility="collapsed",
                    key="sb_quick_pick_group",
                )
                group_syms = next((syms for g, syms in STOCK_GROUPS if g == sb_group), ())
                # 2 columns per row so full ticker names (NVDA, AVGO, BTC-USD) never truncate
                for row_start in range(0, len(group_syms), 2):
                    row_syms = group_syms[row_start : row_start + 2]
                    scols = st.columns(2)
                    for col, sym in zip(scols, row_syms):
                        col.button(
                            sym,
                            key=f"sb_qp_{sb_group}_{sym}",
                            type="primary" if sym == ticker_input else "secondary",
                            width="stretch",
                            on_click=_select_ticker_callback,
                            args=(sym,),
                        )

            with tab_hot:
                group_names = [name for name, _ in STOCK_GROUPS]
                scan_group = st.selectbox(
                    "Scan Group",
                    options=["All Groups"] + group_names,
                    index=0,
                    label_visibility="collapsed",
                    key="sb_radar_group",
                )
                if st.button("Scan 7D Movers", type="primary", width="stretch", key="sb_radar_scan"):
                    target_tickers = [
                        sym
                        for g_name, syms in STOCK_GROUPS
                        if scan_group == "All Groups" or g_name == scan_group
                        for sym in syms
                    ]
                    with st.spinner("Scanning 7D movers..."):
                        try:
                            st.session_state["hot_radar_rows"] = scan_hot_7d_movers(target_tickers, top_n=8)
                        except Exception as exc:
                            st.error(f"Scan failed: {exc}")

                radar_rows: list[HotRadarRow] = st.session_state.get("hot_radar_rows", [])
                if not radar_rows:
                    st.caption("Tap Scan to rank biggest 7-day movers.")
                else:
                    for row in radar_rows:
                        move_txt = f"{row.price_change_7d_pct:+.1f}%"
                        st.button(
                            f"{row.ticker}  ·  {move_txt}",
                            key=f"sb_hot_{row.ticker}",
                            type="primary" if row.ticker == ticker_input else "secondary",
                            width="stretch",
                            on_click=_select_ticker_callback,
                            args=(row.ticker,),
                        )

        debate_rounds = st.slider(
            "Debate Rounds",
            min_value=config.MIN_DEBATE_ROUNDS,
            max_value=config.MAX_DEBATE_ROUNDS,
            value=settings.debate_rounds,
            step=1,
        )

        # 2. Multi-Provider AI Tier Slider Slot
        st.markdown('<div class="sidebar-group-label">AI Model</div>', unsafe_allow_html=True)
        provider_filter_options = [
            "All Providers",
            "Google Gemini",
            "OpenAI",
            "Anthropic Claude",
            "Local Ollama",
        ]
        provider_filter_map: dict[str, ProviderName | None] = {
            "All Providers": None,
            "Google Gemini": "gemini",
            "OpenAI": "openai",
            "Anthropic Claude": "anthropic",
            "Local Ollama": "ollama",
        }
        selected_scope = st.selectbox(
            "Provider",
            options=provider_filter_options,
            index=0,
            label_visibility="collapsed",
        )
        scoped_provider = provider_filter_map[selected_scope]
        available_specs = [
            spec for spec in MODEL_TIERS if scoped_provider is None or spec.provider == scoped_provider
        ]
        slot_labels = [spec.slot_label for spec in available_specs]

        default_slot = find_default_slot_label("gemini", settings.default_model)
        if default_slot not in slot_labels:
            default_slot = slot_labels[0]

        selected_slot_label = st.select_slider(
            "Model Tier",
            options=slot_labels,
            value=default_slot,
        )
        selected_spec = MODEL_BY_SLOT[selected_slot_label]
        selected_provider: ProviderName = selected_spec.provider

        use_custom_model = st.checkbox("Custom Model ID", value=False)
        if use_custom_model:
            effective_model_id = st.text_input(
                "Model ID",
                value=selected_spec.model_id,
                placeholder="e.g., gemini-3.8-flash, qwen3.5:4b",
                label_visibility="collapsed",
            ).strip() or selected_spec.model_id
        else:
            effective_model_id = selected_spec.model_id

        # Dynamic provider credentials / endpoint
        session_key_override = ""
        effective_base_url: str | None = None

        if selected_provider == "ollama":
            effective_base_url = st.text_input(
                "Ollama URL",
                value=config.OLLAMA_BASE_URL,
                placeholder="http://localhost:11434",
            ).strip() or config.OLLAMA_BASE_URL
        elif selected_provider == "gemini":
            env_ready = bool(
                settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here"
            )
            if not env_ready:
                session_key_override = st.text_input(
                    "Gemini API Key",
                    type="password",
                    placeholder="Paste GEMINI_API_KEY...",
                ).strip()
        elif selected_provider == "openai":
            env_ready = bool(
                config.OPENAI_API_KEY and not config.OPENAI_API_KEY.startswith("your_")
            )
            if not env_ready:
                session_key_override = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="Paste OPENAI_API_KEY...",
                ).strip()
            effective_base_url = config.OPENAI_BASE_URL
        elif selected_provider == "anthropic":
            env_ready = bool(
                config.ANTHROPIC_API_KEY and not config.ANTHROPIC_API_KEY.startswith("your_")
            )
            if not env_ready:
                session_key_override = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    placeholder="Paste ANTHROPIC_API_KEY...",
                ).strip()

        auth_ready, auth_label = _provider_auth_ready(selected_provider, settings, session_key_override)
        _render_model_spec_card(
            spec=selected_spec,
            effective_model_id=effective_model_id,
            auth_ready=auth_ready,
            auth_label=auth_label,
            is_custom=use_custom_model and effective_model_id != selected_spec.model_id,
        )

        # 3. Experiment Lab & Selectable Meta-Judge Slots
        st.markdown('<div class="sidebar-group-label">Lab &amp; Judges</div>', unsafe_allow_html=True)
        enable_lab = st.toggle(
            "Run Lab & 3 Meta-Judges",
            value=True,
        )
        with st.expander("Meta-Judge Models", expanded=False):
            j1_label = st.selectbox(
                "Judge 1",
                options=JUDGE_OPTION_LABELS,
                index=JUDGE_OPTION_LABELS.index(_judge_label_for_model(settings.meta_judge_models[0])),
                key="judge_1_slot",
            )
            j2_label = st.selectbox(
                "Judge 2",
                options=JUDGE_OPTION_LABELS,
                index=JUDGE_OPTION_LABELS.index(_judge_label_for_model(settings.meta_judge_models[1])),
                key="judge_2_slot",
            )
            j3_label = st.selectbox(
                "Judge 3",
                options=JUDGE_OPTION_LABELS,
                index=JUDGE_OPTION_LABELS.index(_judge_label_for_model(settings.meta_judge_models[2])),
                key="judge_3_slot",
            )
        judge_specs = (
            JUDGE_OPTION_BY_LABEL[j1_label],
            JUDGE_OPTION_BY_LABEL[j2_label],
            JUDGE_OPTION_BY_LABEL[j3_label],
        )
        judge_models = (judge_specs[0][1], judge_specs[1][1], judge_specs[2][1])

        # 4. Actions
        run_clicked = st.button("Run Simulation", type="primary", width="stretch")
        preview_clicked = st.button("Preview Layout", type="secondary", width="stretch")

        # 5. Forward Check
        st.markdown('<div class="sidebar-group-label">Forward Check</div>', unsafe_allow_html=True)
        eval_clicked = st.button("Check Returns (1D / 5D / 10D)", width="stretch")
        if eval_clicked:
            with st.spinner("Checking realized returns vs SPY..."):
                try:
                    updated, _ = evaluate_all_history_sessions(
                        history_dir=config.HISTORY_DIR,
                        scoreboard_path=config.JUDGE_SCOREBOARD_PATH,
                        default_models=judge_models,
                    )
                    active_fn = st.session_state.get("active_filename")
                    if active_fn and not str(active_fn).startswith("preview_"):
                        st.session_state["active_record"] = load_session(
                            active_fn,
                            history_dir=config.HISTORY_DIR,
                        )
                    st.success(f"Updated {updated} session(s).")
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

        # 6. History (All Live Saved Runs + Session Previews)
        sessions = list_sessions(config.HISTORY_DIR)
        previews: list[dict] = st.session_state.get("preview_history", [])
        total_hist = len(sessions) + len(previews)
        st.markdown(f'<div class="sidebar-group-label">History ({total_hist})</div>', unsafe_allow_html=True)

        if total_hist == 0:
            st.caption("No runs yet.")
        else:
            for item in sessions:
                tag = f" / {item.meta_verdict}" if item.meta_verdict else ""
                label = f"{item.ticker} · {_format_display_time(item.created_at_utc)} · {item.final_verdict}{tag}"
                is_active = st.session_state.get("active_filename") == item.filename
                if st.button(
                    label,
                    key=f"hist_{item.filename}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                ):
                    try:
                        loaded = load_session(item.filename, history_dir=config.HISTORY_DIR)
                        st.session_state["active_record"] = loaded
                        st.session_state["active_filename"] = item.filename
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load session: {exc}")

            for pitem in previews:
                label = f"[Preview] {pitem['ticker']} · {pitem['time']} · {pitem['verdict']}"
                is_active = st.session_state.get("active_filename") == pitem["key"]
                if st.button(
                    label,
                    key=f"hist_{pitem['key']}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                ):
                    st.session_state["active_record"] = pitem["record"]
                    st.session_state["active_filename"] = pitem["key"]
                    st.rerun()

    # Inject Dark Mode CSS in main container AFTER CUSTOM_CSS so it always wins cascade order
    if dark_mode:
        st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)

    # Layout preview mode (recorded in session History without polluting data/history/)
    if preview_clicked:
        preview_rec = build_layout_preview_record(
            ticker=ticker_input,
            model_name=f"{selected_provider}:{effective_model_id}",
            debate_rounds=debate_rounds,
            judge_models=judge_models,
        )
        _record_preview_history(preview_rec)
        st.rerun()

    # Live simulation execution with Game-Style Dark Loading Overlay
    if run_clicked:
        effective_key, resolved_url = _resolve_provider_credentials(
            selected_provider,
            settings,
            session_key_override,
            selected_provider,
            effective_base_url,
        )
        loader_slot = st.empty()
        _render_game_loader(
            loader_slot,
            ticker=ticker_input,
            model_name=effective_model_id,
            stage="Starting Council",
            detail="Gathering latest news & 7-day price data...",
        )

        def _on_progress(stage: str, detail: str) -> None:
            _render_game_loader(
                loader_slot,
                ticker=ticker_input,
                model_name=effective_model_id,
                stage=stage,
                detail=detail,
            )

        try:
            record = asyncio.run(
                _execute_live_simulation(
                    ticker=ticker_input,
                    provider=selected_provider,
                    model_name=effective_model_id,
                    debate_rounds=debate_rounds,
                    api_key=effective_key,
                    base_url=resolved_url,
                    enable_lab=enable_lab,
                    judge_specs=judge_specs,
                    settings=settings,
                    session_key_override=session_key_override,
                    ollama_url_override=effective_base_url,
                    meta_judge_timeout_seconds=settings.meta_judge_timeout_seconds,
                    meta_judge_max_retries=settings.meta_judge_max_retries,
                    on_progress=_on_progress,
                )
            )
        except MissingAPIKeyError as exc:
            loader_slot.empty()
            st.error(str(exc))
        except TickerDataError as exc:
            loader_slot.empty()
            st.error(str(exc))
        except LLMGenerationError as exc:
            loader_slot.empty()
            st.error(str(exc))
        except Exception as exc:
            loader_slot.empty()
            st.error(f"Simulation failed: {exc}")
        else:
            loader_slot.empty()
            st.session_state["active_record"] = record
            refreshed = list_sessions(config.HISTORY_DIR)
            matched = next((s.filename for s in refreshed if s.session_id == record.session_id), None)
            st.session_state["active_filename"] = matched
            st.rerun()

    # Main display area (clean, zero radar clutter)
    active_record: SessionRecord | None = st.session_state.get("active_record")
    if active_record is None:
        st.markdown(
            _html(
                f"""
                <div class="hero-masthead">
                    <div class="hero-app-title">{APP_NAME}</div>
                    <div class="hero-subtitle">
                        Five investor personas read the same news, vote blind, debate live, and let three judges score the final call.
                    </div>
                    <div class="hero-chip-row">
                        <span class="hero-chip">Model: {html.escape(effective_model_id)}</span>
                        <span class="hero-chip">News + 7D Price Only</span>
                        <span class="hero-chip">No Technical Indicators</span>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        steps = [
            ("01", "Market Brief", "Pulls recent headlines and 7-day price change into one clean summary."),
            ("02", "Council Debate", "Five personas vote blind, debate across rounds, and cast final ballots."),
            ("03", "Experiment Lab", "Compares single, isolated, council, and full-herd setups side by side."),
            ("04", "Meta-Judges", "Three independent judges review the session and track 5-day accuracy."),
        ]
        for col, (num, title, desc) in zip((c1, c2, c3, c4), steps):
            with col:
                st.markdown(
                    _html(
                        f"""
                        <div class="step-card">
                            <div class="step-num">{num}</div>
                            <div class="step-title">{title}</div>
                            <div class="step-desc">{desc}</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("Judge Scoreboard")
        scoreboard = load_judge_scoreboard(config.JUDGE_SCOREBOARD_PATH, default_models=judge_models)
        _render_scoreboard_table(scoreboard.judges)

        if st.button("Open Preview", type="primary"):
            preview_rec = build_layout_preview_record(
                ticker=ticker_input,
                model_name=f"{selected_provider}:{effective_model_id}",
                debate_rounds=debate_rounds,
                judge_models=judge_models,
            )
            _record_preview_history(preview_rec)
            st.rerun()
    else:
        _render_session_god_view(active_record)
        st.divider()
        st.subheader("Judge Scoreboard")
        scoreboard = load_judge_scoreboard(config.JUDGE_SCOREBOARD_PATH, default_models=judge_models)
        _render_scoreboard_table(scoreboard.judges)


if __name__ == "__main__":
    main()

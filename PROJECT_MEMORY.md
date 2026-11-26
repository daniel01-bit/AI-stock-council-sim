# PROJECT_MEMORY.md — Stock Council Simulator

**Project Name:** `Stock Council Simulator (P5)`  
**Local Path / Repo:** `c:/Users/ASUS/Desktop/MODEL-2`  
**Current Version / Phase:** `v2.1 (Standalone Behavioral Finance Lab — Live-Verified on AAPL)`  
**Last Updated:** `2026-11-26`

---

## 1. Core Objective
- Build a production-ready Multi-Agent Behavioral Finance Simulator that runs a strict 4-stage Information-Asymmetry pipeline (**Research -> Blind Vote -> Anonymized Debate -> Simultaneous Final Vote**) using **100% news + 7-day price change % only**, with full JSON session replay and a Streamlit God View dashboard.

---

## 2. Non-Goals (Phase 1 / v1.0)
- ❌ No technical indicators (RSI, MACD, SMA/EMA, Bollinger Bands, OHLCV chart pattern analysis).
- ❌ No live brokerage execution or automated order routing.
- ❌ No historical LLM backtesting yet (deferred until live `GEMINI_API_KEY` runs confirm prompt quality, latency, and token cost per simulation; note also that Yahoo Finance `ticker.news` only serves recent headlines, so historical news backtesting requires a separate point-in-time news dataset to avoid look-ahead bias).

---

## 3. Tech Stack & Hard Constraints
- **Core Stack (Verified Local Versions, Python 3.15.2):** `google-genai==3.6.1`, `pydantic==2.14.4`, `yfinance==1.8.0`, `streamlit==1.67.0`, `python-dotenv==1.2.4`, `pytest==9.2.0`, `pytest-asyncio==1.5.0`.
- **SDK Guardrail:** Strictly uses `google-genai` (`from google import genai`) with `client.aio.models.generate_content` and Pydantic `response_schema`. Never `google-generativeai`.
- **Information Asymmetry Firewall:** `build_debate_prompt()` structurally excludes Stage 1 `AgentVote` objects — only `FactBriefing`, anonymized `VoteTally`, and `DebateTurn` history can enter Stage 2.
- **Deterministic Guardrails:** `FactBriefing` binds exact `yfinance` 7-day price numbers server-side so the Researcher LLM cannot drift or hallucinate price changes.

---

## 4. Phase Roadmap & Exit Criteria

### Phase 1: v1.0 — Core Foundation, Pipeline, UI & Verification `[Done & Live-Verified]`
- **Deliverables:**
  - `[x]` Strict Pydantic schemas (`src/schemas.py`) and config loader (`config.py`, `.env.example`)
  - `[x]` News + 7D price change fetcher (`src/data_fetcher.py`) verified against live `yfinance==1.8.0`
  - `[x]` Async `google-genai` structured output client with timeout + retry (`src/llm_client.py`)
  - `[x]` 4-stage orchestrator with Anonymization Firewall & `asyncio.gather()` concurrency (`src/orchestrator.py`)
  - `[x]` Atomic JSON storage & Replay index (`src/storage.py`)
  - `[x]` Streamlit God View + History Replay UI + Mock Preview (`app.py`, `run_dashboard.bat`)
  - `[x]` Unit tests (`tests/test_orchestrator.py`) verifying firewall, concurrency, pipeline, fetcher, and storage
- **Gatekeeping / Exit Criteria:**
  - `[x]` All unit tests pass (`python -m pytest -v` -> 8/8)
  - `[x]` Live `yfinance` smoke test succeeds and handles invalid tickers cleanly
  - `[x]` Streamlit boots cleanly via headless `AppTest`
  - `[x]` 1 live end-to-end simulation with real `GEMINI_API_KEY` (`NVDA`, `gemini-3.5-flash-lite`, 45.8s, `HOLD (Split Decision)`, 2 opinion shifts) saved to `data/history/20261126_061106_NVDA_df20c7d4.json`

### Phase 2: v2.0 + v2.1 — Standalone Behavioral Finance Lab `[Done & Live-Verified]`
- **Deliverables:**
  - `[x]` Self-Amnesia fix (own Stage 1 numeric stance only; zero cross-agent leak) + Probabilistic Voting (`prob_up_5d`, `position_pct`) + Behavioral Metrics (`herding_rate`, `confidence_polarization`, `entropy`)
  - `[x]` Forward-Run Paper Evaluator (`src/evaluator.py`, `T+1D/5D/10D` vs `SPY`, Brier Score, idempotent `data/judge_scoreboard.json`)
  - `[x]` Parallel Counterfactual Lab (4 branches: Baseline, Isolated Crowd, Fog-of-War, Full-Herd)
  - `[x]` Option B: 3 Blank Meta-Judges on 3 distinct models (`gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`) with Shadow Scoring, Anti-Self-Vote Peer Selection, Brier tie-breaker, and 503 failover
- **Gatekeeping / Exit Criteria:**
  - `[x]` 9/9 pytest passing (0 warnings) + v1.0 JSON backward compatibility verified
  - `[x]` Live end-to-end v2.1 Lab run on `AAPL` saved to `data/history/` and `data/judge_scoreboard.json` initialized

### Phase 3: v2.2 — Dynamic Trait Population (10–15 Agents) & Parallel Bulletin Board `[Next]`
- **Deliverables:**
  - `[ ]` Continuous Behavioral Trait Vector generator (flexible retail/inst/bear/contrarian mix)
  - `[ ]` Parallel Bulletin Board + Camp Argument Clustering to scale to 12–15 agents without latency explosion

---

## 5. Current State & Next Immediate Action
- **Completed:** `v2.1` Standalone Behavioral Finance Lab live-verified on `AAPL` (`BUY`, `judge_1`/`gemini-3.6-flash` peer-selected, `data/judge_scoreboard.json` created), `9/9` pytest passing, Streamlit running on `http://localhost:8501`.
- **Active Blocker:** None. (Forward `T+1D/5D/10D` scores will mature automatically as live market days pass — click `Evaluate Forward Returns` in the sidebar to update Win Rates & Brier Scores.)
- **Next Immediate Action:**
  1. Run 5–10 diverse tickers in v2.1 Lab mode (`http://localhost:8501`) to accumulate forward-run evaluation samples across all 3 Meta-Judge models.
  2. Proceed to `v2.2` (Dynamic Trait Population 10–15 Agents + Parallel Bulletin Board) when ready.

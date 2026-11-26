# Changelog & Engineering Decision Record

All notable additions, structural choices, and environment adaptations for **Stock Council Simulator**.

---

## [v2.1.1] — 2026-11-26 (Hardening & Maintainability Pass)

### Fixed & Improved
- **Non-Empty Timeout Diagnostics (`src/llm_client.py`):** Added `_format_exc()` so `asyncio.TimeoutError()` always logs `Timed out after <N>s` instead of a blank string after `:`.
- **Meta-Judge Resilience (`app.py`, `config.py`):** Moved Meta-Judge timeout (`META_JUDGE_TIMEOUT_SECONDS=40`) and retry (`META_JUDGE_MAX_RETRIES=1`) to safe config parsers (`_parse_float` / `_parse_int`), preventing single transient `503` spikes from forcing immediate fallback.
- **Evaluator N+1 Cache & Skip (`src/evaluator.py`):** `evaluate_all_history_sessions()` now skips sessions already `EVALUATED` across all horizons (`1D/5D/10D`) and caches `(symbol, entry_date)` close fetches within each run.
- **UI Maintainability (`app.py`, `src/preview.py`):** Extracted the 206-line mock preview builder to `src/preview.py` and deduplicated Stage 2 debate round rendering into `_render_debate_round()`.
- **Off-Topic News Filtering (`src/prompts.py`) & Git Hygiene (`.gitignore`, `README.md`):** Tightened `RESEARCHER_SYSTEM_PROMPT` to drop unrelated-ticker Yahoo Finance articles, added `data/judge_scoreboard.json` to `.gitignore`, and expanded `pytest` coverage to **10/10 passing tests**.

---

## [v2.1.0] — 2026-11-26 (Standalone Behavioral Finance Research Lab)

### Added & Upgraded
- **429 Free-Tier Rate-Limit & 503 High-Demand Resilience (`src/llm_client.py`, `app.py`):**
  - Added `_extract_retry_delay_seconds()` and `_is_rate_limit_error()` to honor Google's server-requested `retryDelay` on `429 RESOURCE_EXHAUSTED` (Free Tier 15 RPM) while failing fast on hard `limit: 0` models.
  - Added automatic Gemini Flash failover chain (`DEFAULT_GEMINI_FALLBACK_MODELS`) on `GeminiLLMClient` for transient `503`/`429` spikes during Council runs, while passing `fallback_models=()` for Option B Meta-Judges so cross-model benchmarking stays strictly isolated per model.
  - Verified live end-to-end `Run Simulation` via headless `AppTest` on `AMD` (`v2.1 Lab ON`): completed cleanly in ~42s and saved to `data/history/`.

- **Quick Pick & Hot 7D Behavioral Radar (`src/stock_radar.py`, `app.py`, `config.py`):**
  - Fixed `Settings` in `config.py` (`default_provider`, `openai_api_key`, `anthropic_api_key`, `openai_base_url`, `ollama_base_url`) so the Editorial Multi-Provider UI boots and switches tickers/providers with zero errors.
  - Adapted `quant01`'s 4 curated stock groups (`AI & Semiconductors`, `Mega-Cap Tech`, `Value, Finance & Crypto`, `Indices & Safe-Haven`) and `Quick Pick & Hot Radar` into `MODEL-2`'s Editorial style, ranking strictly by `|7D Price Change %|` (zero technical indicators) with 1-click ticker loading into `st.session_state["ticker_input"]`.
- **v2.0 Research Foundation (`src/schemas.py`, `src/prompts.py`, `src/orchestrator.py`):**
  - Fixed **Self-Amnesia** without breaking the Fog-of-War firewall: each agent receives ONLY its own Stage 1 numeric stance (`vote`, `confidence`, `prob_up_5d`, `position_pct`) in Stage 2 & Stage 3 — zero Stage 1 reasoning strings or other agent identities ever leak.
  - Added **Probabilistic Voting** (`prob_up_5d` 0–100%, `position_pct` -100..+100) and deterministic **Behavioral Metrics** (`opinion_shift_rate`, `herding_rate`, `confidence_polarization`, `stage1_entropy`, `stage3_entropy`).
  - Built **Forward-Run Paper Evaluator (`src/evaluator.py`)**: evaluates realized `T+1D / T+5D / T+10D` returns vs `SPY` strictly after `price_end_date` (zero look-ahead bias) and computes directional Hit Rate + 5D Brier Score for Council, all Branches, and all 3 Meta-Judges.
- **v2.1 Parallel Counterfactual Lab & Option B 3 Blank Meta-Judges (`src/orchestrator.py`, `src/evaluator.py`, `app.py`):**
  - Runs 4 parallel counterfactual branches on the exact same `FactBriefing`: `Branch A (Single-Agent Baseline)`, `Branch B (Isolated Crowd — reuses Stage 1 with zero extra calls)`, `Branch C (Fog-of-War Council)`, and `Branch D (Full-Herd Social Pressure)`.
  - Added **3 Character-Free Meta-Judges on 3 Distinct Models (Option B)** (`META_JUDGE_MODELS=gemini-3.6-flash,gemini-3.5-flash,gemini-3.1-flash-lite`), configurable in `.env` and UI:
    1. **Parallel Shadow Scoring** (all 3 models scored equally in Forward Eval to eliminate selection bias; per-model `25s` timeout failover prevents 1 stalled/503 model from crashing the run).
    2. **Peer Selection (Anti-Self-Vote)** with deterministic historical Brier Score / Win Rate tie-breaker.
    3. **Final Judge Synthesis** by the peer-selected winner + idempotent persistent **Judge Scoreboard (`data/judge_scoreboard.json`)**.
- **Dashboard & Tests (`app.py`, `tests/test_orchestrator.py`, `config.py`):**
  - Added v2.1 Lab toggle, 3 Meta-Judge model inputs, 1-click `Evaluate Forward Returns` button, Behavioral Telemetry strip, 4-Branch Comparison Table, Meta-Judge Shadow/Peer/Scoreboard panels, and updated Mock Preview.
  - Preserved multi-provider env keys in `config.py` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`, `OLLAMA_BASE_URL`) alongside `META_JUDGE_MODELS` & `JUDGE_SCOREBOARD_PATH`.
  - Expanded test suite to **9/9 passing tests** (0 warnings) + verified backward compatibility with `v1.0` saved JSON sessions + live end-to-end `AAPL` v2.1 Lab run (`BUY`, `judge_1`/`gemini-3.6-flash` peer-selected, graceful failover when `judge_2` timed out).

---

## [v1.0.0] — 2026-11-26

### Added (Files Created)
- `config.py` & `.env.example` — Centralized env config (`GEMINI_API_KEY`, `DEFAULT_MODEL=gemini-2.5-flash`, `DEBATE_ROUNDS=2`, plus optional timeout/retry/news-cap overrides).
- `.gitignore` — Excludes `.env`, Python caches, `.venv`, and runtime `data/history/*.json` logs.
- `requirements.txt` — Pinned to exact verified versions installed in the local Python 3.15.2 environment (`google-genai==3.6.1`, `pydantic==2.14.4`, `yfinance==1.8.0`, `streamlit==1.67.0`, `python-dotenv==1.2.4`, `pytest==9.2.0`, `pytest-asyncio==1.5.0`).
- `src/schemas.py` — Strict Pydantic v2 contracts for `RawNewsItem`, `MarketContext`, `ResearcherBriefingOutput`, `FactBriefing`, `AgentVoteOutput`, `AgentVote`, `VoteTally`, `DebateMessageOutput`, `DebateTurn`, `OpinionShift`, `SessionRecord`, and `SessionSummary`.
- `src/prompts.py` — System prompts for the non-voting Researcher + 5 distinct Investor Personas (`The Retail FOMO`, `The Fundamentalist`, `The Paranoid Bear`, `The Contrarian`, `The Macro Analyst`) and structurally isolated stage prompt builders.
- `src/data_fetcher.py` — `yfinance`-backed fetcher strictly returning recent deduplicated news + 7-calendar-day price change % (zero technical indicators), handling both `yfinance >= 0.2.50 / 1.x` nested `content` payloads and legacy flat payloads, invalid tickers, and empty news gracefully.
- `src/llm_client.py` — Async `GeminiLLMClient` over official `google-genai` (`client.aio.models.generate_content`) with Pydantic `response_schema`, `asyncio.wait_for` timeout, transient retry backoff, and `StructuredLLMClient` protocol for deterministic testability.
- `src/orchestrator.py` — 4-stage `CouncilOrchestrator` enforcing concurrent `asyncio.gather()` execution for Stages 1 & 3, sequential debate in Stage 2, strict Anonymization Firewall (`VoteTally` only crosses Stage 1 -> Stage 2), deterministic verdict + tie handling (`HOLD (Split Decision)`), and opinion-shift calculation.
- `src/storage.py` — Atomic UTF-8 JSON persistence into `data/history/`, newest-first session indexing resilient to corrupt JSON files, and path-traversal protection on reload.
- `app.py` — Streamlit God View & Session History Replay dashboard with live stage-by-stage progress updates, sidebar ticker/model/round controls, optional session-only API key input, Fog-of-War firewall banner, persona chat transcript grouped by round, and Stage 1 vs. Stage 3 opinion-shift matrix.
- `tests/test_orchestrator.py` & `pytest.ini` — Unit tests verifying Stage 2 anonymization firewall (sentinel leak check), true 5-way concurrency in Stages 1 & 3 vs. sequential Stage 2, full pipeline + opinion shifts + storage roundtrip, split-decision ties, `yfinance` parsing, API key guardrails, and headless Streamlit God View replay rendering (`AppTest`).
- `README.md` & `PROJECT_MEMORY.md` — Architecture Mermaid diagram, setup/model-swap guide, and Phase-Gated local project memory.
- `data/history/.gitkeep` — Tracks history folder cleanly while ignoring generated session JSON files.

- `run_dashboard.bat` — One-click Windows launcher for `python -m streamlit run app.py`.
- `app.py` (UI Layout Preview) — Added **Preview Dashboard Layout (No API Key)** mode so every section of the God View (Verdict Banner, Stage 0 Fact Briefing, Stage 1 Fog-of-War + 5-Agent Cards, Stage 2 Multi-Round Chat, Stage 3 Final Votes + Opinion Shift Table) can be inspected immediately without a `GEMINI_API_KEY` or writing to `data/history/`.

### Live API Verification & Model Calibration (2026-11-26)
- Configured local `.env` (gitignored) with user's `GEMINI_API_KEY`.
- Probed live Gemini API models on this key:
  - `gemini-2.5-flash` returned `404 NOT_FOUND` (retired for new keys).
  - `gemini-3.8-flash` / `3.7-flash` / `3.6-flash` / `3.5-flash` experienced transient `503` high-demand spikes / slow timeouts during probing.
  - `gemini-3.5-flash-lite` responded cleanly in `3.82s` with valid Pydantic structured output.
- Set `DEFAULT_MODEL=gemini-3.5-flash-lite` in `.env` (still freely overrideable in `.env` or UI) and updated model hint text in `app.py`.
- Executed first live end-to-end 4-stage simulation on `NVDA` (`debate_rounds=2`, 21 Gemini calls, `45.8s`, `0` schema errors): Stage 1 `BUY:1 | HOLD:2 | SELL:2` -> Stage 3 `BUY:2 | HOLD:1 | SELL:2` -> Final Verdict `HOLD (Split Decision)` with 2 genuine opinion shifts (`The Retail FOMO`: `SELL->BUY`, `The Macro Analyst`: `HOLD->SELL`), saved to `data/history/20261126_061106_NVDA_df20c7d4.json`.

### Compatibility Fixes Applied During Verification
- Replaced deprecated Streamlit `use_container_width=True` with `width="stretch"` across `app.py` to match `streamlit==1.67.0` with zero deprecation warnings.
- Added `pytest.ini` (`asyncio_default_fixture_loop_scope = function`) to eliminate `pytest-asyncio 1.5.0` loop-scope deprecation warnings.

### Key Architectural & Engineering Decisions
1. **Flat Workspace Root (`c:\Users\ASUS\Desktop\MODEL-2\`) instead of nested `MODEL-2/stock-council-simulator/`:**
   - Avoids double-nesting, matches existing projects in `home-repo`, and enables running `python -m streamlit run app.py` and `python -m pytest -v` directly from the workspace root.
2. **Structural Firewall via Function Signatures (`build_debate_prompt`):**
   - Instead of relying solely on prompt wording, `build_debate_prompt()` literally does not accept `stage1_votes: list[AgentVote]` as a parameter — only `FactBriefing`, anonymized `VoteTally`, and `DebateTurn` history.
3. **Deterministic Server-Side Price Binding (`build_authoritative_briefing`):**
   - `FactBriefing` overwrites `ticker`, `price_start_7d`, `price_end`, `price_change_7d_pct`, and `news_items_reviewed` from `MarketContext` after Stage 0 LLM curation so the Researcher Agent cannot distort or hallucinate price numbers.
4. **Protocol-Based LLM Boundary (`StructuredLLMClient`):**
   - Avoids brittle monkey-patching of `google-genai` SDK internals during testing while verifying real concurrency (`max_concurrent == 5` for Stages 1 & 3, `1` for Stage 2) and exact prompt payloads.
5. **Local Environment Dependency Alignment:**
   - Kept the already-installed `yfinance==1.8.0` and installed compatible cp315 wheels (`google-genai==3.6.1`, `pydantic==2.14.4`, `streamlit==1.67.0`, `python-dotenv==1.2.4`, `pytest==9.2.0`, `pytest-asyncio==1.5.0`) into the active Python 3.15.2 interpreter without disturbing existing scientific/quant packages.
6. **Deferred Live API Run & Backtesting:**
   - `GEMINI_API_KEY` is not yet set on this workstation. All code, tests, live `yfinance` fetching, and Streamlit UI boot were verified without an API key; live Gemini execution and historical backtesting are cleanly deferred.

# Web UI Development Goal

**Status:** Active goal  
**Audience:** Contributor  
**Related:** [WEBUI.md](WEBUI.md) (User tour) · [WEB_API.md](WEB_API.md) · [SETUP.md](SETUP.md) · [STYLE.md](STYLE.md) · `webui/AGENTS.md`

---

## North star

Make the bundled chart workspace at `/app` a **first-class research console** for mtdata: modern, fast, usable on real screens (desktop-first, tablet-capable, mobile-usable), and **feature-complete against the Web API**—with a clear path to expose high-value backend capabilities that still exist only on CLI/MCP.

The UI should not be a thin demo. It should be the default interactive surface for market context, levels, denoising, forecasting, volatility, and backtesting—without requiring users to leave the browser for routine research work.

---

## Why this goal exists

| Reality today | Target |
|---|---|
| SPA is a single full-viewport chart workspace (~23 TS/TSX files, ~130 KB source) | Structured product surface with clear modules, status, and growth path |
| Strong core: OHLC chart, live ticks, pivots/S-R, denoise, forecast/vol/backtest panel | Same core, polished + complete API coverage + responsive UX |
| Web API is **smaller** than full CLI/MCP by design | UI tracks **100% of Web API**; Web API grows deliberately for UI-critical research features |
| Dev quality is light: pure unit tests on libs/API errors only; no lint; AGENTS.md partially stale | Reliable test pyramid for client logic + critical UI paths; docs and scripts stay accurate |
| Layout is desktop-horizontal (fixed 420px side panel, dense toolbar, almost no breakpoints) | Responsive, keyboard-friendly, accessible dark workspace that still feels trading-terminal modern |

---

## Current capability map

### What the UI already does well

- **Chart workspace:** symbol search, timeframes, infinite left history, timezone modes (UTC / local / server)
- **Live market:** tick poll (bid / ask / last), live incomplete candle, reload and empty/error status surface
- **Overlays:** pivot levels, support/resistance, sample-trade indicators (EMA / RSI / MACD / volume panes), forecast series + anchor comparison metrics (MAE, MAPE, RMSE, direction)
- **Preprocessing:** chart-level denoise with method metadata UI
- **Analysis panel:** price forecast, volatility forecast, rolling backtest (with advanced options: dimred, denoise, params)
- **Auth:** in-memory Bearer token for remote/tokenized API access
- **Stack:** React 18, Vite, Tailwind, TanStack Query, lightweight-charts, axios → `/api/v1`

### Web API endpoints vs UI usage

Living matrix: **[WEBUI_API_COVERAGE.md](WEBUI_API_COVERAGE.md)** (route × used / intentional-omit × UI entry).

| Endpoint | UI usage |
|---|---|
| `GET /health`, `GET /ready` | Used — non-blocking `ConnectionStatus` chip |
| `GET /instruments`, `/timeframes` | Used |
| `GET /history`, `/tick` | Used (core chart + live) |
| `GET /pivots`, `/support-resistance` | Used — Levels controls (pivot method, S/R lookback, touches, max levels, tolerance) |
| `GET /denoise/methods`, `/denoise/wavelets` | Used via Denoise modal |
| `GET /methods`, `/volatility/methods`, `/dimred/methods`, `/sktime/estimators` | Used in forecast panel |
| `GET /models` | Used — `ModelsBrowser` in forecast advanced options |
| `POST /forecast/price`, `/forecast/volatility`, `/backtest` | Used |

### Explicit non-parity (historical note, then current boundary)

**Then (dedicated chart routes only):** trading, regimes, patterns, reports, Finviz, options, news, causal tools, and wait-events lived only on CLI/MCP.

**Now:** `GET/POST /api/v1/tools*` plus the SPA **Tools** runner expose almost the full catalog. Dedicated chart widgets remain the comfortable path for candles, levels, denoise, forecast, volatility, and backtest. Two deliberate omissions:

- `forecast_tune_optuna` / `forecast_tune_genetic` — no HTTP progress or cancel contract (run via CLI/MCP).
- `wait_event` — blocking waits have no HTTP progress or cancel contract; see [WAIT_EVENT.md](WAIT_EVENT.md).

Live `trade_*` mutations from the SPA still require an explicit confirm gate, dry-run defaults, and the [trading safety](TRADING_SAFETY.md) rules. That is the safety design; it is not “trading is out of the Web API.”

The goal is not “rebuild every tool as a custom widget on day one.” It is:

1. **Parity with every Web API route and its useful parameters**
2. **Deliberate Web API + UI expansion** for the highest-value research features
3. **Never** expose unsafe live trading from the SPA without confirmations, dry-run defaults, and env guardrails — aligned with [TRADING_SAFETY.md](TRADING_SAFETY.md)

---

## Goal pillars

### 1. Feature parity with the Web API (and intentional expansion)

**Definition of done for parity**

- Every documented Web API route is either (a) reachable from the UI with clear affordances, or (b) listed as intentional omission with rationale.
- Useful query/body parameters are exposed where users need them (not only hardcoded defaults): S/R lookback & method, pivot method, history date range, forecast libraries/estimators/models, backtest extras.
- `GET /models` is discoverable (browse cached/trained models when available).
- Connection health: show API liveness and MT5 readiness (`/ready`) without blocking the whole layout.

**Expansion track (after parity)**  
Prioritize Web API + UI modules that unlock research workflows already mature on the backend:

| Priority | Domain | User value |
|---|---|---|
| P1 | Technical indicators on chart | Everyday charting parity with research notebooks |
| P1 | Richer levels (Fib zones, breakout status from S/R rich mode) | Actionable structure on the chart |
| P2 | Regime overlays / summary | Context for forecast trust |
| P2 | Pattern markers (classic / high-signal only first) | Visual event anchors |
| P2 | Report export / snapshot | Shareable research artifact |
| P3 | Screener / Finviz-style discovery | Symbol discovery beyond search box |
| — | Live order placement | **Out of scope** unless a dedicated safety-reviewed design exists |

Expansion always lands as: **backend route contract → client types → UI surface → tests → docs**.

### 2. Responsive, usable, modern UX

**Principles**

- **Chart-first:** the chart remains the hero; chrome collapses under density pressure.
- **Progressive disclosure:** simple defaults; advanced params behind clear “Advanced” sections (already started in Forecast).
- **Responsive, not just “shrinks”:**
  - Desktop (≥1280): toolbar + side panel + metrics as today
  - Tablet: collapsible tool groups, full-height sheet for forecast
  - Mobile: bottom sheet / full-screen panel, overflow menus for toolbar actions, touch-friendly hit targets (≥44px)
- **Modern look:** keep the dark slate terminal aesthetic; tighten spacing, hierarchy, motion, and empty/loading/error states into one consistent design language (reuse `panel` / `btn` / `input` primitives; avoid one-off styles).
- **Usability:**
  - Keyboard: focus order, Esc closes panels/modals, `/` or `s` focuses symbol search where safe
  - Feedback: toast or inline status for long forecast/backtest jobs; never silent failure
  - Persistence: symbol, timeframe, denoise, panel prefs in localStorage (token never persisted—already correct)
  - Accessibility: labels, `aria-*` on toggles, contrast on overlays, reduced-motion respect

### 3. Performance and perceived speed

- Keep chart interactions at 60fps where possible (lightweight-charts stays the renderer; avoid React re-render thrash on every poll).
- Query discipline: stable React Query keys, abort on symbol/tf change (partially present), sensible live poll intervals by timeframe.
- Avoid refetch storms when toggling overlays or opening panels.
- Bundle: code-split heavy panels (forecast/backtest) if main chunk grows; keep production `dist` lean for FastAPI static serve.
- History: efficient merge of live tip + paginated left history; cap client-side bar retention with clear “load more” semantics.
- Long jobs: optimistic UI + cancel where API allows; clear progress for backtests.

### 4. Engineering quality (close the dev gaps)

| Gap | Target |
|---|---|
| Tests mostly pure lib/API helpers | Expand unit tests for workspace status, time, client; add component tests for critical panels; optional Playwright smoke for `/app` load + symbol select |
| No frontend lint in package scripts | Add ESLint (TS + React hooks) or Biome; wire `npm run lint` |
| AGENTS.md / FILE MAP drift | Keep `webui/AGENTS.md` in sync with real tree (features/, status, vitest) |
| Single mega-hook / large panel components | Feature folders: `chart-workspace`, `forecast`, `overlays`, `connection`; shared UI kit |
| Types hand-maintained vs API | Keep `types.ts` aligned with compact Web API payloads; document any intentional subset |
| No visual regression | Not required day one; snapshot only if it pays for itself |
| Docs | This goal + WEB_API stay the source of truth for “what the UI should expose” |

**Definition of done for quality (baseline)**

- `npm run typecheck` and `npm test` clean in CI-or-manual checklist
- New API client methods ship with types + at least one pure test for error/params shaping when non-trivial
- No `any` without justification (strict TS already expected)

---

## Success criteria

The goal is met when all of the following are true:

1. **API coverage scorecard:** 100% of Web API routes classified (used / intentional omit); zero “forgotten” routes like `/models`.
2. **Core workflows (manual or automated smoke):**
   - Open `/app` → select symbol → see candles
   - Toggle live, bid/ask/last, pivots, S/R, denoise
   - Run price forecast → see overlay + metrics
   - Run volatility + backtest with readable results
   - Auth token path works against tokenized server
3. **Responsive:** usable forecast and toolbar flows at 375px, 768px, and 1440px widths without horizontal page scroll traps.
4. **Performance:** symbol switch and live poll remain snappy on a typical desktop; no unbounded memory growth after long live sessions.
5. **Polish:** consistent empty/loading/error states; modern dark UI with coherent density; keyboard Esc closes overlays.
6. **Docs:** WEB_API and this goal describe the same product surface; `webui/AGENTS.md` matches the tree.

---

## Phased roadmap

Phases are sequential by default; work inside a phase can parallelize.

### Phase 0 — Stabilize & inventory (short)

- Freeze a **coverage matrix** (endpoint × UI entry point × params × tests).
- Refresh `webui/AGENTS.md` and package scripts checklist.
- Baseline metrics: bundle size, Lighthouse-ish manual notes, list of UX pain points (toolbar overflow, fixed panel width).

### Phase 1 — Parity & reliability

- Wire `/models` and readiness/health indicator.
- Expose missing high-value params (pivot method, S/R controls, date range / as-of clarity).
- Harden error surfaces (partial overlay failures, forecast failures, offline API).
- Expand pure tests; fix type drift; optional lint.

### Phase 2 — Responsive & modern shell

- Responsive toolbar (overflow menu / bottom bar).
- Forecast panel as adaptive drawer/sheet.
- Design tokens / shared components (buttons, selects, badges, sheets).
- Keyboard and a11y pass on primary flows.
- Empty states and onboarding (no symbol, MT5 not ready, dist missing already handled server-side).

### Phase 3 — Chart research depth

- Indicator overlays (subset first: MA, ATR band, volume pane if data available).
- Richer S/R / Fib visualization from API rich mode.
- Multi-forecast compare (2–3 methods) if API payloads allow clean overlay semantics.
- Backtest results visualization upgrade (equity-style summary, not only tables).

### Phase 4 — Backend feature bridge

- For each approved domain (regimes, patterns, reports, …): design compact Web API → implement → UI module → docs.
- Prefer read-only research features before any account/trading surfaces.

### Phase 5 — Hardening & speed

- Code-split, query audit, long-session memory check.
- Optional e2e smoke against mock or live local API.
- Performance budget note in this doc or WEB_API.

---

## Working agreements

1. **Transport purity:** UI talks only to Web API (`/api/v1`). No special-casing domain logic in React; adapt and present.
2. **Compact payloads:** Prefer UI-oriented compact responses; request `detail=full` only when the UI shows those diagnostics.
3. **Safety:** No order placement / account mutation from the SPA without an explicit safety design (confirmations, mode gates, env flags)—aligned with trading safety docs.
4. **Token hygiene:** Auth token stays in memory only (current behavior is correct—do not “improve” it into localStorage).
5. **Small PRs:** Prefer vertical slices (one capability usable end-to-end) over giant refactors.
6. **Commit style:** `webui: <imperative summary>` (or `docs:` when only documentation).

---

## Out of scope (for this goal)

- Replacing CLI/MCP as the full automation surface
- Shipping a light-theme redesign as a priority (dark terminal is the product look)
- Mobile-native apps / offline-first PWA (nice-to-have later, not goal-blocking)
- Full visual parity with TradingView or commercial terminals
- Auto-trading dashboards

---

## Tracking

Use this document as the north star. Practical tracking options:

- Keep the **coverage matrix** updated in PRs that touch API or UI surfaces.
- When a phase completes, mark it here with date and short notes.
- Link major implementation PRs under each phase.

| Phase | Status | Notes |
|---|---|---|
| 0 Inventory | Done | Coverage matrix in `docs/WEBUI_API_COVERAGE.md`; `webui/AGENTS.md` refreshed |
| 1 Parity & reliability | Done | `/models`, health/ready chip, pivot method + S/R params, partial-failure banners |
| 2 Responsive & modern shell | Done | Breakpoint helpers; toolbar More overflow; forecast bottom sheet / drawer; Esc closes panels/modals |
| 3 Chart research depth | Partial | On-chart EMA/RSI/MACD/volume, confluence, volume profile, idea TP/SL, read-only exposure (2026-08-18). Regime tint still later. |
| 4 Backend feature bridge | Done | Full MCP catalog via `GET/POST /api/v1/tools*`; SPA Tools runner; watchlist radar + session strip (2026-08-18) |
| 5 Hardening & speed | Partial | Pure unit tests + typecheck/build; no e2e CI |

---

## One-line summary

**Ship a modern, fast, responsive chart research console that fully exercises the Web API, closes frontend engineering gaps, and grows in lockstep with the backend’s highest-value research capabilities—without compromising safety or transport boundaries.**

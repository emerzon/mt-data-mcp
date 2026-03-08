# webui/ — React Frontend

React + Vite + Tailwind single-page app for chart visualization and forecast interaction. Proxies to FastAPI backend on `:8000`.

## FILE MAP

| File | Lines | Purpose |
|------|-------|---------|
| `src/App.tsx` | 547 | Main app component — symbol selector, chart, toolbar, panels |
| `src/main.tsx` | — | React entry point (renders App) |
| `src/types.ts` | — | Shared TypeScript type definitions |
| `src/styles.css` | — | Tailwind base styles |
| **Components** | | |
| `src/components/ChartToolbar.tsx` | 579 | Timeframe, indicators, denoise controls |
| `src/components/ForecastPanel.tsx` | 553 | Forecast configuration and display |
| `src/components/DenoiseModal.tsx` | — | Denoise filter selection modal |
| `src/components/OHLCChart.tsx` | — | OHLC candlestick chart (lightweight-charts) |
| **Data Layer** | | |
| `src/api/client.ts` | — | Axios HTTP client for FastAPI backend |
| `src/hooks/useForecast.ts` | — | React Query hook for forecast data |
| **Utilities** | | |
| `src/lib/storage.ts` | — | LocalStorage helpers |
| `src/lib/time.ts` | — | Time formatting utilities |
| `src/lib/timeframes.ts` | — | Timeframe constants and helpers |
| `src/lib/utils.ts` | — | General utility functions |

## CONVENTIONS

- **TypeScript strict mode** — no `any` casts.
- **PascalCase** filenames for components, `camelCase` for hooks/lib.
- **Tailwind CSS** for all styling — no CSS modules or styled-components.
- **lightweight-charts** (TradingView) for OHLC rendering.
- **@tanstack/react-query** for server state management.

## DEV SETUP

```bash
cd webui && npm install && npm run dev    # Dev server on :5173
cd webui && npm run build                 # Production build → dist/
```

Vite proxies `/api/*` to `localhost:8000` (FastAPI). Production build is served as static files by FastAPI (`web_api_runtime.py` mounts `webui/dist/`).

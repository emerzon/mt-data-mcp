import type { DenoiseSpecUI } from '../types'
import { PivotIcon, RefreshIcon, SRIcon } from '../features/chart-workspace/toolbarIcons'
import {
  DenoiseSelector,
  PriceLinesSelector,
  SymbolSelector,
  TimeframeSelector,
  TimezoneSelector,
} from '../features/chart-workspace/toolbarMenus'

type Props = {
  symbol: string
  timeframe: string
  displayAnchor?: number
  isLoading: boolean
  barsCount: number
  hasPivots: boolean
  hasSR: boolean
  denoise?: DenoiseSpecUI
  showBid: boolean
  showAsk: boolean
  showLast: boolean
  isLive: boolean
  onSymbolChange: (value: string) => void
  onTimeframeChange: (value: string) => void
  onClearAnchor: () => void
  onReload: () => void
  onTogglePivots: () => void
  onToggleSR: () => void
  onDenoiseChange: (value?: DenoiseSpecUI) => void
  onOpenForecast: () => void
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLast: () => void
  onToggleLive: () => void
  timezoneMode: 'utc' | 'local' | 'server'
  onTimezoneChange: (value: 'utc' | 'local' | 'server') => void
}

export function ChartToolbar({
  symbol,
  timeframe,
  displayAnchor,
  isLoading,
  barsCount,
  hasPivots,
  hasSR,
  denoise,
  showBid,
  showAsk,
  showLast,
  isLive,
  onSymbolChange,
  onTimeframeChange,
  onClearAnchor,
  onReload,
  onTogglePivots,
  onToggleSR,
  onDenoiseChange,
  onOpenForecast,
  onToggleBid,
  onToggleAsk,
  onToggleLast,
  onToggleLive,
  timezoneMode,
  onTimezoneChange,
}: Props) {
  return (
    <div className="absolute top-3 left-3 right-3 z-20 flex items-start gap-2">
      <div className="flex items-center gap-1 bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 p-1">
        <SymbolSelector value={symbol} onChange={onSymbolChange} />
        <div className="w-px h-5 bg-slate-700" />
        <TimezoneSelector value={timezoneMode} onChange={onTimezoneChange} />
        <div className="w-px h-5 bg-slate-700" />
        <TimeframeSelector value={timeframe} onChange={onTimeframeChange} />
        <div className="w-px h-5 bg-slate-700" />
        <button className="toolbar-btn" onClick={onReload} disabled={!symbol || isLoading} title="Reload data">
          <RefreshIcon className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="flex-1" />

      {symbol && (
        <div className="bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 px-3 py-1.5 text-xs text-slate-400">
          {barsCount} bars
          {displayAnchor !== undefined && (
            <span className="ml-2 text-amber-400">
              Anchor: {new Date(displayAnchor * 1000).toISOString().slice(11, 19)}
              <button className="ml-1 text-slate-500 hover:text-slate-300" onClick={onClearAnchor}>
                ×
              </button>
            </span>
          )}
        </div>
      )}

      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 p-1">
          <button
            className={`toolbar-btn ${hasPivots ? 'text-amber-400' : ''}`}
            onClick={onTogglePivots}
            disabled={!symbol}
            title="Toggle pivot levels"
          >
            <PivotIcon />
          </button>
          <button
            className={`toolbar-btn ${hasSR ? 'text-emerald-400' : ''}`}
            onClick={onToggleSR}
            disabled={!symbol}
            title="Toggle support/resistance"
          >
            <SRIcon />
          </button>
          <div className="w-px h-5 bg-slate-700" />
          <PriceLinesSelector
            showBid={showBid}
            showAsk={showAsk}
            showLast={showLast}
            isLive={isLive}
            disabled={!symbol}
            onToggleBid={onToggleBid}
            onToggleAsk={onToggleAsk}
            onToggleLast={onToggleLast}
            onToggleLive={onToggleLive}
          />
          <div className="w-px h-5 bg-slate-700" />
          <DenoiseSelector value={denoise} disabled={!symbol} onChange={onDenoiseChange} />
        </div>

        <button
          className="bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium px-4 py-1.5 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          onClick={onOpenForecast}
          disabled={!symbol}
        >
          Forecast
        </button>
      </div>
    </div>
  )
}

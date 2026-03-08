import { useState } from 'react'
import { OHLCChart } from './components/OHLCChart'
import { useChartWorkspace } from './features/chart-workspace/useChartWorkspace'
import { ChartToolbar } from './features/chart-workspace/ChartToolbar'
import { ForecastPanel } from './features/forecast-studio/ForecastPanel'

export default function App() {
  const [showForecastPanel, setShowForecastPanel] = useState(false)
  const workspace = useChartWorkspace()

  return (
    <div className="h-full flex flex-col bg-slate-950">
      {/* Full-screen chart area */}
      <main className="flex-1 relative min-h-0">
        {/* Floating toolbar */}
        <ChartToolbar
          symbol={workspace.symbol}
          timeframe={workspace.timeframe}
          anchor={workspace.anchor}
          isLoading={workspace.isFetching || workspace.isLoadingMore}
          onSymbolChange={workspace.handleSymbolChange}
          onTimeframeChange={workspace.handleTimeframeChange}
          onClearAnchor={workspace.clearAnchor}
          onReload={workspace.reload}
          onTogglePivots={workspace.handlePivotToggle}
          onToggleSR={workspace.handleSRToggle}
          onDenoiseChange={workspace.handleDenoiseChange}
          onOpenForecast={() => setShowForecastPanel(true)}
          hasPivots={!!workspace.pivotLevels}
          hasSR={!!workspace.srLevels}
          denoise={workspace.chartDenoise}
          barsCount={workspace.bars.length}
          showBid={workspace.showBid}
          showAsk={workspace.showAsk}
          showLast={workspace.showLast}
          isLive={workspace.isLive}
          timezoneMode={workspace.timezoneMode}
          onToggleBid={workspace.toggleBid}
          onToggleAsk={workspace.toggleAsk}
          onToggleLast={workspace.toggleLast}
          onToggleLive={workspace.toggleLive}
          onTimezoneChange={workspace.setTimezoneMode}
        />

        {/* Chart */}
        <div className="absolute inset-0">
          <OHLCChart
            data={workspace.displayBars}
            onAnchor={workspace.handleAnchorSelect}
            onNeedMoreLeft={workspace.earliest ? workspace.handleNeedMoreLeft : undefined}
            anchorTime={workspace.displayAnchor}
            overlays={workspace.displayOverlays}
            priceLines={workspace.priceLines}
          />
        </div>

        {/* Metrics overlay */}
        {workspace.metrics && (
          <div className="absolute bottom-4 left-4 flex gap-2 z-20">
            <MetricBadge label="n" value={String(workspace.metrics.overlap)} />
            <MetricBadge label="MAE" value={workspace.metrics.mae.toFixed(4)} />
            <MetricBadge label="MAPE" value={`${workspace.metrics.mape.toFixed(1)}%`} />
            <MetricBadge label="RMSE" value={workspace.metrics.rmse.toFixed(4)} />
            <MetricBadge 
              label="Dir" 
              value={`${workspace.metrics.dirAcc.toFixed(0)}%`}
              variant={workspace.metrics.dirAcc >= 60 ? 'success' : workspace.metrics.dirAcc >= 50 ? 'warning' : 'error'}
            />
          </div>
        )}

        {/* Forecast panel (slide-in from right) */}
        <ForecastPanel
          open={showForecastPanel}
          onClose={() => setShowForecastPanel(false)}
          symbol={workspace.symbol}
          timeframe={workspace.timeframe}
          anchor={workspace.anchor}
          onResult={workspace.handleForecastResult}
        />
      </main>
    </div>
  )
}

function MetricBadge({ 
  label, 
  value, 
  variant = 'default' 
}: { 
  label: string
  value: string
  variant?: 'default' | 'success' | 'warning' | 'error'
}) {
  const colors = {
    default: 'bg-slate-800/90 text-slate-300 border-slate-700',
    success: 'bg-emerald-950/90 text-emerald-300 border-emerald-800',
    warning: 'bg-amber-950/90 text-amber-300 border-amber-800',
    error: 'bg-rose-950/90 text-rose-300 border-rose-800',
  }
  return (
    <div className={`px-2 py-1 rounded border text-xs font-medium backdrop-blur-sm ${colors[variant]}`}>
      <span className="text-slate-500 mr-1">{label}</span>
      {value}
    </div>
  )
}

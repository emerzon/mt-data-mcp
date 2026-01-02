import { useState, useCallback, useMemo, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getHistory, getPivots, getSupportResistance, getTick } from './api/client'
import { OHLCChart, type PriceLineSpec } from './components/OHLCChart'
import { ChartToolbar } from './components/ChartToolbar'
import { ForecastPanel } from './components/ForecastPanel'
import type {
  ForecastPayload,
  HistoryBar,
  SupportResistanceLevel,
  DenoiseSpecUI,
  ChartOverlay,
  AnchorMetrics,
  PivotLevel,
} from './types'
import { toUtcSec } from './lib/time'
import { tfSeconds } from './lib/timeframes'
import { loadJSON, saveJSON } from './lib/storage'

const QUERY_LIMIT = 1000

export default function App() {
  // Core state
  const [symbol, setSymbol] = useState(() => loadJSON<string>('last_symbol') || '')
  const [timeframe, setTimeframe] = useState('H1')
  // extraHistory stores older bars fetched via infinite scroll
  const [extraHistory, setExtraHistory] = useState<HistoryBar[]>([])
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  
  const [end, setEnd] = useState<string | undefined>(undefined)
  const [anchor, setAnchor] = useState<number | undefined>(undefined)
  const [showBid, setShowBid] = useState(false)
  const [showAsk, setShowAsk] = useState(false)
  const [showLast, setShowLast] = useState(true)
  const [isLive, setIsLive] = useState(true)
  const [timezoneMode, setTimezoneMode] = useState<'utc' | 'local' | 'server'>('utc')
  const [serverOffset, setServerOffset] = useState(0)

  // Chart overlays
  const [forecastOverlays, setForecastOverlays] = useState<ChartOverlay[]>([])
  const [chartDenoise, setChartDenoise] = useState<DenoiseSpecUI | undefined>(undefined)
  const [pivotLevels, setPivotLevels] = useState<PivotLevel[] | null>(null)
  const [srLevels, setSrLevels] = useState<SupportResistanceLevel[] | null>(null)

  // UI state
  const [showForecastPanel, setShowForecastPanel] = useState(false)
  const [metrics, setMetrics] = useState<AnchorMetrics | null>(null)

  // Data fetching
  const { data: histDataResponse, refetch, isFetching } = useQuery({
    queryKey: ['hist', symbol, timeframe, QUERY_LIMIT, end, JSON.stringify(chartDenoise || {}), isLive],
    queryFn: () => getHistory({ symbol, timeframe, limit: QUERY_LIMIT, end, denoise: chartDenoise, include_incomplete: isLive }),
    enabled: !!symbol,
  })

  // Update server offset if available
  useEffect(() => {
    if (histDataResponse?.meta?.server_tz_offset !== undefined) {
      setServerOffset(histDataResponse.meta.server_tz_offset)
    }
  }, [histDataResponse])

  const { data: liveDataResponse } = useQuery({
    queryKey: ['hist-live', symbol, timeframe],
    queryFn: () => getHistory({ symbol, timeframe, limit: 2, include_incomplete: true }),
    enabled: isLive && !!symbol && !end,
    refetchInterval: 2000,
  })

  const { data: tickData } = useQuery({
    queryKey: ['tick', symbol],
    queryFn: () => getTick(symbol),
    enabled: !!symbol,
    refetchInterval: 2000,
  })

  const bars = useMemo(() => {
    const base = (histDataResponse?.bars ?? []) as HistoryBar[]
    const live = (liveDataResponse?.bars ?? []) as HistoryBar[]
    
    let combined = base

    // Prepend extraHistory
    if (extraHistory.length) {
      // Dedup: filter extraHistory to ensure all bars are older than base[0]
      const mainStart = base.length ? base[0].time : Infinity
      const older = extraHistory.filter(b => b.time < mainStart)
      combined = [...older, ...base]
    }

    if (!isLive || !live.length || !combined.length || end) return combined
    
    const merged = [...combined]
    // Merge live tail
    live.forEach(bar => {
      const lastIndex = merged.length - 1
      if (lastIndex >= 0) {
        const last = merged[lastIndex]
        // Allow small floating point diffs in time (though typically integers)
        if (Math.abs(bar.time - last.time) < 0.1) {
          merged[lastIndex] = bar
        } else if (bar.time > last.time) {
          merged.push(bar)
        }
      } else {
        merged.push(bar)
      }
    })
    return merged
  }, [histDataResponse, liveDataResponse, isLive, end, extraHistory])

  // Compute timezone-adjusted bars for display
  const displayBars = useMemo(() => {
    // bars are in Canonical UTC (from API)
    // We need to shift them to the target display time
    
    // Target: UTC (no shift)
    if (timezoneMode === 'utc') {
      return bars
    }
    
    // Target: Server (Exchange) - shift by server offset
    if (timezoneMode === 'server') {
      return bars.map(b => ({ ...b, time: b.time + serverOffset }))
    }

    // Target: Local - shift by local offset
    if (timezoneMode === 'local') {
      // Local offset in seconds (e.g. UTC-6 -> -21600)
      const localOffset = -new Date().getTimezoneOffset() * 60
      return bars.map(b => ({ ...b, time: b.time + localOffset }))
    }

    return bars
  }, [bars, timezoneMode, serverOffset])

  const getCanonicalTime = useCallback((displayTime: number) => {
    // Convert Display Time -> Canonical UTC
    if (timezoneMode === 'utc') return displayTime
    
    if (timezoneMode === 'server') {
      return displayTime - serverOffset
    }
    
    if (timezoneMode === 'local') {
      const localOffset = -new Date().getTimezoneOffset() * 60
      return displayTime - localOffset
    }
    return displayTime
  }, [timezoneMode, serverOffset])

  const getDisplayTime = useCallback((utcTime: number) => {
    // Convert Canonical UTC -> Display Time
    if (timezoneMode === 'utc') return utcTime
    
    if (timezoneMode === 'server') {
      return utcTime + serverOffset
    }
    
    if (timezoneMode === 'local') {
      const localOffset = -new Date().getTimezoneOffset() * 60
      return utcTime + localOffset
    }
    return utcTime
  }, [timezoneMode, serverOffset])

  // Handler for anchor selection from chart (receives display time)
  const handleAnchorSelect = useCallback((t: number) => {
    setAnchor(getCanonicalTime(t))
  }, [getCanonicalTime])

  const earliest = bars.length ? bars[0].time : undefined

  // Reset state on symbol/timeframe change
  const handleSymbolChange = useCallback((newSymbol: string) => {
    setSymbol(newSymbol)
    setEnd(undefined)
    setExtraHistory([])
    setForecastOverlays([])
    setAnchor(undefined)
    setPivotLevels(null)
    setSrLevels(null)
    setMetrics(null)
    
    // Save state
    saveJSON('last_symbol', newSymbol)
    
    // Update recent symbols
    if (newSymbol) {
      const recent = loadJSON<string[]>('recent_symbols') || []
      const updated = [newSymbol, ...recent.filter(s => s !== newSymbol)].slice(0, 10)
      saveJSON('recent_symbols', updated)
    }

    if (newSymbol && timeframe) {
      const saved = loadJSON<DenoiseSpecUI | undefined>(`chart_dn:${newSymbol}:${timeframe}`)
      setChartDenoise(saved || undefined)
    }
  }, [timeframe])

  const handleTimeframeChange = useCallback((newTf: string) => {
    setTimeframe(newTf)
    setEnd(undefined)
    setExtraHistory([])
    setForecastOverlays([])
    setAnchor(undefined)
    setPivotLevels(null)
    setSrLevels(null)
    setMetrics(null)
    if (symbol && newTf) {
      const saved = loadJSON<DenoiseSpecUI | undefined>(`chart_dn:${symbol}:${newTf}`)
      setChartDenoise(saved || undefined)
    }
  }, [symbol])

  const handleNeedMoreLeft = useCallback(async (tEarliestDisplay: number) => {
    if (!symbol || isLoadingMore || isFetching) return
    setIsLoadingMore(true)
    try {
      // Convert display time back to UTC for the API query
      const tUtc = getCanonicalTime(tEarliestDisplay)
      const dt = new Date((tUtc - 1) * 1000).toISOString().slice(0, 19).replace('T', ' ')
      
      const older = await getHistory({ 
        symbol, 
        timeframe, 
        limit: QUERY_LIMIT, 
        end: dt, 
        denoise: chartDenoise 
      })
      if (older.bars.length) {
        setExtraHistory(prev => {
           // Simple prepend; useMemo will dedup based on time
           return [...older.bars, ...prev]
        })
      }
    } catch (err) {
      console.error('Failed to load more history:', err)
    } finally {
      setIsLoadingMore(false)
    }
  }, [symbol, timeframe, chartDenoise, isLoadingMore, isFetching, getCanonicalTime])

  const handlePivotToggle = useCallback(async () => {
    if (!symbol) return
    if (pivotLevels) {
      setPivotLevels(null)
      return
    }
    try {
      const data = await getPivots({ symbol, timeframe, method: 'classic' })
      const levels = (data.levels || [])
        .map(row => ({ level: String(row.level), value: Number(row.value) }))
        .filter(row => Number.isFinite(row.value))
      if (levels.length) setPivotLevels(levels)
    } catch (err) {
      console.error('Failed to fetch pivots:', err)
    }
  }, [symbol, timeframe, pivotLevels])

  const handleSRToggle = useCallback(async () => {
    if (!symbol) return
    if (srLevels) {
      setSrLevels(null)
      return
    }
    try {
      const data = await getSupportResistance({ symbol, timeframe, limit: QUERY_LIMIT })
      const levels = (data.levels || []).filter(row => Number.isFinite(row?.value))
      if (levels.length) setSrLevels(levels)
    } catch (err) {
      console.error('Failed to fetch S/R:', err)
    }
  }, [symbol, timeframe, srLevels])

  const handleDenoiseChange = useCallback((denoise?: DenoiseSpecUI) => {
    setChartDenoise(denoise)
    if (symbol && timeframe) saveJSON(`chart_dn:${symbol}:${timeframe}`, denoise)
  }, [symbol, timeframe])

  const handleForecastResult = useCallback((res: ForecastPayload) => {
    const main = res.forecast_price ?? res.forecast_return ?? []
    let times: number[] = []
    
    if (res.forecast_epoch && res.forecast_epoch.length === main.length) {
      times = res.forecast_epoch.map(t => toUtcSec(t))
    } else {
      const step = tfSeconds(timeframe)
      const anchorOverride = res.__anchor ? Number(res.__anchor) : undefined
      if (anchorOverride && step) {
        times = Array.from({ length: main.length }, (_, i) => anchorOverride + step * (i + 1))
      } else {
        const last = bars.length ? bars[bars.length - 1].time : undefined
        if (last && step) {
          times = Array.from({ length: main.length }, (_, i) => last + step * (i + 1))
        } else {
          const fallback = (res.forecast_time || res.times || []) as (number | string)[]
          times = fallback.map(t => toUtcSec(t))
        }
      }
    }

    const overlays: ChartOverlay[] = [
      { name: 'forecast', points: times.map((t, i) => ({ time: t, value: main[i] })), color: '#60a5fa', lineWidth: 2 }
    ]
    if (res.lower_price && res.upper_price) {
      overlays.push({
        name: 'lower',
        points: times.map((t, i) => ({ time: t, value: res.lower_price![i] })),
        color: '#64748b',
        lineStyle: 'dashed',
      })
      overlays.push({
        name: 'upper',
        points: times.map((t, i) => ({ time: t, value: res.upper_price![i] })),
        color: '#64748b',
        lineStyle: 'dashed',
      })
    }
    setForecastOverlays(overlays)

    // Calculate metrics for partial forecasts
    if (res.__kind === 'partial' && anchor && bars.length) {
      const closeByTime = new Map<number, number>()
      for (const bar of bars) closeByTime.set(Math.floor(bar.time), bar.close)
      
      const yPred: number[] = []
      const yAct: number[] = []
      for (let i = 0; i < times.length; i++) {
        const actual = closeByTime.get(Math.floor(times[i]))
        if (actual !== undefined && Number.isFinite(main[i])) {
          yPred.push(Number(main[i]))
          yAct.push(Number(actual))
        }
      }
      
      if (yPred.length) {
        const n = yPred.length
        const diffs = yPred.map((p, i) => p - yAct[i])
        const mae = diffs.reduce((acc, d) => acc + Math.abs(d), 0) / n
        const mape = (yPred.reduce((acc, _, i) => acc + Math.abs((yPred[i] - yAct[i]) / (Math.abs(yAct[i]) || 1)), 0) / n) * 100
        const rmse = Math.sqrt(diffs.reduce((acc, d) => acc + d * d, 0) / n)
        const anchorClose = bars.find(b => Math.floor(b.time) === Math.floor(anchor))?.close ?? yAct[0]
        let correct = 0
        for (let i = 0; i < n; i++) {
          const prev = i === 0 ? anchorClose : yAct[i - 1]
          if (Math.sign(yPred[i] - prev) === Math.sign(yAct[i] - prev)) correct++
        }
        setMetrics({ overlap: n, mae, mape, rmse, dirAcc: (correct / n) * 100 })
      } else {
        setMetrics(null)
      }
    } else {
      setMetrics(null)
    }
  }, [timeframe, bars, anchor])

  // Build chart overlays
  const chartOverlays = useMemo(() => {
    const map = new Map<string, ChartOverlay>()
    const add = (ov: ChartOverlay) => {
      if (ov?.name && Array.isArray(ov.points)) map.set(ov.name, ov)
    }

    forecastOverlays.forEach(add)

    const startTime = bars.length ? bars[0].time : undefined
    const lastBarTime = bars.length ? bars[bars.length - 1].time : undefined
    const tfStep = tfSeconds(timeframe) || 60
    
    let maxTime = lastBarTime
    forecastOverlays.forEach(ov => {
      ov.points?.forEach(pt => {
        if (Number.isFinite(pt?.time)) {
          maxTime = maxTime === undefined ? pt.time : Math.max(maxTime, pt.time)
        }
      })
    })
    const lineEnd = maxTime !== undefined ? maxTime + tfStep : undefined

    // Denoised line
    if (histDataResponse?.bars.length && 'close_dn' in histDataResponse.bars[0]) {
      const dnPoints = histDataResponse.bars
        .filter((bar): bar is HistoryBar & { close_dn: number } => 
          Number.isFinite(bar.time) && Number.isFinite(bar.close_dn))
        .map(bar => ({ time: bar.time, value: bar.close_dn }))
      if (dnPoints.length) {
        add({ name: 'denoise:close', points: dnPoints, color: '#f59e0b', lineWidth: 2 })
      }
    }

    // Pivot levels
    if (pivotLevels?.length && startTime !== undefined && lineEnd !== undefined) {
      pivotLevels.forEach(level => {
        if (!Number.isFinite(level.value)) return
        const color = level.level.startsWith('R') ? '#f97316' : level.level.startsWith('S') ? '#38bdf8' : '#facc15'
        add({
          name: `pivot-${level.level}`,
          points: [{ time: startTime, value: level.value }, { time: lineEnd, value: level.value }],
          color,
          lineStyle: 'dashed',
          lineWidth: 1,
          label: level.level,
        })
      })
    }

    // S/R levels
    if (srLevels?.length && startTime !== undefined && lineEnd !== undefined) {
      srLevels.forEach((level, idx) => {
        if (!Number.isFinite(level?.value)) return
        add({
          name: `sr-${level.type}-${idx}`,
          points: [{ time: startTime, value: level.value }, { time: lineEnd, value: level.value }],
          color: level.type === 'resistance' ? '#f87171' : '#34d399',
          lineWidth: 2,
          lineStyle: 'dotted',
          label: `${level.type === 'resistance' ? 'Res' : 'Sup'} (${level.touches})`,
        })
      })
    }

    return Array.from(map.values())
  }, [forecastOverlays, bars, pivotLevels, srLevels, timeframe])

  const priceLines: PriceLineSpec[] = useMemo(() => {
    if (!tickData) return []
    const lines: PriceLineSpec[] = []
    if (showBid) lines.push({ price: tickData.bid, color: '#ef4444', title: 'Bid' })
    if (showAsk) lines.push({ price: tickData.ask, color: '#22c55e', title: 'Ask' })
    
    if (showLast) {
      // Always show Last price line, labeled 'Last'
      // Prefer tick data, fallback to last bar close
      let lastPrice = tickData?.last
      if (!lastPrice && bars.length > 0) {
        lastPrice = bars[bars.length - 1].close
      }

      if (lastPrice && lastPrice > 0) {
        lines.push({ price: lastPrice, color: '#facc15', title: 'Last' })
      }
    }
    
    return lines
  }, [tickData, showBid, showAsk, showLast, bars])

  return (
    <div className="h-full flex flex-col bg-slate-950">
      {/* Full-screen chart area */}
      <main className="flex-1 relative min-h-0">
        {/* Floating toolbar */}
        <ChartToolbar
          symbol={symbol}
          timeframe={timeframe}
          anchor={anchor}
          isLoading={isFetching || isLoadingMore}
          onSymbolChange={handleSymbolChange}
          onTimeframeChange={handleTimeframeChange}
          onClearAnchor={() => setAnchor(undefined)}
          onReload={() => { setEnd(undefined); setExtraHistory([]); refetch() }}
          onTogglePivots={handlePivotToggle}
          onToggleSR={handleSRToggle}
          onDenoiseChange={handleDenoiseChange}
          onOpenForecast={() => setShowForecastPanel(true)}
          hasPivots={!!pivotLevels}
          hasSR={!!srLevels}
          denoise={chartDenoise}
          barsCount={bars.length}
          showBid={showBid}
          showAsk={showAsk}
          showLast={showLast}
          isLive={isLive}
          timezoneMode={timezoneMode}
          onToggleBid={() => setShowBid(prev => !prev)}
          onToggleAsk={() => setShowAsk(prev => !prev)}
          onToggleLast={() => setShowLast(prev => !prev)}
          onToggleLive={() => setIsLive(prev => !prev)}
          onTimezoneChange={setTimezoneMode}
        />

        {/* Chart */}
        <div className="absolute inset-0">
          <OHLCChart
            data={displayBars}
            onAnchor={handleAnchorSelect}
            onNeedMoreLeft={earliest ? handleNeedMoreLeft : undefined}
            anchorTime={anchor ? getDisplayTime(anchor) : undefined}
            overlays={chartOverlays.map(ov => ({
              ...ov,
              points: ov.points.map(p => ({ ...p, time: getDisplayTime(p.time) }))
            }))}
            priceLines={priceLines}
          />
        </div>

        {/* Metrics overlay */}
        {metrics && (
          <div className="absolute bottom-4 left-4 flex gap-2 z-20">
            <MetricBadge label="n" value={String(metrics.overlap)} />
            <MetricBadge label="MAE" value={metrics.mae.toFixed(4)} />
            <MetricBadge label="MAPE" value={`${metrics.mape.toFixed(1)}%`} />
            <MetricBadge label="RMSE" value={metrics.rmse.toFixed(4)} />
            <MetricBadge 
              label="Dir" 
              value={`${metrics.dirAcc.toFixed(0)}%`}
              variant={metrics.dirAcc >= 60 ? 'success' : metrics.dirAcc >= 50 ? 'warning' : 'error'}
            />
          </div>
        )}

        {/* Forecast panel (slide-in from right) */}
        <ForecastPanel
          open={showForecastPanel}
          onClose={() => setShowForecastPanel(false)}
          symbol={symbol}
          timeframe={timeframe}
          anchor={anchor}
          onResult={handleForecastResult}
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

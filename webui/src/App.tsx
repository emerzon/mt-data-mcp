import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getHistory, getPivots, getSupportResistance } from './api/client'
import { InstrumentPicker } from './components/InstrumentPicker'
import { TimeframePicker } from './components/TimeframePicker'
import { OHLCChart } from './components/OHLCChart'
import { ForecastControls } from './components/ForecastControls'
import { VolatilityControls } from './components/VolatilityControls'
import { ChartDenoiseControls } from './components/ChartDenoiseControls'
import type { ForecastPayload, HistoryBar, SupportResistanceLevel } from './types'
import { toUtcSec } from './lib/time'
import { tfSeconds } from './lib/timeframes'
import type { DenoiseSpecUI } from './components/DenoiseModal'
import { loadJSON, saveJSON } from './lib/storage'

type AnchorMetrics = {
  overlap: number
  mae: number
  mape: number
  rmse: number
  dirAcc: number
}

export default function App() {
  const [tab, setTab] = useState<'price' | 'vol'>('price')
  const [symbol, setSymbol] = useState('')
  const [timeframe, setTimeframe] = useState('H1')
  const [limit, setLimit] = useState(800)
  const [end, setEnd] = useState<string | undefined>(undefined)
  const [anchor, setAnchor] = useState<number | undefined>(undefined)
  const [forecastOverlays, setForecastOverlays] = useState<any[]>([])
  const [chartDenoise, setChartDenoise] = useState<DenoiseSpecUI | undefined>(undefined)
  const [pivotLevels, setPivotLevels] = useState<{ level: string; value: number }[] | null>(null)
  const [pivotLoading, setPivotLoading] = useState(false)
  const [pivotError, setPivotError] = useState<string | null>(null)
  const [pivotMeta, setPivotMeta] = useState<{ method: string; period?: { start?: string; end?: string } } | null>(null)
  const [srLevels, setSrLevels] = useState<SupportResistanceLevel[] | null>(null)
  const [srLoading, setSrLoading] = useState(false)
  const [srError, setSrError] = useState<string | null>(null)
  const [srMeta, setSrMeta] = useState<{ method: string; tolerance_pct: number; min_touches: number; window?: { start?: string | null; end?: string | null } } | null>(null)
  const [metrics, setMetrics] = useState<AnchorMetrics | null>(null)

  const { data, refetch, isFetching } = useQuery({
    queryKey: ['hist', symbol, timeframe, limit, end, JSON.stringify(chartDenoise || {})],
    queryFn: () =>
      getHistory({
        symbol,
        timeframe,
        limit,
        end,
        denoise: chartDenoise,
      }),
    enabled: !!symbol,
  })

  const bars = (data ?? []) as HistoryBar[]
  const earliest = bars.length ? bars[0].time : undefined

  useEffect(() => {
    setForecastOverlays([])
    setAnchor(undefined)
    setPivotLevels(null)
    setPivotMeta(null)
    setPivotError(null)
    setPivotLoading(false)
    setSrLevels(null)
    setSrMeta(null)
    setSrError(null)
    setSrLoading(false)
    if (symbol && timeframe) {
      const saved = loadJSON<DenoiseSpecUI | undefined>(`chart_dn:${symbol}:${timeframe}`)
      setChartDenoise(saved || undefined)
    }
  }, [symbol, timeframe])

  function onNeedMoreLeft(tEarliest: number) {
    if (!symbol || isFetching) return
    const dt = new Date((tEarliest - 1) * 1000)
    const fmt = dt.toISOString().slice(0, 19).replace('T', ' ')
    setEnd(fmt)
    setLimit((prev) => Math.min(20000, Math.floor(prev * 1.2)))
    setTimeout(() => refetch(), 50)
  }

  function onAnchor(t: number) {
    setAnchor(t)
  }

  const handlePivotToggle = async () => {
    if (!symbol) return
    if (pivotLevels) {
      setPivotLevels(null)
      setPivotMeta(null)
      setPivotError(null)
      return
    }
    try {
      setPivotLoading(true)
      setPivotError(null)
      const data = await getPivots({ symbol, timeframe, method: 'classic' })
      const levels = (data.levels || [])
        .map((row: any) => ({ level: String(row.level), value: Number(row.value) }))
        .filter((row) => Number.isFinite(row.value))
      if (!levels.length) {
        setPivotError('No pivot levels returned')
        setPivotLevels(null)
        setPivotMeta(null)
        return
      }
      setPivotLevels(levels)
      setPivotMeta({ method: data.method ?? 'classic', period: data.period })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch pivot levels'
      setPivotError(message)
      setPivotLevels(null)
      setPivotMeta(null)
    } finally {
      setPivotLoading(false)
    }
  }

  const handleSupportToggle = async () => {
    if (!symbol) return
    if (srLevels) {
      setSrLevels(null)
      setSrMeta(null)
      setSrError(null)
      return
    }
    try {
      setSrLoading(true)
      setSrError(null)
      const data = await getSupportResistance({ symbol, timeframe, limit })
      const levels = ((data.levels || []) as SupportResistanceLevel[]).filter((row) => Number.isFinite(row?.value))
      if (!levels.length) {
        setSrError('No support/resistance levels detected')
        setSrLevels(null)
        setSrMeta(null)
        return
      }
      setSrLevels(levels)
      setSrMeta({
        method: data.method ?? 'swing',
        tolerance_pct: data.tolerance_pct ?? 0,
        min_touches: data.min_touches ?? 2,
        window: data.window,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch S/R levels'
      setSrError(message)
      setSrLevels(null)
      setSrMeta(null)
    } finally {
      setSrLoading(false)
    }
  }

  function onPriceResult(res: ForecastPayload) {
    const main = res.forecast_price ?? res.forecast_return ?? []
    let times: number[] = []
    if (res.forecast_epoch && res.forecast_epoch.length === main.length) {
      times = res.forecast_epoch.map((t) => toUtcSec(t as any))
    } else {
      const step = tfSeconds(timeframe)
      const anchorOverride = (res as any).__anchor ? Number((res as any).__anchor) : undefined
      if (anchorOverride && step) {
        times = Array.from({ length: main.length }, (_, i) => anchorOverride + step * (i + 1))
      } else {
        const last = bars.length ? bars[bars.length - 1].time : undefined
        if (last && step) {
          times = Array.from({ length: main.length }, (_, i) => last + step * (i + 1))
        } else {
          const fallback = (res.forecast_time || res.times || []) as any[]
          times = fallback.map((t) => toUtcSec(t as any))
        }
      }
    }
    const overlay = times.map((t, i) => ({ time: t, value: main[i] }))
    const overlays = [{ name: 'forecast', points: overlay, color: '#60a5fa' }]
    if (res.lower_price && res.upper_price) {
      overlays.push({
        name: 'lower',
        points: times.map((t, i) => ({ time: t, value: res.lower_price![i] })),
        color: '#94a3b8',
      })
      overlays.push({
        name: 'upper',
        points: times.map((t, i) => ({ time: t, value: res.upper_price![i] })),
        color: '#94a3b8',
      })
    }
    setForecastOverlays(overlays)

    const isPartial = (res as any).__kind === 'partial'
    if (isPartial && anchor && bars.length) {
      const closeByTime = new Map<number, number>()
      for (const bar of bars) {
        closeByTime.set(Math.floor(bar.time), bar.close)
      }
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
        const mape =
          (yPred.reduce((acc, _, i) => {
            const denom = Math.abs(yAct[i]) || 1
            return acc + Math.abs((yPred[i] - yAct[i]) / denom)
          }, 0) /
            n) *
          100
        const rmse = Math.sqrt(diffs.reduce((acc, d) => acc + d * d, 0) / n)
        const anchorClose = bars.find((b) => Math.floor(b.time) === Math.floor(anchor))?.close ?? yAct[0]
        let correct = 0
        for (let i = 0; i < n; i++) {
          const prev = i === 0 ? anchorClose : yAct[i - 1]
          const dp = Math.sign(yPred[i] - prev)
          const da = Math.sign(yAct[i] - prev)
          if (dp === da) correct += 1
        }
        const dirAcc = (correct / n) * 100
        setMetrics({
          overlap: n,
          mae,
          mape,
          rmse,
          dirAcc,
        })
      } else {
        setMetrics(null)
      }
    } else {
      setMetrics(null)
    }
  }

  const chartAdvancedSummary =
    [
      chartDenoise?.method ? `denoise:${chartDenoise.method}` : null,
      pivotLevels ? `pivots:${pivotMeta?.method ?? 'classic'}` : null,
      srLevels ? `sr:${srMeta?.method ?? 'swing'}` : null,
    ]
      .filter(Boolean)
      .join(' | ') || 'None'

  const chartOverlays = useMemo(() => {
    const map = new Map<string, any>()
    const addOverlay = (ov: any) => {
      if (!ov?.name || !Array.isArray(ov.points)) return
      map.set(ov.name, ov)
    }

    forecastOverlays.forEach(addOverlay)

    const startTime = bars.length ? bars[0].time : undefined
    const lastBarTime = bars.length ? bars[bars.length - 1].time : undefined
    const tfStep = tfSeconds(timeframe) || 0
    const fallbackStep = tfStep || (bars.length >= 2 ? Math.max(1, bars[1].time - bars[0].time) : 60)
    let maxTime = lastBarTime
    forecastOverlays.forEach((ov) => {
      if (!ov?.points) return
      ov.points.forEach((pt: any) => {
        if (pt?.time === undefined) return
        const t = Number(pt.time)
        if (!Number.isFinite(t)) return
        maxTime = maxTime === undefined ? t : Math.max(maxTime, t)
      })
    })
    const lineEnd = maxTime !== undefined ? maxTime + fallbackStep : undefined

    if (bars.length && 'close_dn' in (bars[0] as any)) {
      const dnPoints = bars
        .filter((bar: any) => Number.isFinite(bar.time) && Number.isFinite(bar.close_dn))
        .map((bar: any) => ({ time: bar.time, value: bar.close_dn }))
      if (dnPoints.length) {
        addOverlay({ name: 'denoise:close', points: dnPoints, color: '#f59e0b', lineWidth: 2 })
      }
    }

    if (pivotLevels && pivotLevels.length && startTime !== undefined && lineEnd !== undefined) {
      const colorForLevel = (level: string) => {
        if (level.startsWith('R')) return '#f97316'
        if (level.startsWith('S')) return '#38bdf8'
        return '#facc15'
      }
      pivotLevels.forEach((level) => {
        if (!Number.isFinite(level.value)) return
        addOverlay({
          name: `pivot-${level.level}`,
          points: [
            { time: startTime, value: level.value },
            { time: lineEnd, value: level.value },
          ],
          color: colorForLevel(level.level),
          lineStyle: 'dashed',
          lineWidth: 1.5,
        })
      })
    }

    if (srLevels && srLevels.length && startTime !== undefined && lineEnd !== undefined) {
      srLevels.forEach((level, idx) => {
        if (!Number.isFinite(level?.value)) return
        const color = level.type === 'resistance' ? '#f87171' : '#34d399'
        addOverlay({
          name: `sr-${level.type}-${idx}`,
          points: [
            { time: startTime, value: level.value },
            { time: lineEnd, value: level.value },
          ],
          color,
          lineWidth: 2,
          lineStyle: 'dotted',
        })
      })
    }

    return Array.from(map.values())
  }, [forecastOverlays, bars, pivotLevels, srLevels, timeframe])

  const pivotButtonLabel = pivotLevels ? 'Hide Pivot Levels' : pivotLoading ? 'Loading...' : 'Plot Pivot Levels'
  const supportButtonLabel = srLevels ? 'Hide S/R Levels' : srLoading ? 'Loading...' : 'Plot S/R Levels'

  return (
    <div className="h-full flex flex-col">
      <header className="p-3 border-b border-slate-800 bg-slate-900/60">
        <div className="max-w-7xl mx-auto flex items-center gap-3">
          <h1 className="text-lg font-semibold text-slate-200">MTData WebUI</h1>
          <nav className="ml-6 flex gap-2">
            <button
              className={`btn ${tab === 'price' ? 'bg-sky-600' : 'bg-slate-700 hover:bg-slate-600'}`}
              onClick={() => setTab('price')}
            >
              Price / Returns
            </button>
            <button
              className={`btn ${tab === 'vol' ? 'bg-sky-600' : 'bg-slate-700 hover:bg-slate-600'}`}
              onClick={() => setTab('vol')}
            >
              Volatility
            </button>
          </nav>
          <div className="ml-auto text-xs text-slate-400">{symbol ? `${symbol} / ${timeframe}` : 'Select symbol'}</div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto w-full p-4 flex flex-col gap-4">
        <section className="panel p-4 space-y-4">
          <h2 className="text-sm font-semibold text-slate-200">Chart Settings</h2>
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <div className="label">Instrument</div>
              <InstrumentPicker value={symbol} onChange={setSymbol} />
            </div>
            <div>
              <div className="label">Timeframe</div>
              <TimeframePicker value={timeframe} onChange={setTimeframe} />
            </div>
            <label className="flex flex-col">
              <span className="label">Anchor (click chart to set)</span>
              <input
                className="input w-56"
                value={anchor ? new Date(anchor * 1000).toISOString().slice(0, 19).replace('T', ' ') : ''}
                readOnly
                placeholder="YYYY-MM-DD HH:MM:SS"
              />
            </label>
            <div className="ml-auto flex items-end gap-2">
              <label className="flex flex-col">
                <span className="label">Bars</span>
                <input
                  className="input w-24"
                  type="number"
                  min={100}
                  max={20000}
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                />
              </label>
              <button
                className="btn"
                disabled={!symbol}
                onClick={() => {
                  setEnd(undefined)
                  refetch()
                }}
              >
                Reload
              </button>
            </div>
          </div>
          <details className="border border-slate-800 rounded-md" open={!!chartDenoise?.method || !!pivotLevels || !!srLevels}>
            <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium text-slate-200 hover:bg-slate-800 flex justify-between items-center">
              <span>Advanced Settings</span>
              <span className="text-xs text-slate-400">{chartAdvancedSummary}</span>
            </summary>
            <div className="p-3 space-y-3">
              <ChartDenoiseControls
                value={chartDenoise}
                onChange={(v) => {
                  setChartDenoise(v)
                  if (symbol && timeframe) saveJSON(`chart_dn:${symbol}:${timeframe}`, v)
                }}
              />
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                  <button className="btn" onClick={handlePivotToggle} disabled={pivotLoading || !symbol}>
                    {pivotButtonLabel}
                  </button>
                  {pivotError && <span className="text-xs text-rose-400">{pivotError}</span>}
                </div>
                <div className="flex items-center gap-2">
                  <button className="btn" onClick={handleSupportToggle} disabled={srLoading || !symbol}>
                    {supportButtonLabel}
                  </button>
                  {srError && <span className="text-xs text-rose-400">{srError}</span>}
                </div>
              </div>
              {pivotLevels && pivotMeta?.period && (
                <div className="text-xs text-slate-400">Pivot source: {pivotMeta.period.start}{' -> '}{pivotMeta.period.end}</div>
              )}
              {srLevels && (
                <div className="text-xs text-slate-400 space-y-1">
                  <div>
                    S/R ({srLevels.length}) - tol +/- {((srMeta?.tolerance_pct ?? 0) * 100).toFixed(2)}% - min touches {srMeta?.min_touches ?? 2}
                  </div>
                  {srMeta?.window && (srMeta.window.start || srMeta.window.end) && (
                    <div>Window: {srMeta.window.start ?? 'n/a'}{' -> '}{srMeta.window.end ?? 'n/a'}</div>
                  )}
                  <div className="grid gap-1 sm:grid-cols-2">
                    {srLevels.map((lvl, idx) => {
                      const formatted = lvl.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })
                      const hue = lvl.type === 'resistance' ? 'text-rose-300' : 'text-emerald-300'
                      const last = lvl.last_touch ? ` - last ${lvl.last_touch}` : ''
                      return (
                        <div key={`sr-chip-${lvl.type}-${idx}`} className="flex justify-between gap-2">
                          <span className={`capitalize ${hue}`}>{lvl.type} @ {formatted}</span>
                          <span>{lvl.touches} touches{last}</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          </details>
        </section>

        <section className="panel p-4 space-y-4">
          <h2 className="text-sm font-semibold text-slate-200">Forecast Settings</h2>
          {tab === 'price' ? (
            <ForecastControls symbol={symbol} timeframe={timeframe} anchor={anchor} onResult={onPriceResult} />
          ) : (
            <VolatilityControls symbol={symbol} timeframe={timeframe} anchor={anchor} onResult={() => {}} />
          )}
        </section>

        <section className="panel p-2">
          <OHLCChart
            data={bars}
            onAnchor={onAnchor}
            onNeedMoreLeft={earliest ? onNeedMoreLeft : undefined}
            anchorTime={anchor}
            overlays={chartOverlays}
          />
        </section>

        {metrics && (
          <section className="panel p-3 flex flex-wrap items-center gap-4">
            <span className="text-xs text-slate-400">Anchor forecast metrics (n={metrics.overlap})</span>
            <MetricChip label="MAE" value={metrics.mae} unit="" severity={sevPct((metrics.mae / (bars[bars.length - 1]?.close || 1)) * 100)} fmt="abs" />
            <MetricChip label="MAPE" value={metrics.mape} unit="%" severity={sevPct(metrics.mape)} fmt="pct" />
            <MetricChip label="RMSE" value={metrics.rmse} unit="" severity={sevPct((metrics.rmse / (bars[bars.length - 1]?.close || 1)) * 100)} fmt="abs" />
            <MetricChip label="Dir Acc" value={metrics.dirAcc} unit="%" severity={sevDir(metrics.dirAcc)} fmt="pct" />
          </section>
        )}
      </main>
    </div>
  )
}

function sevPct(pct: number): 'good' | 'med' | 'bad' {
  const v = Math.abs(pct)
  if (v < 0.5) return 'good'
  if (v < 1.5) return 'med'
  return 'bad'
}

function sevDir(accPct: number): 'good' | 'med' | 'bad' {
  if (accPct >= 60) return 'good'
  if (accPct >= 50) return 'med'
  return 'bad'
}

function MetricChip({
  label,
  value,
  unit,
  severity,
  fmt,
}: {
  label: string
  value: number
  unit: string
  severity: 'good' | 'med' | 'bad'
  fmt: 'pct' | 'abs'
}) {
  const color = severity === 'good' ? 'bg-emerald-600' : severity === 'med' ? 'bg-amber-600' : 'bg-rose-600'
  const txt = fmt === 'pct' ? `${value.toFixed(1)}${unit}` : `${value.toFixed(5)}`
  return (
    <span className={`inline-flex items-center gap-2 ${color} text-white px-2 py-1 rounded-md text-xs`}>
      <strong className="font-semibold">{label}</strong>
      <span>{txt}</span>
    </span>
  )
}

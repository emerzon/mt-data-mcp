import { useState, useEffect, useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  getHistory,
  getMethods,
  getPivots,
  getSupportResistance,
  forecastPrice,
  getErrorMessage,
} from '../api/client'
import type {
  HistoryBar,
  ForecastPayload,
  DenoiseSpecUI,
  PivotLevel,
  SupportResistanceLevel,
  ChartOverlay,
  AnchorMetrics,
  ForecastPriceBody,
} from '../types'
import { loadJSON, saveJSON } from '../lib/storage'
import { tfSeconds } from '../lib/timeframes'
import { toUtcSec } from '../lib/time'
import { formatDateTime } from '../lib/utils'

// ============================================================================
// Chart Data Hook
// ============================================================================

export type UseChartDataOptions = {
  symbol: string
  timeframe: string
  limit: number
  end?: string
  denoise?: DenoiseSpecUI
}

export function useChartData(options: UseChartDataOptions) {
  const { symbol, timeframe, limit, end, denoise } = options

  const query = useQuery({
    queryKey: ['history', symbol, timeframe, limit, end, JSON.stringify(denoise ?? {})],
    queryFn: () => getHistory({ symbol, timeframe, limit, end, denoise }),
    enabled: !!symbol,
    staleTime: 30000,
  })

  return {
    bars: query.data ?? [],
    isLoading: query.isFetching,
    error: query.error ? getErrorMessage(query.error) : null,
    refetch: query.refetch,
  }
}

// ============================================================================
// Forecast Methods Hook
// ============================================================================

export function useForecastMethods() {
  const query = useQuery({
    queryKey: ['methods'],
    queryFn: getMethods,
    staleTime: 60000,
  })

  return {
    methods: query.data?.methods ?? [],
    isLoading: query.isLoading,
    error: query.error ? getErrorMessage(query.error) : null,
    refetch: query.refetch,
  }
}

// ============================================================================
// Pivot Levels Hook
// ============================================================================

export function usePivotLevels(symbol: string, timeframe: string) {
  const [levels, setLevels] = useState<PivotLevel[] | null>(null)
  const [meta, setMeta] = useState<{ method: string; period?: { start?: string; end?: string } } | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const toggle = useCallback(async () => {
    if (!symbol) return

    if (levels) {
      setLevels(null)
      setMeta(null)
      setError(null)
      return
    }

    try {
      setIsLoading(true)
      setError(null)
      const data = await getPivots({ symbol, timeframe, method: 'classic' })
      const parsed = (data.levels || [])
        .map(row => ({ level: String(row.level), value: Number(row.value) }))
        .filter(row => Number.isFinite(row.value))

      if (!parsed.length) {
        setError('No pivot levels returned')
        setLevels(null)
        setMeta(null)
        return
      }

      setLevels(parsed)
      setMeta({ method: data.method ?? 'classic', period: data.period })
    } catch (err) {
      setError(getErrorMessage(err))
      setLevels(null)
      setMeta(null)
    } finally {
      setIsLoading(false)
    }
  }, [symbol, timeframe, levels])

  const reset = useCallback(() => {
    setLevels(null)
    setMeta(null)
    setError(null)
  }, [])

  return { levels, meta, isLoading, error, toggle, reset }
}

// ============================================================================
// Support/Resistance Hook
// ============================================================================

export function useSupportResistance(symbol: string, timeframe: string, limit: number) {
  const [levels, setLevels] = useState<SupportResistanceLevel[] | null>(null)
  const [meta, setMeta] = useState<{
    method: string
    tolerance_pct: number
    min_touches: number
    window?: { start?: string | null; end?: string | null }
  } | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const toggle = useCallback(async () => {
    if (!symbol) return

    if (levels) {
      setLevels(null)
      setMeta(null)
      setError(null)
      return
    }

    try {
      setIsLoading(true)
      setError(null)
      const data = await getSupportResistance({ symbol, timeframe, limit })
      const parsed = (data.levels || []).filter(row => Number.isFinite(row?.value))

      if (!parsed.length) {
        setError('No support/resistance levels detected')
        setLevels(null)
        setMeta(null)
        return
      }

      setLevels(parsed)
      setMeta({
        method: data.method ?? 'swing',
        tolerance_pct: data.tolerance_pct ?? 0,
        min_touches: data.min_touches ?? 2,
        window: data.window,
      })
    } catch (err) {
      setError(getErrorMessage(err))
      setLevels(null)
      setMeta(null)
    } finally {
      setIsLoading(false)
    }
  }, [symbol, timeframe, limit, levels])

  const reset = useCallback(() => {
    setLevels(null)
    setMeta(null)
    setError(null)
  }, [])

  return { levels, meta, isLoading, error, toggle, reset }
}

// ============================================================================
// Forecast State Hook
// ============================================================================

const DIMRED_METHODS = new Set([
  'mlf_rf', 'mlf_lightgbm', 'nhits', 'nbeatsx', 'tft', 'patchtst',
  'chronos_bolt', 'timesfm', 'lag_llama', 'gt_deepar', 'gt_sfeedforward',
  'gt_prophet', 'gt_tft', 'gt_wavenet', 'gt_deepnpts', 'gt_mqf2', 'gt_npts', 'ensemble',
])

export type ForecastSettings = {
  method: string
  horizon: number
  lookback: number | ''
  target: 'price' | 'return'
  ci_alpha: number
  methodParams: Record<string, unknown>
  denoise?: DenoiseSpecUI
  dimredMethod?: string
  dimredParams?: Record<string, unknown>
}

export function useForecastSettings(symbol: string, timeframe: string) {
  const [settings, setSettings] = useState<ForecastSettings>({
    method: 'theta',
    horizon: 12,
    lookback: '',
    target: 'price',
    ci_alpha: 0.1,
    methodParams: {},
  })

  const storageKey = symbol && timeframe ? `fc:${symbol}:${timeframe}` : null

  // Load saved settings when symbol/timeframe changes
  useEffect(() => {
    if (!storageKey) return
    const saved = loadJSON<Partial<ForecastSettings>>(storageKey)
    if (saved) {
      setSettings(prev => ({
        ...prev,
        method: saved.method ?? prev.method,
        horizon: saved.horizon ?? prev.horizon,
        lookback: saved.lookback ?? prev.lookback,
        target: saved.target ?? prev.target,
        ci_alpha: saved.ci_alpha ?? prev.ci_alpha,
        methodParams: saved.methodParams ?? prev.methodParams,
        denoise: saved.denoise,
        dimredMethod: saved.dimredMethod,
        dimredParams: saved.dimredParams,
      }))
    }
  }, [storageKey])

  // Save settings when they change
  useEffect(() => {
    if (!storageKey) return
    saveJSON(storageKey, settings)
  }, [storageKey, settings])

  const supportsDimred = DIMRED_METHODS.has(settings.method)

  // Clear dimred when method doesn't support it
  useEffect(() => {
    if (!supportsDimred && (settings.dimredMethod || settings.dimredParams)) {
      setSettings(prev => ({ ...prev, dimredMethod: undefined, dimredParams: undefined }))
    }
  }, [supportsDimred, settings.dimredMethod, settings.dimredParams])

  return { settings, setSettings, supportsDimred }
}

// ============================================================================
// Forecast Execution Hook
// ============================================================================

export function useForecast(
  symbol: string,
  timeframe: string,
  settings: ForecastSettings,
  bars: HistoryBar[],
  onResult: (payload: ForecastPayload, metrics: AnchorMetrics | null) => void
) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = useCallback(
    async (kind: 'full' | 'partial', anchor?: number) => {
      if (!symbol) return

      setIsLoading(true)
      setError(null)

      try {
        const body: ForecastPriceBody = {
          symbol,
          timeframe,
          method: settings.method,
          horizon: settings.horizon,
          lookback: settings.lookback === '' ? undefined : Number(settings.lookback),
          ci_alpha: settings.ci_alpha,
          target: settings.target,
          as_of: kind === 'full' ? undefined : anchor ? formatDateTime(anchor) : undefined,
          params: settings.methodParams,
          denoise: settings.denoise,
          dimred_method: settings.dimredMethod,
          dimred_params: settings.dimredParams,
        }

        const res = await forecastPrice(body)
        const payload: ForecastPayload = {
          ...res,
          __anchor: kind === 'full' ? undefined : anchor,
          __kind: kind,
        }

        const metrics = calculateMetrics(payload, bars, anchor, timeframe)
        onResult(payload, metrics)
      } catch (err) {
        setError(getErrorMessage(err))
      } finally {
        setIsLoading(false)
      }
    },
    [symbol, timeframe, settings, bars, onResult]
  )

  return { run, isLoading, error }
}

// ============================================================================
// Chart Overlays Builder
// ============================================================================

export function useChartOverlays(
  bars: HistoryBar[],
  forecastOverlays: ChartOverlay[],
  pivotLevels: PivotLevel[] | null,
  srLevels: SupportResistanceLevel[] | null,
  timeframe: string
): ChartOverlay[] {
  return useMemo(() => {
    const map = new Map<string, ChartOverlay>()

    const addOverlay = (ov: ChartOverlay) => {
      if (!ov?.name || !Array.isArray(ov.points)) return
      map.set(ov.name, ov)
    }

    // Add forecast overlays
    forecastOverlays.forEach(addOverlay)

    // Calculate time boundaries
    const startTime = bars.length ? bars[0].time : undefined
    const lastBarTime = bars.length ? bars[bars.length - 1].time : undefined
    const tfStep = tfSeconds(timeframe) || 0
    const fallbackStep = tfStep || (bars.length >= 2 ? Math.max(1, bars[1].time - bars[0].time) : 60)

    let maxTime = lastBarTime
    forecastOverlays.forEach(ov => {
      ov.points?.forEach(pt => {
        if (pt?.time !== undefined && Number.isFinite(pt.time)) {
          maxTime = maxTime === undefined ? pt.time : Math.max(maxTime, pt.time)
        }
      })
    })
    const lineEnd = maxTime !== undefined ? maxTime + fallbackStep : undefined

    // Add denoised line if present
    if (bars.length && 'close_dn' in bars[0]) {
      const dnPoints = bars
        .filter((bar): bar is HistoryBar & { close_dn: number } => 
          Number.isFinite(bar.time) && Number.isFinite(bar.close_dn)
        )
        .map(bar => ({ time: bar.time, value: bar.close_dn }))

      if (dnPoints.length) {
        addOverlay({ name: 'denoise:close', points: dnPoints, color: '#f59e0b', lineWidth: 2 })
      }
    }

    // Add pivot levels
    if (pivotLevels?.length && startTime !== undefined && lineEnd !== undefined) {
      const colorForLevel = (level: string) => {
        if (level.startsWith('R')) return '#f97316'
        if (level.startsWith('S')) return '#38bdf8'
        return '#facc15'
      }

      pivotLevels.forEach(level => {
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

    // Add support/resistance levels
    if (srLevels?.length && startTime !== undefined && lineEnd !== undefined) {
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
  }, [bars, forecastOverlays, pivotLevels, srLevels, timeframe])
}

// ============================================================================
// Metrics Calculator
// ============================================================================

function calculateMetrics(
  res: ForecastPayload,
  bars: HistoryBar[],
  anchor: number | undefined,
  timeframe: string
): AnchorMetrics | null {
  const main = res.forecast_price ?? res.forecast_return ?? []
  const isPartial = res.__kind === 'partial'

  if (!isPartial || !anchor || !bars.length) return null

  // Build times array
  let times: number[] = []
  if (res.forecast_epoch?.length === main.length) {
    times = res.forecast_epoch.map(t => toUtcSec(t))
  } else {
    const step = tfSeconds(timeframe)
    if (anchor && step) {
      times = Array.from({ length: main.length }, (_, i) => anchor + step * (i + 1))
    }
  }

  // Map bar times to close prices
  const closeByTime = new Map<number, number>()
  for (const bar of bars) {
    closeByTime.set(Math.floor(bar.time), bar.close)
  }

  // Collect prediction/actual pairs
  const yPred: number[] = []
  const yAct: number[] = []
  for (let i = 0; i < times.length; i++) {
    const actual = closeByTime.get(Math.floor(times[i]))
    if (actual !== undefined && Number.isFinite(main[i])) {
      yPred.push(Number(main[i]))
      yAct.push(Number(actual))
    }
  }

  if (!yPred.length) return null

  const n = yPred.length
  const diffs = yPred.map((p, i) => p - yAct[i])
  const mae = diffs.reduce((acc, d) => acc + Math.abs(d), 0) / n
  const mape =
    (yPred.reduce((acc, _, i) => {
      const denom = Math.abs(yAct[i]) || 1
      return acc + Math.abs((yPred[i] - yAct[i]) / denom)
    }, 0) / n) * 100
  const rmse = Math.sqrt(diffs.reduce((acc, d) => acc + d * d, 0) / n)

  const anchorClose = bars.find(b => Math.floor(b.time) === Math.floor(anchor))?.close ?? yAct[0]
  let correct = 0
  for (let i = 0; i < n; i++) {
    const prev = i === 0 ? anchorClose : yAct[i - 1]
    const dp = Math.sign(yPred[i] - prev)
    const da = Math.sign(yAct[i] - prev)
    if (dp === da) correct += 1
  }
  const dirAcc = (correct / n) * 100

  return { overlap: n, mae, mape, rmse, dirAcc }
}

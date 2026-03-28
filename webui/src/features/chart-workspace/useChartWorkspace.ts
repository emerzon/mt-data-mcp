import { useCallback, useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getHistory, getTick } from '../../api/client'
import { useChartOverlays, usePivotLevels, useSupportResistance } from '../../hooks/useForecast'
import { loadJSON, saveJSON } from '../../lib/storage'
import { toUtcSec } from '../../lib/time'
import { tfSeconds } from '../../lib/timeframes'
import type {
  AnchorMetrics,
  ChartOverlay,
  DenoiseSpecUI,
  ForecastPayload,
  HistoryBar,
} from '../../types'
import type { PriceLineSpec } from '../../components/OHLCChart'

export type TimezoneMode = 'utc' | 'local' | 'server'

const QUERY_LIMIT = 1000

export function useChartWorkspace() {
  const [symbol, setSymbol] = useState(() => loadJSON<string>('last_symbol') || '')
  const [timeframe, setTimeframe] = useState('H1')
  const [extraHistory, setExtraHistory] = useState<HistoryBar[]>([])
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [end, setEnd] = useState<string | undefined>(undefined)
  const [anchor, setAnchor] = useState<number | undefined>(undefined)
  const [showBid, setShowBid] = useState(false)
  const [showAsk, setShowAsk] = useState(false)
  const [showLast, setShowLast] = useState(true)
  const [isLive, setIsLive] = useState(true)
  const [timezoneMode, setTimezoneMode] = useState<TimezoneMode>('utc')
  const [serverOffset, setServerOffset] = useState(0)
  const [forecastOverlays, setForecastOverlays] = useState<ChartOverlay[]>([])
  const [chartDenoise, setChartDenoise] = useState<DenoiseSpecUI | undefined>(undefined)
  const [metrics, setMetrics] = useState<AnchorMetrics | null>(null)

  const pivotState = usePivotLevels(symbol, timeframe)
  const srState = useSupportResistance(symbol, timeframe, QUERY_LIMIT)

  const { data: histDataResponse, refetch, isFetching } = useQuery({
    queryKey: ['hist', symbol, timeframe, QUERY_LIMIT, end, JSON.stringify(chartDenoise || {}), isLive],
    queryFn: () =>
      getHistory({
        symbol,
        timeframe,
        limit: QUERY_LIMIT,
        end,
        denoise: chartDenoise,
        include_incomplete: isLive,
      }),
    enabled: !!symbol,
  })

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

  useEffect(() => {
    const serverOffsetSeconds = histDataResponse?.meta?.runtime?.timezone?.server?.offset_seconds
    if (serverOffsetSeconds !== undefined) {
      setServerOffset(serverOffsetSeconds)
    }
  }, [histDataResponse])

  useEffect(() => {
    if (!symbol || !timeframe) {
      setChartDenoise(undefined)
      return
    }
    const saved = loadJSON<DenoiseSpecUI | undefined>(`chart_dn:${symbol}:${timeframe}`)
    setChartDenoise(saved || undefined)
  }, [symbol, timeframe])

  const bars = useMemo(() => {
    const base = (histDataResponse?.data ?? []) as HistoryBar[]
    const live = (liveDataResponse?.data ?? []) as HistoryBar[]

    let combined = base

    if (extraHistory.length) {
      const mainStart = base.length ? base[0].time : Infinity
      const older = extraHistory.filter((bar) => bar.time < mainStart)
      combined = [...older, ...base]
    }

    if (!isLive || !live.length || !combined.length || end) return combined

    const merged = [...combined]
    live.forEach((bar) => {
      const lastIndex = merged.length - 1
      if (lastIndex < 0) {
        merged.push(bar)
        return
      }
      const last = merged[lastIndex]
      if (Math.abs(bar.time - last.time) < 0.1) {
        merged[lastIndex] = bar
      } else if (bar.time > last.time) {
        merged.push(bar)
      }
    })
    return merged
  }, [end, extraHistory, histDataResponse, isLive, liveDataResponse])

  const displayBars = useMemo(() => {
    if (timezoneMode === 'utc') return bars

    if (timezoneMode === 'server') {
      return bars.map((bar) => ({ ...bar, time: bar.time + serverOffset }))
    }

    const localOffset = -new Date().getTimezoneOffset() * 60
    return bars.map((bar) => ({ ...bar, time: bar.time + localOffset }))
  }, [bars, timezoneMode, serverOffset])

  const getCanonicalTime = useCallback(
    (displayTime: number) => {
      if (timezoneMode === 'utc') return displayTime
      if (timezoneMode === 'server') return displayTime - serverOffset
      const localOffset = -new Date().getTimezoneOffset() * 60
      return displayTime - localOffset
    },
    [serverOffset, timezoneMode]
  )

  const getDisplayTime = useCallback(
    (utcTime: number) => {
      if (timezoneMode === 'utc') return utcTime
      if (timezoneMode === 'server') return utcTime + serverOffset
      const localOffset = -new Date().getTimezoneOffset() * 60
      return utcTime + localOffset
    },
    [serverOffset, timezoneMode]
  )

  const handleAnchorSelect = useCallback(
    (displayTime: number) => {
      setAnchor(getCanonicalTime(displayTime))
    },
    [getCanonicalTime]
  )

  const resetWorkspaceView = useCallback(() => {
    setEnd(undefined)
    setExtraHistory([])
    setForecastOverlays([])
    setAnchor(undefined)
    setMetrics(null)
    pivotState.reset()
    srState.reset()
  }, [pivotState, srState])

  const handleSymbolChange = useCallback(
    (newSymbol: string) => {
      setSymbol(newSymbol)
      resetWorkspaceView()
      saveJSON('last_symbol', newSymbol)

      if (!newSymbol) return

      const recent = loadJSON<string[]>('recent_symbols') || []
      const updated = [newSymbol, ...recent.filter((item) => item !== newSymbol)].slice(0, 10)
      saveJSON('recent_symbols', updated)
    },
    [resetWorkspaceView]
  )

  const handleTimeframeChange = useCallback(
    (newTimeframe: string) => {
      setTimeframe(newTimeframe)
      resetWorkspaceView()
    },
    [resetWorkspaceView]
  )

  const handleNeedMoreLeft = useCallback(
    async (earliestDisplayTime: number) => {
      if (!symbol || isLoadingMore || isFetching) return
      setIsLoadingMore(true)
      try {
        const utcTime = getCanonicalTime(earliestDisplayTime)
        const before = new Date((utcTime - 1) * 1000).toISOString().slice(0, 19).replace('T', ' ')
        const older = await getHistory({
          symbol,
          timeframe,
          limit: QUERY_LIMIT,
          end: before,
          denoise: chartDenoise,
        })
        if (older.data.length) {
          setExtraHistory((previous) => [...older.data, ...previous])
        }
      } catch (error) {
        console.error('Failed to load more history:', error)
      } finally {
        setIsLoadingMore(false)
      }
    },
    [chartDenoise, getCanonicalTime, isFetching, isLoadingMore, symbol, timeframe]
  )

  const handleDenoiseChange = useCallback(
    (denoise?: DenoiseSpecUI) => {
      setChartDenoise(denoise)
      if (symbol && timeframe) {
        saveJSON(`chart_dn:${symbol}:${timeframe}`, denoise)
      }
    },
    [symbol, timeframe]
  )

  const handleForecastResult = useCallback(
    (result: ForecastPayload) => {
      const main = result.forecast_price ?? result.forecast_return ?? []
      let times: number[] = []

      if (result.forecast_epoch && result.forecast_epoch.length === main.length) {
        times = result.forecast_epoch.map((value) => toUtcSec(value))
      } else {
        const step = tfSeconds(timeframe)
        const anchorOverride = result.__anchor ? Number(result.__anchor) : undefined
        if (anchorOverride && step) {
          times = Array.from({ length: main.length }, (_, index) => anchorOverride + step * (index + 1))
        } else {
          const last = bars.length ? bars[bars.length - 1].time : undefined
          if (last && step) {
            times = Array.from({ length: main.length }, (_, index) => last + step * (index + 1))
          } else {
            const fallback = (result.forecast_time || result.times || []) as (number | string)[]
            times = fallback.map((value) => toUtcSec(value))
          }
        }
      }

      const overlays: ChartOverlay[] = [
        {
          name: 'forecast',
          points: times.map((time, index) => ({ time, value: main[index] })),
          color: '#60a5fa',
          lineWidth: 2,
        },
      ]

      if (result.lower_price && result.upper_price) {
        overlays.push({
          name: 'lower',
          points: times.map((time, index) => ({ time, value: result.lower_price![index] })),
          color: '#64748b',
          lineStyle: 'dashed',
        })
        overlays.push({
          name: 'upper',
          points: times.map((time, index) => ({ time, value: result.upper_price![index] })),
          color: '#64748b',
          lineStyle: 'dashed',
        })
      }
      setForecastOverlays(overlays)

      if (result.__kind === 'partial' && anchor && bars.length) {
        const closeByTime = new Map<number, number>()
        for (const bar of bars) closeByTime.set(Math.floor(bar.time), bar.close)

        const yPred: number[] = []
        const yAct: number[] = []
        for (let index = 0; index < times.length; index += 1) {
          const actual = closeByTime.get(Math.floor(times[index]))
          if (actual !== undefined && Number.isFinite(main[index])) {
            yPred.push(Number(main[index]))
            yAct.push(Number(actual))
          }
        }

        if (yPred.length) {
          const n = yPred.length
          const diffs = yPred.map((prediction, index) => prediction - yAct[index])
          const mae = diffs.reduce((total, diff) => total + Math.abs(diff), 0) / n
          const mape =
            (yPred.reduce((total, _, index) => {
              const denom = Math.abs(yAct[index]) || 1
              return total + Math.abs((yPred[index] - yAct[index]) / denom)
            }, 0) /
              n) *
            100
          const rmse = Math.sqrt(diffs.reduce((total, diff) => total + diff * diff, 0) / n)
          const anchorClose = bars.find((bar) => Math.floor(bar.time) === Math.floor(anchor))?.close ?? yAct[0]

          let correct = 0
          for (let index = 0; index < n; index += 1) {
            const previous = index === 0 ? anchorClose : yAct[index - 1]
            if (Math.sign(yPred[index] - previous) === Math.sign(yAct[index] - previous)) {
              correct += 1
            }
          }
          setMetrics({ overlap: n, mae, mape, rmse, dirAcc: (correct / n) * 100 })
        } else {
          setMetrics(null)
        }
      } else {
        setMetrics(null)
      }
    },
    [anchor, bars, timeframe]
  )

  const chartOverlays = useChartOverlays(
    bars,
    forecastOverlays,
    pivotState.levels,
    srState.levels,
    timeframe
  )

  const priceLines: PriceLineSpec[] = useMemo(() => {
    if (!tickData) return []

    const lines: PriceLineSpec[] = []
    if (showBid) lines.push({ price: tickData.bid, color: '#ef4444', title: 'Bid' })
    if (showAsk) lines.push({ price: tickData.ask, color: '#22c55e', title: 'Ask' })

    if (showLast) {
      let lastPrice = tickData.last
      if (!lastPrice && bars.length > 0) {
        lastPrice = bars[bars.length - 1].close
      }
      if (lastPrice && lastPrice > 0) {
        lines.push({ price: lastPrice, color: '#facc15', title: 'Last' })
      }
    }

    return lines
  }, [bars, showAsk, showBid, showLast, tickData])

  const earliest = bars.length ? bars[0].time : undefined

  return {
    symbol,
    timeframe,
    anchor,
    showBid,
    showAsk,
    showLast,
    isLive,
    timezoneMode,
    chartDenoise,
    bars,
    displayBars,
    chartOverlays,
    priceLines,
    metrics,
    pivotLevels: pivotState.levels,
    srLevels: srState.levels,
    isFetching,
    isLoadingMore,
    earliest,
    setTimezoneMode,
    handleAnchorSelect,
    handleSymbolChange,
    handleTimeframeChange,
    handleNeedMoreLeft,
    handleDenoiseChange,
    handleForecastResult,
    handlePivotToggle: pivotState.toggle,
    handleSRToggle: srState.toggle,
    reload: () => {
      setEnd(undefined)
      setExtraHistory([])
      void refetch()
    },
    toggleBid: () => setShowBid((value) => !value),
    toggleAsk: () => setShowAsk((value) => !value),
    toggleLast: () => setShowLast((value) => !value),
    toggleLive: () => setIsLive((value) => !value),
    clearAnchor: () => setAnchor(undefined),
    displayAnchor: anchor ? getDisplayTime(anchor) : undefined,
    displayOverlays: chartOverlays.map((overlay) => ({
      ...overlay,
      points: overlay.points.map((point) => ({ ...point, time: getDisplayTime(point.time) })),
    })),
  }
}

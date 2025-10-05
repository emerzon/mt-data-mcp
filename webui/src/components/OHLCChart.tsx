import { createChart, IChartApi, LineStyle, Time, UTCTimestamp } from 'lightweight-charts'
import { useEffect, useRef } from 'react'
import type { HistoryBar } from '../types'

export type OHLCChartProps = {
  data: HistoryBar[]
  onAnchor?: (t: number) => void
  onNeedMoreLeft?: (earliestTime: number) => void
  overlays?: { name: string; points: { time: number; value: number }[]; color?: string; lineWidth?: number; lineStyle?: 'solid' | 'dashed' | 'dotted'; priceScaleId?: string }[]
  anchorTime?: number
}

export function OHLCChart({ data, onAnchor, onNeedMoreLeft, overlays, anchorTime }: OHLCChartProps) {
  const ref = useRef<HTMLDivElement | null>(null)
  const apiRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<any>(null)
  const anchorRef = useRef<any>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      autoSize: true,
      layout: { background: { color: '#0f172a' }, textColor: '#a3b3c7' },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: '#1f2937' },
      timeScale: { borderColor: '#1f2937' },
    })
    const series = chart.addCandlestickSeries({ upColor: '#22c55e', downColor: '#ef4444', borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444' })
    apiRef.current = chart
    candleRef.current = series

    const clickSub = chart.subscribeClick((p) => {
      if (!p || p.time === undefined) return
      const t = (p.time as UTCTimestamp) as number
      onAnchor?.(t)
    })
    const rangeSub = chart.timeScale().subscribeVisibleTimeRangeChange((r) => {
      if (!r || !onNeedMoreLeft) return
      const from = r.from as number | undefined
      if (from && data.length > 0) {
        const earliest = data[0]?.time
        // When user scrolls close to the left edge, request more
        if (from <= earliest + 2) {
          onNeedMoreLeft(earliest)
        }
      }
    })

    return () => {
      chart.unsubscribeClick(clickSub)
      chart.timeScale().unsubscribeVisibleTimeRangeChange(rangeSub)
      chart.remove()
      apiRef.current = null
      candleRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!candleRef.current) return
    const series = candleRef.current
    const points = data.map((b) => ({ time: b.time as Time, open: b.open, high: b.high, low: b.low, close: b.close }))
    series.setData(points)
  }, [data])

  useEffect(() => {
    if (!apiRef.current) return
    const chart = apiRef.current
    const overlaySeries: any[] = []
    overlays?.forEach((ov) => {
      if (!ov?.points?.length) return
      const style =
        ov.lineStyle === 'dashed'
          ? LineStyle.Dashed
          : ov.lineStyle === 'dotted'
          ? LineStyle.Dotted
          : LineStyle.Solid
      const series = chart.addLineSeries({
        color: ov.color || '#60a5fa',
        lineWidth: ov.lineWidth ?? 2,
        lineStyle: style,
        priceScaleId: ov.priceScaleId || 'right',
      })
      series.setData(
        ov.points
          .filter((p) => Number.isFinite(p.time) && Number.isFinite(p.value))
          .map((p) => ({
            time: (Math.floor(p.time) as unknown as UTCTimestamp) as Time,
            value: p.value,
          })),
      )
      overlaySeries.push(series)
    })
    return () => {
      overlaySeries.forEach((series) => chart.removeSeries(series))
    }
  }, [overlays])

  // Anchor vertical marker using a 1-candle series with long wick spanning min..max
  useEffect(() => {
    const chart = apiRef.current
    if (!chart) return
    if (anchorRef.current) {
      chart.removeSeries(anchorRef.current)
      anchorRef.current = null
    }
    if (!data?.length || !Number.isFinite(anchorTime as number)) return
    const minLow = Math.min(...data.map(d => d.low))
    const maxHigh = Math.max(...data.map(d => d.high))
    const center = (minLow + maxHigh) / 2
    const s = chart.addCandlestickSeries({
      upColor: '#facc15', downColor: '#facc15', borderVisible: false, wickUpColor: '#facc15', wickDownColor: '#facc15',
      priceScaleId: 'right',
    })
    s.setData([{ time: (Math.floor(anchorTime as number) as unknown as UTCTimestamp) as Time, open: center, high: maxHigh, low: minLow, close: center }])
    anchorRef.current = s
    return () => {
      if (anchorRef.current) {
        chart.removeSeries(anchorRef.current)
        anchorRef.current = null
      }
    }
  }, [anchorTime, data])

  return <div className="w-full h-[520px]" ref={ref} />
}

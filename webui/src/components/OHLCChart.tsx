import { createChart, IChartApi, ISeriesApi, LineStyle, Time, IPriceLine } from 'lightweight-charts'
import { useEffect, useRef } from 'react'
import type { HistoryBar, ChartOverlay } from '../types'

export type PriceLineSpec = {
  price: number
  color: string
  title: string
}

export type OHLCChartProps = {
  data: HistoryBar[]
  onAnchor?: (t: number) => void
  onNeedMoreLeft?: (earliestTime: number) => void
  overlays?: ChartOverlay[]
  priceLines?: PriceLineSpec[]
  anchorTime?: number
}

export function OHLCChart({ data, onAnchor, onNeedMoreLeft, overlays, priceLines, anchorTime }: OHLCChartProps) {
  const ref = useRef<HTMLDivElement | null>(null)
  const apiRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const anchorRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const priceLinesRef = useRef<IPriceLine[]>([])

  // Keep refs up to date for event handlers
  const dataRef = useRef(data)
  const onNeedMoreLeftRef = useRef(onNeedMoreLeft)
  const onAnchorRef = useRef(onAnchor)

  useEffect(() => {
    dataRef.current = data
    onNeedMoreLeftRef.current = onNeedMoreLeft
    onAnchorRef.current = onAnchor
  }, [data, onNeedMoreLeft, onAnchor])

  useEffect(() => {
    if (!ref.current) return

    const chart = createChart(ref.current, {
      autoSize: true,
      layout: { background: { color: '#0f172a' }, textColor: '#a3b3c7' },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: '#1f2937' },
      timeScale: { borderColor: '#1f2937', timeVisible: true, secondsVisible: true },
    })

    const series = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      lastValueVisible: false,
      priceLineVisible: false,
    })

    apiRef.current = chart
    candleRef.current = series

    const clickHandler = (p: { time?: Time }) => {
      if (!p || p.time === undefined) return
      const t = Number(p.time)
      if (!Number.isFinite(t)) return
      onAnchorRef.current?.(t)
    }

    const rangeHandler = (r: { from: Time; to: Time } | null) => {
      const currentData = dataRef.current
      const needMore = onNeedMoreLeftRef.current
      
      if (!r || !needMore || !currentData.length) return
      
      const from = r.from as number | undefined
      if (from) {
        const earliest = currentData[0].time
        // Trigger if visible range start is at or before the earliest bar
        if (from <= earliest) {
          needMore(earliest)
        }
      }
    }

    chart.subscribeClick(clickHandler)
    chart.timeScale().subscribeVisibleTimeRangeChange(rangeHandler)

    return () => {
      chart.unsubscribeClick(clickHandler)
      chart.timeScale().unsubscribeVisibleTimeRangeChange(rangeHandler)
      chart.remove()
      apiRef.current = null
      candleRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!candleRef.current) return
    const series = candleRef.current
    const points = data.map(b => ({
      time: b.time as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))
    series.setData(points)

    // Adjust seconds visibility based on detected bar spacing
    if (apiRef.current && data.length >= 2) {
      const dt = Math.abs(Math.floor(data[1].time) - Math.floor(data[0].time))
      const secondsVisible = dt < 60
      apiRef.current.applyOptions({ timeScale: { timeVisible: true, secondsVisible } })
    }
  }, [data])

  useEffect(() => {
    if (!apiRef.current) return
    const chart = apiRef.current
    const overlaySeries: ISeriesApi<'Line'>[] = []

    overlays?.forEach(ov => {
      if (!ov?.points?.length) return

      const style =
        ov.lineStyle === 'dashed'
          ? LineStyle.Dashed
          : ov.lineStyle === 'dotted'
          ? LineStyle.Dotted
          : LineStyle.Solid

      const series = chart.addLineSeries({
        color: ov.color || '#60a5fa',
        lineWidth: (ov.lineWidth ?? 2) as 1 | 2 | 3 | 4,
        lineStyle: style,
        priceScaleId: ov.priceScaleId || 'right',
      })

      series.setData(
        ov.points
          .filter(p => Number.isFinite(p.time) && Number.isFinite(p.value))
          .map(p => ({
            time: Math.floor(p.time) as unknown as Time,
            value: p.value,
          }))
      )

      if (ov.label && ov.points.length > 0) {
        const last = ov.points[ov.points.length - 1]
        series.createPriceLine({
          price: last.value,
          color: ov.color || '#60a5fa',
          lineWidth: (ov.lineWidth ?? 2) as 1 | 2 | 3 | 4,
          lineStyle: style,
          axisLabelVisible: true,
          title: ov.label,
        })
      }

      overlaySeries.push(series)
    })

    return () => {
      overlaySeries.forEach(series => chart.removeSeries(series))
    }
  }, [overlays])

  // Manage Price Lines (Bid/Ask)
  useEffect(() => {
    if (!candleRef.current) return
    const series = candleRef.current
    
    // Clear old lines
    priceLinesRef.current.forEach(l => series.removePriceLine(l))
    priceLinesRef.current = []

    // Add new lines
    priceLines?.forEach(pl => {
      const line = series.createPriceLine({
        price: pl.price,
        color: pl.color,
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: pl.title,
      })
      priceLinesRef.current.push(line)
    })
    
    return () => {
      if (candleRef.current) {
        priceLinesRef.current.forEach(l => candleRef.current?.removePriceLine(l))
      }
      priceLinesRef.current = []
    }
  }, [priceLines])

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
      upColor: '#facc15',
      downColor: '#facc15',
      borderVisible: false,
      wickUpColor: '#facc15',
      wickDownColor: '#facc15',
      priceScaleId: 'right',
    })

    s.setData([
      {
        time: Math.floor(anchorTime as number) as unknown as Time,
        open: center,
        high: maxHigh,
        low: minLow,
        close: center,
      },
    ])

    anchorRef.current = s

    return () => {
      if (anchorRef.current && chart) {
        chart.removeSeries(anchorRef.current)
        anchorRef.current = null
      }
    }
  }, [anchorTime, data])

  return <div className="w-full h-full" ref={ref} />
}

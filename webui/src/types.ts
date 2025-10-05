export type Timeframe = string

export type Instrument = {
  name: string
  group?: string
  description?: string
}

export type HistoryBar = {
  time: number // epoch seconds UTC
  open: number
  high: number
  low: number
  close: number
  tick_volume?: number
}

export type SupportResistanceLevel = {
  type: 'support' | 'resistance'
  value: number
  touches: number
  score?: number
  first_touch?: string | null
  last_touch?: string | null
}

export type ForecastPayload = {
  times?: number[]
  forecast_epoch?: number[]
  forecast_time?: string[]
  forecast_price?: number[]
  forecast_return?: number[]
  lower_price?: number[]
  upper_price?: number[]
  forecast_quantiles?: Record<string, number[]>
  // client-only context
  __anchor?: number
  __kind?: 'full' | 'partial' | 'backtest'
}

export type MethodsMeta = {
  methods: {
    method: string
    available: boolean
    requires: string[]
    description: string
    params: { name: string; type: string; default?: any; description?: string }[]
    supports: { price?: boolean; return?: boolean; ci?: boolean }
  }[]
}

export type VolatilityMethodsMeta = {
  methods: {
    method: string
    available: boolean
    requires: string[]
    description?: string
    params: { name: string; type: string; default?: any; description?: string }[]
  }[]
}


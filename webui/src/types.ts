// ============================================================================
// Core Data Types
// ============================================================================

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
  close_dn?: number // denoised close (when denoising applied)
}

// ============================================================================
// Support/Resistance & Pivot Types
// ============================================================================

export type SupportResistanceLevel = {
  type: 'support' | 'resistance'
  value: number
  touches: number
  score?: number
  first_touch?: string | null
  last_touch?: string | null
}

export type PivotLevel = {
  level: string
  value: number
}

export type PivotResponse = {
  levels: PivotLevel[]
  period?: { start?: string; end?: string }
  method: string
  symbol: string
  timeframe: string
}

export type SupportResistanceResponse = {
  symbol: string
  timeframe: string
  limit: number
  method: string
  tolerance_pct: number
  min_touches: number
  window?: { start?: string | null; end?: string | null }
  levels: SupportResistanceLevel[]
}

// ============================================================================
// Forecast Types
// ============================================================================

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

export type VolatilityPayload = {
  symbol: string
  timeframe: string
  method: string
  horizon: number
  forecast_epoch?: number[]
  forecast_time?: string[]
  forecast_vol?: number[]
  annualized_vol?: number
}

// ============================================================================
// Method Metadata Types
// ============================================================================

export type ParamDef = {
  name: string
  type: string
  default?: unknown
  description?: string
}

export type MethodInfo = {
  method: string
  available: boolean
  requires: string[]
  description: string
  params: ParamDef[]
  supports?: { price?: boolean; return?: boolean; ci?: boolean }
}

export type MethodsMeta = {
  methods: MethodInfo[]
}

export type VolatilityMethodInfo = {
  method: string
  available: boolean
  requires: string[]
  description?: string
  params: ParamDef[]
}

export type VolatilityMethodsMeta = {
  methods: VolatilityMethodInfo[]
}

export type DenoiseMethodInfo = {
  method: string
  available: boolean
  requires?: string
  description: string
  params: ParamDef[]
}

export type DenoiseMethodsMeta = {
  methods: DenoiseMethodInfo[]
}

export type DimredMethodInfo = {
  method: string
  available: boolean
  description: string
  params: ParamDef[]
}

export type DimredMethodsMeta = {
  methods: DimredMethodInfo[]
}

export type WaveletsResponse = {
  available: boolean
  families: string[]
  wavelets: string[]
  by_family: Record<string, string[]>
}

export type SktimeEstimator = {
  name: string
  class_path: string
}

export type SktimeEstimatorsResponse = {
  available: boolean
  estimators: SktimeEstimator[]
  error?: string
}

// ============================================================================
// Denoise Spec (for UI forms)
// ============================================================================

export type DenoiseSpecUI = {
  method?: string
  params?: Record<string, unknown>
  columns?: string | string[]
  when?: 'pre_ti' | 'post_ti'
  causality?: 'zero_phase' | 'causal'
  keep_original?: boolean
}

// ============================================================================
// API Request Bodies
// ============================================================================

export type ForecastPriceBody = {
  symbol: string
  timeframe?: string
  method?: string
  horizon?: number
  lookback?: number
  as_of?: string
  params?: Record<string, unknown>
  ci_alpha?: number
  quantity?: 'price' | 'return' | 'volatility'
  target?: 'price' | 'return'
  denoise?: DenoiseSpecUI
  features?: Record<string, unknown>
  dimred_method?: string
  dimred_params?: Record<string, unknown>
  target_spec?: Record<string, unknown>
}

export type ForecastVolBody = {
  symbol: string
  timeframe?: string
  horizon?: number
  method?: string
  proxy?: string
  params?: Record<string, unknown>
  as_of?: string
  denoise?: DenoiseSpecUI
}

export type BacktestBody = {
  symbol: string
  timeframe?: string
  horizon?: number
  steps?: number
  spacing?: number
  methods?: string[]
  params_per_method?: Record<string, unknown>
  quantity?: 'price' | 'return'
  target?: 'price' | 'return'
  denoise?: DenoiseSpecUI
  params?: Record<string, unknown>
  features?: Record<string, unknown>
  dimred_method?: string
  dimred_params?: Record<string, unknown>
  slippage_bps?: number
  trade_threshold?: number
}

export type BacktestResult = {
  symbol: string
  timeframe: string
  horizon: number
  steps: number
  spacing: number
  results: {
    method: string
    mae?: number
    mape?: number
    rmse?: number
    direction_accuracy?: number
    predictions?: number[][]
    actuals?: number[][]
    anchor_times?: number[]
  }[]
}

// ============================================================================
// Chart Overlay Types
// ============================================================================

export type ChartOverlay = {
  name: string
  points: { time: number; value: number }[]
  color?: string
  lineWidth?: number
  lineStyle?: 'solid' | 'dashed' | 'dotted'
  priceScaleId?: string
  label?: string
}

// ============================================================================
// Metrics
// ============================================================================

export type AnchorMetrics = {
  overlap: number
  mae: number
  mape: number
  rmse: number
  dirAcc: number
}

export type Tick = {
  symbol: string
  time: number
  bid: number
  ask: number
  last: number
  volume: number
}

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

export type RuntimeTimezoneMeta = {
  output?: {
    tz?: {
      value?: string | null
      hint?: string
    }
  }
  server?: {
    source?: string
    tz?: {
      configured?: string | null
      resolved?: string | null
      offset_seconds?: number
    }
  }
  client?: {
    tz?: {
      configured?: string | null
      resolved?: string | null
    }
  }
  utc?: {
    now?: string
  }
  local?: {
    tz?: {
      name?: string | null
    }
  }
}

export type HistoryResponse = {
  bars: HistoryBar[]
  meta?: {
    runtime?: {
      timezone?: RuntimeTimezoneMeta
    }
  }
}

// ============================================================================
// Support/Resistance & Pivot Types
// ============================================================================

export type SupportResistanceLevel = {
  type: 'support' | 'resistance'
  value: number
  touches: number
  episodes?: number
  score?: number
  distance?: number | null
  distance_pct?: number | null
  zone_low?: number | null
  zone_high?: number | null
  zone_width?: number | null
  zone_width_atr?: number | null
  first_touch?: string | null
  last_touch?: string | null
  dominant_source?: 'support' | 'resistance' | 'mixed'
  status?: string
  source_tests?: {
    support: number
    resistance: number
  }
  source_episodes?: {
    support: number
    resistance: number
  }
  avg_bounce_atr?: number | null
  avg_pretest_adx?: number | null
  breakout_analysis?: {
    decisive_break_count: number
    avg_breach_atr?: number | null
    last_break_time?: string | null
    role_reversal_count: number
  }
  score_breakdown?: {
    base?: number
    retests?: number
    bounce?: number
    adx?: number
    breakout_penalty?: number
    role_reversal_bonus?: number
    mtf_confirmation_bonus?: number
    total?: number
  }
  source_timeframes?: string[]
  merge_details?: {
    cross_timeframe_dedupe_count: number
    deduped_timeframes: string[]
  }
  episode_details?: Array<{
    type: 'support' | 'resistance' | 'mixed'
    touches: number
    first_touch?: string | null
    last_touch?: string | null
  }>
  timeframe_contributions?: Array<{
    timeframe: string
    weight: number
    raw_score: number
    weighted_score: number
    touches: number
    episodes?: number
    merge_mode?: 'full' | 'deduped'
  }>
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
  mode?: string
  timeframes_analyzed?: string[]
  timeframe_weights?: Record<string, number>
  per_timeframe?: Array<{
    timeframe: string
    supports: number
    resistances: number
    current_price?: number | null
    window?: { start?: string | null; end?: string | null }
    effective_tolerance_pct?: number
    effective_reaction_bars?: number
    volatility_ratio?: number
    current_atr_pct?: number | null
    baseline_atr_pct?: number | null
  }>
  limit: number
  method: string
  tolerance_pct: number
  effective_tolerance_pct?: number
  min_touches: number
  qualification_basis?: 'episodes'
  max_levels?: number
  reaction_bars?: number
  effective_reaction_bars?: number
  adx_period?: number
  adaptive_mode?: 'atr_regime'
  volatility_ratio?: number
  current_atr_pct?: number | null
  baseline_atr_pct?: number | null
  current_price?: number | null
  window?: { start?: string | null; end?: string | null }
  levels: SupportResistanceLevel[]
  supports?: SupportResistanceLevel[]
  resistances?: SupportResistanceLevel[]
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
  quantity?: 'price' | 'return' | 'volatility'
  denoise?: DenoiseSpecUI
  params?: Record<string, unknown>
  features?: Record<string, unknown>
  dimred_method?: string
  dimred_params?: Record<string, unknown>
  slippage_bps?: number
  trade_threshold?: number
  detail?: 'compact' | 'full'
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

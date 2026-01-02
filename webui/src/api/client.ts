import axios, { AxiosError } from 'axios'
import type {
  HistoryBar,
  Instrument,
  Tick,
  MethodsMeta,
  VolatilityMethodsMeta,
  DenoiseMethodsMeta,
  DimredMethodsMeta,
  WaveletsResponse,
  SktimeEstimatorsResponse,
  ForecastPayload,
  VolatilityPayload,
  PivotResponse,
  SupportResistanceResponse,
  DenoiseSpecUI,
  ForecastPriceBody,
  ForecastVolBody,
  BacktestBody,
  BacktestResult,
} from '../types'

// Use environment variable or default to empty (same origin)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const baseURL = (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_BASE) || ''

export const api = axios.create({ baseURL })

/**
 * Standardized error extraction from axios errors.
 */
export function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    return error.response?.data?.detail || error.response?.data?.message || error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unknown error occurred'
}

// ============================================================================
// Timeframes & Instruments
// ============================================================================

export async function getTimeframes(): Promise<string[]> {
  const { data } = await api.get<{ timeframes: string[] }>('/api/timeframes')
  return data.timeframes ?? []
}

export async function searchInstruments(search?: string, limit?: number): Promise<Instrument[]> {
  const { data } = await api.get<{ items: Instrument[] }>('/api/instruments', {
    params: { search, limit },
  })
  return data.items ?? []
}

// ============================================================================
// History Data
// ============================================================================

export type HistoryParams = {
  symbol: string
  timeframe: string
  limit: number
  start?: string
  end?: string
  denoise?: DenoiseSpecUI
  include_incomplete?: boolean
}

export async function getHistory(params: HistoryParams): Promise<HistoryBar[]> {
  const query: Record<string, unknown> = {
    symbol: params.symbol,
    timeframe: params.timeframe,
    limit: params.limit,
    start: params.start,
    end: params.end,
    include_incomplete: params.include_incomplete,
  }

  const dn = params.denoise
  if (dn?.method) {
    query.denoise_method = dn.method
    const extras: Record<string, unknown> = {}
    if (dn.params) extras.params = dn.params
    if (dn.columns) extras.columns = dn.columns
    if (dn.when) extras.when = dn.when
    if (dn.causality) extras.causality = dn.causality
    if (typeof dn.keep_original === 'boolean') extras.keep_original = dn.keep_original
    if (Object.keys(extras).length) {
      query.denoise_params = JSON.stringify(extras)
    }
  }

  const { data } = await api.get<{ bars: HistoryBar[] }>('/api/history', { params: query })
  return data.bars ?? []
}

export async function getTick(symbol: string): Promise<Tick> {
  const { data } = await api.get<Tick>('/api/tick', { params: { symbol } })
  return data
}

// ============================================================================
// Forecast Methods Metadata
// ============================================================================

export async function getMethods(): Promise<MethodsMeta> {
  const { data } = await api.get<MethodsMeta>('/api/methods')
  return data
}

export async function getVolatilityMethods(): Promise<VolatilityMethodsMeta> {
  const { data } = await api.get<VolatilityMethodsMeta>('/api/volatility/methods')
  return data
}

export async function getDenoiseMethods(): Promise<DenoiseMethodsMeta> {
  const { data } = await api.get<DenoiseMethodsMeta>('/api/denoise/methods')
  return data
}

export async function getDimredMethods(): Promise<DimredMethodsMeta> {
  const { data } = await api.get<DimredMethodsMeta>('/api/dimred/methods')
  return data
}

export async function getWavelets(): Promise<WaveletsResponse> {
  const { data } = await api.get<WaveletsResponse>('/api/denoise/wavelets')
  return data
}

export async function getSktimeEstimators(): Promise<SktimeEstimatorsResponse> {
  const { data } = await api.get<SktimeEstimatorsResponse>('/api/sktime/estimators')
  return data
}

// ============================================================================
// Forecasting
// ============================================================================

export async function forecastPrice(body: ForecastPriceBody): Promise<ForecastPayload> {
  const { data } = await api.post<ForecastPayload>('/api/forecast/price', body)
  return data
}

export async function forecastVolatility(body: ForecastVolBody): Promise<VolatilityPayload> {
  const { data } = await api.post<VolatilityPayload>('/api/forecast/volatility', body)
  return data
}

export async function runBacktest(body: BacktestBody): Promise<BacktestResult> {
  const { data } = await api.post<BacktestResult>('/api/backtest', body)
  return data
}

// ============================================================================
// Technical Analysis
// ============================================================================

export type PivotParams = {
  symbol: string
  timeframe: string
  method?: 'classic' | 'fibonacci' | 'camarilla' | 'woodie' | 'demark'
}

export async function getPivots(params: PivotParams): Promise<PivotResponse> {
  const { data } = await api.get<PivotResponse>('/api/pivots', { params })
  return data
}

export type SupportResistanceParams = {
  symbol: string
  timeframe: string
  limit?: number
  tolerance_pct?: number
  min_touches?: number
  max_levels?: number
}

export async function getSupportResistance(
  params: SupportResistanceParams
): Promise<SupportResistanceResponse> {
  const { data } = await api.get<SupportResistanceResponse>('/api/support-resistance', { params })
  return data
}

// ============================================================================
// Health Check
// ============================================================================

export async function healthCheck(): Promise<{ service: string; status: string }> {
  const { data } = await api.get<{ service: string; status: string }>('/')
  return data
}

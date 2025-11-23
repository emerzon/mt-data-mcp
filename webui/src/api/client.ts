import axios from 'axios'
import type { DenoiseSpecUI } from '../components/DenoiseModal'
import type { SupportResistanceLevel, VolatilityMethodsMeta } from '../types'

const baseURL = import.meta.env.VITE_API_BASE || ''

export const api = axios.create({ baseURL })

export async function getTimeframes(): Promise<string[]> {
  const { data } = await api.get('/api/timeframes')
  return data.timeframes ?? []
}

export async function searchInstruments(q?: string, limit?: number) {
  const { data } = await api.get('/api/instruments', { params: { search: q, limit } })
  return data.items as { name: string; group?: string; description?: string }[]
}

export async function getHistory(params: { symbol: string; timeframe: string; limit: number; end?: string; denoise?: DenoiseSpecUI }) {
  const query: any = { symbol: params.symbol, timeframe: params.timeframe, limit: params.limit, end: params.end }
  const dn = params.denoise
  if (dn?.method) {
    query.denoise_method = dn.method
    const extras: Record<string, any> = {}
    if (dn.params) extras.params = dn.params
    if (dn.columns) extras.columns = dn.columns
    if (dn.when) extras.when = dn.when
    if (dn.causality) extras.causality = dn.causality
    if (typeof dn.keep_original === 'boolean') extras.keep_original = dn.keep_original
    if (Object.keys(extras).length) query.denoise_params = JSON.stringify(extras)
  }
  const { data } = await api.get('/api/history', { params: query })
  return data.bars as { time: number; open: number; high: number; low: number; close: number }[]
}

export async function getMethods() {
  const { data } = await api.get('/api/methods')
  return data as import('../types').MethodsMeta
}

export async function getVolatilityMethods() {
  const { data } = await api.get('/api/volatility/methods')
  return data as VolatilityMethodsMeta
}

export async function getDenoiseMethods() {
  const { data } = await api.get('/api/denoise/methods')
  return data as { methods: { method: string; description: string; params: { name: string; type: string; default?: any; description?: string }[] }[] }
}

export async function getDimredMethods() {
  const { data } = await api.get('/api/dimred/methods')
  return data as { methods: { method: string; available: boolean; description: string; params: { name: string; type: string; default?: any; description?: string }[] }[] }
}

export async function getWavelets() {
  const { data } = await api.get('/api/denoise/wavelets')
  return data as { available: boolean; families: string[]; wavelets: string[]; by_family: Record<string, string[]> }
}

export async function getSktimeEstimators() {
  const { data } = await api.get('/api/sktime/estimators')
  return data as { available: boolean; estimators: { name: string; class_path: string }[]; error?: string }
}

export async function forecastPrice(body: any) {
  const { data } = await api.post('/api/forecast/price', body)
  return data as import('../types').ForecastPayload
}

export async function forecastVolatility(body: any) {
  const { data } = await api.post('/api/forecast/volatility', body)
  return data
}

export async function getPivots(params: { symbol: string; timeframe: string; method?: string }) {
  const { data } = await api.get('/api/pivots', { params })
  return data as { levels: { level: string; value: number }[]; period?: any; method: string; symbol: string; timeframe: string }
}

export async function getSupportResistance(params: { symbol: string; timeframe: string; limit?: number; tolerance_pct?: number; min_touches?: number; max_levels?: number }) {
  const { data } = await api.get('/api/support-resistance', { params })
  return data as {
    symbol: string
    timeframe: string
    limit: number
    method: string
    tolerance_pct: number
    min_touches: number
    window?: { start?: string | null; end?: string | null }
    levels: SupportResistanceLevel[]
  }
}


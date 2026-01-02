import { useState, useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getMethods, forecastPrice, getVolatilityMethods, forecastVolatility, runBacktest, getErrorMessage, getDimredMethods } from '../api/client'
import type { ForecastPayload, DenoiseSpecUI, ForecastPriceBody, BacktestResult } from '../types'
import { loadJSON, saveJSON } from '../lib/storage'
import { formatDateTime, coerce } from '../lib/utils'
import { DenoiseModal } from './DenoiseModal'

type Props = {
  open: boolean
  onClose: () => void
  symbol: string
  timeframe: string
  anchor?: number
  onResult: (res: ForecastPayload) => void
}

type Tab = 'forecast' | 'volatility' | 'backtest'

export function ForecastPanel({ open, onClose, symbol, timeframe, anchor, onResult }: Props) {
  const [tab, setTab] = useState<Tab>('forecast')

  if (!open) return null

  return (
    <div className="absolute top-0 right-0 bottom-0 w-[420px] bg-slate-900/98 backdrop-blur-sm border-l border-slate-800 z-30 flex flex-col shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <div className="flex gap-1">
          {(['forecast', 'volatility', 'backtest'] as Tab[]).map((t) => (
            <button
              key={t}
              className={`px-3 py-1 text-xs font-medium rounded ${
                tab === t ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
              onClick={() => setTab(t)}
            >
              {t === 'forecast' ? 'Price' : t === 'volatility' ? 'Volatility' : 'Backtest'}
            </button>
          ))}
        </div>
        <button className="text-slate-400 hover:text-slate-200 p-1" onClick={onClose}>
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {tab === 'forecast' && (
          <ForecastTab symbol={symbol} timeframe={timeframe} anchor={anchor} onResult={onResult} />
        )}
        {tab === 'volatility' && (
          <VolatilityTab symbol={symbol} timeframe={timeframe} anchor={anchor} />
        )}
        {tab === 'backtest' && (
          <BacktestTab symbol={symbol} timeframe={timeframe} />
        )}
      </div>
    </div>
  )
}

// ============================================================================
// Forecast Tab
// ============================================================================

const DIMRED_METHODS = new Set([
  'mlf_rf', 'mlf_lightgbm', 'nhits', 'nbeatsx', 'tft', 'patchtst',
  'chronos_bolt', 'timesfm', 'lag_llama', 'gt_deepar', 'gt_sfeedforward',
  'gt_prophet', 'gt_tft', 'gt_wavenet', 'gt_deepnpts', 'gt_mqf2', 'gt_npts', 'ensemble',
])

function ForecastTab({ symbol, timeframe, anchor, onResult }: { symbol: string; timeframe: string; anchor?: number; onResult: (res: ForecastPayload) => void }) {
  const { data: methods } = useQuery({ queryKey: ['methods'], queryFn: getMethods })
  const { data: dimredMethods } = useQuery({ queryKey: ['dimred_methods'], queryFn: getDimredMethods })
  
  const [method, setMethod] = useState('theta')
  const [horizon, setHorizon] = useState(12)
  const [target, setTarget] = useState<'price' | 'return'>('price')
  const [lookback, setLookback] = useState<number | ''>('')
  const [ciAlpha, setCiAlpha] = useState(0.1)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showDenoise, setShowDenoise] = useState(false)
  const [denoise, setDenoise] = useState<DenoiseSpecUI | undefined>()
  const [dimredMethod, setDimredMethod] = useState<string>('')
  const [methodParams, setMethodParams] = useState<Record<string, unknown>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const storageKey = symbol && timeframe ? `fc2:${symbol}:${timeframe}` : null

  // Load saved settings
  useEffect(() => {
    if (!storageKey) return
    const saved = loadJSON<any>(storageKey)
    if (saved) {
      if (saved.method) setMethod(saved.method)
      if (typeof saved.horizon === 'number') setHorizon(saved.horizon)
      if (saved.target) setTarget(saved.target)
      if (typeof saved.lookback === 'number' || saved.lookback === '') setLookback(saved.lookback)
      if (typeof saved.ciAlpha === 'number') setCiAlpha(saved.ciAlpha)
      setDenoise(saved.denoise)
      setDimredMethod(saved.dimredMethod || '')
      setMethodParams(saved.methodParams || {})
    }
  }, [storageKey])

  // Save settings
  useEffect(() => {
    if (!storageKey) return
    saveJSON(storageKey, { method, horizon, target, lookback, ciAlpha, denoise, dimredMethod, methodParams })
  }, [storageKey, method, horizon, target, lookback, ciAlpha, denoise, dimredMethod, methodParams])

  const selectedMeta = useMemo(() => methods?.methods?.find(m => m.method === method), [methods, method])
  const supportsDimred = DIMRED_METHODS.has(method)
  const availableDimred = (dimredMethods?.methods ?? []).filter(m => m.available)

  const run = async (kind: 'full' | 'partial') => {
    if (!symbol) return
    setIsLoading(true)
    setError(null)
    try {
      const body: ForecastPriceBody = {
        symbol,
        timeframe,
        method,
        horizon,
        lookback: lookback === '' ? undefined : Number(lookback),
        ci_alpha: ciAlpha,
        target,
        as_of: kind === 'full' ? undefined : anchor ? formatDateTime(anchor) : undefined,
        params: Object.keys(methodParams).length ? methodParams : undefined,
        denoise,
        dimred_method: supportsDimred && dimredMethod ? dimredMethod : undefined,
      }
      const res = await forecastPrice(body)
      onResult({ ...res, __anchor: kind === 'full' ? undefined : anchor, __kind: kind })
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Method */}
      <div>
        <label className="text-xs text-slate-400 mb-1 block">Method</label>
        <select
          className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
          value={method}
          onChange={(e) => setMethod(e.target.value)}
        >
          {methods?.methods?.map(m => (
            <option key={m.method} value={m.method} disabled={!m.available}>
              {m.method}{!m.available ? ' (unavailable)' : ''}
            </option>
          ))}
        </select>
        {selectedMeta && !selectedMeta.available && (
          <p className="text-xs text-amber-400 mt-1">
            Requires: {selectedMeta.requires?.join(', ') || 'additional dependencies'}
          </p>
        )}
      </div>

      {/* Basic params row */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Horizon</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Target</label>
          <select
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={target}
            onChange={(e) => setTarget(e.target.value as 'price' | 'return')}
          >
            <option value="price">Price</option>
            <option value="return">Return</option>
          </select>
        </div>
      </div>

      {/* Advanced toggle */}
      <button
        className="w-full text-left text-xs text-slate-400 hover:text-slate-300 flex items-center justify-between py-2 border-t border-slate-800"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        <span>Advanced Options</span>
        <span>{showAdvanced ? '−' : '+'}</span>
      </button>

      {showAdvanced && (
        <div className="space-y-3 pb-3 border-b border-slate-800">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 mb-1 block">Lookback</label>
              <input
                type="number"
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={lookback}
                onChange={(e) => setLookback(e.target.value === '' ? '' : Number(e.target.value))}
                placeholder="auto"
                min={50}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 mb-1 block">CI Alpha</label>
              <input
                type="number"
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={ciAlpha}
                onChange={(e) => setCiAlpha(Number(e.target.value))}
                step={0.01}
                min={0}
                max={0.5}
              />
            </div>
          </div>

          {/* Denoising */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">
              Forecast Denoise: <span className="text-slate-300">{denoise?.method || 'None'}</span>
            </span>
            <button className="text-xs text-sky-400 hover:text-sky-300" onClick={() => setShowDenoise(true)}>
              Configure
            </button>
          </div>

          {/* Dimred */}
          {supportsDimred && (
            <div>
              <label className="text-xs text-slate-400 mb-1 block">Dim. Reduction</label>
              <select
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={dimredMethod}
                onChange={(e) => setDimredMethod(e.target.value)}
              >
                <option value="">None</option>
                {availableDimred.map(m => (
                  <option key={m.method} value={m.method}>{m.method}</option>
                ))}
              </select>
            </div>
          )}

          {/* Method params */}
          {selectedMeta?.params && selectedMeta.params.length > 0 && (
            <div>
              <div className="text-xs text-slate-400 mb-2">Method Parameters</div>
              <div className="grid grid-cols-2 gap-2">
                {selectedMeta.params.map(p => (
                  <div key={p.name}>
                    <label className="text-xs text-slate-500 mb-0.5 block">{p.name}</label>
                    <input
                      className="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                      value={String(methodParams[p.name] ?? '')}
                      onChange={(e) => setMethodParams({ ...methodParams, [p.name]: coerce(e.target.value) })}
                      placeholder={String(p.default ?? '')}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        <button
          className="flex-1 bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
          onClick={() => run('full')}
          disabled={!symbol || !selectedMeta?.available || isLoading}
        >
          {isLoading ? 'Running...' : 'Full Forecast'}
        </button>
        <button
          className="flex-1 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
          onClick={() => run('partial')}
          disabled={!symbol || !selectedMeta?.available || !anchor || isLoading}
        >
          From Anchor
        </button>
      </div>

      <DenoiseModal
        open={showDenoise}
        title="Forecast Denoising"
        value={denoise}
        onClose={() => setShowDenoise(false)}
        onApply={(d) => { setDenoise(d); setShowDenoise(false) }}
      />
    </div>
  )
}

// ============================================================================
// Volatility Tab
// ============================================================================

function VolatilityTab({ symbol, timeframe, anchor }: { symbol: string; timeframe: string; anchor?: number }) {
  const { data: methods } = useQuery({ queryKey: ['vol_methods'], queryFn: getVolatilityMethods })
  
  const [method, setMethod] = useState('ewma')
  const [horizon, setHorizon] = useState(12)
  const [proxy, setProxy] = useState('squared_return')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)

  const run = async () => {
    if (!symbol) return
    setIsLoading(true)
    setError(null)
    try {
      const res = await forecastVolatility({
        symbol,
        timeframe,
        method,
        horizon,
        proxy,
        as_of: anchor ? formatDateTime(anchor) : undefined,
      })
      setResult(res)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs text-slate-400 mb-1 block">Method</label>
        <select
          className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
          value={method}
          onChange={(e) => setMethod(e.target.value)}
        >
          {methods?.methods?.map(m => (
            <option key={m.method} value={m.method} disabled={!m.available}>
              {m.method}{!m.available ? ' (unavailable)' : ''}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Horizon</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Proxy</label>
          <select
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={proxy}
            onChange={(e) => setProxy(e.target.value)}
          >
            <option value="squared_return">Squared Return</option>
            <option value="abs_return">Abs Return</option>
            <option value="log_r2">Log R²</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      <button
        className="w-full bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
        onClick={run}
        disabled={!symbol || isLoading}
      >
        {isLoading ? 'Running...' : 'Run Volatility Forecast'}
      </button>

      {result && (
        <div className="bg-slate-800/50 rounded-lg p-3 text-sm">
          <div className="text-slate-400 text-xs mb-2">Result</div>
          <div className="text-slate-200">
            Annualized Vol: <span className="font-mono">{(result.annualized_vol * 100).toFixed(2)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Backtest Tab
// ============================================================================

function BacktestTab({ symbol, timeframe }: { symbol: string; timeframe: string }) {
  const { data: methods } = useQuery({ queryKey: ['methods'], queryFn: getMethods })
  
  const [selectedMethods, setSelectedMethods] = useState<string[]>(['theta'])
  const [horizon, setHorizon] = useState(12)
  const [steps, setSteps] = useState(5)
  const [spacing, setSpacing] = useState(20)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)

  const availableMethods = useMemo(() => methods?.methods?.filter(m => m.available) ?? [], [methods])

  const toggleMethod = (m: string) => {
    setSelectedMethods(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m])
  }

  const run = async () => {
    if (!symbol || !selectedMethods.length) return
    setIsLoading(true)
    setError(null)
    try {
      const res = await runBacktest({ symbol, timeframe, horizon, steps, spacing, methods: selectedMethods })
      setResult(res)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Horizon</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Steps</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Spacing</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={spacing}
            onChange={(e) => setSpacing(Number(e.target.value))}
            min={1}
          />
        </div>
      </div>

      <div>
        <div className="text-xs text-slate-400 mb-2">Methods to compare</div>
        <div className="flex flex-wrap gap-1">
          {availableMethods.map(m => (
            <button
              key={m.method}
              className={`px-2 py-1 text-xs rounded ${
                selectedMethods.includes(m.method)
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
              onClick={() => toggleMethod(m.method)}
            >
              {m.method}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/50 border border-rose-800 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      <button
        className="w-full bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded-lg disabled:opacity-50"
        onClick={run}
        disabled={!symbol || !selectedMethods.length || isLoading}
      >
        {isLoading ? 'Running Backtest...' : 'Run Backtest'}
      </button>

      {result && (
        <div className="bg-slate-800/50 rounded-lg overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left px-2 py-2">Method</th>
                <th className="text-right px-2 py-2">MAPE</th>
                <th className="text-right px-2 py-2">Dir%</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map(r => (
                <tr key={r.method} className="border-b border-slate-700/50">
                  <td className="px-2 py-1.5 text-slate-200">{r.method}</td>
                  <td className="text-right px-2 py-1.5 text-slate-300 font-mono">
                    {r.mape?.toFixed(2) ?? '-'}
                  </td>
                  <td className={`text-right px-2 py-1.5 font-mono ${
                    (r.direction_accuracy ?? 0) >= 60 ? 'text-emerald-400' :
                    (r.direction_accuracy ?? 0) >= 50 ? 'text-amber-400' : 'text-rose-400'
                  }`}>
                    {r.direction_accuracy?.toFixed(0) ?? '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

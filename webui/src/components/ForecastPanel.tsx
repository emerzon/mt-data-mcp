import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  getDimredMethods,
  getVolatilityMethods,
  forecastVolatility,
  runBacktest,
  getErrorMessage,
} from '../api/client'
import { useForecast, useForecastMethods, useForecastSettings } from '../hooks/useForecast'
import type { BacktestResult, ForecastPayload } from '../types'
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
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <div className="flex gap-1">
          {(['forecast', 'volatility', 'backtest'] as Tab[]).map((item) => (
            <button
              key={item}
              className={`px-3 py-1 text-xs font-medium rounded ${
                tab === item ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
              onClick={() => setTab(item)}
            >
              {item === 'forecast' ? 'Price' : item === 'volatility' ? 'Volatility' : 'Backtest'}
            </button>
          ))}
        </div>
        <button className="text-slate-400 hover:text-slate-200 p-1" onClick={onClose}>
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {tab === 'forecast' && (
          <ForecastTab symbol={symbol} timeframe={timeframe} anchor={anchor} onResult={onResult} />
        )}
        {tab === 'volatility' && <VolatilityTab symbol={symbol} timeframe={timeframe} anchor={anchor} />}
        {tab === 'backtest' && <BacktestTab symbol={symbol} timeframe={timeframe} />}
      </div>
    </div>
  )
}

function ForecastTab({
  symbol,
  timeframe,
  anchor,
  onResult,
}: {
  symbol: string
  timeframe: string
  anchor?: number
  onResult: (res: ForecastPayload) => void
}) {
  const { methods } = useForecastMethods()
  const { settings, setSettings, supportsDimred } = useForecastSettings(symbol, timeframe)
  const { data: dimredMethods } = useQuery({ queryKey: ['dimred_methods'], queryFn: getDimredMethods })
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showDenoise, setShowDenoise] = useState(false)
  const { run, isLoading, error } = useForecast(symbol, timeframe, settings, onResult)

  const selectedMeta = useMemo(
    () => methods.find((method) => method.method === settings.method),
    [methods, settings.method]
  )
  const availableDimred = (dimredMethods?.methods ?? []).filter((method) => method.available)

  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs text-slate-400 mb-1 block">Method</label>
        <select
          className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
          value={settings.method}
          onChange={(event) =>
            setSettings((previous) => ({
              ...previous,
              method: event.target.value,
            }))
          }
        >
          {methods.map((method) => (
            <option key={method.method} value={method.method} disabled={!method.available}>
              {method.method}
              {!method.available ? ' (unavailable)' : ''}
            </option>
          ))}
        </select>
        {selectedMeta && !selectedMeta.available && (
          <p className="text-xs text-amber-400 mt-1">
            Requires: {selectedMeta.requires?.join(', ') || 'additional dependencies'}
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Horizon</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={settings.horizon}
            onChange={(event) =>
              setSettings((previous) => ({
                ...previous,
                horizon: Number(event.target.value),
              }))
            }
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Quantity</label>
          <select
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={settings.quantity}
            onChange={(event) =>
              setSettings((previous) => ({
                ...previous,
                quantity: event.target.value as 'price' | 'return',
              }))
            }
          >
            <option value="price">Price</option>
            <option value="return">Return</option>
          </select>
        </div>
      </div>

      <button
        className="w-full text-left text-xs text-slate-400 hover:text-slate-300 flex items-center justify-between py-2 border-t border-slate-800"
        onClick={() => setShowAdvanced((value) => !value)}
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
                value={settings.lookback}
                onChange={(event) =>
                  setSettings((previous) => ({
                    ...previous,
                    lookback: event.target.value === '' ? '' : Number(event.target.value),
                  }))
                }
                placeholder="auto"
                min={50}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 mb-1 block">CI Alpha</label>
              <input
                type="number"
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={settings.ci_alpha}
                onChange={(event) =>
                  setSettings((previous) => ({
                    ...previous,
                    ci_alpha: Number(event.target.value),
                  }))
                }
                step={0.01}
                min={0}
                max={0.5}
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">
              Forecast Denoise: <span className="text-slate-300">{settings.denoise?.method || 'None'}</span>
            </span>
            <button className="text-xs text-sky-400 hover:text-sky-300" onClick={() => setShowDenoise(true)}>
              Configure
            </button>
          </div>

          {supportsDimred && (
            <div>
              <label className="text-xs text-slate-400 mb-1 block">Dim. Reduction</label>
              <select
                className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
                value={settings.dimredMethod ?? ''}
                onChange={(event) =>
                  setSettings((previous) => ({
                    ...previous,
                    dimredMethod: event.target.value || undefined,
                  }))
                }
              >
                <option value="">None</option>
                {availableDimred.map((method) => (
                  <option key={method.method} value={method.method}>
                    {method.method}
                  </option>
                ))}
              </select>
            </div>
          )}

          {selectedMeta?.params && selectedMeta.params.length > 0 && (
            <div>
              <div className="text-xs text-slate-400 mb-2">Method Parameters</div>
              <div className="grid grid-cols-2 gap-2">
                {selectedMeta.params.map((param) => (
                  <div key={param.name}>
                    <label className="text-xs text-slate-500 mb-0.5 block">{param.name}</label>
                    <input
                      className="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1.5 border border-slate-700"
                      value={String(settings.params[param.name] ?? '')}
                      onChange={(event) =>
                        setSettings((previous) => ({
                          ...previous,
                          params: {
                            ...previous.params,
                            [param.name]: coerce(event.target.value),
                          },
                        }))
                      }
                      placeholder={String(param.default ?? '')}
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
          onClick={() => run('partial', anchor)}
          disabled={!symbol || !selectedMeta?.available || !anchor || isLoading}
        >
          From Anchor
        </button>
      </div>

      <DenoiseModal
        open={showDenoise}
        title="Forecast Denoising"
        value={settings.denoise}
        onClose={() => setShowDenoise(false)}
        onApply={(denoise) => {
          setSettings((previous) => ({
            ...previous,
            denoise,
          }))
          setShowDenoise(false)
        }}
      />
    </div>
  )
}

function VolatilityTab({ symbol, timeframe, anchor }: { symbol: string; timeframe: string; anchor?: number }) {
  const { data: methods } = useQuery({ queryKey: ['vol_methods'], queryFn: getVolatilityMethods })

  const [method, setMethod] = useState('ewma')
  const [horizon, setHorizon] = useState(12)
  const [proxy, setProxy] = useState('squared_return')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ annualized_vol?: number } | null>(null)

  const run = async () => {
    if (!symbol) return
    setIsLoading(true)
    setError(null)
    try {
      const response = await forecastVolatility({
        symbol,
        timeframe,
        method,
        horizon,
        proxy,
        as_of: anchor ? formatDateTime(anchor) : undefined,
      })
      setResult(response)
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
          onChange={(event) => setMethod(event.target.value)}
        >
          {methods?.methods?.map((item) => (
            <option key={item.method} value={item.method} disabled={!item.available}>
              {item.method}
              {!item.available ? ' (unavailable)' : ''}
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
            onChange={(event) => setHorizon(Number(event.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Proxy</label>
          <select
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-3 py-2 border border-slate-700"
            value={proxy}
            onChange={(event) => setProxy(event.target.value)}
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
            Annualized Vol: <span className="font-mono">{((result.annualized_vol ?? 0) * 100).toFixed(2)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

function BacktestTab({ symbol, timeframe }: { symbol: string; timeframe: string }) {
  const { methods } = useForecastMethods()

  const [selectedMethods, setSelectedMethods] = useState<string[]>(['theta'])
  const [horizon, setHorizon] = useState(12)
  const [steps, setSteps] = useState(5)
  const [spacing, setSpacing] = useState(20)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)

  const availableMethods = useMemo(() => methods.filter((method) => method.available), [methods])

  const toggleMethod = (method: string) => {
    setSelectedMethods((previous) =>
      previous.includes(method) ? previous.filter((item) => item !== method) : [...previous, method]
    )
  }

  const run = async () => {
    if (!symbol || !selectedMethods.length) return
    setIsLoading(true)
    setError(null)
    try {
      const response = await runBacktest({ symbol, timeframe, horizon, steps, spacing, methods: selectedMethods })
      setResult(response)
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
            onChange={(event) => setHorizon(Number(event.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Steps</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-slate-400 mb-1 block">Spacing</label>
          <input
            type="number"
            className="w-full bg-slate-800 text-slate-200 text-sm rounded-lg px-2 py-2 border border-slate-700"
            value={spacing}
            onChange={(event) => setSpacing(Number(event.target.value))}
            min={1}
          />
        </div>
      </div>

      <div>
        <div className="text-xs text-slate-400 mb-2">Methods to compare</div>
        <div className="flex flex-wrap gap-1">
          {availableMethods.map((method) => (
            <button
              key={method.method}
              className={`px-2 py-1 text-xs rounded ${
                selectedMethods.includes(method.method)
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
              onClick={() => toggleMethod(method.method)}
            >
              {method.method}
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
              {result.results.map((item) => (
                <tr key={item.method} className="border-b border-slate-700/50">
                  <td className="px-2 py-1.5 text-slate-200">{item.method}</td>
                  <td className="text-right px-2 py-1.5 text-slate-300 font-mono">
                    {item.mape?.toFixed(2) ?? '-'}
                  </td>
                  <td
                    className={`text-right px-2 py-1.5 font-mono ${
                      (item.direction_accuracy ?? 0) >= 60
                        ? 'text-emerald-400'
                        : (item.direction_accuracy ?? 0) >= 50
                          ? 'text-amber-400'
                          : 'text-rose-400'
                    }`}
                  >
                    {item.direction_accuracy?.toFixed(0) ?? '-'}
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

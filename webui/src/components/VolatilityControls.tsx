import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { forecastVolatility, getVolatilityMethods, getErrorMessage } from '../api/client'
import type { VolatilityMethodsMeta, ParamDef } from '../types'
import { loadJSON, saveJSON } from '../lib/storage'
import { parseParamValue, stringifyValue, isEmptyValue, formatDateTime } from '../lib/utils'

type Props = {
  symbol?: string
  timeframe: string
  anchor?: number
  onResult: (res: unknown) => void
}

type ParamsMap = Record<string, Record<string, unknown>>

function buildDefaults(meta?: VolatilityMethodsMeta['methods'][number]): Record<string, unknown> {
  if (!meta) return {}
  const out: Record<string, unknown> = {}
  for (const param of meta.params || []) {
    if (param.default !== undefined && param.default !== null) {
      out[param.name] = param.default
    }
  }
  return out
}

export function VolatilityControls({ symbol, timeframe, anchor, onResult }: Props) {
  const { data: methodsData, refetch: refetchMethods } = useQuery({
    queryKey: ['volMethods'],
    queryFn: getVolatilityMethods,
  })

  const methods = methodsData?.methods ?? []

  const [method, setMethod] = useState<string>('')
  const [horizon, setHorizon] = useState<number>(12)
  const [proxy, setProxy] = useState<string>('squared_return')
  const [paramsByMethod, setParamsByMethod] = useState<ParamsMap>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const storageKey = symbol && timeframe ? `vol:${symbol}:${timeframe}` : null

  // Load saved preferences when symbol/timeframe changes
  useEffect(() => {
    if (!storageKey) {
      setMethod('')
      setParamsByMethod({})
      setHorizon(12)
      setProxy('squared_return')
      return
    }
    const saved = loadJSON<{
      method?: string
      horizon?: number
      proxy?: string
      paramsByMethod?: ParamsMap
    }>(storageKey)
    if (saved) {
      if (typeof saved.method === 'string') setMethod(saved.method)
      if (typeof saved.horizon === 'number') setHorizon(saved.horizon)
      if (typeof saved.proxy === 'string') setProxy(saved.proxy)
      if (saved.paramsByMethod && typeof saved.paramsByMethod === 'object') {
        setParamsByMethod(saved.paramsByMethod)
      } else {
        setParamsByMethod({})
      }
    } else {
      setMethod('')
      setParamsByMethod({})
      setHorizon(12)
      setProxy('squared_return')
    }
  }, [storageKey])

  // Ensure a valid method is selected once metadata arrives
  useEffect(() => {
    if (!methods.length) return
    setMethod(prev => {
      if (prev && methods.some(m => m.method === prev)) return prev
      const preferred = methods.find(m => m.available)?.method ?? methods[0].method
      return preferred
    })
  }, [methods])

  const selectedMeta = useMemo(() => methods.find(m => m.method === method), [methods, method])

  // Seed defaults for the active method if absent
  useEffect(() => {
    if (!method || !selectedMeta) return
    setParamsByMethod(prev => {
      if (prev[method]) return prev
      const defaults = buildDefaults(selectedMeta)
      if (Object.keys(defaults).length === 0) {
        return prev
      }
      return { ...prev, [method]: defaults }
    })
  }, [method, selectedMeta])

  // Persist preferences
  useEffect(() => {
    if (!storageKey || !method) return
    const payload = { method, horizon, proxy, paramsByMethod }
    saveJSON(storageKey, payload)
  }, [storageKey, method, horizon, proxy, paramsByMethod])

  const activeParams = paramsByMethod[method] ?? {}

  const advancedSummary = useMemo(() => {
    if (!selectedMeta) return 'None'
    const entries: string[] = []
    for (const param of selectedMeta.params || []) {
      const value = activeParams[param.name]
      if (isEmptyValue(value)) continue
      if (Array.isArray(value)) {
        entries.push(`${param.name}=[${value.join(', ')}]`)
      } else {
        entries.push(`${param.name}=${value}`)
      }
    }
    return entries.length ? entries.join(' | ') : 'Defaults'
  }, [selectedMeta, activeParams])

  function updateParam(name: string, rawValue: string, param: ParamDef) {
    setParamsByMethod(prev => {
      const current = { ...(prev[method] ?? {}) }
      const parsed = parseParamValue(rawValue, param.type)
      if (isEmptyValue(parsed)) {
        delete current[name]
      } else {
        current[name] = parsed
      }
      return { ...prev, [method]: current }
    })
  }

  function toggleBoolean(name: string, value: boolean) {
    setParamsByMethod(prev => {
      const current = { ...(prev[method] ?? {}) }
      current[name] = value
      return { ...prev, [method]: current }
    })
  }

  const canRun = Boolean(symbol) && selectedMeta?.available !== false

  async function run() {
    if (!symbol || !selectedMeta) return

    setIsLoading(true)
    setError(null)

    try {
      const paramsRaw = paramsByMethod[method] ?? {}
      const params: Record<string, unknown> = {}
      for (const [key, value] of Object.entries(paramsRaw)) {
        if (isEmptyValue(value)) continue
        params[key] = value
      }
      const body = {
        symbol,
        timeframe,
        method,
        horizon,
        proxy,
        params,
        as_of:
          anchor && !Number.isNaN(anchor)
            ? formatDateTime(anchor)
            : undefined,
      }
      const res = await forecastVolatility(body)
      onResult(res)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col">
          <span className="label">Method</span>
          <div className="flex gap-2 items-center">
            <select className="select w-48" value={method} onChange={e => setMethod(e.target.value)}>
              {methods.map(m => (
                <option
                  key={m.method}
                  value={m.method}
                  disabled={!m.available}
                  title={!m.available && m.requires?.length ? `Requires: ${m.requires.join(', ')}` : undefined}
                >
                  {m.method}
                  {!m.available ? ' (unavailable)' : ''}
                </option>
              ))}
            </select>
            <button type="button" className="btn" onClick={() => refetchMethods()}>
              Refresh
            </button>
          </div>
        </label>
        <label className="flex flex-col">
          <span className="label">Horizon</span>
          <input
            className="input w-24"
            type="number"
            min={1}
            value={horizon}
            onChange={e => setHorizon(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col">
          <span className="label">Proxy</span>
          <select className="select w-40" value={proxy} onChange={e => setProxy(e.target.value)}>
            <option value="squared_return">squared_return</option>
            <option value="abs_return">abs_return</option>
            <option value="log_r2">log_r2</option>
          </select>
        </label>
        <button className="btn ml-auto" disabled={!canRun || isLoading} onClick={run}>
          {isLoading ? 'Running...' : 'Run Vol Forecast'}
        </button>
      </div>

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/30 border border-rose-800 rounded-md px-3 py-2">
          {error}
        </div>
      )}

      {!selectedMeta?.available && (
        <div className="text-xs text-amber-400">
          {selectedMeta?.method} is not available.{' '}
          {selectedMeta?.requires?.length ? `Requires: ${selectedMeta.requires.join(', ')}` : 'Install the optional dependency.'}
        </div>
      )}
      {selectedMeta?.description && <div className="text-xs text-slate-400 max-w-3xl">{selectedMeta.description}</div>}

      {selectedMeta && (
        <details
          className="border border-slate-800 rounded-md"
          open={selectedMeta.params?.length ? advancedSummary !== 'Defaults' : false}
        >
          <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium text-slate-200 hover:bg-slate-800 flex justify-between items-center">
            <span>Method Parameters</span>
            <span className="text-xs text-slate-400">{selectedMeta.params?.length ? advancedSummary : 'None'}</span>
          </summary>
          {selectedMeta.params?.length ? (
            <div className="p-3 grid gap-3 sm:grid-cols-2">
              {selectedMeta.params.map(param => {
                const lower = param.type?.toLowerCase() ?? ''
                const value = activeParams[param.name]
                if (lower === 'bool') {
                  const boolVal = typeof value === 'boolean' ? value : value === 'true'
                  return (
                    <label key={param.name} className="flex flex-col gap-1" title={param.description || ''}>
                      <span className="label">{param.name}</span>
                      <select
                        className="select"
                        value={boolVal ? 'true' : 'false'}
                        onChange={e => toggleBoolean(param.name, e.target.value === 'true')}
                      >
                        <option value="true">true</option>
                        <option value="false">false</option>
                      </select>
                      {param.description && <span className="text-xs text-slate-400">{param.description}</span>}
                    </label>
                  )
                }
                return (
                  <label key={param.name} className="flex flex-col gap-1" title={param.description || ''}>
                    <span className="label">
                      {param.name}
                      {param.type && <em className="not-italic text-slate-500"> ({param.type})</em>}
                    </span>
                    <input
                      className="input"
                      value={stringifyValue(value, param.type)}
                      onChange={e => updateParam(param.name, e.target.value, param)}
                      placeholder={param.default !== undefined && param.default !== null ? stringifyValue(param.default, param.type) : ''}
                    />
                    {param.description && <span className="text-xs text-slate-400">{param.description}</span>}
                  </label>
                )
              })}
            </div>
          ) : (
            <div className="p-3 text-xs text-slate-500">Selected method has no configurable parameters.</div>
          )}
        </details>
      )}
    </div>
  )
}

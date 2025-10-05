import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { forecastVolatility, getVolatilityMethods } from '../api/client'
import type { VolatilityMethodsMeta } from '../types'
import { loadJSON, saveJSON } from '../lib/storage'

type Props = {
  symbol?: string
  timeframe: string
  anchor?: number
  onResult: (res: any) => void
}

type ParamDef = {
  name: string
  type: string
  default?: any
  description?: string
}

type ParamsMap = Record<string, Record<string, any>>

function buildDefaults(meta?: VolatilityMethodsMeta['methods'][number]): Record<string, any> {
  if (!meta) return {}
  const out: Record<string, any> = {}
  for (const param of meta.params || []) {
    if (param.default !== undefined && param.default !== null) {
      out[param.name] = param.default
    }
  }
  return out
}

function stringifyValue(value: any, type?: string): string {
  if (value === undefined || value === null) return ''
  const lower = type?.toLowerCase() ?? ''
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return value.join(', ')
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : ''
  if (lower.startsWith('json') && typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch {
      return ''
    }
  }
  return String(value)
}

function parseParamValue(input: string, type?: string) {
  const raw = input.trim()
  if (!raw) {
    return ''
  }
  const t = (type || '').toLowerCase()
  if (t === 'int') {
    const v = Number.parseInt(raw, 10)
    return Number.isNaN(v) ? raw : v
  }
  if (t === 'float') {
    const v = Number.parseFloat(raw)
    return Number.isNaN(v) ? raw : v
  }
  if (t === 'bool') {
    if (raw === 'true' || raw === '1') return true
    if (raw === 'false' || raw === '0') return false
    return raw
  }
  if (t.startsWith('list')) {
    const inner = t.includes('[') ? t.slice(t.indexOf('[') + 1, t.lastIndexOf(']')).trim() : 'str'
    const parts = raw.split(/[\s,]+/).map((p) => p.trim()).filter(Boolean)
    if (inner === 'int') {
      const vals = parts.map((p) => Number.parseInt(p, 10)).filter((v) => !Number.isNaN(v))
      return vals
    }
    if (inner === 'float') {
      const vals = parts.map((p) => Number.parseFloat(p)).filter((v) => !Number.isNaN(v))
      return vals
    }
    return parts
  }
  if (t === 'json') {
    try {
      return JSON.parse(raw)
    } catch {
      return raw
    }
  }
  return raw
}

function isEmptyParamValue(value: any): boolean {
  if (value === '' || value === undefined || value === null) return true
  if (Array.isArray(value)) return value.length === 0
  return false
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
    const saved = loadJSON<any>(storageKey)
    if (saved) {
      if (typeof saved.method === 'string') setMethod(saved.method)
      if (typeof saved.horizon === 'number') setHorizon(saved.horizon)
      if (typeof saved.proxy === 'string') setProxy(saved.proxy)
      if (saved.paramsByMethod && typeof saved.paramsByMethod === 'object') {
        setParamsByMethod(saved.paramsByMethod as ParamsMap)
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
    setMethod((prev) => {
      if (prev && methods.some((m) => m.method === prev)) return prev
      const preferred = methods.find((m) => m.available)?.method ?? methods[0].method
      return preferred
    })
  }, [methods])

  const selectedMeta = useMemo(() => methods.find((m) => m.method === method), [methods, method])

  // Seed defaults for the active method if absent
  useEffect(() => {
    if (!method || !selectedMeta) return
    setParamsByMethod((prev) => {
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
    const payload = {
      method,
      horizon,
      proxy,
      paramsByMethod,
    }
    saveJSON(storageKey, payload)
  }, [storageKey, method, horizon, proxy, paramsByMethod])

  const activeParams = paramsByMethod[method] ?? {}

  const advancedSummary = useMemo(() => {
    if (!selectedMeta) return 'None'
    const entries: string[] = []
    for (const param of selectedMeta.params || []) {
      const value = activeParams[param.name]
      if (isEmptyParamValue(value)) continue
      if (Array.isArray(value)) {
        entries.push(`${param.name}=[${value.join(', ')}]`)
      } else {
        entries.push(`${param.name}=${value}`)
      }
    }
    return entries.length ? entries.join(' | ') : 'Defaults'
  }, [selectedMeta, activeParams])

  function updateParam(name: string, rawValue: string, param: ParamDef) {
    setParamsByMethod((prev) => {
      const current = { ...(prev[method] ?? {}) }
      const parsed = parseParamValue(rawValue, param.type)
      if (isEmptyParamValue(parsed)) {
        delete current[name]
      } else {
        current[name] = parsed
      }
      return { ...prev, [method]: current }
    })
  }

  function toggleBoolean(name: string, value: boolean) {
    setParamsByMethod((prev) => {
      const current = { ...(prev[method] ?? {}) }
      current[name] = value
      return { ...prev, [method]: current }
    })
  }

  const canRun = Boolean(symbol) && selectedMeta?.available !== false

  async function run() {
    if (!symbol || !selectedMeta) return
    const paramsRaw = paramsByMethod[method] ?? {}
    const params: Record<string, any> = {}
    for (const [key, value] of Object.entries(paramsRaw)) {
      if (isEmptyParamValue(value)) continue
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
          ? new Date(anchor * 1000).toISOString().slice(0, 19).replace('T', ' ')
          : undefined,
    }
    const res = await forecastVolatility(body)
    onResult(res)
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col">
          <span className="label">Method</span>
          <div className="flex gap-2 items-center">
            <select className="select w-48" value={method} onChange={(e) => setMethod(e.target.value)}>
              {methods.map((m) => (
                <option key={m.method} value={m.method} disabled={!m.available} title={!m.available && m.requires?.length ? `Requires: ${m.requires.join(', ')}` : undefined}>
                  {m.method}{!m.available ? ' (unavailable)' : ''}
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
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col">
          <span className="label">Proxy</span>
          <select className="select w-40" value={proxy} onChange={(e) => setProxy(e.target.value)}>
            <option value="squared_return">squared_return</option>
            <option value="abs_return">abs_return</option>
            <option value="log_r2">log_r2</option>
          </select>
        </label>
        <button className="btn ml-auto" disabled={!canRun} onClick={run}>
          Run Vol Forecast
        </button>
      </div>
      {!selectedMeta?.available && (
        <div className="text-xs text-amber-400">
          {selectedMeta?.method} is not available.{' '}
          {selectedMeta?.requires?.length ? `Requires: ${selectedMeta.requires.join(', ')}` : 'Install the optional dependency.'}
        </div>
      )}
      {selectedMeta?.description && (
        <div className="text-xs text-slate-400 max-w-3xl">{selectedMeta.description}</div>
      )}
      {selectedMeta && (
        <details className="border border-slate-800 rounded-md" open={selectedMeta.params?.length ? advancedSummary !== 'Defaults' : false}>
          <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium text-slate-200 hover:bg-slate-800 flex justify-between items-center">
            <span>Method Parameters</span>
            <span className="text-xs text-slate-400">{selectedMeta.params?.length ? advancedSummary : 'None'}</span>
          </summary>
          {selectedMeta.params?.length ? (
            <div className="p-3 grid gap-3 sm:grid-cols-2">
              {selectedMeta.params.map((param) => {
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
                        onChange={(e) => toggleBoolean(param.name, e.target.value === 'true')}
                      >
                        <option value="true">true</option>
                        <option value="false">false</option>
                      </select>
                      {param.description ? <span className="text-xs text-slate-400">{param.description}</span> : null}
                    </label>
                  )
                }
                return (
                  <label key={param.name} className="flex flex-col gap-1" title={param.description || ''}>
                    <span className="label">{param.name}{param.type ? <em className="not-italic text-slate-500"> ({param.type})</em> : null}</span>
                    <input
                      className="input"
                      value={stringifyValue(value, param.type)}
                      onChange={(e) => updateParam(param.name, e.target.value, param)}
                      placeholder={param.default !== undefined && param.default !== null ? stringifyValue(param.default, param.type) : ''}
                    />
                    {param.description ? <span className="text-xs text-slate-400">{param.description}</span> : null}
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

import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getMethods, forecastPrice } from '../api/client'
import type { ForecastPayload, MethodsMeta } from '../types'
import { AdvancedPanel } from './AdvancedPanel'
import type { DenoiseSpecUI } from './DenoiseModal'
import { loadJSON, saveJSON } from '../lib/storage'

const DIMRED_METHODS = new Set<string>([
  'mlf_rf',
  'mlf_lightgbm',
  'nhits',
  'nbeatsx',
  'tft',
  'patchtst',
  'chronos_bolt',
  'timesfm',
  'lag_llama',
  'gt_deepar',
  'gt_sfeedforward',
  'gt_prophet',
  'gt_tft',
  'gt_wavenet',
  'gt_deepnpts',
  'gt_mqf2',
  'gt_npts',
  'ensemble',
])

export function ForecastControls({ symbol, timeframe, anchor, onResult }: { symbol?: string; timeframe: string; anchor?: number; onResult: (res: ForecastPayload) => void }) {
  const { data: methods, refetch: refetchMethods } = useQuery({ queryKey: ['methods'], queryFn: getMethods, refetchOnWindowFocus: true })
  const defaultMethod = useMemo(() => methods?.methods?.find((m) => m.method === 'theta')?.method || methods?.methods?.[0]?.method || 'theta', [methods])
  const [method, setMethod] = useState<string>(defaultMethod)
  const [horizon, setHorizon] = useState<number>(12)
  const [ci, setCi] = useState<number>(0.1)
  const [target, setTarget] = useState<'price' | 'return'>('price')
  const [lookback, setLookback] = useState<number | ''>('')
  const [methodParams, setMethodParams] = useState<any>({})
  const [denoise, setDenoise] = useState<DenoiseSpecUI | undefined>(undefined)
  const [dimredMethod, setDimredMethod] = useState<string | undefined>(undefined)
  const [dimredParams, setDimredParams] = useState<any>(undefined)

  useEffect(() => {
    if (defaultMethod) setMethod(defaultMethod)
  }, [defaultMethod])

  const selectedMeta = useMemo(() => methods?.methods?.find((m) => m.method === method), [methods, method])
  const canRun = !!symbol && !!selectedMeta?.available
  const supportsDimred = DIMRED_METHODS.has(method)

  const advancedSummary = [
    denoise?.method ? `denoise:${denoise.method}` : null,
    supportsDimred && dimredMethod ? `dimred:${dimredMethod}` : null,
  ]
    .filter(Boolean)
    .join(' • ') || 'None'

  useEffect(() => {
    if (!symbol || !timeframe) return
    const key = `fc:${symbol}:${timeframe}`
    const saved = loadJSON<any>(key)
    if (!saved) return
    if (saved.method) setMethod(saved.method)
    if (typeof saved.horizon === 'number') setHorizon(saved.horizon)
    if (typeof saved.ci === 'number') setCi(saved.ci)
    if (saved.target) setTarget(saved.target)
    if (saved.lookback === '' || typeof saved.lookback === 'number') setLookback(saved.lookback)
    if (saved.methodParams) setMethodParams(saved.methodParams)
    setDenoise(saved.denoise)
    setDimredMethod(saved.dimredMethod)
    setDimredParams(saved.dimredParams)
  }, [symbol, timeframe])

  useEffect(() => {
    if (!symbol || !timeframe) return
    const key = `fc:${symbol}:${timeframe}`
    const payload = { method, horizon, ci, target, lookback, methodParams, denoise, dimredMethod, dimredParams }
    saveJSON(key, payload)
  }, [symbol, timeframe, method, horizon, ci, target, lookback, methodParams, denoise, dimredMethod, dimredParams])

  useEffect(() => {
    if (!supportsDimred) {
      setDimredMethod(undefined)
      setDimredParams(undefined)
    }
  }, [supportsDimred])

  async function run(kind: 'full' | 'partial' | 'backtest') {
    if (!symbol) return
    const body = {
      symbol,
      timeframe,
      method,
      horizon,
      lookback: lookback === '' ? undefined : Number(lookback),
      ci_alpha: ci,
      target,
      as_of: kind === 'full' ? undefined : anchor ? new Date(anchor * 1000).toISOString().slice(0, 19).replace('T', ' ') : undefined,
      params: methodParams,
      denoise,
      dimred_method: supportsDimred ? dimredMethod : undefined,
      dimred_params: supportsDimred ? dimredParams : undefined,
    }
    const res = await forecastPrice(body)
    onResult({ ...res, __anchor: kind === 'full' ? undefined : anchor, __kind: kind } as any)
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col">
          <span className="label">Method</span>
          <div className="flex gap-2 items-center">
            <select className="select w-48" value={method} onChange={(e) => setMethod(e.target.value)}>
              {methods?.methods?.map((m) => (
                <option key={m.method} value={m.method} disabled={!m.available} title={!m.available ? `Requires: ${m.requires?.join(', ') || 'extra dependencies'}` : ''}>
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
          <input className="input w-24" type="number" min={1} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} />
        </label>
        <label className="flex flex-col">
          <span className="label">Lookback Bars</span>
          <input
            className="input w-28"
            type="number"
            min={50}
            value={lookback}
            onChange={(e) => {
              const val = e.target.value
              setLookback(val === '' ? '' : Number(val))
            }}
            placeholder="auto"
          />
        </label>
        <label className="flex flex-col">
          <span className="label">Target</span>
          <select className="select w-28" value={target} onChange={(e) => setTarget(e.target.value as any)}>
            <option value="price">price</option>
            <option value="return">return</option>
          </select>
        </label>
        <label className="flex flex-col" title="Tail probability for confidence intervals. Example: 0.10 -> 90% interval, 0.05 -> 95%.">
          <span className="label">CI alpha</span>
          <input className="input w-24" type="number" step="0.01" min={0} max={0.5} value={ci} onChange={(e) => setCi(Number(e.target.value))} />
        </label>
        <div className="flex gap-2 ml-auto items-end">
          {!selectedMeta?.available && (
            <div className="text-xs text-amber-400 max-w-[24rem]">
              {selectedMeta?.method} is not available. {selectedMeta?.requires?.length ? `Requires: ${selectedMeta?.requires.join(', ')}` : 'Install the required package.'}
            </div>
          )}
          <button className="btn" disabled={!canRun} onClick={() => run('full')}>
            Full Forecast
          </button>
          <button className="btn" disabled={!canRun || !anchor} onClick={() => run('partial')}>
            Forecast from Anchor
          </button>
        </div>
      </div>
      <details className="border border-slate-800 rounded-md" open={!!denoise?.method || (!!dimredMethod && supportsDimred)}>
        <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium text-slate-200 hover:bg-slate-800 flex justify-between items-center">
          <span>Advanced Settings</span>
          <span className="text-xs text-slate-400">{advancedSummary}</span>
        </summary>
        <div className="p-3 space-y-3">
          <AdvancedPanel
            methods={methods as MethodsMeta}
            method={method}
            methodParams={methodParams}
            onMethodParams={setMethodParams}
            denoise={denoise}
            onDenoise={setDenoise}
            dimredMethod={dimredMethod}
            dimredParams={dimredParams}
            onDimred={(m, p) => {
              setDimredMethod(m)
              setDimredParams(p)
            }}
            showDimred={supportsDimred}
          />
        </div>
      </details>
    </div>
  )
}

import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getMethods, runBacktest, getErrorMessage } from '../api/client'
import type { BacktestBody, BacktestResult } from '../types'
import { loadJSON, saveJSON } from '../lib/storage'

type Props = {
  symbol?: string
  timeframe: string
  onResult?: (result: BacktestResult) => void
}

export function BacktestControls({ symbol, timeframe, onResult }: Props) {
  const { data: methods } = useQuery({
    queryKey: ['methods'],
    queryFn: getMethods,
  })

  const [selectedMethods, setSelectedMethods] = useState<string[]>(['theta'])
  const [horizon, setHorizon] = useState(12)
  const [steps, setSteps] = useState(5)
  const [spacing, setSpacing] = useState(20)
  const [target, setTarget] = useState<'price' | 'return'>('price')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)

  const storageKey = symbol && timeframe ? `bt:${symbol}:${timeframe}` : null

  // Load saved settings
  useEffect(() => {
    if (!storageKey) return
    const saved = loadJSON<{
      selectedMethods?: string[]
      horizon?: number
      steps?: number
      spacing?: number
      target?: 'price' | 'return'
    }>(storageKey)
    if (saved) {
      if (saved.selectedMethods?.length) setSelectedMethods(saved.selectedMethods)
      if (typeof saved.horizon === 'number') setHorizon(saved.horizon)
      if (typeof saved.steps === 'number') setSteps(saved.steps)
      if (typeof saved.spacing === 'number') setSpacing(saved.spacing)
      if (saved.target) setTarget(saved.target)
    }
  }, [storageKey])

  // Save settings
  useEffect(() => {
    if (!storageKey) return
    saveJSON(storageKey, { selectedMethods, horizon, steps, spacing, target })
  }, [storageKey, selectedMethods, horizon, steps, spacing, target])

  const availableMethods = useMemo(
    () => methods?.methods?.filter(m => m.available) ?? [],
    [methods]
  )

  const toggleMethod = (method: string) => {
    setSelectedMethods(prev =>
      prev.includes(method) ? prev.filter(m => m !== method) : [...prev, method]
    )
  }

  const canRun = !!symbol && selectedMethods.length > 0

  async function run() {
    if (!symbol || !selectedMethods.length) return

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const body: BacktestBody = {
        symbol,
        timeframe,
        horizon,
        steps,
        spacing,
        methods: selectedMethods,
        target,
      }
      const res = await runBacktest(body)
      setResult(res)
      onResult?.(res)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3">
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
          <span className="label">Backtest Steps</span>
          <input
            className="input w-24"
            type="number"
            min={1}
            max={50}
            value={steps}
            onChange={e => setSteps(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col">
          <span className="label">Spacing (bars)</span>
          <input
            className="input w-24"
            type="number"
            min={1}
            value={spacing}
            onChange={e => setSpacing(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col">
          <span className="label">Target</span>
          <select className="select w-28" value={target} onChange={e => setTarget(e.target.value as 'price' | 'return')}>
            <option value="price">price</option>
            <option value="return">return</option>
          </select>
        </label>
        <button className="btn ml-auto" disabled={!canRun || isLoading} onClick={run}>
          {isLoading ? 'Running...' : 'Run Backtest'}
        </button>
      </div>

      <div>
        <div className="label mb-2">Select Methods to Compare</div>
        <div className="flex flex-wrap gap-2">
          {availableMethods.map(m => (
            <button
              key={m.method}
              className={`btn text-xs ${
                selectedMethods.includes(m.method)
                  ? 'bg-sky-600'
                  : 'bg-slate-700 hover:bg-slate-600'
              }`}
              onClick={() => toggleMethod(m.method)}
            >
              {m.method}
            </button>
          ))}
        </div>
        <div className="text-xs text-slate-400 mt-2">
          {selectedMethods.length} method(s) selected
        </div>
      </div>

      {error && (
        <div className="text-sm text-rose-400 bg-rose-950/30 border border-rose-800 rounded-md px-3 py-2">
          {error}
        </div>
      )}

      {result && (
        <div className="border border-slate-800 rounded-md overflow-hidden">
          <div className="bg-slate-800/50 px-3 py-2 text-sm font-medium text-slate-200">
            Backtest Results ({result.steps} steps, {result.spacing} bar spacing)
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left px-3 py-2 text-slate-400 font-medium">Method</th>
                  <th className="text-right px-3 py-2 text-slate-400 font-medium">MAE</th>
                  <th className="text-right px-3 py-2 text-slate-400 font-medium">MAPE %</th>
                  <th className="text-right px-3 py-2 text-slate-400 font-medium">RMSE</th>
                  <th className="text-right px-3 py-2 text-slate-400 font-medium">Dir Acc %</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r, idx) => (
                  <tr key={r.method} className={idx % 2 === 0 ? 'bg-slate-900/30' : ''}>
                    <td className="px-3 py-2 text-slate-200 font-medium">{r.method}</td>
                    <td className="text-right px-3 py-2 text-slate-300">
                      {r.mae !== undefined ? r.mae.toFixed(4) : '-'}
                    </td>
                    <td className="text-right px-3 py-2 text-slate-300">
                      {r.mape !== undefined ? r.mape.toFixed(2) : '-'}
                    </td>
                    <td className="text-right px-3 py-2 text-slate-300">
                      {r.rmse !== undefined ? r.rmse.toFixed(4) : '-'}
                    </td>
                    <td className="text-right px-3 py-2">
                      <span
                        className={
                          r.direction_accuracy !== undefined
                            ? r.direction_accuracy >= 60
                              ? 'text-emerald-400'
                              : r.direction_accuracy >= 50
                              ? 'text-amber-400'
                              : 'text-rose-400'
                            : 'text-slate-500'
                        }
                      >
                        {r.direction_accuracy !== undefined ? r.direction_accuracy.toFixed(1) : '-'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

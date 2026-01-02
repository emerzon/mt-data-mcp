import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getDimredMethods, getSktimeEstimators } from '../api/client'
import { DenoiseModal } from './DenoiseModal'
import type { MethodsMeta, DenoiseSpecUI, ParamDef } from '../types'
import { coerce } from '../lib/utils'

type Props = {
  methods?: MethodsMeta
  method: string
  methodParams: Record<string, unknown>
  onMethodParams: (p: Record<string, unknown>) => void
  denoise?: DenoiseSpecUI
  onDenoise: (d?: DenoiseSpecUI) => void
  dimredMethod?: string
  dimredParams?: Record<string, unknown>
  onDimred: (m?: string, p?: Record<string, unknown>) => void
  showDimred?: boolean
}

export function AdvancedPanel({
  methods,
  method,
  methodParams,
  onMethodParams,
  denoise,
  onDenoise,
  dimredMethod,
  dimredParams,
  onDimred,
  showDimred = true,
}: Props) {
  const [showDenoise, setShowDenoise] = useState(false)
  const meta = useMemo(() => methods?.methods.find(m => m.method === method), [methods, method])
  const { data: dr } = useQuery({ queryKey: ['dr_methods'], queryFn: getDimredMethods })
  const drMethods = (dr?.methods ?? []).filter(m => m.available)
  const drParams: ParamDef[] = drMethods.find(m => m.method === (dimredMethod || ''))?.params || []
  const isSktime = method === 'sktime'
  const { data: skt } = useQuery({ queryKey: ['sktime_estimators'], queryFn: getSktimeEstimators, enabled: isSktime })

  const denoiseSummary = denoise?.method ? `${denoise.method}${denoise.params ? ' • params' : ''}` : 'None'

  return (
    <div className="space-y-4">
      <div>
        <div className="text-xs text-slate-400 mb-2">Method Parameters</div>
        {isSktime && (
          <div className="mb-2 flex items-center gap-2">
            <span className="text-xs text-slate-400">Preset:</span>
            <select
              className="select"
              onChange={e => {
                const v = e.target.value
                if (!v) return
                const presets: Record<string, Record<string, unknown>> = {
                  naive: { estimator: 'sktime.forecasting.naive.NaiveForecaster', estimator_params: { strategy: 'last' } },
                  snaive: { estimator: 'sktime.forecasting.naive.NaiveForecaster', estimator_params: { strategy: 'last' } },
                  theta: { estimator: 'sktime.forecasting.theta.ThetaForecaster' },
                  autoets: { estimator: 'sktime.forecasting.ets.AutoETS' },
                  arima: { estimator: 'sktime.forecasting.arima.ARIMA' },
                  autoarima: { estimator: 'sktime.forecasting.arima.AutoARIMA' },
                }
                onMethodParams({ ...methodParams, ...(presets[v] || {}) })
              }}
              defaultValue=""
            >
              <option value="">select…</option>
              <option value="naive">Naive (last)</option>
              <option value="snaive">Seasonal Naive</option>
              <option value="theta">Theta</option>
              <option value="autoets">AutoETS</option>
              <option value="arima">ARIMA</option>
              <option value="autoarima">AutoARIMA</option>
            </select>
            <span className="text-xs text-slate-500">Fills estimator fields below</span>
          </div>
        )}
        {isSktime && (
          <div className="mb-2 flex items-center gap-2">
            <span className="text-xs text-slate-400">Estimator:</span>
            <select
              className="select w-96"
              value={(methodParams?.estimator as string) || ''}
              onChange={e => onMethodParams({ ...methodParams, estimator: e.target.value })}
            >
              <option value="">select class…</option>
              {(skt?.estimators || []).map(it => (
                <option key={it.class_path} value={it.class_path}>
                  {it.class_path}
                </option>
              ))}
            </select>
            {!skt?.available && (
              <span className="text-xs text-amber-500">
                sktime not installed{skt?.error ? ` (${skt.error})` : ''}
              </span>
            )}
          </div>
        )}
        <div className="grid grid-cols-2 gap-2">
          {meta?.params?.map(p => (
            <label key={p.name} className="flex flex-col gap-1" title={p.description || ''}>
              <span className="label">
                {p.name} {p.type && <em className="not-italic text-slate-500">({p.type})</em>}
              </span>
              <input
                className="input"
                value={String(methodParams?.[p.name] ?? '')}
                onChange={e => onMethodParams({ ...methodParams, [p.name]: coerce(e.target.value) })}
                placeholder={String(p.default ?? '')}
              />
              {p.description && <span className="text-xs text-slate-400">{p.description}</span>}
            </label>
          ))}
          {(!meta?.params || meta.params.length === 0) && (
            <div className="text-xs text-slate-500 col-span-2">Selected method has no tunable parameters.</div>
          )}
        </div>
      </div>

      <details className="border border-slate-800 rounded-md" open={!!denoise?.method}>
        <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium text-slate-200 hover:bg-slate-800 flex justify-between items-center">
          <span>Denoising</span>
          <span className="text-xs text-slate-400">{denoiseSummary}</span>
        </summary>
        <div className="p-3">
          <div className="flex items-center justify-between">
            <div className="text-xs text-slate-400">Configure forecast-level denoising.</div>
            <button type="button" className="btn" onClick={() => setShowDenoise(true)}>
              {denoise?.method ? 'Edit' : 'Configure'}
            </button>
          </div>
        </div>
      </details>

      {showDimred && (
        <div>
          <div className="text-xs text-slate-400 mb-2">Dimensionality Reduction</div>
          <label className="label">Method</label>
          <select
            className="select"
            value={dimredMethod || ''}
            onChange={e => onDimred(e.target.value || undefined, dimredParams)}
          >
            <option value="">none</option>
            {drMethods.map(m => (
              <option key={m.method} value={m.method}>
                {m.method}
              </option>
            ))}
          </select>
          {dimredMethod && (
            <div className="text-xs text-slate-400 mt-1">
              {drMethods.find(m => m.method === dimredMethod)?.description}
            </div>
          )}
          {drParams.length > 0 && (
            <div className="mt-2 grid grid-cols-2 gap-2">
              {drParams.map(p => (
                <label key={p.name} className="flex flex-col gap-1" title={p.description || ''}>
                  <span className="label">
                    {p.name} {p.type && <em className="not-italic text-slate-500">({p.type})</em>}
                  </span>
                  <input
                    className="input"
                    value={String(dimredParams?.[p.name] ?? '')}
                    onChange={e =>
                      onDimred(dimredMethod, { ...(dimredParams || {}), [p.name]: coerce(e.target.value) })
                    }
                    placeholder={String(p.default ?? '')}
                  />
                  {p.description && <span className="text-xs text-slate-400">{p.description}</span>}
                </label>
              ))}
            </div>
          )}
        </div>
      )}

      <DenoiseModal
        open={showDenoise}
        title="Forecast Denoising"
        value={denoise}
        onClose={() => setShowDenoise(false)}
        onApply={onDenoise}
      />
    </div>
  )
}

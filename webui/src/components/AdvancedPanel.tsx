import { useMemo, useState } from 'react'
import type { MethodsMeta } from '../types'
import { useQuery } from '@tanstack/react-query'
import { getDimredMethods } from '../api/client'
import { DenoiseModal, DenoiseSpecUI } from './DenoiseModal'

type ParamDef = { name: string; type: string; default?: any; description?: string }

type Props = {
  methods?: MethodsMeta
  method: string
  methodParams: any
  onMethodParams: (p: any) => void
  denoise?: DenoiseSpecUI
  onDenoise: (d?: DenoiseSpecUI) => void
  dimredMethod?: string
  dimredParams?: any
  onDimred: (m?: string, p?: any) => void
  showDimred?: boolean
}

export function AdvancedPanel({ methods, method, methodParams, onMethodParams, denoise, onDenoise, dimredMethod, dimredParams, onDimred, showDimred = true }: Props) {
  const [showDenoise, setShowDenoise] = useState(false)
  const meta = useMemo(() => methods?.methods.find(m => m.method === method), [methods, method])
  const { data: dr } = useQuery({ queryKey: ['dr_methods'], queryFn: getDimredMethods })
  const drMethods = (dr?.methods ?? []).filter((m: any) => m.available)
  const drParams: ParamDef[] = (drMethods.find((m: any) => m.method === (dimredMethod || ''))?.params) || []

  const denoiseSummary = denoise?.method
    ? `${denoise.method}${denoise.params ? ' • params' : ''}`
    : 'None'

  return (
    <div className="space-y-4">
      <div>
        <div className="text-xs text-slate-400 mb-2">Method Parameters</div>
        <div className="grid grid-cols-2 gap-2">
          {meta?.params?.map(p => (
            <label key={p.name} className="flex flex-col gap-1" title={p.description || ''}>
              <span className="label">{p.name} {p.type ? <em className="not-italic text-slate-500">({p.type})</em> : null}</span>
              <input
                className="input"
                value={methodParams?.[p.name] ?? ''}
                onChange={e => onMethodParams({ ...methodParams, [p.name]: coerce(e.target.value) })}
                placeholder={String(p.default ?? '')}
              />
              {p.description ? <span className="text-xs text-slate-400">{p.description}</span> : null}
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
            <button type="button" className="btn" onClick={() => setShowDenoise(true)}>{denoise?.method ? 'Edit' : 'Configure'}</button>
          </div>
        </div>
      </details>

      {showDimred && (
        <div>
          <div className="text-xs text-slate-400 mb-2">Dimensionality Reduction</div>
          <label className="label">Method</label>
          <select className="select" value={dimredMethod || ''} onChange={e => onDimred(e.target.value || undefined, dimredParams)}>
            <option value="">none</option>
            {drMethods.map((m: any) => <option key={m.method} value={m.method}>{m.method}</option>)}
          </select>
          {dimredMethod ? (
            <div className="text-xs text-slate-400 mt-1">{drMethods.find((m: any) => m.method === dimredMethod)?.description}</div>
          ) : null}
          {drParams.length > 0 && (
            <div className="mt-2 grid grid-cols-2 gap-2">
              {drParams.map(p => (
                <label key={p.name} className="flex flex-col gap-1" title={p.description || ''}>
                  <span className="label">{p.name} {p.type ? <em className="not-italic text-slate-500">({p.type})</em> : null}</span>
                  <input
                    className="input"
                    value={dimredParams?.[p.name] ?? ''}
                    onChange={e => onDimred(dimredMethod, { ...(dimredParams || {}), [p.name]: coerce(e.target.value) })}
                    placeholder={String(p.default ?? '')}
                  />
                  {p.description ? <span className="text-xs text-slate-400">{p.description}</span> : null}
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

function coerce(v: string) {
  const t = v.trim()
  if (t === '') return ''
  if (!Number.isNaN(Number(t))) return Number(t)
  if (t === 'true') return true
  if (t === 'false') return false
  return t
}


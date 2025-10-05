import { Fragment, useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { useQuery } from '@tanstack/react-query'
import { getDenoiseMethods, getWavelets } from '../api/client'

export type DenoiseSpecUI = {
  method?: string
  params?: Record<string, any>
  columns?: string | string[]
  when?: string
  causality?: string
  keep_original?: boolean
}

type ParamDef = { name: string; type: string; default?: any; description?: string }

type Props = {
  open: boolean
  title?: string
  value?: DenoiseSpecUI
  onClose: () => void
  onApply: (value?: DenoiseSpecUI) => void
}

export function DenoiseModal({ open, title = 'Configure Denoising', value, onClose, onApply }: Props) {
  const { data: methodData } = useQuery({ queryKey: ['dn_methods'], queryFn: getDenoiseMethods })
  const methods = methodData?.methods ?? []
  const [method, setMethod] = useState<string>(value?.method || '')
  const [params, setParams] = useState<Record<string, any>>(value?.params || {})
  const [columns, setColumns] = useState<string>(Array.isArray(value?.columns) ? value?.columns.join(',') : (value?.columns as string) || 'close')
  const [when, setWhen] = useState<string>(value?.when || 'post_ti')
  const [causality, setCausality] = useState<string>(value?.causality || 'zero_phase')
  const [keepOriginal, setKeepOriginal] = useState<boolean>(value?.keep_original ?? true)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)

  const paramDefs: ParamDef[] = useMemo(() => methods.find((m: any) => m.method === method)?.params || [], [methods, method])

  const waveletEnabled = method === 'wavelet'
  const { data: wv } = useQuery({ queryKey: ['wavelets'], queryFn: getWavelets, enabled: waveletEnabled })
  const wavelets: string[] = (wv?.available ? (wv?.wavelets || []) : [])

  useEffect(() => {
    if (!open) return
    setMethod(value?.method || '')
    setParams(value?.params || {})
    setColumns(Array.isArray(value?.columns) ? value?.columns.join(',') : (value?.columns as string) || 'close')
    setWhen(value?.when || 'post_ti')
    setCausality(value?.causality || 'zero_phase')
    setKeepOriginal(value?.keep_original ?? true)
    const advancedActive = Boolean(value?.columns || value?.when && value?.when !== 'post_ti' || value?.causality && value?.causality !== 'zero_phase' || value?.keep_original === false)
    setShowAdvanced(advancedActive)
  }, [open, value?.method, value?.params, value?.columns, value?.when, value?.causality, value?.keep_original])

  if (!open) return null

  const portalTarget = document.getElementById('modal-root') || document.body

  const apply = () => {
    if (!method) {
      onApply(undefined)
      onClose()
      return
    }
    const cols = columns?.split(',').map(c => c.trim()).filter(Boolean)
    onApply({
      method,
      params,
      columns: cols && cols.length ? cols : undefined,
      when,
      causality,
      keep_original: keepOriginal,
    })
    onClose()
  }

  const clear = () => {
    setMethod('')
    setParams({})
    onApply(undefined)
    onClose()
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80">
      <div className="panel w-[640px] max-h-[90vh] overflow-y-auto p-5 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-slate-200">{title}</h2>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
        <div className="space-y-3">
          <label className="flex flex-col gap-1">
            <span className="label">Method</span>
            <select className="select" value={method} onChange={e => { setMethod(e.target.value); setParams({}); setShowAdvanced(false) }}>
              <option value="">none</option>
              {methods.map((m: any) => (
                <option key={m.method} value={m.method}>{m.method}</option>
              ))}
            </select>
            {method && <span className="text-xs text-slate-400">{methods.find((m: any) => m.method === method)?.description}</span>}
          </label>

          {method && (
            <Fragment>
              <div className="border border-slate-800 rounded-md">
                <button
                  type="button"
                  className="w-full text-left px-3 py-2 text-sm font-medium text-slate-200 flex justify-between items-center hover:bg-slate-800"
                  onClick={() => setShowAdvanced(v => !v)}
                >
                  <span>Advanced settings</span>
                  <span>{showAdvanced ? '−' : '+'}</span>
                </button>
                {showAdvanced && (
                  <div className="grid grid-cols-2 gap-3 px-3 pb-3">
                    <label className="flex flex-col gap-1">
                      <span className="label">Columns</span>
                      <input className="input" value={columns} onChange={e => setColumns(e.target.value)} placeholder="close" />
                      <span className="text-xs text-slate-400">Comma separated (default: close)</span>
                    </label>
                    <label className="flex flex-col gap-1">
                      <span className="label">When</span>
                      <select className="select" value={when} onChange={e => setWhen(e.target.value)}>
                        <option value="pre_ti">pre_ti</option>
                        <option value="post_ti">post_ti</option>
                      </select>
                    </label>
                    <label className="flex flex-col gap-1">
                      <span className="label">Causality</span>
                      <select className="select" value={causality} onChange={e => setCausality(e.target.value)}>
                        <option value="zero_phase">zero_phase</option>
                        <option value="causal">causal</option>
                      </select>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={keepOriginal} onChange={e => setKeepOriginal(e.target.checked)} />
                      <span className="label">Keep original columns</span>
                    </label>
                  </div>
                )}
              </div>

              {paramDefs.length > 0 && (
                <div className="grid grid-cols-2 gap-3">
                  {paramDefs.map((p) => {
                    if (waveletEnabled && p.name === 'wavelet') {
                      return (
                        <label key={p.name} className="flex flex-col gap-1" title={p.description || ''}>
                          <span className="label">wavelet</span>
                          <select className="select" value={params?.wavelet ?? ''} onChange={e => setParams({ ...params, wavelet: e.target.value })}>
                            <option value="">db4 (default)</option>
                            {wavelets.map(w => <option key={w} value={w}>{w}</option>)}
                          </select>
                          {p.description ? <span className="text-xs text-slate-400">{p.description}</span> : null}
                        </label>
                      )
                    }
                    if (waveletEnabled && p.name === 'mode') {
                      return (
                        <label key={p.name} className="flex flex-col gap-1" title={p.description || ''}>
                          <span className="label">mode</span>
                          <select className="select" value={params?.mode ?? 'soft'} onChange={e => setParams({ ...params, mode: e.target.value })}>
                            <option value="soft">soft</option>
                            <option value="hard">hard</option>
                          </select>
                          {p.description ? <span className="text-xs text-slate-400">{p.description}</span> : null}
                        </label>
                      )
                    }
                    return (
                      <label key={p.name} className="flex flex-col gap-1" title={p.description || ''}>
                        <span className="label">{p.name} {p.type ? <em className="not-italic text-slate-500">({p.type})</em> : null}</span>
                        <input className="input" value={params?.[p.name] ?? ''} onChange={e => setParams({ ...params, [p.name]: coerce(e.target.value) })} placeholder={String(p.default ?? '')} />
                        {p.description ? <span className="text-xs text-slate-400">{p.description}</span> : null}
                      </label>
                    )
                  })}
                </div>
              )}
            </Fragment>
          )}
        </div>

        <div className="flex justify-between items-center pt-2 border-t border-slate-800">
          <button className="btn bg-rose-600 hover:bg-rose-500" onClick={clear}>Disable</button>
          <div className="flex gap-2">
            <button className="btn bg-slate-600 hover:bg-slate-500" onClick={onClose}>Cancel</button>
            <button className="btn" onClick={apply} disabled={!method}>Apply</button>
          </div>
        </div>
      </div>
    </div>,
    portalTarget
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

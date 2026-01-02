import { useState } from 'react'
import { DenoiseModal } from './DenoiseModal'
import type { DenoiseSpecUI } from '../types'

type Props = {
  value?: DenoiseSpecUI
  onChange: (v?: DenoiseSpecUI) => void
}

export function ChartDenoiseControls({ value, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const summary = value?.method
    ? `${value.method}${value.columns ? ` • cols:${Array.isArray(value.columns) ? value.columns.join(',') : value.columns}` : ''}${value.params ? ' • params' : ''}`
    : 'None'

  return (
    <div className="flex items-center justify-between">
      <div>
        <span className="label">Chart Denoise</span>
        <div className="text-xs text-slate-400">{summary}</div>
      </div>
      <button className="btn" onClick={() => setOpen(true)}>{value?.method ? 'Edit' : 'Configure'}</button>
      <DenoiseModal
        open={open}
        title="Chart Denoising"
        value={value}
        onClose={() => setOpen(false)}
        onApply={onChange}
      />
    </div>
  )
}

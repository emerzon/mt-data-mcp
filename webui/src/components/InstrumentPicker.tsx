import { useQuery } from '@tanstack/react-query'
import { searchInstruments } from '../api/client'
import { useMemo, useState } from 'react'

export function InstrumentPicker({ value, onChange }: { value?: string; onChange: (v: string) => void }) {
  const [q, setQ] = useState('')
  const { data } = useQuery({ queryKey: ['inst', q], queryFn: () => searchInstruments(q || undefined, 50) })
  const items = data ?? []
  const grouped = useMemo(() => items, [items])

  return (
    <div className="flex items-center gap-2">
      <input className="input w-40" placeholder="Search" value={q} onChange={e => setQ(e.target.value)} />
      <select className="select w-56" value={value} onChange={e => onChange(e.target.value)}>
        <option value="">Select instrument</option>
        {grouped.map((s) => (
          <option key={s.name} value={s.name}>{s.name}{s.description ? ` — ${s.description}` : ''}</option>
        ))}
      </select>
    </div>
  )
}


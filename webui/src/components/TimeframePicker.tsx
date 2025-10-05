import { useQuery } from '@tanstack/react-query'
import { getTimeframes } from '../api/client'

export function TimeframePicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data } = useQuery({ queryKey: ['tfs'], queryFn: getTimeframes })
  const tfs = data ?? []
  return (
    <select className="select" value={value} onChange={e => onChange(e.target.value)}>
      {tfs.map(tf => <option key={tf} value={tf}>{tf}</option>)}
    </select>
  )
}


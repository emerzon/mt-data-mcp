import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getDenoiseMethods, getTimeframes, getWavelets, searchInstruments } from '../../api/client'
import { loadJSON } from '../../lib/storage'
import type { DenoiseSpecUI } from '../../types'
import { ChevronDown } from './toolbarIcons'

function useDismissiblePanel(onClose: () => void) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [onClose])

  return ref
}

export function SymbolSelector({
  value,
  onChange,
}: {
  value: string
  onChange: (value: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')

  return (
    <div className="relative">
      <button className="toolbar-btn min-w-[120px] justify-between" onClick={() => setOpen((value) => !value)}>
        <span className={value ? 'text-slate-200' : 'text-slate-500'}>{value || 'Symbol'}</span>
        <ChevronDown />
      </button>
      {open && (
        <SymbolDropdown
          value={value}
          search={search}
          onSearchChange={setSearch}
          onSelect={(symbol) => {
            onChange(symbol)
            setOpen(false)
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function SymbolDropdown({
  value,
  search,
  onSearchChange,
  onSelect,
  onClose,
}: {
  value: string
  search: string
  onSearchChange: (value: string) => void
  onSelect: (value: string) => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)
  const { data: searchResults } = useQuery({
    queryKey: ['instruments', search],
    queryFn: () => searchInstruments(search || undefined, 50),
    enabled: !!search,
  })

  const items = useMemo(() => {
    if (search) return searchResults ?? []
    const recent = loadJSON<string[]>('recent_symbols') || []
    return recent.map((name) => ({ name, description: 'Recent' }))
  }, [search, searchResults])

  return (
    <div ref={ref} className="absolute top-full left-0 mt-1 w-72 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <input
        className="w-full px-3 py-2 bg-slate-800 text-sm text-slate-200 border-b border-slate-700 focus:outline-none"
        placeholder="Search instruments..."
        value={search}
        onChange={(event) => onSearchChange(event.target.value)}
        autoFocus
      />
      <div className="max-h-64 overflow-y-auto">
        {items.map((item) => (
          <button
            key={item.name}
            className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${
              item.name === value ? 'bg-slate-800 text-sky-400' : 'text-slate-300'
            }`}
            onClick={() => onSelect(item.name)}
          >
            <span className="font-medium">{item.name}</span>
            {item.description && <span className="ml-2 text-slate-500 text-xs">{item.description}</span>}
          </button>
        ))}
        {items.length === 0 && <div className="px-3 py-4 text-sm text-slate-500 text-center">No instruments found</div>}
      </div>
    </div>
  )
}

export function TimezoneSelector({
  value,
  onChange,
}: {
  value: 'utc' | 'local' | 'server'
  onChange: (value: 'utc' | 'local' | 'server') => void
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        className="toolbar-btn min-w-[60px] justify-between text-slate-400"
        onClick={() => setOpen((value) => !value)}
        title="Timezone"
      >
        <span className="text-xs font-medium uppercase">{value === 'server' ? 'Exch' : value}</span>
        <ChevronDown />
      </button>
      {open && (
        <TimezoneDropdown
          value={value}
          onChange={(mode) => {
            onChange(mode)
            setOpen(false)
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function TimezoneDropdown({
  value,
  onChange,
  onClose,
}: {
  value: 'utc' | 'local' | 'server'
  onChange: (value: 'utc' | 'local' | 'server') => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)

  return (
    <div ref={ref} className="absolute top-full left-0 mt-1 w-32 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">Timezone</div>
      {(['utc', 'server', 'local'] as const).map((mode) => (
        <button
          key={mode}
          className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${
            value === mode ? 'text-sky-400' : 'text-slate-300'
          }`}
          onClick={() => onChange(mode)}
        >
          {mode === 'server' ? 'Exchange' : mode.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

export function TimeframeSelector({
  value,
  onChange,
}: {
  value: string
  onChange: (value: string) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useDismissiblePanel(() => setOpen(false))
  const { data } = useQuery({ queryKey: ['timeframes'], queryFn: getTimeframes })
  const timeframes = data ?? []

  return (
    <div ref={ref} className="relative">
      <button className="toolbar-btn min-w-[50px] justify-between" onClick={() => setOpen((value) => !value)}>
        <span>{value}</span>
        <ChevronDown />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 w-24 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50 max-h-64 overflow-y-auto">
          {timeframes.map((timeframe) => (
            <button
              key={timeframe}
              className={`w-full text-left px-3 py-1.5 text-sm hover:bg-slate-800 ${
                timeframe === value ? 'bg-slate-800 text-sky-400' : 'text-slate-300'
              }`}
              onClick={() => {
                onChange(timeframe)
                setOpen(false)
              }}
            >
              {timeframe}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function DenoiseSelector({
  value,
  disabled,
  onChange,
}: {
  value?: DenoiseSpecUI
  disabled: boolean
  onChange: (value?: DenoiseSpecUI) => void
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        className={`toolbar-btn ${value?.method ? 'text-orange-400' : ''}`}
        onClick={() => setOpen((value) => !value)}
        disabled={disabled}
        title="Chart denoising"
      >
        <span>Filter</span>
        <ChevronDown />
      </button>
      {open && (
        <DenoiseDropdown
          value={value}
          onChange={(denoise) => {
            onChange(denoise)
            setOpen(false)
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function DenoiseDropdown({
  value,
  onChange,
  onClose,
}: {
  value?: DenoiseSpecUI
  onChange: (value?: DenoiseSpecUI) => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)
  const { data: methodsData } = useQuery({ queryKey: ['denoise_methods'], queryFn: getDenoiseMethods })
  const { data: waveletsData } = useQuery({
    queryKey: ['wavelets'],
    queryFn: getWavelets,
    enabled: value?.method === 'wavelet',
  })

  const methods = (methodsData?.methods ?? []).filter((method) => method.available)
  const wavelets = waveletsData?.wavelets ?? []

  const handleMethodSelect = (method: string) => {
    if (method === '') {
      onChange(undefined)
      return
    }
    onChange({ method, params: {} })
  }

  return (
    <div ref={ref} className="absolute top-full right-0 mt-1 w-56 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">Chart Denoising</div>
      <div className="max-h-64 overflow-y-auto">
        <button
          className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${
            !value?.method ? 'text-sky-400' : 'text-slate-300'
          }`}
          onClick={() => handleMethodSelect('')}
        >
          None (raw data)
        </button>
        {methods.map((method) => (
          <button
            key={method.method}
            className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${
              method.method === value?.method ? 'text-sky-400' : 'text-slate-300'
            }`}
            onClick={() => handleMethodSelect(method.method)}
          >
            {method.method}
            <span className="ml-2 text-slate-500 text-xs">{method.description}</span>
          </button>
        ))}
      </div>
      {value?.method === 'wavelet' && wavelets.length > 0 && (
        <div className="border-t border-slate-700 px-3 py-2">
          <label className="text-xs text-slate-400">Wavelet</label>
          <select
            className="w-full mt-1 bg-slate-800 text-slate-200 text-sm rounded px-2 py-1 border border-slate-700"
            value={(value.params?.wavelet as string) || 'db4'}
            onChange={(event) =>
              onChange({ ...value, params: { ...value.params, wavelet: event.target.value } })
            }
          >
            {wavelets.map((wavelet) => (
              <option key={wavelet} value={wavelet}>
                {wavelet}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}

export function PriceLinesSelector({
  showBid,
  showAsk,
  showLast,
  isLive,
  disabled,
  onToggleBid,
  onToggleAsk,
  onToggleLast,
  onToggleLive,
}: {
  showBid: boolean
  showAsk: boolean
  showLast: boolean
  isLive: boolean
  disabled: boolean
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLast: () => void
  onToggleLive: () => void
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button
        className={`toolbar-btn ${showBid || showAsk || showLast ? 'text-sky-400' : ''}`}
        onClick={() => setOpen((value) => !value)}
        disabled={disabled}
        title="Price Lines"
      >
        <span>Lines</span>
        <ChevronDown />
      </button>
      {open && (
        <PriceLinesDropdown
          showBid={showBid}
          showAsk={showAsk}
          showLast={showLast}
          isLive={isLive}
          onToggleBid={onToggleBid}
          onToggleAsk={onToggleAsk}
          onToggleLast={onToggleLast}
          onToggleLive={onToggleLive}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )
}

function PriceLinesDropdown({
  showBid,
  showAsk,
  showLast,
  isLive,
  onToggleBid,
  onToggleAsk,
  onToggleLast,
  onToggleLive,
  onClose,
}: {
  showBid: boolean
  showAsk: boolean
  showLast: boolean
  isLive: boolean
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLast: () => void
  onToggleLive: () => void
  onClose: () => void
}) {
  const ref = useDismissiblePanel(onClose)

  return (
    <div ref={ref} className="absolute top-full right-0 mt-1 w-48 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">Chart Settings</div>
      <div className="p-1">
        <ToolbarCheckbox checked={isLive} label="Live Chart" onChange={onToggleLive} />
        <div className="h-px bg-slate-700 my-1" />
        <ToolbarCheckbox checked={showLast} label="Show Last" onChange={onToggleLast} />
        <ToolbarCheckbox checked={showBid} label="Show Bid" onChange={onToggleBid} />
        <ToolbarCheckbox checked={showAsk} label="Show Ask" onChange={onToggleAsk} />
      </div>
    </div>
  )
}

function ToolbarCheckbox({
  checked,
  label,
  onChange,
}: {
  checked: boolean
  label: string
  onChange: () => void
}) {
  return (
    <label className="flex items-center gap-2 px-2 py-1.5 hover:bg-slate-800 rounded cursor-pointer">
      <input
        type="checkbox"
        className="rounded border-slate-600 bg-slate-700 text-sky-500 focus:ring-offset-slate-900"
        checked={checked}
        onChange={onChange}
      />
      <span className="text-sm text-slate-300">{label}</span>
    </label>
  )
}

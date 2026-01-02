import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { searchInstruments, getTimeframes, getDenoiseMethods, getWavelets } from '../api/client'
import type { DenoiseSpecUI } from '../types'

type Props = {
  symbol: string
  timeframe: string
  anchor?: number
  limit: number
  isLoading: boolean
  barsCount: number
  hasPivots: boolean
  hasSR: boolean
  denoise?: DenoiseSpecUI
  showBid: boolean
  showAsk: boolean
  isLive: boolean
  onSymbolChange: (s: string) => void
  onTimeframeChange: (tf: string) => void
  onLimitChange: (n: number) => void
  onClearAnchor: () => void
  onReload: () => void
  onTogglePivots: () => void
  onToggleSR: () => void
  onDenoiseChange: (d?: DenoiseSpecUI) => void
  onOpenForecast: () => void
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLive: () => void
}

export function ChartToolbar({
  symbol,
  timeframe,
  anchor,
  limit,
  isLoading,
  barsCount,
  hasPivots,
  hasSR,
  denoise,
  showBid,
  showAsk,
  isLive,
  onSymbolChange,
  onTimeframeChange,
  onLimitChange,
  onClearAnchor,
  onReload,
  onTogglePivots,
  onToggleSR,
  onDenoiseChange,
  onOpenForecast,
  onToggleBid,
  onToggleAsk,
  onToggleLive,
}: Props) {
  const [showSymbolMenu, setShowSymbolMenu] = useState(false)
  const [showDenoiseMenu, setShowDenoiseMenu] = useState(false)
  const [showPriceMenu, setShowPriceMenu] = useState(false)
  const [symbolSearch, setSymbolSearch] = useState('')

  return (
    <div className="absolute top-3 left-3 right-3 z-20 flex items-start gap-2">
      {/* Left side: Symbol & Timeframe */}
      <div className="flex items-center gap-1 bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 p-1">
        {/* Symbol selector */}
        <div className="relative">
          <button
            className="toolbar-btn min-w-[120px] justify-between"
            onClick={() => setShowSymbolMenu(!showSymbolMenu)}
          >
            <span className={symbol ? 'text-slate-200' : 'text-slate-500'}>
              {symbol || 'Symbol'}
            </span>
            <ChevronDown />
          </button>
          {showSymbolMenu && (
            <SymbolDropdown
              value={symbol}
              search={symbolSearch}
              onSearchChange={setSymbolSearch}
              onSelect={(s) => { onSymbolChange(s); setShowSymbolMenu(false) }}
              onClose={() => setShowSymbolMenu(false)}
            />
          )}
        </div>

        <div className="w-px h-5 bg-slate-700" />

        {/* Timeframe selector */}
        <TimeframeDropdown value={timeframe} onChange={onTimeframeChange} />

        <div className="w-px h-5 bg-slate-700" />

        {/* Bars count */}
        <input
          type="number"
          className="w-16 bg-transparent text-xs text-slate-300 text-center focus:outline-none"
          value={limit}
          onChange={(e) => onLimitChange(Number(e.target.value))}
          min={100}
          max={20000}
          title="Number of bars"
        />

        <button className="toolbar-btn" onClick={onReload} disabled={!symbol || isLoading} title="Reload data">
          <RefreshIcon className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Center: Info */}
      <div className="flex-1" />
      
      {symbol && (
        <div className="bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 px-3 py-1.5 text-xs text-slate-400">
          {barsCount} bars
          {anchor && (
            <span className="ml-2 text-amber-400">
              Anchor: {new Date(anchor * 1000).toISOString().slice(11, 19)}
              <button className="ml-1 text-slate-500 hover:text-slate-300" onClick={onClearAnchor}>×</button>
            </span>
          )}
        </div>
      )}

      {/* Right side: Chart options & Forecast */}
      <div className="flex items-center gap-2">
        {/* Chart menu */}
        <div className="relative">
          <div className="flex items-center gap-1 bg-slate-900/95 backdrop-blur-sm rounded-lg border border-slate-800 p-1">
            <button
              className={`toolbar-btn ${hasPivots ? 'text-amber-400' : ''}`}
              onClick={onTogglePivots}
              disabled={!symbol}
              title="Toggle pivot levels"
            >
              <PivotIcon />
            </button>
            <button
              className={`toolbar-btn ${hasSR ? 'text-emerald-400' : ''}`}
              onClick={onToggleSR}
              disabled={!symbol}
              title="Toggle support/resistance"
            >
              <SRIcon />
            </button>
            <div className="w-px h-5 bg-slate-700" />
            
            <div className="relative">
              <button
                className={`toolbar-btn ${showBid || showAsk ? 'text-sky-400' : ''}`}
                onClick={() => setShowPriceMenu(!showPriceMenu)}
                disabled={!symbol}
                title="Price Lines"
              >
                <LinesIcon />
                <ChevronDown />
              </button>
              {showPriceMenu && (
                <PriceLinesDropdown
                  showBid={showBid}
                  showAsk={showAsk}
                  isLive={isLive}
                  onToggleBid={onToggleBid}
                  onToggleAsk={onToggleAsk}
                  onToggleLive={onToggleLive}
                  onClose={() => setShowPriceMenu(false)}
                />
              )}
            </div>

            <div className="w-px h-5 bg-slate-700" />
            <button
              className={`toolbar-btn ${denoise?.method ? 'text-orange-400' : ''}`}
              onClick={() => setShowDenoiseMenu(!showDenoiseMenu)}
              disabled={!symbol}
              title="Chart denoising"
            >
              <DenoiseIcon />
              <ChevronDown />
            </button>
          </div>
          {showDenoiseMenu && (
            <DenoiseDropdown
              value={denoise}
              onChange={(d) => { onDenoiseChange(d); setShowDenoiseMenu(false) }}
              onClose={() => setShowDenoiseMenu(false)}
            />
          )}
        </div>

        {/* Forecast button */}
        <button
          className="bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium px-4 py-1.5 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          onClick={onOpenForecast}
          disabled={!symbol}
        >
          Forecast
        </button>
      </div>
    </div>
  )
}

// Symbol dropdown
function SymbolDropdown({
  value,
  search,
  onSearchChange,
  onSelect,
  onClose,
}: {
  value: string
  search: string
  onSearchChange: (s: string) => void
  onSelect: (s: string) => void
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { data } = useQuery({
    queryKey: ['instruments', search],
    queryFn: () => searchInstruments(search || undefined, 50),
  })
  const items = data ?? []

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [onClose])

  return (
    <div ref={ref} className="absolute top-full left-0 mt-1 w-72 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <input
        className="w-full px-3 py-2 bg-slate-800 text-sm text-slate-200 border-b border-slate-700 focus:outline-none"
        placeholder="Search instruments..."
        value={search}
        onChange={(e) => onSearchChange(e.target.value)}
        autoFocus
      />
      <div className="max-h-64 overflow-y-auto">
        {items.map((s) => (
          <button
            key={s.name}
            className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${s.name === value ? 'bg-slate-800 text-sky-400' : 'text-slate-300'}`}
            onClick={() => onSelect(s.name)}
          >
            <span className="font-medium">{s.name}</span>
            {s.description && <span className="ml-2 text-slate-500 text-xs">{s.description}</span>}
          </button>
        ))}
        {items.length === 0 && (
          <div className="px-3 py-4 text-sm text-slate-500 text-center">No instruments found</div>
        )}
      </div>
    </div>
  )
}

// Timeframe dropdown
function TimeframeDropdown({ value, onChange }: { value: string; onChange: (tf: string) => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const { data } = useQuery({ queryKey: ['timeframes'], queryFn: getTimeframes })
  const tfs = data ?? []

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  return (
    <div ref={ref} className="relative">
      <button className="toolbar-btn min-w-[50px] justify-between" onClick={() => setOpen(!open)}>
        <span>{value}</span>
        <ChevronDown />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 w-24 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50 max-h-64 overflow-y-auto">
          {tfs.map((tf) => (
            <button
              key={tf}
              className={`w-full text-left px-3 py-1.5 text-sm hover:bg-slate-800 ${tf === value ? 'bg-slate-800 text-sky-400' : 'text-slate-300'}`}
              onClick={() => { onChange(tf); setOpen(false) }}
            >
              {tf}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// Denoise dropdown
function DenoiseDropdown({
  value,
  onChange,
  onClose,
}: {
  value?: DenoiseSpecUI
  onChange: (d?: DenoiseSpecUI) => void
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { data: methodsData } = useQuery({ queryKey: ['denoise_methods'], queryFn: getDenoiseMethods })
  const { data: waveletsData } = useQuery({ queryKey: ['wavelets'], queryFn: getWavelets, enabled: value?.method === 'wavelet' })
  
  const methods = (methodsData?.methods ?? []).filter(m => m.available)
  const wavelets = waveletsData?.wavelets ?? []

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [onClose])

  const handleMethodSelect = (method: string) => {
    if (method === '') {
      onChange(undefined)
    } else {
      onChange({ method, params: {} })
    }
  }

  return (
    <div ref={ref} className="absolute top-full right-0 mt-1 w-56 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">
        Chart Denoising
      </div>
      <div className="max-h-64 overflow-y-auto">
        <button
          className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${!value?.method ? 'text-sky-400' : 'text-slate-300'}`}
          onClick={() => handleMethodSelect('')}
        >
          None (raw data)
        </button>
        {methods.map((m) => (
          <button
            key={m.method}
            className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-800 ${m.method === value?.method ? 'text-sky-400' : 'text-slate-300'}`}
            onClick={() => handleMethodSelect(m.method)}
          >
            {m.method}
            <span className="ml-2 text-slate-500 text-xs">{m.description}</span>
          </button>
        ))}
      </div>
      {value?.method === 'wavelet' && wavelets.length > 0 && (
        <div className="border-t border-slate-700 px-3 py-2">
          <label className="text-xs text-slate-400">Wavelet</label>
          <select
            className="w-full mt-1 bg-slate-800 text-slate-200 text-sm rounded px-2 py-1 border border-slate-700"
            value={(value.params?.wavelet as string) || 'db4'}
            onChange={(e) => onChange({ ...value, params: { ...value.params, wavelet: e.target.value } })}
          >
            {wavelets.map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}

// Price Lines dropdown
function PriceLinesDropdown({
  showBid,
  showAsk,
  isLive,
  onToggleBid,
  onToggleAsk,
  onToggleLive,
  onClose,
}: {
  showBid: boolean
  showAsk: boolean
  isLive: boolean
  onToggleBid: () => void
  onToggleAsk: () => void
  onToggleLive: () => void
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [onClose])

  return (
    <div ref={ref} className="absolute top-full right-0 mt-1 w-48 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-50">
      <div className="px-3 py-2 text-xs text-slate-400 border-b border-slate-700 font-medium">
        Chart Settings
      </div>
      <div className="p-1">
        <label className="flex items-center gap-2 px-2 py-1.5 hover:bg-slate-800 rounded cursor-pointer">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-slate-700 text-sky-500 focus:ring-offset-slate-900"
            checked={isLive}
            onChange={onToggleLive}
          />
          <span className="text-sm text-slate-300">Live Chart</span>
        </label>
        <div className="h-px bg-slate-700 my-1" />
        <label className="flex items-center gap-2 px-2 py-1.5 hover:bg-slate-800 rounded cursor-pointer">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-slate-700 text-sky-500 focus:ring-offset-slate-900"
            checked={showBid}
            onChange={onToggleBid}
          />
          <span className="text-sm text-slate-300">Show Bid</span>
        </label>
        <label className="flex items-center gap-2 px-2 py-1.5 hover:bg-slate-800 rounded cursor-pointer">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-slate-700 text-sky-500 focus:ring-offset-slate-900"
            checked={showAsk}
            onChange={onToggleAsk}
          />
          <span className="text-sm text-slate-300">Show Ask</span>
        </label>
      </div>
    </div>
  )
}

// Icons
function ChevronDown() {
  return (
    <svg className="w-3 h-3 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  )
}

function RefreshIcon({ className = '' }: { className?: string }) {
  return (
    <svg className={`w-4 h-4 ${className}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  )
}

function PivotIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  )
}

function SRIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  )
}

function DenoiseIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  )
}

function LinesIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
    </svg>
  )
}

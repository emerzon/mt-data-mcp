import { describe, expect, it } from 'vitest'
import {
  addWatchlistSymbol,
  moveWatchlistSymbol,
  normalizeWatchlist,
  removeWatchlistSymbol,
  seedWatchlist,
} from './watchlist'

describe('watchlist helpers', () => {
  it('normalizes, dedupes, and caps names', () => {
    expect(normalizeWatchlist(['eurusd', 'EURUSD', ' gbpusd ', '', 'XAUUSD'], 2)).toEqual([
      'EURUSD',
      'GBPUSD',
    ])
  })

  it('seeds last symbol plus broker majors', () => {
    expect(seedWatchlist('xauusd', ['EURUSD', 'XAUUSD', 'US500'])).toEqual(['XAUUSD', 'EURUSD'])
  })

  it('falls back to catalog names when majors are absent', () => {
    expect(seedWatchlist(undefined, ['US500', 'DE40', 'JP225'])).toEqual(['US500', 'DE40', 'JP225'])
  })

  it('adds, removes, and reorders', () => {
    const added = addWatchlistSymbol(['EURUSD'], 'gbpusd')
    expect(added).toEqual(['EURUSD', 'GBPUSD'])
    expect(removeWatchlistSymbol(added, 'EURUSD')).toEqual(['GBPUSD'])
    expect(moveWatchlistSymbol(['EURUSD', 'GBPUSD', 'USDJPY'], 'USDJPY', -1)).toEqual([
      'EURUSD',
      'USDJPY',
      'GBPUSD',
    ])
  })
})

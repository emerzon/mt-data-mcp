import { describe, expect, it } from 'vitest'
import { formatEpochTime, toUtcSec } from './time'

describe('toUtcSec', () => {
  it('passes through finite epoch seconds', () => {
    expect(toUtcSec(1_700_000_000)).toBe(1_700_000_000)
    expect(toUtcSec(1_700_000_000.9)).toBe(1_700_000_000)
  })

  it('parses date-only and datetime strings as UTC', () => {
    expect(toUtcSec('2024-01-02')).toBe(Date.parse('2024-01-02T00:00:00Z') / 1000)
    expect(toUtcSec('2024-01-02 15:30')).toBe(Date.parse('2024-01-02T15:30:00Z') / 1000)
    expect(toUtcSec('2024-01-02 15:30:45')).toBe(Date.parse('2024-01-02T15:30:45Z') / 1000)
  })

  it('rejects invalid strings', () => {
    expect(() => toUtcSec('not-a-date')).toThrow(/Invalid date/)
  })
})

describe('formatEpochTime', () => {
  it('formats epoch seconds in the requested IANA timezone', () => {
    // 2024-06-01 12:00:00 UTC
    const epoch = Date.parse('2024-06-01T12:00:00Z') / 1000
    const utc = formatEpochTime(epoch, 'UTC')
    expect(utc).toMatch(/12:00:00/)

    const ny = formatEpochTime(epoch, 'America/New_York')
    // EDT (UTC-4) → 08:00:00
    expect(ny).toMatch(/08:00:00/)
  })
})

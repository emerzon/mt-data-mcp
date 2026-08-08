import { describe, expect, it } from 'vitest'
import { resolveConnectionStatus } from './connectionStatus'

describe('resolveConnectionStatus', () => {
  it('reports checking while both probes are unknown', () => {
    const status = resolveConnectionStatus({ healthOk: null, readyOk: null })
    expect(status.kind).toBe('checking')
    expect(status.label).toMatch(/connect/i)
  })

  it('reports api-down when health fails', () => {
    const status = resolveConnectionStatus({
      healthOk: false,
      readyOk: null,
      healthError: 'Network Error',
    })
    expect(status.kind).toBe('api-down')
    expect(status.label).toMatch(/API/i)
    expect(status.hint).toContain('Network Error')
  })

  it('prefers api-down over ready failure', () => {
    const status = resolveConnectionStatus({
      healthOk: false,
      readyOk: false,
      healthError: 'offline',
      readyMessage: 'mt5 down',
    })
    expect(status.kind).toBe('api-down')
  })

  it('reports mt5-not-ready when API is up but ready fails', () => {
    const status = resolveConnectionStatus({
      healthOk: true,
      readyOk: false,
      readyMessage: 'MT5 connection failed',
    })
    expect(status.kind).toBe('mt5-not-ready')
    expect(status.hint).toContain('MT5 connection failed')
  })

  it('reports ok when both probes succeed', () => {
    const status = resolveConnectionStatus({ healthOk: true, readyOk: true })
    expect(status.kind).toBe('ok')
    expect(status.label).toMatch(/connected/i)
  })

  it('reports checking when health ok but ready still unknown', () => {
    const status = resolveConnectionStatus({ healthOk: true, readyOk: null })
    expect(status.kind).toBe('checking')
  })
})

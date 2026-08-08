import { describe, expect, it } from 'vitest'
import { resolveChartWorkspaceStatus } from './workspaceStatus'

describe('resolveChartWorkspaceStatus', () => {
  it('prompts for a symbol when none is selected', () => {
    const status = resolveChartWorkspaceStatus({
      symbol: '  ',
      isLoading: false,
      barsCount: 0,
      historyError: null,
    })
    expect(status.kind).toBe('prompt-symbol')
    expect(status.message).toMatch(/symbol/i)
    expect(status.hint).toBeTruthy()
  })

  it('surfaces primary history errors with a recovery hint', () => {
    const status = resolveChartWorkspaceStatus({
      symbol: 'EURUSD',
      isLoading: false,
      barsCount: 0,
      historyError: 'Symbol not found',
    })
    expect(status).toEqual({
      kind: 'error',
      message: 'Symbol not found',
      hint: expect.stringMatching(/reload/i),
    })
  })

  it('shows loading while the first history fetch is in flight', () => {
    const status = resolveChartWorkspaceStatus({
      symbol: 'XAUUSD',
      isLoading: true,
      isInitialLoading: true,
      barsCount: 0,
      historyError: null,
    })
    expect(status.kind).toBe('loading')
    expect(status.message).toContain('XAUUSD')
  })

  it('shows empty when history settled with zero bars', () => {
    const status = resolveChartWorkspaceStatus({
      symbol: 'EURUSD',
      isLoading: false,
      isInitialLoading: false,
      barsCount: 0,
      historyError: null,
    })
    expect(status.kind).toBe('empty')
    expect(status.message).toMatch(/No history bars/)
    expect(status.hint).toMatch(/timeframe|MT5/i)
  })

  it('is ready when bars are present', () => {
    const status = resolveChartWorkspaceStatus({
      symbol: 'EURUSD',
      isLoading: true,
      barsCount: 120,
      historyError: null,
    })
    expect(status).toEqual({ kind: 'ready' })
  })
})

import { describe, expect, it, vi, beforeEach } from 'vitest'

const getMock = vi.fn()

vi.mock('axios', () => {
  const instance = {
    get: (...args: unknown[]) => getMock(...args),
    post: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  }
  return {
    default: {
      create: () => instance,
      isAxiosError: (error: unknown) =>
        Boolean(error && typeof error === 'object' && (error as { isAxiosError?: boolean }).isAxiosError),
    },
  }
})

describe('getHistory indicator contract', () => {
  beforeEach(() => {
    getMock.mockReset()
    vi.resetModules()
  })

  it('forwards indicators and ohlcv on the history query', async () => {
    getMock.mockResolvedValueOnce({
      data: {
        data: [{ time: 1, open: 1, high: 1, low: 1, close: 1, ema_20: 1.1 }],
        indicator_columns: ['ema_20'],
      },
    })
    const { getHistory } = await import('./client')
    const result = await getHistory({
      symbol: 'EURUSD',
      timeframe: 'H1',
      limit: 200,
      indicators: 'EMA(20)',
      ohlcv: 'ohlcv',
    })
    const [, config] = getMock.mock.calls[0]
    expect(config.params.indicators).toBe('EMA(20)')
    expect(config.params.ohlcv).toBe('ohlcv')
    expect(config.params.timestamp_format).toBe('epoch')
    expect(result.indicator_columns).toEqual(['ema_20'])
  })

  it('omits empty indicator/ohlcv params', async () => {
    getMock.mockResolvedValueOnce({ data: { data: [] } })
    const { getHistory } = await import('./client')
    await getHistory({ symbol: 'EURUSD', timeframe: 'H1', limit: 50 })
    const [, config] = getMock.mock.calls[0]
    expect(config.params.indicators).toBeUndefined()
    expect(config.params.ohlcv).toBeUndefined()
  })
})

import { describe, expect, it } from 'vitest'
import { chartDenoiseFromMethod, defaultDenoiseCausality } from './denoiseSpec'

describe('chartDenoiseFromMethod', () => {
  it('clears denoise for empty/none', () => {
    expect(chartDenoiseFromMethod('')).toBeUndefined()
    expect(chartDenoiseFromMethod('none')).toBeUndefined()
  })

  it('opts into zero_phase for non-causal methods like l1_trend', () => {
    const spec = chartDenoiseFromMethod('l1_trend', {
      method: 'l1_trend',
      available: true,
      description: 'L1 trend',
      params: [],
      supports_causal: false,
      requires_causality_opt_in: true,
      supports: { causality: ['zero_phase'] },
    })
    expect(spec).toEqual(
      expect.objectContaining({
        method: 'l1_trend',
        causality: 'zero_phase',
        when: 'post_ti',
        keep_original: true,
      })
    )
  })

  it('defaults causal methods to causal', () => {
    const spec = chartDenoiseFromMethod('ema', {
      method: 'ema',
      available: true,
      description: 'EMA',
      params: [],
      supports_causal: true,
      requires_causality_opt_in: false,
      supports: { causality: ['causal', 'zero_phase'] },
      defaults: { causality: 'causal' },
    })
    expect(spec?.causality).toBe('causal')
  })

  it('without metadata opts into zero_phase so non-causal methods still work', () => {
    const spec = chartDenoiseFromMethod('l1_trend')
    expect(spec?.method).toBe('l1_trend')
    expect(spec?.causality).toBe('zero_phase')
  })

  it('without metadata defaults dual-mode methods to causal', () => {
    const spec = chartDenoiseFromMethod('ema')
    expect(spec?.method).toBe('ema')
    expect(spec?.causality).toBe('causal')
  })
})

describe('defaultDenoiseCausality', () => {
  it('uses causal for dual-mode methods without metadata', () => {
    expect(defaultDenoiseCausality('kalman')).toBe('causal')
    expect(defaultDenoiseCausality('savgol')).toBe('causal')
    expect(defaultDenoiseCausality('supersmoother')).toBe('causal')
    expect(defaultDenoiseCausality('kama')).toBe('causal')
    expect(defaultDenoiseCausality('kalman_robust')).toBe('causal')
    expect(defaultDenoiseCausality('preaverage')).toBe('causal')
  })

  it('uses zero_phase for known non-causal methods without metadata', () => {
    expect(defaultDenoiseCausality('wavelet')).toBe('zero_phase')
  })
})

describe('ensureChartDenoiseCausality', () => {
  it('fills missing causality on legacy saved specs', async () => {
    const { ensureChartDenoiseCausality } = await import('./denoiseSpec')
    const fixed = ensureChartDenoiseCausality({ method: 'l1_trend', params: {} })
    expect(fixed?.causality).toBe('zero_phase')
    expect(fixed?.method).toBe('l1_trend')
  })

  it('preserves explicit causality', async () => {
    const { ensureChartDenoiseCausality } = await import('./denoiseSpec')
    const kept = ensureChartDenoiseCausality({
      method: 'ema',
      causality: 'causal',
      params: { alpha: 0.2 },
    })
    expect(kept?.causality).toBe('causal')
  })
})

/**
 * Pure helpers for chart-workspace denoise specs (history API query shape).
 */

import type { DenoiseMethodInfo, DenoiseSpecUI } from '../types'

/** Methods that cannot run causally. Keep in sync with `_DENOISE_METHOD_CAUSALITY_SUPPORT`. */
export const ZERO_PHASE_ONLY_METHODS = new Set([
  'lowpass_fft',
  'wavelet',
  'wavelet_packet',
  'hp',
  'whittaker',
  'l1_trend',
  'gaussian',
  'loess',
  'stl',
  'tv',
  'ssa',
  'vmd',
  'emd',
  'eemd',
  'ceemdan',
])

export function defaultDenoiseCausality(
  method: string,
  methodMeta?: DenoiseMethodInfo | null
): 'causal' | 'zero_phase' {
  if (
    methodMeta?.requires_causality_opt_in === true ||
    methodMeta?.supports_causal === false ||
    (Array.isArray(methodMeta?.supports?.causality) &&
      !methodMeta.supports.causality.includes('causal'))
  ) {
    return 'zero_phase'
  }
  if (methodMeta?.defaults?.causality === 'causal' || methodMeta?.defaults?.causality === 'zero_phase') {
    return methodMeta.defaults.causality
  }
  const name = String(method || '').trim().toLowerCase()
  return ZERO_PHASE_ONLY_METHODS.has(name) ? 'zero_phase' : 'causal'
}

/**
 * Build a denoise spec when the user picks a method from the chart Filter menu.
 * Non-causal methods (e.g. l1_trend) require explicit causality='zero_phase' opt-in
 * for the history API; dual-mode methods default to causal.
 */
export function chartDenoiseFromMethod(
  method: string,
  methodMeta?: DenoiseMethodInfo | null,
  previous?: DenoiseSpecUI
): DenoiseSpecUI | undefined {
  const name = String(method || '').trim()
  if (!name || name.toLowerCase() === 'none') return undefined

  const resolvedDefault = defaultDenoiseCausality(name, methodMeta)
  let causality: 'zero_phase' | 'causal'
  if (resolvedDefault === 'zero_phase') {
    causality = 'zero_phase'
  } else if (previous?.method === name && (previous.causality === 'causal' || previous.causality === 'zero_phase')) {
    causality = previous.causality
  } else {
    causality = resolvedDefault
  }

  return {
    method: name,
    params: previous?.method === name && previous.params ? { ...previous.params } : {},
    columns: previous?.columns,
    when: previous?.when ?? 'post_ti',
    causality,
    keep_original: previous?.keep_original ?? true,
  }
}

/**
 * Ensure a stored/loaded denoise spec includes causality before history requests.
 * Older UI saves only sent `{ method, params }` which 400s for non-causal methods.
 */
export function ensureChartDenoiseCausality(spec?: DenoiseSpecUI | null): DenoiseSpecUI | undefined {
  if (!spec?.method) return undefined
  if (spec.causality === 'causal' || spec.causality === 'zero_phase') return spec
  return chartDenoiseFromMethod(spec.method, null, spec)
}

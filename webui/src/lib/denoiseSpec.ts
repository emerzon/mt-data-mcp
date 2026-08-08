/**
 * Pure helpers for chart-workspace denoise specs (history API query shape).
 */

import type { DenoiseMethodInfo, DenoiseSpecUI } from '../types'

/**
 * Build a denoise spec when the user picks a method from the chart Filter menu.
 * Non-causal methods (e.g. l1_trend) require explicit causality='zero_phase' opt-in
 * for the history API; chart research always opts in for retrospective overlays.
 */
export function chartDenoiseFromMethod(
  method: string,
  methodMeta?: DenoiseMethodInfo | null,
  previous?: DenoiseSpecUI
): DenoiseSpecUI | undefined {
  const name = String(method || '').trim()
  if (!name || name.toLowerCase() === 'none') return undefined

  const hasMeta = Boolean(methodMeta)
  const requiresOptIn =
    methodMeta?.requires_causality_opt_in === true ||
    methodMeta?.supports_causal === false ||
    (Array.isArray(methodMeta?.supports?.causality) &&
      !methodMeta!.supports!.causality!.includes('causal'))

  // Chart overlays are retrospective. Non-causal methods must opt into zero_phase;
  // when method metadata is not loaded yet, prefer zero_phase so filters like
  // l1_trend do not hit the history API without consent.
  let causality: 'zero_phase' | 'causal'
  if (requiresOptIn || !hasMeta) {
    causality = 'zero_phase'
  } else if (previous?.method === name && (previous.causality === 'causal' || previous.causality === 'zero_phase')) {
    causality = previous.causality
  } else if (methodMeta?.defaults?.causality === 'causal' || methodMeta?.defaults?.causality === 'zero_phase') {
    causality = methodMeta.defaults.causality
  } else {
    causality = 'causal'
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

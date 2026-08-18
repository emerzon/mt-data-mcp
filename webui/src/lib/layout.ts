/**
 * Pure layout helpers for responsive chart workspace chrome.
 * Widths align with WEBUI_GOAL: ~375 mobile, ~768 tablet, ~1440 desktop.
 */

export type LayoutBreakpoint = 'mobile' | 'tablet' | 'desktop'

/** Tailwind-aligned: <768 mobile, <1280 tablet, else desktop. */
export function resolveLayoutBreakpoint(widthPx: number): LayoutBreakpoint {
  if (!Number.isFinite(widthPx) || widthPx < 0) return 'desktop'
  if (widthPx < 768) return 'mobile'
  if (widthPx < 1280) return 'tablet'
  return 'desktop'
}

/**
 * Forecast / analysis panel placement classes (Tailwind).
 * Mobile: bottom sheet; tablet/desktop: right drawer with capped width.
 */
export function forecastPanelPlacementClass(bp: LayoutBreakpoint): string {
  if (bp === 'mobile') {
    return [
      'fixed inset-x-0 bottom-0 top-auto z-30',
      'max-h-[min(92vh,100%)] w-full',
      'rounded-t-xl border border-slate-800 border-b-0',
      'bg-slate-900/98 backdrop-blur-sm shadow-2xl',
      'flex flex-col',
    ].join(' ')
  }
  return [
    'absolute top-0 right-0 bottom-0 z-30',
    'w-full max-w-md sm:w-[min(420px,100%)]',
    'border-l border-slate-800',
    'bg-slate-900/98 backdrop-blur-sm shadow-2xl',
    'flex flex-col',
  ].join(' ')
}

/** Watchlist / radar placement: left drawer on desktop, bottom sheet on mobile. */
export function radarPanelPlacementClass(bp: LayoutBreakpoint): string {
  if (bp === 'mobile') {
    return forecastPanelPlacementClass(bp)
  }
  return [
    'absolute top-0 left-0 bottom-0 z-30',
    'w-full max-w-sm sm:w-[min(360px,100%)]',
    'border-r border-slate-800',
    'bg-slate-900/98 backdrop-blur-sm shadow-2xl',
    'flex flex-col',
  ].join(' ')
}

/** Whether the toolbar should collapse secondary actions into a More menu. */
export function toolbarUsesOverflowMenu(bp: LayoutBreakpoint): boolean {
  return bp === 'mobile' || bp === 'tablet'
}

/** Touch-friendly min size for primary toolbar controls on compact layouts. */
export function toolbarControlMinClass(bp: LayoutBreakpoint): string {
  return bp === 'mobile' ? 'min-h-11 min-w-11' : ''
}

import { describe, expect, it } from 'vitest'
import {
  forecastPanelPlacementClass,
  resolveLayoutBreakpoint,
  toolbarUsesOverflowMenu,
} from './layout'

describe('resolveLayoutBreakpoint', () => {
  it('maps goal reference widths', () => {
    expect(resolveLayoutBreakpoint(375)).toBe('mobile')
    expect(resolveLayoutBreakpoint(768)).toBe('tablet')
    expect(resolveLayoutBreakpoint(1440)).toBe('desktop')
  })

  it('uses 768 and 1280 as boundaries', () => {
    expect(resolveLayoutBreakpoint(767)).toBe('mobile')
    expect(resolveLayoutBreakpoint(1279)).toBe('tablet')
    expect(resolveLayoutBreakpoint(1280)).toBe('desktop')
  })
})

describe('forecastPanelPlacementClass', () => {
  it('uses bottom-sheet classes on mobile', () => {
    const cls = forecastPanelPlacementClass('mobile')
    expect(cls).toContain('bottom-0')
    expect(cls).toContain('fixed')
    expect(cls).toMatch(/max-h/)
  })

  it('uses right drawer classes on tablet/desktop', () => {
    const tablet = forecastPanelPlacementClass('tablet')
    const desktop = forecastPanelPlacementClass('desktop')
    expect(tablet).toContain('right-0')
    expect(desktop).toContain('right-0')
    expect(tablet).not.toContain('fixed inset-x-0 bottom-0')
  })
})

describe('toolbarUsesOverflowMenu', () => {
  it('collapses chrome on mobile and tablet only', () => {
    expect(toolbarUsesOverflowMenu('mobile')).toBe(true)
    expect(toolbarUsesOverflowMenu('tablet')).toBe(true)
    expect(toolbarUsesOverflowMenu('desktop')).toBe(false)
  })
})

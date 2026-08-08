import { useEffect, useState } from 'react'
import { resolveLayoutBreakpoint, type LayoutBreakpoint } from '../lib/layout'

function readWidth(): number {
  if (typeof window === 'undefined') return 1440
  return window.innerWidth
}

/** Live layout breakpoint for responsive chrome (mobile / tablet / desktop). */
export function useViewportBreakpoint(): LayoutBreakpoint {
  const [bp, setBp] = useState<LayoutBreakpoint>(() => resolveLayoutBreakpoint(readWidth()))

  useEffect(() => {
    const onResize = () => setBp(resolveLayoutBreakpoint(window.innerWidth))
    onResize()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  return bp
}

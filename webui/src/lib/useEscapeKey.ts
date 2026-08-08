import { useEffect } from 'react'

/**
 * Call `onEscape` when Escape is pressed while `enabled` is true.
 * Used by forecast panel, denoise modal, and other sheets.
 */
export function useEscapeKey(enabled: boolean, onEscape: () => void): void {
  useEffect(() => {
    if (!enabled) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onEscape()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [enabled, onEscape])
}

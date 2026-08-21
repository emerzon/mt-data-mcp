/**
 * Coerce a string value to the appropriate JS type.
 * Used for parsing form inputs.
 */
export function coerce(v: string): string | number | boolean {
  const t = v.trim()
  if (t === '') return ''
  if (!Number.isNaN(Number(t))) return Number(t)
  if (t === 'true') return true
  if (t === 'false') return false
  return t
}

/**
 * Format a Date object or epoch timestamp to ISO-like string (YYYY-MM-DD HH:MM:SS).
 */
export function formatDateTime(input: Date | number): string {
  const date = typeof input === 'number' ? new Date(input * 1000) : input
  return date.toISOString().slice(0, 19).replace('T', ' ')
}

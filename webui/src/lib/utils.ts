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
 * Parse a parameter value from string input based on its type definition.
 */
export function parseParamValue(input: string, type?: string): unknown {
  const raw = input.trim()
  if (!raw) return ''
  
  const t = (type || '').toLowerCase()
  
  if (t === 'int') {
    const v = Number.parseInt(raw, 10)
    return Number.isNaN(v) ? raw : v
  }
  
  if (t === 'float') {
    const v = Number.parseFloat(raw)
    return Number.isNaN(v) ? raw : v
  }
  
  if (t === 'bool') {
    if (raw === 'true' || raw === '1') return true
    if (raw === 'false' || raw === '0') return false
    return raw
  }
  
  if (t.startsWith('list')) {
    const inner = t.includes('[') 
      ? t.slice(t.indexOf('[') + 1, t.lastIndexOf(']')).trim() 
      : 'str'
    const parts = raw.split(/[\s,]+/).map(p => p.trim()).filter(Boolean)
    
    if (inner === 'int') {
      return parts.map(p => Number.parseInt(p, 10)).filter(v => !Number.isNaN(v))
    }
    if (inner === 'float') {
      return parts.map(p => Number.parseFloat(p)).filter(v => !Number.isNaN(v))
    }
    return parts
  }
  
  if (t === 'json') {
    try {
      return JSON.parse(raw)
    } catch {
      return raw
    }
  }
  
  return raw
}

/**
 * Convert a value to string for display in form inputs.
 */
export function stringifyValue(value: unknown, type?: string): string {
  if (value === undefined || value === null) return ''
  
  const lower = type?.toLowerCase() ?? ''
  
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return value.join(', ')
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : ''
  
  if (lower.startsWith('json') && typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch {
      return ''
    }
  }
  
  return String(value)
}

/**
 * Check if a parameter value is "empty" (should be omitted from API calls).
 */
export function isEmptyValue(value: unknown): boolean {
  if (value === '' || value === undefined || value === null) return true
  if (Array.isArray(value)) return value.length === 0
  return false
}

/**
 * Format a Date object or epoch timestamp to ISO-like string (YYYY-MM-DD HH:MM:SS).
 */
export function formatDateTime(input: Date | number): string {
  const date = typeof input === 'number' ? new Date(input * 1000) : input
  return date.toISOString().slice(0, 19).replace('T', ' ')
}

/**
 * Format a number with locale-aware formatting.
 */
export function formatNumber(
  value: number,
  options?: { minDecimals?: number; maxDecimals?: number }
): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: options?.minDecimals ?? 2,
    maximumFractionDigits: options?.maxDecimals ?? 6,
  })
}

/**
 * Clamp a number between min and max values.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/**
 * Classnames helper (simple version).
 */
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ')
}

/**
 * Debounce a function.
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

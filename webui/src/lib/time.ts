export function toUtcSec(input: number | string): number {
  if (typeof input === 'number' && Number.isFinite(input)) return Math.floor(input)
  const s = String(input).trim()
  // Accept formats like 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD HH:MM:SS'
  // Convert to ISO by replacing space with 'T' and appending 'Z'
  let iso = s.replace(' ', 'T')
  // If time lacks seconds, add :00
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(iso)) iso += ':00'
  // If only date is given, make it midnight
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) iso += 'T00:00:00'
  if (!iso.endsWith('Z')) iso += 'Z'
  const ms = Date.parse(iso)
  if (Number.isNaN(ms)) throw new Error(`Invalid date string=${input}`)
  return Math.floor(ms / 1000)
}


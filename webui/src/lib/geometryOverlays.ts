import type { PriceLineSpec } from '../components/OHLCChart'

export type ConfluenceLevel = {
  price: number
  type?: string
  score?: number
  range?: { low?: number; high?: number }
}

export type VolumeProfileLevels = {
  poc?: number
  vah?: number
  val?: number
}

export type ExposureRow = {
  ticket?: number | string
  type?: string
  volume?: number
  price?: number
  sl?: number
  tp?: number
}

export type IdeaGeometry = {
  entry?: number
  take_profit?: number
  stop_loss?: number
  direction?: string
}

function line(price: number | undefined, color: string, title: string): PriceLineSpec | null {
  if (typeof price !== 'number' || !Number.isFinite(price) || price <= 0) return null
  return { price, color, title }
}

export function confluencePriceLines(levels: ConfluenceLevel[] | null | undefined): PriceLineSpec[] {
  if (!levels?.length) return []
  return levels
    .map((level) => {
      const kind = String(level.type || '').toLowerCase()
      const color = kind === 'support' ? '#34d399' : kind === 'resistance' ? '#fb7185' : '#94a3b8'
      const title = kind ? `Conf ${kind}` : 'Confluence'
      return line(level.price, color, title)
    })
    .filter((item): item is PriceLineSpec => item !== null)
}

export function volumeProfilePriceLines(
  profile: VolumeProfileLevels | null | undefined
): PriceLineSpec[] {
  if (!profile) return []
  return [
    line(profile.poc, '#a78bfa', 'POC'),
    line(profile.vah, '#818cf8', 'VAH'),
    line(profile.val, '#818cf8', 'VAL'),
  ].filter((item): item is PriceLineSpec => item !== null)
}

export function ideaGeometryPriceLines(geometry: IdeaGeometry | null | undefined): PriceLineSpec[] {
  if (!geometry) return []
  return [
    line(geometry.entry, '#38bdf8', 'Idea entry'),
    line(geometry.take_profit, '#22c55e', 'Idea TP'),
    line(geometry.stop_loss, '#ef4444', 'Idea SL'),
  ].filter((item): item is PriceLineSpec => item !== null)
}

export function exposurePriceLines(
  positions: ExposureRow[] | null | undefined,
  pending: ExposureRow[] | null | undefined
): PriceLineSpec[] {
  const lines: PriceLineSpec[] = []
  for (const row of positions ?? []) {
    const ticket = row.ticket != null ? String(row.ticket) : 'pos'
    const open = line(row.price, '#f59e0b', `Pos ${ticket}`)
    if (open) lines.push(open)
    const sl = line(row.sl, '#ef4444', `Pos SL ${ticket}`)
    if (sl) lines.push(sl)
    const tp = line(row.tp, '#22c55e', `Pos TP ${ticket}`)
    if (tp) lines.push(tp)
  }
  for (const row of pending ?? []) {
    const ticket = row.ticket != null ? String(row.ticket) : 'pend'
    const pendingLine = line(row.price, '#94a3b8', `Pend ${ticket}`)
    if (pendingLine) lines.push(pendingLine)
  }
  return lines
}

export function tfSeconds(tf: string): number {
  const T: Record<string, number> = {
    'M1': 60, 'M2': 120, 'M3': 180, 'M4': 240, 'M5': 300,
    'M10': 600, 'M12': 720, 'M15': 900, 'M20': 1200, 'M30': 1800,
    'H1': 3600, 'H2': 7200, 'H3': 10800, 'H4': 14400, 'H6': 21600, 'H8': 28800, 'H12': 43200,
    'D1': 86400, 'W1': 604800, 'MN1': 2629800, // approximate month length
  }
  return T[tf?.toUpperCase()] ?? 3600
}


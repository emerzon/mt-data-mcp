export function loadJSON<T>(key: string): T | undefined {
  try {
    const s = localStorage.getItem(key)
    if (!s) return undefined
    return JSON.parse(s) as T
  } catch {
    return undefined
  }
}

export function saveJSON(key: string, value: any) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // ignore
  }
}


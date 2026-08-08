/**
 * Pure catalog / param / safety helpers for the schema-driven Tools runner.
 */

export type ToolSurface = 'dedicated_ui' | 'generic_runner' | 'intentional_omit'

export type ToolField = {
  name: string
  required: boolean
  default?: unknown
  type?: string
  description?: string | null
}

export type ToolSafety = {
  requires_confirmation?: boolean
  is_live_trade_mutation?: boolean
  surface?: ToolSurface
  dedicated_path?: string
  omit_rationale?: string
  warning?: string
}

export type ToolCatalogEntry = {
  name: string
  category?: string
  description?: string
  surface?: ToolSurface
  parameters?: Record<string, string>
  fields?: ToolField[]
  safety?: ToolSafety
  enabled?: boolean
  enable_env?: string
  status?: string
  why_disabled?: string
  related_tools?: string[]
}

export type ToolParamValues = Record<string, string>

/** Human label from snake_case tool/param names. */
export function humanizeIdentifier(raw: string): string {
  const text = String(raw || '').trim()
  if (!text) return ''
  return text
    .replace(/[_.-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase())
}

export function filterToolCatalog(
  tools: ToolCatalogEntry[],
  opts: { search?: string; category?: string } = {}
): ToolCatalogEntry[] {
  const search = (opts.search ?? '').trim().toLowerCase()
  const category = (opts.category ?? '').trim().toLowerCase()
  return tools.filter((tool) => {
    if (category && String(tool.category || '').toLowerCase() !== category) return false
    if (!search) return true
    const hay = [tool.name, tool.category, tool.description]
      .map((part) => String(part || '').toLowerCase())
      .join(' ')
    return hay.includes(search)
  })
}

export function uniqueCategories(tools: ToolCatalogEntry[]): string[] {
  const set = new Set<string>()
  for (const tool of tools) {
    const c = String(tool.category || '').trim()
    if (c) set.add(c)
  }
  return Array.from(set).sort((a, b) => a.localeCompare(b))
}

/** Build initial form values from field metadata (defaults as strings for inputs). */
export function defaultParamValues(fields: ToolField[] | undefined | null): ToolParamValues {
  const out: ToolParamValues = {}
  for (const field of fields ?? []) {
    if (!field?.name) continue
    if (field.default === undefined || field.default === null) {
      out[field.name] = ''
      continue
    }
    if (typeof field.default === 'string') {
      out[field.name] = field.default
    } else {
      try {
        out[field.name] = JSON.stringify(field.default)
      } catch {
        out[field.name] = String(field.default)
      }
    }
  }
  return out
}

/**
 * Coerce form strings into JSON-friendly argument values for POST /tools/{name}/invoke.
 * Empty optional strings are omitted; required empties stay as "" so the server can reject.
 */
export function shapeInvokeArguments(
  fields: ToolField[] | undefined | null,
  values: ToolParamValues
): Record<string, unknown> {
  const args: Record<string, unknown> = {}
  const fieldMap = new Map((fields ?? []).map((f) => [f.name, f]))

  for (const [name, raw] of Object.entries(values)) {
    const text = raw ?? ''
    const field = fieldMap.get(name)
    const trimmed = text.trim()
    if (!trimmed) {
      if (field?.required) {
        args[name] = ''
      }
      continue
    }
    args[name] = coerceParamValue(trimmed, field?.type)
  }
  return args
}

export function coerceParamValue(text: string, typeHint?: string): unknown {
  const t = (typeHint || '').toLowerCase()
  const value = text.trim()
  if (!value) return value

  if (t.includes('bool')) {
    const lower = value.toLowerCase()
    if (['true', '1', 'yes', 'on'].includes(lower)) return true
    if (['false', '0', 'no', 'off'].includes(lower)) return false
  }

  if (
    (value.startsWith('{') && value.endsWith('}')) ||
    (value.startsWith('[') && value.endsWith(']'))
  ) {
    try {
      return JSON.parse(value) as unknown
    } catch {
      // fall through
    }
  }

  if (t.includes('int') && !t.includes('float') && /^-?\d+$/.test(value)) {
    return Number(value)
  }
  if ((t.includes('float') || t.includes('number')) && /^-?\d+(\.\d+)?$/.test(value)) {
    return Number(value)
  }
  if (!t && /^-?\d+$/.test(value)) return Number(value)
  if (!t && /^-?\d+\.\d+$/.test(value)) return Number(value)

  return value
}

export function formatToolResult(result: unknown): string {
  if (result === undefined) return ''
  if (typeof result === 'string') return result
  try {
    return JSON.stringify(result, null, 2)
  } catch {
    return String(result)
  }
}

export function toolIsRunnable(tool: ToolCatalogEntry | null | undefined): boolean {
  if (!tool?.name) return false
  if (tool.surface === 'intentional_omit') return false
  if (tool.enabled === false) return false
  return true
}

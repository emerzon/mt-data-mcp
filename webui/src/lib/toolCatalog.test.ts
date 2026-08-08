import { describe, expect, it } from 'vitest'
import {
  coerceParamValue,
  defaultParamValues,
  filterToolCatalog,
  formatToolResult,
  humanizeIdentifier,
  shapeInvokeArguments,
  toolIsRunnable,
  uniqueCategories,
  type ToolCatalogEntry,
  type ToolField,
} from './toolCatalog'

const SAMPLE: ToolCatalogEntry[] = [
  {
    name: 'forecast_generate',
    category: 'forecast',
    description: 'Generate forecasts',
    surface: 'dedicated_ui',
  },
  {
    name: 'trade_place',
    category: 'trading',
    description: 'Place live order',
    surface: 'generic_runner',
    safety: { requires_confirmation: true, is_live_trade_mutation: true },
  },
  {
    name: 'market_depth_fetch',
    category: 'market',
    description: 'DOM depth',
    surface: 'generic_runner',
    enabled: false,
  },
]

describe('humanizeIdentifier', () => {
  it('turns snake_case into title words', () => {
    expect(humanizeIdentifier('trade_place')).toBe('Trade Place')
    expect(humanizeIdentifier('ci_alpha')).toBe('Ci Alpha')
  })
})

describe('filterToolCatalog', () => {
  it('filters by search and category using shipped entries', () => {
    expect(filterToolCatalog(SAMPLE, { search: 'trade' }).map((t) => t.name)).toEqual([
      'trade_place',
    ])
    expect(filterToolCatalog(SAMPLE, { category: 'forecast' }).map((t) => t.name)).toEqual([
      'forecast_generate',
    ])
  })
})

describe('uniqueCategories', () => {
  it('returns sorted unique categories', () => {
    expect(uniqueCategories(SAMPLE)).toEqual(['forecast', 'market', 'trading'])
  })
})

describe('defaultParamValues + shapeInvokeArguments', () => {
  const fields: ToolField[] = [
    { name: 'symbol', required: true, type: 'str' },
    { name: 'horizon', required: false, default: 12, type: 'int' },
    { name: 'params', required: false, type: 'Dict[str, Any]' },
    { name: 'async_mode', required: false, default: false, type: 'bool' },
  ]

  it('seeds defaults as form strings', () => {
    expect(defaultParamValues(fields)).toEqual({
      symbol: '',
      horizon: '12',
      params: '',
      async_mode: 'false',
    })
  })

  it('shapes invoke payload: omit empty optionals, coerce types', () => {
    const shaped = shapeInvokeArguments(fields, {
      symbol: 'EURUSD',
      horizon: '24',
      params: '{"alpha": 0.1}',
      async_mode: 'true',
    })
    expect(shaped).toEqual({
      symbol: 'EURUSD',
      horizon: 24,
      params: { alpha: 0.1 },
      async_mode: true,
    })
  })

  it('keeps required empty string so validation can fail server-side', () => {
    const shaped = shapeInvokeArguments(fields, {
      symbol: '',
      horizon: '',
      params: '',
      async_mode: '',
    })
    expect(shaped).toEqual({ symbol: '' })
  })
})

describe('coerceParamValue', () => {
  it('parses bools, ints, and json', () => {
    expect(coerceParamValue('yes', 'bool')).toBe(true)
    expect(coerceParamValue('off', 'boolean')).toBe(false)
    expect(coerceParamValue('42', 'int')).toBe(42)
    expect(coerceParamValue('[1,2]', 'list')).toEqual([1, 2])
  })
})

describe('formatToolResult', () => {
  it('pretty-prints objects', () => {
    expect(formatToolResult({ ok: true })).toContain('"ok"')
    expect(formatToolResult('plain')).toBe('plain')
  })
})

describe('toolIsRunnable', () => {
  it('blocks omit and disabled tools', () => {
    expect(toolIsRunnable(SAMPLE[0])).toBe(true)
    expect(toolIsRunnable(SAMPLE[2])).toBe(false)
    expect(toolIsRunnable({ name: 'x', surface: 'intentional_omit' })).toBe(false)
  })
})

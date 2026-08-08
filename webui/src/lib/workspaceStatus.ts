/**
 * Pure chart-workspace status resolution (no React / DOM).
 * Used by the shell for loading / error / empty affordances and unit-tested in-repo.
 */

export type ChartWorkspaceStatusKind =
  | 'prompt-symbol'
  | 'loading'
  | 'error'
  | 'empty'
  | 'ready'

export type ChartWorkspaceStatus = {
  kind: ChartWorkspaceStatusKind
  /** Primary user-facing message for error / empty / prompt states. */
  message?: string
  /** Optional recovery hint shown under the primary message. */
  hint?: string
}

export type ChartWorkspaceStatusInput = {
  symbol: string
  /** True while the primary history query is in flight (including background refetch). */
  isLoading: boolean
  /** True when React Query has not yet settled the first history fetch. */
  isInitialLoading?: boolean
  barsCount: number
  /** Primary history failure message, if any. */
  historyError: string | null
}

/**
 * Resolve the primary chart surface state for symbol/history failures and empty series.
 */
export function resolveChartWorkspaceStatus(
  input: ChartWorkspaceStatusInput
): ChartWorkspaceStatus {
  const symbol = (input.symbol ?? '').trim()
  if (!symbol) {
    return {
      kind: 'prompt-symbol',
      message: 'Select a symbol to load the chart',
      hint: 'Use the symbol search in the toolbar to start.',
    }
  }

  if (input.historyError) {
    return {
      kind: 'error',
      message: input.historyError,
      hint: 'Check the symbol, API auth token, and that mtdata-webapi can reach MT5. Then reload.',
    }
  }

  const initial = input.isInitialLoading ?? input.isLoading
  if (initial && input.barsCount === 0) {
    return {
      kind: 'loading',
      message: `Loading ${symbol} history…`,
    }
  }

  if (input.barsCount === 0) {
    return {
      kind: 'empty',
      message: `No history bars for ${symbol}`,
      hint: 'Try another timeframe or confirm the symbol is available in MT5.',
    }
  }

  return { kind: 'ready' }
}

/**
 * Pure connection / readiness status for the chart workspace chrome.
 * Driven by /api/v1/health and /api/v1/ready responses (or fetch failures).
 */

export type ConnectionStatusKind = 'checking' | 'ok' | 'api-down' | 'mt5-not-ready'

export type ConnectionStatus = {
  kind: ConnectionStatusKind
  /** Short toolbar label */
  label: string
  /** Longer hint for title / tooltip */
  hint?: string
}

export type ConnectionStatusInput = {
  /** null = still loading / unknown */
  healthOk: boolean | null
  /** null = still loading / unknown; only meaningful when healthOk is true */
  readyOk: boolean | null
  healthError?: string | null
  readyMessage?: string | null
}

/**
 * Resolve API liveness + MT5 readiness into a non-blocking chrome status.
 * Health failure wins over readiness (cannot talk to API).
 */
export function resolveConnectionStatus(input: ConnectionStatusInput): ConnectionStatus {
  if (input.healthOk === null && input.readyOk === null) {
    return {
      kind: 'checking',
      label: 'Connecting…',
      hint: 'Checking API health and MT5 readiness.',
    }
  }

  if (input.healthOk === false) {
    return {
      kind: 'api-down',
      label: 'API down',
      hint: input.healthError?.trim() || 'Cannot reach /api/v1/health. Is mtdata-webapi running?',
    }
  }

  if (input.healthOk === true && input.readyOk === null) {
    return {
      kind: 'checking',
      label: 'API ok…',
      hint: 'API is up; checking MT5 readiness.',
    }
  }

  if (input.readyOk === false) {
    return {
      kind: 'mt5-not-ready',
      label: 'MT5 not ready',
      hint:
        input.readyMessage?.trim() ||
        'API is up but MT5 readiness failed. Terminal may be closed or credentials missing.',
    }
  }

  return {
    kind: 'ok',
    label: 'Connected',
    hint: 'API healthy and MT5 readiness reported OK.',
  }
}

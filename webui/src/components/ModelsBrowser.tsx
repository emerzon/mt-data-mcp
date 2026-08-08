import { useQuery } from '@tanstack/react-query'
import { getErrorMessage, getModels } from '../api/client'
import type { StoredModelInfo } from '../types'

type Props = {
  methodFilter?: string
  compact?: boolean
}

function modelLabel(model: StoredModelInfo): string {
  const id = String(model.model_id ?? model.id ?? model.path ?? 'model')
  const method = model.method ? String(model.method) : ''
  const symbol = model.symbol ? String(model.symbol) : ''
  const tf = model.timeframe ? String(model.timeframe) : ''
  const parts = [id, method, [symbol, tf].filter(Boolean).join(' ')].filter(Boolean)
  return parts.join(' · ')
}

/**
 * Discoverable browser for GET /api/v1/models (cached / trained models).
 */
export function ModelsBrowser({ methodFilter, compact }: Props) {
  const { data, error, isFetching, refetch, isFetched } = useQuery({
    queryKey: ['models', methodFilter || ''],
    queryFn: ({ signal }) => getModels(methodFilter || undefined, signal),
    staleTime: 30_000,
  })

  const models = data?.models ?? []
  const count = data?.count ?? models.length

  return (
    <div className="space-y-2" data-models-browser>
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs text-slate-400">
          Stored models
          {methodFilter ? (
            <span className="text-slate-500"> · filter: {methodFilter}</span>
          ) : null}
        </div>
        <button
          type="button"
          className="text-xs text-sky-400 hover:text-sky-300"
          onClick={() => void refetch()}
          disabled={isFetching}
        >
          {isFetching ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="text-xs text-rose-300 bg-rose-950/40 border border-rose-900 rounded px-2 py-1.5">
          {getErrorMessage(error)}
        </div>
      )}

      {!error && isFetched && count === 0 && (
        <p className="text-xs text-slate-500">
          No stored models found{methodFilter ? ` for ${methodFilter}` : ''}. Train or cache models
          via CLI/MCP to list them here.
        </p>
      )}

      {models.length > 0 && (
        <ul
          className={`space-y-1 overflow-y-auto ${compact ? 'max-h-28' : 'max-h-40'} rounded border border-slate-800 bg-slate-950/50 p-1`}
        >
          {models.map((model, index) => {
            const key = String(model.model_id ?? model.id ?? model.path ?? index)
            return (
              <li
                key={key}
                className="px-2 py-1.5 text-[11px] text-slate-300 rounded hover:bg-slate-800/80 truncate"
                title={modelLabel(model)}
              >
                {modelLabel(model)}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

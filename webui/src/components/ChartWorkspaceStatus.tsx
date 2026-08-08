import type { ChartWorkspaceStatus } from '../lib/workspaceStatus'

type Props = {
  status: ChartWorkspaceStatus
  onReload?: () => void
}

/**
 * Full-viewport feedback for the primary chart path (prompt / loading / error / empty).
 */
export function ChartWorkspaceStatusView({ status, onReload }: Props) {
  if (status.kind === 'ready') return null

  const showReload = status.kind === 'error' || status.kind === 'empty'

  return (
    <div
      className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950/80 backdrop-blur-[1px] p-6"
      role="status"
      aria-live="polite"
      data-workspace-status={status.kind}
    >
      <div className="max-w-md w-full rounded-xl border border-slate-800 bg-slate-900/95 px-5 py-4 shadow-xl animate-fade-in">
        <div className="text-[11px] font-semibold tracking-wider uppercase text-sky-400 mb-2">
          MTData · Chart Workspace
        </div>
        {status.kind === 'loading' ? (
          <div className="flex items-center gap-3 text-slate-200">
            <span className="spinner text-sky-400" aria-hidden />
            <span className="text-sm font-medium">{status.message}</span>
          </div>
        ) : (
          <>
            <div
              className={
                status.kind === 'error'
                  ? 'text-sm font-medium text-rose-200'
                  : 'text-sm font-medium text-slate-100'
              }
            >
              {status.message}
            </div>
            {status.hint ? (
              <p className="mt-2 text-xs leading-relaxed text-slate-400">{status.hint}</p>
            ) : null}
            {showReload && onReload ? (
              <button type="button" className="btn mt-4" onClick={onReload}>
                Reload chart
              </button>
            ) : null}
          </>
        )}
      </div>
    </div>
  )
}

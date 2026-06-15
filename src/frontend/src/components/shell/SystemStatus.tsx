import { useQuery } from '@tanstack/react-query'
import { getHealth, getPerformanceStats, getStats } from '@/services/api'

// Slim app-wide status line. This is where retrieval/index telemetry lives now —
// out of the page heroes, where users were reading "p95 latency" instead of jobs.
// Lazy-loaded by the shell so it stays out of the entry chunk (api.ts comes with it).
export default function SystemStatus() {
  const stats = useQuery({ queryKey: ['stats'], queryFn: getStats, staleTime: 5 * 60 * 1000 })
  const perf = useQuery({
    queryKey: ['performanceStats'],
    queryFn: () => getPerformanceStats(30),
    staleTime: 5 * 60 * 1000,
  })
  const health = useQuery({ queryKey: ['health'], queryFn: getHealth, staleTime: 60 * 1000 })

  // Three-state, not a boolean: don't assert "healthy" while the probe is loading
  // or has errored — only claim it when the health query actually succeeded.
  const status: 'healthy' | 'limited' | 'unknown' = health.isSuccess
    ? health.data.degraded
      ? 'limited'
      : 'healthy'
    : 'unknown'

  const facts = [
    stats.data && `${stats.data.total_jobs.toLocaleString()} jobs indexed`,
    stats.data && `${stats.data.embedding_coverage_pct.toFixed(0)}% with embeddings`,
    perf.data && `search p95 ${perf.data.p95_ms.toFixed(0)}ms`,
    stats.data?.model_version,
  ].filter(Boolean) as string[]

  const statusDot =
    status === 'healthy'
      ? 'bg-[color:var(--color-success-500)]'
      : status === 'limited'
        ? 'bg-[color:var(--color-warning-500)]'
        : 'bg-[color:var(--ink-subtle)]'
  const statusLabel =
    status === 'healthy'
      ? 'Search healthy'
      : status === 'limited'
        ? 'Search running in limited mode'
        : 'Checking search status…'

  return (
    <footer className="mt-10 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-[color:var(--border)] pt-4 text-xs text-[color:var(--ink-subtle)]">
      <span className="inline-flex items-center gap-1.5">
        <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${statusDot}`} />
        {statusLabel}
      </span>
      {facts.map((fact) => (
        <span key={fact} className="inline-flex items-center gap-2">
          <span aria-hidden className="text-[color:var(--ink-subtle)]">·</span>
          {fact}
        </span>
      ))}
    </footer>
  )
}

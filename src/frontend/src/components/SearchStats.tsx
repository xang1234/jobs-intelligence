import type { SearchResponse } from '@/types/api'
import { Card, Chip } from '@/components/ui'

export default function SearchStats({ data }: { data: SearchResponse }) {
  // Only surface a speed cue when results are genuinely fast — never the raw 17s latency (issue #1).
  const fast = data.search_time_ms < 2000
  return (
    <Card elevation={1} radius="lg" className="p-4">
      <div className="flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
        <span>
          <span className="font-semibold text-[color:var(--ink)]">
            {data.total_candidates.toLocaleString()}
          </span>{' '}
          matching jobs
        </span>
        {fast && (
          <span className="text-[color:var(--ink-subtle)]">
            in {(data.search_time_ms / 1000).toFixed(1)}s
          </span>
        )}
        {/* ponytail: `cached` is an internal detail — dev-only (issue #1). */}
        {import.meta.env.DEV && data.cache_hit && <Chip intent="success" size="sm">cached</Chip>}
        {data.degraded && <Chip intent="warning" size="sm">keyword fallback</Chip>}
      </div>

      {data.query_expansion && data.query_expansion.length > 0 && (
        <div className="mt-4 rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-3">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
            Query expansion inspector
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {data.query_expansion.map((term) => (
              <Chip key={term} intent="neutral" size="sm">
                {term}
              </Chip>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

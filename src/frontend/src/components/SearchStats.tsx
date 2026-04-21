import type { SearchResponse } from '@/types/api'
import { Card, Chip } from '@/components/ui'

export default function SearchStats({ data }: { data: SearchResponse }) {
  return (
    <Card elevation={1} radius="lg" className="p-4">
      <div className="flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
        <span>
          <span className="font-semibold text-[color:var(--ink)]">
            {data.total_candidates.toLocaleString()}
          </span>{' '}
          candidates
        </span>
        <span className="text-[color:var(--ink-subtle)]">{data.search_time_ms.toFixed(0)}ms</span>
        {data.cache_hit && <Chip intent="success" size="sm">cached</Chip>}
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

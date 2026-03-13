import type { SearchResponse } from '@/types/api'

export default function SearchStats({ data }: { data: SearchResponse }) {
  return (
    <div className="rounded-[24px] border border-[color:var(--border)] bg-white/90 p-4 shadow-[0_12px_36px_rgba(15,23,42,0.08)]">
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
        <span>
          <span className="font-semibold text-[color:var(--ink)]">{data.total_candidates.toLocaleString()}</span>{' '}
          candidates
        </span>
        <span>{data.search_time_ms.toFixed(0)}ms</span>
        {data.cache_hit && (
          <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-900">
            cached
          </span>
        )}
        {data.degraded && (
          <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
            keyword fallback
          </span>
        )}
      </div>

      {data.query_expansion && data.query_expansion.length > 0 && (
        <div className="mt-4 rounded-[18px] bg-[color:var(--surface-strong)] p-3">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            Query expansion inspector
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {data.query_expansion.map((term) => (
              <span
                key={term}
                className="rounded-full border border-[color:var(--border)] bg-white px-3 py-1 text-xs font-medium text-slate-700"
              >
                {term}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

import { useQuery } from '@tanstack/react-query'
import {
  getOverview,
  getPerformanceStats,
  getPopularQueries,
  getStats,
} from '@/services/api'

function formatMoney(value: number | null): string {
  if (value == null) return 'N/A'
  return `$${value.toLocaleString()}`
}

export default function OverviewPage() {
  const overview = useQuery({
    queryKey: ['overview'],
    queryFn: () => getOverview(12),
  })
  const stats = useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
  })
  const popular = useQuery({
    queryKey: ['popularQueries'],
    queryFn: () => getPopularQueries(30, 8),
  })
  const performance = useQuery({
    queryKey: ['performanceStats'],
    queryFn: () => getPerformanceStats(30),
  })

  const data = overview.data

  return (
    <div className="space-y-8">
      <section className="grid gap-6 rounded-[32px] border border-[color:var(--border)] bg-[linear-gradient(135deg,rgba(248,240,214,0.96),rgba(211,236,231,0.82))] p-8 shadow-[0_28px_90px_rgba(15,23,42,0.10)] lg:grid-cols-[1.35fr_0.65fr]">
        <div className="space-y-4">
          <p className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-600">
            Singapore hiring-market intelligence
          </p>
          <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-[color:var(--ink)] sm:text-5xl">
            Track demand shifts, salary movement, and the skill graph behind the market.
          </h1>
          <p className="max-w-2xl text-base leading-7 text-slate-700">
            This platform surfaces recruiter-grade job-market signals on top of hybrid search,
            embeddings, related-skill neighborhoods, and profile-to-role matching.
          </p>
        </div>

        <div className="rounded-[28px] bg-white/80 p-6 backdrop-blur">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            Retrieval stack health
          </p>
          <dl className="mt-4 grid gap-4 text-sm text-slate-600">
            <div className="flex items-center justify-between">
              <dt>Jobs indexed</dt>
              <dd className="font-semibold text-[color:var(--ink)]">
                {stats.data?.total_jobs.toLocaleString() ?? '...'}
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt>Embedding coverage</dt>
              <dd className="font-semibold text-[color:var(--ink)]">
                {stats.data ? `${stats.data.embedding_coverage_pct.toFixed(1)}%` : '...'}
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt>Median latency p95</dt>
              <dd className="font-semibold text-[color:var(--ink)]">
                {performance.data ? `${performance.data.p95_ms.toFixed(0)}ms` : '...'}
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt>Model</dt>
              <dd className="font-semibold text-[color:var(--ink)]">
                {stats.data?.model_version ?? '...'}
              </dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <article className="rounded-[26px] border border-[color:var(--border)] bg-white/90 p-5">
          <p className="text-sm text-slate-500">Tracked jobs</p>
          <p className="mt-2 text-3xl font-semibold text-[color:var(--ink)]">
            {data?.headline_metrics.total_jobs.toLocaleString() ?? '...'}
          </p>
        </article>
        <article className="rounded-[26px] border border-[color:var(--border)] bg-white/90 p-5">
          <p className="text-sm text-slate-500">This month</p>
          <p className="mt-2 text-3xl font-semibold text-[color:var(--ink)]">
            {data?.headline_metrics.current_month_jobs.toLocaleString() ?? '...'}
          </p>
        </article>
        <article className="rounded-[26px] border border-[color:var(--border)] bg-white/90 p-5">
          <p className="text-sm text-slate-500">Companies</p>
          <p className="mt-2 text-3xl font-semibold text-[color:var(--ink)]">
            {data?.headline_metrics.unique_companies.toLocaleString() ?? '...'}
          </p>
        </article>
        <article className="rounded-[26px] border border-[color:var(--border)] bg-white/90 p-5">
          <p className="text-sm text-slate-500">Average annual salary</p>
          <p className="mt-2 text-3xl font-semibold text-[color:var(--ink)]">
            {formatMoney(data?.headline_metrics.avg_salary_annual ?? null)}
          </p>
        </article>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_1.1fr_0.8fr]">
        <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Fastest-rising skills</p>
              <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]">Demand momentum</h2>
            </div>
          </div>
          <div className="mt-5 space-y-3">
            {data?.rising_skills.map((item) => (
              <div key={item.name} className="flex items-center justify-between rounded-[20px] bg-[color:var(--surface)] px-4 py-3">
                <div>
                  <p className="font-semibold text-[color:var(--ink)]">{item.name}</p>
                  <p className="text-xs text-slate-500">{item.job_count.toLocaleString()} jobs</p>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-semibold ${item.momentum >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                    {item.momentum >= 0 ? '+' : ''}{item.momentum.toFixed(1)}%
                  </p>
                  <p className="text-xs text-slate-500">{formatMoney(item.median_salary_annual)}</p>
                </div>
              </div>
            )) ?? <p className="text-sm text-slate-500">Loading skill movers...</p>}
          </div>
        </article>

        <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Fastest-rising companies</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]">Hiring velocity</h2>
          <div className="mt-5 space-y-3">
            {data?.rising_companies.map((item) => (
              <div key={item.name} className="flex items-center justify-between rounded-[20px] bg-[color:var(--surface)] px-4 py-3">
                <div>
                  <p className="font-semibold text-[color:var(--ink)]">{item.name}</p>
                  <p className="text-xs text-slate-500">{item.job_count.toLocaleString()} jobs</p>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-semibold ${item.momentum >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                    {item.momentum >= 0 ? '+' : ''}{item.momentum.toFixed(1)}%
                  </p>
                  <p className="text-xs text-slate-500">{formatMoney(item.median_salary_annual)}</p>
                </div>
              </div>
            )) ?? <p className="text-sm text-slate-500">Loading company movers...</p>}
          </div>
        </article>

        <article className="space-y-6">
          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Salary movement</p>
            <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]">
              {formatMoney(data?.salary_movement.current_median_salary_annual ?? null)}
            </h2>
            <p className={`mt-3 text-sm font-semibold ${(data?.salary_movement.change_pct ?? 0) >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
              {(data?.salary_movement.change_pct ?? 0) >= 0 ? '+' : ''}
              {data?.salary_movement.change_pct.toFixed(1) ?? '0.0'}% vs prior month
            </p>
          </div>

          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Recent search demand</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {popular.data?.map((item) => (
                <span key={item.query} className="rounded-full bg-[color:var(--surface)] px-3 py-1.5 text-xs font-medium text-slate-700">
                  {item.query} ({item.count})
                </span>
              )) ?? <p className="text-sm text-slate-500">No analytics yet.</p>}
            </div>
          </div>
        </article>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {data?.market_insights.map((insight) => (
          <article key={insight.label} className="rounded-[24px] border border-[color:var(--border)] bg-white/90 p-5">
            <p className="text-sm text-slate-500">{insight.label}</p>
            <div className="mt-2 flex items-end justify-between gap-4">
              <p className="text-2xl font-semibold text-[color:var(--ink)]">
                {insight.value != null ? insight.value.toLocaleString() : 'N/A'}
              </p>
              <p className={`text-sm font-semibold ${insight.delta >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                {insight.delta >= 0 ? '+' : ''}{insight.delta.toFixed(1)}%
              </p>
            </div>
          </article>
        )) ?? null}
      </section>
    </div>
  )
}

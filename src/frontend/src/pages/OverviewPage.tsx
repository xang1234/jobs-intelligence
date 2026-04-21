import { useQuery } from '@tanstack/react-query'
import { ArrowDownRightIcon, ArrowUpRightIcon, ChartBarIcon } from '@heroicons/react/20/solid'
import MetricCard from '@/components/overview/MetricCard'
import { Card, Chip, EmptyState, Skeleton, SkeletonText } from '@/components/ui'
import {
  getOverview,
  getPerformanceStats,
  getPopularQueries,
  getStats,
} from '@/services/api'
import type { MomentumCard } from '@/types/api'

function formatMoney(value: number | null): string {
  if (value == null) return 'N/A'
  return `$${value.toLocaleString()}`
}

export default function OverviewPage() {
  const overview = useQuery({ queryKey: ['overview'], queryFn: () => getOverview(12) })
  const stats = useQuery({ queryKey: ['stats'], queryFn: getStats })
  const popular = useQuery({
    queryKey: ['popularQueries'],
    queryFn: () => getPopularQueries(30, 8),
  })
  const performance = useQuery({
    queryKey: ['performanceStats'],
    queryFn: () => getPerformanceStats(30),
  })

  const data = overview.data
  const salaryDelta = data?.salary_movement.change_pct ?? null

  return (
    <div className="space-y-8">
      <Card
        as="section"
        radius="2xl"
        elevation={2}
        className="grid gap-6 p-8 lg:grid-cols-[1.35fr_0.65fr]"
        style={{
          background:
            'linear-gradient(135deg, color-mix(in srgb, var(--color-accent-100) 70%, var(--surface-1-alpha)), color-mix(in srgb, var(--color-brand-100) 50%, var(--surface-1-alpha)))',
        }}
      >
        <div className="space-y-4">
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[color:var(--ink-muted)]">
            Singapore hiring-market intelligence
          </p>
          <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-[color:var(--ink)] sm:text-5xl">
            Track demand shifts, salary movement, and the skill graph behind the market.
          </h1>
          <p className="max-w-2xl text-base leading-7 text-[color:var(--ink-muted)]">
            This platform surfaces recruiter-grade job-market signals on top of hybrid search,
            embeddings, related-skill neighborhoods, and profile-to-role matching.
          </p>
        </div>

        <Card radius="xl" elevation={0} className="p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-subtle)]">
            Retrieval stack health
          </p>
          <dl className="mt-4 grid gap-4 text-sm text-[color:var(--ink-muted)]">
            <StackRow
              label="Jobs indexed"
              value={stats.data?.total_jobs.toLocaleString()}
              loading={stats.isLoading}
            />
            <StackRow
              label="Embedding coverage"
              value={
                stats.data
                  ? `${stats.data.embedding_coverage_pct.toFixed(1)}%`
                  : undefined
              }
              loading={stats.isLoading}
            />
            <StackRow
              label="Median latency p95"
              value={
                performance.data ? `${performance.data.p95_ms.toFixed(0)}ms` : undefined
              }
              loading={performance.isLoading}
            />
            <StackRow
              label="Model"
              value={stats.data?.model_version}
              loading={stats.isLoading}
            />
          </dl>
        </Card>
      </Card>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Tracked jobs"
          value={data?.headline_metrics.total_jobs.toLocaleString()}
          loading={overview.isLoading}
          elevation={2}
        />
        <MetricCard
          label="This month"
          value={data?.headline_metrics.current_month_jobs.toLocaleString()}
          loading={overview.isLoading}
        />
        <MetricCard
          label="Companies"
          value={data?.headline_metrics.unique_companies.toLocaleString()}
          loading={overview.isLoading}
        />
        <MetricCard
          label="Average annual salary"
          value={formatMoney(data?.headline_metrics.avg_salary_annual ?? null)}
          loading={overview.isLoading}
          deltaPct={salaryDelta}
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_1.1fr_0.8fr]">
        <MomentumList
          eyebrow="Fastest-rising skills"
          title="Demand momentum"
          items={data?.rising_skills}
          loading={overview.isLoading}
        />
        <MomentumList
          eyebrow="Fastest-rising companies"
          title="Hiring velocity"
          items={data?.rising_companies}
          loading={overview.isLoading}
        />

        <div className="space-y-6">
          <Card radius="xl" className="p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
              Salary movement
            </p>
            {overview.isLoading ? (
              <div className="mt-2 space-y-3">
                <Skeleton height={28} width="70%" />
                <Skeleton height={16} width="50%" />
              </div>
            ) : (
              <>
                <h2 className="mt-1 text-2xl font-semibold text-[color:var(--ink)]">
                  {formatMoney(data?.salary_movement.current_median_salary_annual ?? null)}
                </h2>
                {salaryDelta != null && (
                  <Chip
                    intent={salaryDelta >= 0 ? 'success' : 'danger'}
                    size="sm"
                    leftIcon={
                      salaryDelta >= 0 ? (
                        <ArrowUpRightIcon className="h-3 w-3" />
                      ) : (
                        <ArrowDownRightIcon className="h-3 w-3" />
                      )
                    }
                    className="mt-3"
                  >
                    {salaryDelta >= 0 ? '+' : ''}
                    {salaryDelta.toFixed(1)}% vs prior month
                  </Chip>
                )}
              </>
            )}
          </Card>

          <Card radius="xl" className="p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
              Recent search demand
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {popular.isLoading ? (
                <SkeletonText lines={2} />
              ) : popular.data && popular.data.length > 0 ? (
                popular.data.map((item) => (
                  <Chip key={item.query} intent="neutral" size="sm">
                    {item.query} <span className="opacity-60">({item.count})</span>
                  </Chip>
                ))
              ) : (
                <p className="text-sm text-[color:var(--ink-subtle)]">No analytics yet.</p>
              )}
            </div>
          </Card>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {overview.isLoading
          ? Array.from({ length: 4 }).map((_, i) => (
              <Card key={i} radius="xl" className="p-5">
                <Skeleton height={14} width="35%" />
                <div className="mt-2">
                  <Skeleton height={32} width="55%" />
                </div>
              </Card>
            ))
          : data?.market_insights.map((insight) => (
              <MetricCard
                key={insight.label}
                label={insight.label}
                value={insight.value != null ? insight.value.toLocaleString() : 'N/A'}
                deltaPct={insight.delta}
              />
            )) ?? (
              <EmptyState
                icon={<ChartBarIcon />}
                title="No market insights yet"
                description="Insight cards will appear once enough jobs are tracked this month."
                compact
              />
            )}
      </section>
    </div>
  )
}

function StackRow({
  label,
  value,
  loading,
}: {
  label: string
  value: string | undefined
  loading: boolean
}) {
  return (
    <div className="flex items-center justify-between">
      <dt>{label}</dt>
      <dd className="font-semibold text-[color:var(--ink)]">
        {loading || value == null ? <Skeleton height={16} width={80} /> : value}
      </dd>
    </div>
  )
}

function MomentumList({
  eyebrow,
  title,
  items,
  loading,
}: {
  eyebrow: string
  title: string
  items: MomentumCard[] | undefined
  loading: boolean
}) {
  return (
    <Card as="article" radius="xl" className="p-6">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
        {eyebrow}
      </p>
      <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]">{title}</h2>
      <div className="mt-5 space-y-3">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={56} rounded="lg" />
          ))
        ) : items && items.length > 0 ? (
          items.map((item) => {
            const up = item.momentum >= 0
            return (
              <div
                key={item.name}
                className="flex items-center justify-between rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] px-4 py-3"
              >
                <div className="min-w-0">
                  <p className="truncate font-semibold text-[color:var(--ink)]">{item.name}</p>
                  <p className="text-xs text-[color:var(--ink-subtle)]">
                    {item.job_count.toLocaleString()} jobs
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <Chip
                    intent={up ? 'success' : 'danger'}
                    size="sm"
                    leftIcon={
                      up ? (
                        <ArrowUpRightIcon className="h-3 w-3" />
                      ) : (
                        <ArrowDownRightIcon className="h-3 w-3" />
                      )
                    }
                  >
                    {up ? '+' : ''}
                    {item.momentum.toFixed(1)}%
                  </Chip>
                  <p className="text-xs text-[color:var(--ink-subtle)]">
                    {formatMoney(item.median_salary_annual)}
                  </p>
                </div>
              </div>
            )
          })
        ) : (
          <p className="text-sm text-[color:var(--ink-subtle)]">No data yet.</p>
        )}
      </div>
    </Card>
  )
}

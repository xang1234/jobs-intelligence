import { useQuery } from '@tanstack/react-query'
import { ArrowDownRightIcon, ArrowUpRightIcon, ChartBarIcon } from '@heroicons/react/20/solid'
import MetricCard from '@/components/overview/MetricCard'
import PageHero from '@/components/shell/PageHero'
import { Card, Chip, EmptyState, Skeleton, SkeletonText } from '@/components/ui'
import { useIdleEnabled } from '@/hooks/useIdleEnabled'
import { getOverview, getPopularQueries } from '@/services/api'
import type { MomentumCard } from '@/types/api'

function formatMoney(value: number | null): string {
  if (value == null) return 'N/A'
  return `$${value.toLocaleString()}/yr`
}

export default function OverviewPage() {
  const showSecondaryData = useIdleEnabled()
  const overview = useQuery({ queryKey: ['overview', 3], queryFn: () => getOverview(3) })
  const popular = useQuery({
    queryKey: ['popularQueries'],
    queryFn: () => getPopularQueries(30, 8),
    enabled: showSecondaryData,
  })

  const data = overview.data
  const salaryDelta = data?.salary_movement.change_pct ?? null

  return (
    <div className="space-y-8">
      <PageHero
        tone="brand"
        eyebrow="Singapore hiring market · last 90 days"
        title="What's hiring, what it pays, and what changed."
        subtitle="A quick read on demand, salaries, and the companies and skills moving fastest right now."
      />

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
              {popular.isLoading || popular.isPending ? (
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

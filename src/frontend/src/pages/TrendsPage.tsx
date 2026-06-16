import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowDownRightIcon, ArrowRightIcon, ArrowUpRightIcon } from '@heroicons/react/20/solid'
import TrendSparkline from '@/components/TrendSparkline'
import PageHero from '@/components/shell/PageHero'
import { Button, Card, Chip, EmptyState, Input, Select, Skeleton } from '@/components/ui'
import type { ChipIntent, SelectOption } from '@/components/ui'
import { findSimilarCompanies, getCompanyTrend, getOverview, getRoleTrend, getSkillTrends } from '@/services/api'
import { getMomentumSignal, TREND_WINDOW_OPTIONS } from '@/services/trendSignals'
import type { SkillTrendSeries, TrendPoint } from '@/types/api'

function formatMoney(value: number | null): string {
  if (value == null) return 'N/A'
  return `$${value.toLocaleString()}/yr`
}

function formatMonth(value: string | null | undefined): string {
  if (!value) return 'No month'
  const [year, month] = value.split('-')
  const date = new Date(Number(year), Number(month) - 1, 1)
  return date.toLocaleDateString(undefined, { month: 'short', year: 'numeric' })
}

function activeMonths(points: TrendPoint[]): number {
  return points.filter((point) => point.job_count > 0).length
}

function latestPoint(points: TrendPoint[]): TrendPoint | null {
  return points.at(-1) ?? null
}

function trendSummary(label: string, latest: TrendPoint | null, points: TrendPoint[]): string {
  if (!latest || latest.job_count === 0) {
    return `No matching postings for ${label} in the last few months. Try a broader keyword or remove filters.`
  }
  const month = formatMonth(latest.month)
  const signal = getMomentumSignal(latest)
  if (!signal.showPercent) {
    return `${latest.job_count.toLocaleString()} postings in ${month}; treat this as current demand, not growth.`
  }
  return `${latest.job_count.toLocaleString()} postings in ${month}, ${signal.detail.toLowerCase()}. Active in ${activeMonths(points)} of ${points.length} months.`
}

function MomentumChip({ point }: { point: TrendPoint | null | undefined }) {
  const signal = getMomentumSignal(point)
  const icon =
    point?.momentum_status === 'up' ? (
      <ArrowUpRightIcon className="h-3 w-3" />
    ) : point?.momentum_status === 'down' ? (
      <ArrowDownRightIcon className="h-3 w-3" />
    ) : signal.showPercent ? (
      <ArrowRightIcon className="h-3 w-3" />
    ) : null

  return (
    <Chip intent={signal.intent as ChipIntent} size="sm" leftIcon={icon}>
      {signal.label}
    </Chip>
  )
}

function SignalMetric({
  label,
  value,
  detail,
}: {
  label: string
  value: string
  detail?: string
}) {
  return (
    <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] px-4 py-3">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-subtle)]">
        {label}
      </p>
      <p className="mt-2 text-xl font-semibold text-[color:var(--ink)]">{value}</p>
      {detail ? <p className="mt-1 text-xs text-[color:var(--ink-subtle)]">{detail}</p> : null}
    </div>
  )
}

function SkillSignalCard({ series }: { series: SkillTrendSeries }) {
  const latest = series.latest
  return (
    <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold text-[color:var(--ink)]">{series.skill}</h2>
          <p className="mt-1 text-sm leading-6 text-[color:var(--ink-muted)]">
            {trendSummary(series.skill, latest, series.series)}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <MomentumChip point={latest} />
          <p className="text-xs text-[color:var(--ink-subtle)]">
            {formatMoney(latest?.median_salary_annual ?? null)}
          </p>
        </div>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <SignalMetric label="Current jobs" value={(latest?.job_count ?? 0).toLocaleString()} />
        <SignalMetric
          label="Market share"
          value={`${(latest?.market_share ?? 0).toFixed(2)}%`}
          detail={formatMonth(latest?.month)}
        />
        <SignalMetric
          label="Months active"
          value={`${activeMonths(series.series)}/${series.series.length}`}
          detail="months active"
        />
      </div>
      <div className="mt-4">
        <TrendSparkline
          points={series.series}
          ariaLabel={`${series.skill} postings over the hosted trend window`}
        />
      </div>
    </div>
  )
}

const MONTHS_OPTIONS = TREND_WINDOW_OPTIONS as ReadonlyArray<SelectOption<number>>

export default function TrendsPage() {
  const [skillInput, setSkillInput] = useState('Customer Service, Microsoft Excel, Communication Skills')
  const [roleInput, setRoleInput] = useState('customer service')
  const [companyInput, setCompanyInput] = useState('DBS BANK LTD.')
  const [months, setMonths] = useState<number>(3)
  const [employmentType, setEmploymentType] = useState('')
  const [region, setRegion] = useState('')
  const [similarProfilesCompany, setSimilarProfilesCompany] = useState<string | null>(null)
  const companyName = companyInput.trim()
  const similarProfilesRequested = similarProfilesCompany === companyName

  const skills = useMemo(
    () => skillInput.split(',').map((item) => item.trim()).filter(Boolean).slice(0, 3),
    [skillInput],
  )

  const overview = useQuery({
    queryKey: ['overview', months],
    queryFn: () => getOverview(months),
  })

  const skillTrends = useQuery({
    queryKey: ['skillTrends', skills, months, employmentType, region],
    queryFn: () =>
      getSkillTrends({
        skills,
        months,
        employment_type: employmentType || null,
        region: region || null,
      }),
    enabled: skills.length > 0,
  })

  const roleTrend = useQuery({
    queryKey: ['roleTrend', roleInput, months, employmentType, region],
    queryFn: () =>
      getRoleTrend({
        query: roleInput,
        months,
        employment_type: employmentType || null,
        region: region || null,
      }),
    enabled: roleInput.trim().length > 0,
  })

  const companyTrend = useQuery({
    queryKey: ['companyTrend', companyName, months],
    queryFn: () => getCompanyTrend(companyName, months, false),
    enabled: companyName.length > 0,
  })

  const similarCompanies = useQuery({
    queryKey: ['similarCompanies', companyName],
    queryFn: () => findSimilarCompanies({ company_name: companyName, limit: 6 }),
    enabled: similarProfilesRequested && companyName.length > 0,
    staleTime: 10 * 60 * 1000,
  })

  const roleLatest = roleTrend.data?.latest ?? null
  const companyLatest = latestPoint(companyTrend.data?.series ?? [])
  const companyActiveMonths = activeMonths(companyTrend.data?.series ?? [])
  const companySkillSnapshots = companyTrend.data?.top_skills_by_month.filter((snapshot) => snapshot.skills.length) ?? []
  const similarEmployerProfiles = similarProfilesRequested ? (similarCompanies.data ?? []) : []

  return (
    <div className="space-y-8">
      <PageHero
        eyebrow={`${months}-month market signals`}
        title="Decide what to emphasize in your CV and search."
        subtitle={`These charts cover the last ${months} month${months === 1 ? '' : 's'} of postings. When there isn't enough history to call a real trend, we show current demand instead of a misleading growth number.`}
        actions={
          <div className="grid gap-3 sm:grid-cols-3">
            <Select<number>
              label="Time window"
              value={months}
              onChange={(v) => setMonths(v ?? 3)}
              options={MONTHS_OPTIONS}
              clearable={false}
            />
            <Input
              label="Employment type"
              value={employmentType}
              onChange={(e) => setEmploymentType(e.target.value)}
              placeholder="Full Time"
            />
            <Input
              label="Region"
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              placeholder="Singapore"
            />
          </div>
        }
      >
        <div className="grid gap-3 sm:grid-cols-3">
          <SignalMetric
            label="Jobs in latest month"
            value={(overview.data?.headline_metrics.current_month_jobs ?? 0).toLocaleString()}
            detail={overview.isLoading ? 'loading' : `${months}-month hosted window`}
          />
          <SignalMetric
            label="Average salary"
            value={formatMoney(overview.data?.headline_metrics.avg_salary_annual ?? null)}
            detail="annualized midpoint"
          />
          <SignalMetric
            label="How to read this"
            value="Demand first"
            detail="growth shown only with enough history"
          />
        </div>
      </PageHero>

      <section className="grid items-start gap-6 xl:grid-cols-2">
        <Card as="article" radius="xl" className="p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
            Skills in demand
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-[color:var(--ink)]">Which skills should your CV lead with?</h2>
          <div className="mt-4">
            <Input
              value={skillInput}
              onChange={(e) => setSkillInput(e.target.value)}
              placeholder="Customer Service, Microsoft Excel, Communication Skills"
              hint="Up to 3 comma-separated skills"
              aria-label="Skill list"
            />
          </div>
          <div className="mt-5 grid gap-4">
            {skillTrends.isLoading
              ? Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} height={210} rounded="lg" />
                ))
              : skillTrends.data?.length
                ? skillTrends.data.map((series) => <SkillSignalCard key={series.skill} series={series} />)
                : (
                    <EmptyState
                      title="No skill signals found"
                      description="Try broader skill names or remove one of the filters."
                    />
                  )}
          </div>
        </Card>

        <Card as="article" radius="xl" className="p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
            Is this role hiring?
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-[color:var(--ink)]">Is this role still hiring?</h2>
          <div className="mt-4">
            <Input
              value={roleInput}
              onChange={(e) => setRoleInput(e.target.value)}
              placeholder="customer service"
              aria-label="Role query"
            />
          </div>
          {roleTrend.isLoading ? (
            <div className="mt-5">
              <Skeleton height={240} rounded="lg" />
            </div>
          ) : roleTrend.data ? (
            <div className="mt-5 rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0">
                  <h2 className="text-lg font-semibold text-[color:var(--ink)]">{roleTrend.data.query}</h2>
                  <p className="mt-1 text-sm leading-6 text-[color:var(--ink-muted)]">
                    {trendSummary(roleTrend.data.query, roleLatest, roleTrend.data.series)}
                  </p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1">
                  <MomentumChip point={roleLatest} />
                  <p className="text-xs text-[color:var(--ink-subtle)]">
                    {formatMoney(roleLatest?.median_salary_annual ?? null)}
                  </p>
                </div>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <SignalMetric label="Current jobs" value={(roleLatest?.job_count ?? 0).toLocaleString()} />
                <SignalMetric label="Market share" value={`${(roleLatest?.market_share ?? 0).toFixed(2)}%`} />
                <SignalMetric label="Months active" value={`${activeMonths(roleTrend.data.series)}/${roleTrend.data.series.length}`} detail="months active" />
              </div>
              <div className="mt-4">
                <TrendSparkline points={roleTrend.data.series} ariaLabel={`${roleTrend.data.query} postings over time`} />
              </div>
            </div>
          ) : (
            <EmptyState title="No role signal yet" description="Enter a role or keyword to compare current demand." />
          )}
        </Card>
      </section>

      <Card as="section" radius="xl" className="p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
              Who's hiring
            </p>
            <h2 className="mt-2 text-2xl font-semibold text-[color:var(--ink)]">Who is hiring right now?</h2>
          </div>
          <div className="lg:w-96">
            <Input
              value={companyInput}
              onChange={(e) => setCompanyInput(e.target.value)}
              placeholder="RECRUIT EXPERT PTE. LTD."
              aria-label="Company name"
            />
          </div>
        </div>

        {companyTrend.isLoading && (
          <div className="mt-5">
            <Skeleton height={280} rounded="lg" />
          </div>
        )}

        {companyTrend.data && (
          <div className="mt-5 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0">
                  <h2 className="text-lg font-semibold text-[color:var(--ink)]">
                    {companyTrend.data.company_name}
                  </h2>
                  <p className="mt-1 text-sm leading-6 text-[color:var(--ink-muted)]">
                    {trendSummary(companyTrend.data.company_name, companyLatest, companyTrend.data.series)}
                  </p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1">
                  <MomentumChip point={companyLatest} />
                  <p className="text-xs text-[color:var(--ink-subtle)]">
                    {formatMoney(companyLatest?.median_salary_annual ?? null)}
                  </p>
                </div>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <SignalMetric label="Current jobs" value={(companyLatest?.job_count ?? 0).toLocaleString()} />
                <SignalMetric label="Market share" value={`${(companyLatest?.market_share ?? 0).toFixed(2)}%`} />
                <SignalMetric label="Months active" value={`${companyActiveMonths}/${companyTrend.data.series.length}`} detail="months active" />
              </div>
              <div className="mt-4">
                <TrendSparkline
                  points={companyTrend.data.series}
                  ariaLabel={`${companyTrend.data.company_name} postings over time`}
                />
              </div>
            </div>

            <div className="grid gap-4">
              <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
                <p className="text-sm font-semibold text-[color:var(--ink)]">Skill mix in active months</p>
                <div className="mt-4 space-y-3">
                  {companySkillSnapshots.length ? (
                    companySkillSnapshots.slice(-3).map((snapshot) => (
                      <div key={snapshot.month}>
                        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
                          {formatMonth(snapshot.month)}
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {snapshot.skills.slice(0, 5).map((skill) => (
                            <Chip key={`${snapshot.month}-${skill.skill}`} intent="neutral" size="sm">
                              {skill.skill} <span className="opacity-60">({skill.job_count})</span>
                            </Chip>
                          ))}
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-[color:var(--ink-subtle)]">
                      No skill mix available for this employer in the hosted window.
                    </p>
                  )}
                </div>
              </div>

              <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold text-[color:var(--ink)]">Similar hiring profiles</p>
                  <Button
                    variant="secondary"
                    size="sm"
                    loading={similarCompanies.isFetching}
                    disabled={!companyName}
                    onClick={() => {
                      if (similarProfilesRequested) {
                        void similarCompanies.refetch()
                      } else {
                        setSimilarProfilesCompany(companyName)
                      }
                    }}
                  >
                    {similarProfilesRequested ? 'Refresh profiles' : 'Load similar profiles'}
                  </Button>
                </div>
                <div className="mt-4 space-y-2">
                  {!similarProfilesRequested ? (
                    <p className="text-sm text-[color:var(--ink-subtle)]">
                      Load employer look-alikes only when you need the heavier comparison.
                    </p>
                  ) : similarCompanies.isLoading ? (
                    Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} height={58} rounded="md" />)
                  ) : similarEmployerProfiles.length ? (
                    similarEmployerProfiles.map((company) => (
                      <Card
                        key={company.company_name}
                        radius="md"
                        elevation={0}
                        interactive
                        className="flex items-center justify-between bg-[color:var(--surface-1)] px-4 py-3"
                      >
                        <div className="min-w-0">
                          <p className="truncate font-medium text-[color:var(--ink)]">{company.company_name}</p>
                          <p className="text-xs text-[color:var(--ink-subtle)]">{company.job_count} jobs</p>
                        </div>
                        <p className="ml-3 shrink-0 text-sm font-semibold text-[color:var(--brand)]">
                          {(company.similarity_score * 100).toFixed(0)}%
                        </p>
                      </Card>
                    ))
                  ) : (
                    <p className="text-sm text-[color:var(--ink-subtle)]">
                      Similar employers need at least one matching company profile.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

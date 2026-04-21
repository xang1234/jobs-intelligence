import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowDownRightIcon, ArrowUpRightIcon } from '@heroicons/react/20/solid'
import TrendSparkline from '@/components/TrendSparkline'
import { Card, Chip, Input, Select, Skeleton } from '@/components/ui'
import type { SelectOption } from '@/components/ui'
import { getCompanyTrend, getRoleTrend, getSkillTrends } from '@/services/api'

function formatMoney(value: number | null): string {
  if (value == null) return 'N/A'
  return `$${value.toLocaleString()}`
}

const MONTHS_OPTIONS: ReadonlyArray<SelectOption<number>> = [
  { value: 6, label: '6 months' },
  { value: 12, label: '12 months' },
  { value: 18, label: '18 months' },
  { value: 24, label: '24 months' },
]

function MomentumChip({ value }: { value: number }) {
  const up = value >= 0
  return (
    <Chip
      intent={up ? 'success' : 'danger'}
      size="sm"
      leftIcon={up ? <ArrowUpRightIcon className="h-3 w-3" /> : <ArrowDownRightIcon className="h-3 w-3" />}
    >
      {up ? '+' : ''}
      {value.toFixed(1)}%
    </Chip>
  )
}

export default function TrendsPage() {
  const [skillInput, setSkillInput] = useState('Python, SQL, Machine Learning')
  const [roleInput, setRoleInput] = useState('data scientist')
  const [companyInput, setCompanyInput] = useState('DBS BANK LTD.')
  const [months, setMonths] = useState<number>(12)
  const [employmentType, setEmploymentType] = useState('')
  const [region, setRegion] = useState('')

  const skills = useMemo(
    () => skillInput.split(',').map((item) => item.trim()).filter(Boolean).slice(0, 3),
    [skillInput],
  )

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
    queryKey: ['companyTrend', companyInput, months],
    queryFn: () => getCompanyTrend(companyInput, months),
    enabled: companyInput.trim().length > 0,
  })

  return (
    <div className="space-y-8">
      <Card as="section" radius="2xl" className="p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[color:var(--ink-subtle)]">
              Trends explorer
            </p>
            <h1 className="mt-2 text-4xl font-semibold tracking-tight text-[color:var(--ink)]">
              Compare skills, roles, and hiring companies over time.
            </h1>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Select<number>
              label="Window"
              value={months}
              onChange={(v) => setMonths(v ?? 12)}
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
              placeholder="Central"
            />
          </div>
        </div>
      </Card>

      <section className="grid gap-6 xl:grid-cols-2">
        <Card as="article" radius="xl" className="p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
            Skill comparison
          </p>
          <div className="mt-3">
            <Input
              value={skillInput}
              onChange={(e) => setSkillInput(e.target.value)}
              placeholder="Python, SQL, Machine Learning"
              hint="Up to 3 comma-separated skills"
              aria-label="Skill list"
            />
          </div>
          <div className="mt-5 grid gap-4">
            {skillTrends.isLoading
              ? Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} height={160} rounded="lg" />
                ))
              : skillTrends.data?.map((series) => (
                  <div
                    key={series.skill}
                    className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-4"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-lg font-semibold text-[color:var(--ink)]">
                          {series.skill}
                        </h2>
                        <p className="text-sm text-[color:var(--ink-subtle)]">
                          {series.latest?.job_count.toLocaleString() ?? 0} jobs this month
                        </p>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <MomentumChip value={series.latest?.momentum ?? 0} />
                        <p className="text-xs text-[color:var(--ink-subtle)]">
                          {formatMoney(series.latest?.median_salary_annual ?? null)}
                        </p>
                      </div>
                    </div>
                    <div className="mt-3">
                      <TrendSparkline points={series.series} />
                    </div>
                  </div>
                )) ?? <p className="text-sm text-[color:var(--ink-subtle)]">No trends yet.</p>}
          </div>
        </Card>

        <Card as="article" radius="xl" className="p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
            Role trend
          </p>
          <div className="mt-3">
            <Input
              value={roleInput}
              onChange={(e) => setRoleInput(e.target.value)}
              placeholder="data scientist"
              aria-label="Role query"
            />
          </div>
          {roleTrend.isLoading ? (
            <div className="mt-5">
              <Skeleton height={160} rounded="lg" />
            </div>
          ) : roleTrend.data ? (
            <div className="mt-5 rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-[color:var(--ink)]">
                    {roleTrend.data.query}
                  </h2>
                  <p className="text-sm text-[color:var(--ink-subtle)]">
                    Market share {roleTrend.data.latest?.market_share.toFixed(1) ?? '0.0'}%
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <MomentumChip value={roleTrend.data.latest?.momentum ?? 0} />
                  <p className="text-xs text-[color:var(--ink-subtle)]">
                    {formatMoney(roleTrend.data.latest?.median_salary_annual ?? null)}
                  </p>
                </div>
              </div>
              <div className="mt-4">
                <TrendSparkline points={roleTrend.data.series} />
              </div>
            </div>
          ) : (
            <p className="mt-5 text-sm text-[color:var(--ink-subtle)]">No role trend yet.</p>
          )}
        </Card>
      </section>

      <Card as="section" radius="xl" className="p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
          Company hiring pattern
        </p>
        <div className="mt-3">
          <Input
            value={companyInput}
            onChange={(e) => setCompanyInput(e.target.value)}
            placeholder="DBS BANK LTD."
            aria-label="Company name"
          />
        </div>

        {companyTrend.isLoading && (
          <div className="mt-5">
            <Skeleton height={280} rounded="lg" />
          </div>
        )}

        {companyTrend.data && (
          <div className="mt-5 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-[color:var(--ink)]">
                    {companyTrend.data.company_name}
                  </h2>
                  <p className="text-sm text-[color:var(--ink-subtle)]">
                    {companyTrend.data.series.at(-1)?.job_count.toLocaleString() ?? 0} jobs this month
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <MomentumChip value={companyTrend.data.series.at(-1)?.momentum ?? 0} />
                  <p className="text-xs text-[color:var(--ink-subtle)]">
                    {formatMoney(companyTrend.data.series.at(-1)?.median_salary_annual ?? null)}
                  </p>
                </div>
              </div>
              <div className="mt-4">
                <TrendSparkline points={companyTrend.data.series} />
              </div>
            </div>

            <div className="grid gap-4">
              <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
                <p className="text-sm font-semibold text-[color:var(--ink)]">Top skills by month</p>
                <div className="mt-4 space-y-3">
                  {companyTrend.data.top_skills_by_month.slice(-4).map((snapshot) => (
                    <div key={snapshot.month}>
                      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
                        {snapshot.month}
                      </p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {snapshot.skills.slice(0, 5).map((skill) => (
                          <Chip
                            key={`${snapshot.month}-${skill.skill}`}
                            intent="neutral"
                            size="sm"
                          >
                            {skill.skill} <span className="opacity-60">({skill.job_count})</span>
                          </Chip>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface-2)] p-5">
                <p className="text-sm font-semibold text-[color:var(--ink)]">Similar hiring profiles</p>
                <div className="mt-4 space-y-2">
                  {companyTrend.data.similar_companies.map((company) => (
                    <Card
                      key={company.company_name}
                      radius="md"
                      elevation={0}
                      interactive
                      className="flex items-center justify-between bg-[color:var(--surface-1)] px-4 py-3"
                    >
                      <div>
                        <p className="font-medium text-[color:var(--ink)]">{company.company_name}</p>
                        <p className="text-xs text-[color:var(--ink-subtle)]">{company.job_count} jobs</p>
                      </div>
                      <p className="text-sm font-semibold text-[color:var(--brand)]">
                        {(company.similarity_score * 100).toFixed(0)}%
                      </p>
                    </Card>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

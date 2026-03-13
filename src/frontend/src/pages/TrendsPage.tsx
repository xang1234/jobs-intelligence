import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import TrendSparkline from '@/components/TrendSparkline'
import {
  getCompanyTrend,
  getRoleTrend,
  getSkillTrends,
} from '@/services/api'

function formatMoney(value: number | null): string {
  if (value == null) return 'N/A'
  return `$${value.toLocaleString()}`
}

export default function TrendsPage() {
  const [skillInput, setSkillInput] = useState('Python, SQL, Machine Learning')
  const [roleInput, setRoleInput] = useState('data scientist')
  const [companyInput, setCompanyInput] = useState('DBS BANK LTD.')
  const [months, setMonths] = useState(12)
  const [employmentType, setEmploymentType] = useState('')
  const [region, setRegion] = useState('')

  const skills = useMemo(
    () => skillInput.split(',').map((item) => item.trim()).filter(Boolean).slice(0, 3),
    [skillInput],
  )

  const skillTrends = useQuery({
    queryKey: ['skillTrends', skills, months, employmentType, region],
    queryFn: () => getSkillTrends({
      skills,
      months,
      employment_type: employmentType || null,
      region: region || null,
    }),
    enabled: skills.length > 0,
  })

  const roleTrend = useQuery({
    queryKey: ['roleTrend', roleInput, months, employmentType, region],
    queryFn: () => getRoleTrend({
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
      <section className="rounded-[32px] border border-[color:var(--border)] bg-white/90 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-500">Trends explorer</p>
            <h1 className="mt-2 text-4xl font-semibold tracking-tight text-[color:var(--ink)]">
              Compare skills, roles, and hiring companies over time.
            </h1>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <label className="text-sm text-slate-600">
              Months
              <select
                value={months}
                onChange={(event) => setMonths(Number(event.target.value))}
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-2"
              >
                <option value={6}>6</option>
                <option value={12}>12</option>
                <option value={18}>18</option>
                <option value={24}>24</option>
              </select>
            </label>
            <label className="text-sm text-slate-600">
              Employment type
              <input
                value={employmentType}
                onChange={(event) => setEmploymentType(event.target.value)}
                placeholder="Full Time"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-2"
              />
            </label>
            <label className="text-sm text-slate-600">
              Region
              <input
                value={region}
                onChange={(event) => setRegion(event.target.value)}
                placeholder="Central"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-2"
              />
            </label>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Skill comparison</p>
          <input
            value={skillInput}
            onChange={(event) => setSkillInput(event.target.value)}
            placeholder="Python, SQL, Machine Learning"
            className="mt-3 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3 text-sm"
          />
          <div className="mt-5 grid gap-4">
            {skillTrends.data?.map((series) => (
              <div key={series.skill} className="rounded-[24px] bg-[color:var(--surface)] p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-[color:var(--ink)]">{series.skill}</h2>
                    <p className="text-sm text-slate-500">
                      {series.latest?.job_count.toLocaleString() ?? 0} jobs this month
                    </p>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-semibold ${(series.latest?.momentum ?? 0) >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {(series.latest?.momentum ?? 0) >= 0 ? '+' : ''}
                      {series.latest?.momentum.toFixed(1) ?? '0.0'}%
                    </p>
                    <p className="text-xs text-slate-500">{formatMoney(series.latest?.median_salary_annual ?? null)}</p>
                  </div>
                </div>
                <div className="mt-3">
                  <TrendSparkline points={series.series} />
                </div>
              </div>
            )) ?? <p className="text-sm text-slate-500">Loading skill trends...</p>}
          </div>
        </article>

        <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Role trend</p>
          <input
            value={roleInput}
            onChange={(event) => setRoleInput(event.target.value)}
            placeholder="data scientist"
            className="mt-3 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3 text-sm"
          />
          {roleTrend.data ? (
            <div className="mt-5 rounded-[24px] bg-[color:var(--surface)] p-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-[color:var(--ink)]">{roleTrend.data.query}</h2>
                  <p className="text-sm text-slate-500">
                    Market share {roleTrend.data.latest?.market_share.toFixed(1) ?? '0.0'}%
                  </p>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-semibold ${(roleTrend.data.latest?.momentum ?? 0) >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                    {(roleTrend.data.latest?.momentum ?? 0) >= 0 ? '+' : ''}
                    {roleTrend.data.latest?.momentum.toFixed(1) ?? '0.0'}%
                  </p>
                  <p className="text-xs text-slate-500">{formatMoney(roleTrend.data.latest?.median_salary_annual ?? null)}</p>
                </div>
              </div>
              <div className="mt-4">
                <TrendSparkline points={roleTrend.data.series} />
              </div>
            </div>
          ) : (
            <p className="mt-5 text-sm text-slate-500">Loading role trend...</p>
          )}
        </article>
      </section>

      <section className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Company hiring pattern</p>
        <input
          value={companyInput}
          onChange={(event) => setCompanyInput(event.target.value)}
          placeholder="DBS BANK LTD."
          className="mt-3 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3 text-sm"
        />

        {companyTrend.data && (
          <div className="mt-5 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="rounded-[24px] bg-[color:var(--surface)] p-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-[color:var(--ink)]">{companyTrend.data.company_name}</h2>
                  <p className="text-sm text-slate-500">
                    {companyTrend.data.series.at(-1)?.job_count.toLocaleString() ?? 0} jobs this month
                  </p>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-semibold ${(companyTrend.data.series.at(-1)?.momentum ?? 0) >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                    {(companyTrend.data.series.at(-1)?.momentum ?? 0) >= 0 ? '+' : ''}
                    {companyTrend.data.series.at(-1)?.momentum.toFixed(1) ?? '0.0'}%
                  </p>
                  <p className="text-xs text-slate-500">
                    {formatMoney(companyTrend.data.series.at(-1)?.median_salary_annual ?? null)}
                  </p>
                </div>
              </div>
              <div className="mt-4">
                <TrendSparkline points={companyTrend.data.series} />
              </div>
            </div>

            <div className="grid gap-4">
              <div className="rounded-[24px] bg-[color:var(--surface)] p-5">
                <p className="text-sm font-semibold text-[color:var(--ink)]">Top skills by month</p>
                <div className="mt-4 space-y-3">
                  {companyTrend.data.top_skills_by_month.slice(-4).map((snapshot) => (
                    <div key={snapshot.month}>
                      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{snapshot.month}</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {snapshot.skills.slice(0, 5).map((skill) => (
                          <span key={`${snapshot.month}-${skill.skill}`} className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                            {skill.skill} ({skill.job_count})
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[24px] bg-[color:var(--surface)] p-5">
                <p className="text-sm font-semibold text-[color:var(--ink)]">Similar hiring profiles</p>
                <div className="mt-4 space-y-2">
                  {companyTrend.data.similar_companies.map((company) => (
                    <div key={company.company_name} className="flex items-center justify-between rounded-2xl bg-white px-4 py-3">
                      <div>
                        <p className="font-medium text-[color:var(--ink)]">{company.company_name}</p>
                        <p className="text-xs text-slate-500">{company.job_count} jobs</p>
                      </div>
                      <p className="text-sm font-semibold text-[color:var(--brand)]">
                        {(company.similarity_score * 100).toFixed(0)}%
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}

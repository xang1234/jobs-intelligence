import type { CareerDeltaBaseline } from '@/types/api'
import { formatCurrency, titleCaseFromKey } from './helpers'
import { SummaryMetric } from './primitives'

export function BaselineInsightCard({ baseline }: { baseline: CareerDeltaBaseline }) {
  return (
    <article className="rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Market position</p>
          <h3 className="mt-2 text-3xl font-semibold capitalize text-[color:var(--ink)]">
            {baseline.position}
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[color:var(--ink-muted)]">
            {baseline.reachable_jobs.toLocaleString()} reachable roles across{' '}
            {baseline.total_candidates.toLocaleString()} considered candidates, with median fit{' '}
            {(baseline.fit_median * 100).toFixed(0)}%.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {baseline.thin_market ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              limited data
            </span>
          ) : null}
          {baseline.degraded ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              rough estimate
            </span>
          ) : null}
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <SummaryMetric label="Reachable jobs" value={baseline.reachable_jobs.toLocaleString()} accent />
        <SummaryMetric label="Median fit" value={`${(baseline.fit_median * 100).toFixed(0)}%`} />
        <SummaryMetric label="P90 fit" value={`${(baseline.fit_p90 * 100).toFixed(0)}%`} />
        <SummaryMetric label="Skill coverage" value={`${(baseline.skill_coverage * 100).toFixed(0)}%`} />
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Salary band</p>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
            <span className="rounded-full bg-[color:var(--surface-1)] px-3 py-1 font-medium text-[color:var(--ink)]">
              Min {formatCurrency(baseline.salary_band.min_annual)}
            </span>
            <span className="rounded-full bg-[color:var(--surface-1)] px-3 py-1 font-medium text-[color:var(--ink)]">
              Median {formatCurrency(baseline.salary_band.median_annual)}
            </span>
            <span className="rounded-full bg-[color:var(--surface-1)] px-3 py-1 font-medium text-[color:var(--ink)]">
              Max {formatCurrency(baseline.salary_band.max_annual)}
            </span>
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Baseline notes</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {baseline.notes.length ? (
              baseline.notes.map((note) => (
                <span key={note} className="rounded-full bg-[color:var(--surface-1)] px-3 py-1 text-xs font-medium text-[color:var(--ink-muted)]">
                  {note}
                </span>
              ))
            ) : (
              <span className="text-sm text-[color:var(--ink-subtle)]">No special caveats on the baseline pass.</span>
            )}
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Extracted skills</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {baseline.extracted_skills.length ? (
              baseline.extracted_skills.map((skill) => (
                <span key={skill} className="rounded-full bg-[color:var(--surface-1)] px-3 py-1 text-xs font-medium text-[color:var(--ink-muted)]">
                  {skill}
                </span>
              ))
            ) : (
              <span className="text-sm text-[color:var(--ink-subtle)]">No extracted baseline skills surfaced.</span>
            )}
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Top skill gaps</p>
          <div className="mt-3 space-y-2">
            {baseline.top_skill_gaps.length ? (
              baseline.top_skill_gaps.map((gap) => (
                <div key={gap.name} className="flex items-center justify-between rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm">
                  <span className="font-medium text-[color:var(--ink)]">{gap.name}</span>
                  <span className="text-[color:var(--ink-subtle)]">
                    {gap.job_count} jobs · {gap.share_pct.toFixed(0)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-[color:var(--ink-subtle)]">No recurring skill gaps surfaced in the baseline pool.</p>
            )}
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Top industries</p>
          <div className="mt-3 space-y-2">
            {baseline.top_industries.length ? (
              baseline.top_industries.map((industry) => (
                <div key={industry.name} className="flex items-center justify-between rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm">
                  <span className="font-medium text-[color:var(--ink)]">{titleCaseFromKey(industry.name)}</span>
                  <span className="text-[color:var(--ink-subtle)]">
                    {industry.job_count} jobs · {industry.share_pct.toFixed(0)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-[color:var(--ink-subtle)]">Industry concentration is still unclear for this profile.</p>
            )}
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Top companies</p>
          <div className="mt-3 space-y-2">
            {baseline.top_companies.length ? (
              baseline.top_companies.map((company) => (
                <div key={company.name} className="flex items-center justify-between rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm">
                  <span className="font-medium text-[color:var(--ink)]">{company.name}</span>
                  <span className="text-[color:var(--ink-subtle)]">
                    {company.job_count} jobs · {company.share_pct.toFixed(0)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-[color:var(--ink-subtle)]">No dominant employer cluster surfaced in the baseline pool.</p>
            )}
          </div>
        </div>
      </div>
    </article>
  )
}

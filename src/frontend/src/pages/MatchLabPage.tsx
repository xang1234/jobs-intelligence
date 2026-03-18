import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import JobCard from '@/components/JobCard'
import { analyzeCareerDelta, matchProfile } from '@/services/api'
import { buildCareerDeltaAnalysisRequest, buildProfileMatchRequest } from '@/services/matchLab'
import type {
  CareerDeltaAnalysisResponse,
  CareerDeltaBaseline,
  CareerDeltaFilteredScenario,
  CareerDeltaScenarioChange,
  CareerDeltaScenarioSignal,
  CareerDeltaScenarioSummary,
  MatchLabSharedInputs,
} from '@/types/api'

type MatchLabTab = 'match' | 'what-if'

function tabButtonClass(isActive: boolean): string {
  return isActive
    ? 'bg-[color:var(--brand)] text-white shadow-lg'
    : 'bg-[color:var(--surface)] text-slate-600 hover:text-[color:var(--brand)]'
}

function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return 'n/a'
  }
  return `${value > 0 ? '+' : ''}${value.toFixed(0)}%`
}

function formatRatioPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return 'n/a'
  }
  return `${value > 0 ? '+' : ''}${(value * 100).toFixed(0)}%`
}

function formatConfidence(score: number): string {
  return `${(score * 100).toFixed(0)}%`
}

function formatCurrency(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return 'n/a'
  }
  return new Intl.NumberFormat('en-SG', {
    style: 'currency',
    currency: 'SGD',
    maximumFractionDigits: 0,
  }).format(value)
}

function titleCaseFromKey(value: string | null | undefined): string {
  if (!value) {
    return 'Unknown'
  }
  return value
    .replaceAll('_', ' ')
    .replaceAll('/', ' / ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function compactList(items: string[], emptyLabel: string): string {
  return items.length ? items.join(', ') : emptyLabel
}

function describeScenarioChange(change: CareerDeltaScenarioChange | null): string[] {
  if (!change) {
    return []
  }

  const lines: string[] = []

  if (change.added_skills.length) {
    lines.push(`Add ${compactList(change.added_skills, '')}`)
  }
  if (change.replaced_skills.length) {
    lines.push(
      `Swap ${change.replaced_skills
        .map((replacement) => `${replacement.from_skill} -> ${replacement.to_skill}`)
        .join(', ')}`,
    )
  }
  if (change.target_title_family && change.target_title_family !== change.source_title_family) {
    lines.push(
      `Shift title focus from ${titleCaseFromKey(change.source_title_family)} to ${titleCaseFromKey(change.target_title_family)}`,
    )
  }
  if (change.target_industry && change.target_industry !== change.source_industry) {
    lines.push(
      `Pivot sector from ${titleCaseFromKey(change.source_industry)} to ${titleCaseFromKey(change.target_industry)}`,
    )
  }

  return lines
}

function bestScenarioSignal(signals: CareerDeltaScenarioSignal[]): CareerDeltaScenarioSignal | null {
  if (!signals.length) {
    return null
  }
  return [...signals].sort((left, right) => right.supporting_jobs - left.supporting_jobs)[0]
}

function SummaryMetric({
  label,
  value,
  accent = false,
}: {
  label: string
  value: string
  accent?: boolean
}) {
  return (
    <div className="rounded-[20px] bg-[color:var(--surface)] px-4 py-3">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={`mt-2 text-lg font-semibold ${accent ? 'text-[color:var(--brand)]' : 'text-[color:var(--ink)]'}`}>
        {value}
      </p>
    </div>
  )
}

function StatePanel({
  eyebrow,
  title,
  message,
  tone = 'neutral',
  actionLabel,
  onAction,
}: {
  eyebrow: string
  title: string
  message: string
  tone?: 'neutral' | 'warning' | 'danger'
  actionLabel?: string
  onAction?: () => void
}) {
  const toneClass =
    tone === 'danger'
      ? 'border-rose-200 bg-rose-50 text-rose-900'
      : tone === 'warning'
        ? 'border-amber-200 bg-amber-50 text-amber-900'
        : 'border-[color:var(--border)] bg-white/80 text-slate-700'

  return (
    <section className={`rounded-[28px] border p-6 ${toneClass}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] opacity-70">{eyebrow}</p>
      <h3 className="mt-2 text-xl font-semibold">{title}</h3>
      <p className="mt-3 max-w-3xl text-sm leading-6 opacity-90">{message}</p>
      {actionLabel && onAction ? (
        <button
          type="button"
          onClick={onAction}
          className="mt-4 rounded-full border border-current px-4 py-2 text-sm font-semibold transition hover:bg-white/40"
        >
          {actionLabel}
        </button>
      ) : null}
    </section>
  )
}

function BaselineInsightCard({ baseline }: { baseline: CareerDeltaBaseline }) {
  return (
    <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Market position</p>
          <h3 className="mt-2 text-3xl font-semibold capitalize text-[color:var(--ink)]">
            {baseline.position}
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
            {baseline.reachable_jobs.toLocaleString()} reachable roles across{' '}
            {baseline.total_candidates.toLocaleString()} considered candidates, with median fit{' '}
            {(baseline.fit_median * 100).toFixed(0)}%.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {baseline.thin_market ? (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
              thin market
            </span>
          ) : null}
          {baseline.degraded ? (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
              degraded retrieval
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
        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Salary band</p>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-slate-600">
            <span className="rounded-full bg-white px-3 py-1 font-medium text-[color:var(--ink)]">
              Min {formatCurrency(baseline.salary_band.min_annual)}
            </span>
            <span className="rounded-full bg-white px-3 py-1 font-medium text-[color:var(--ink)]">
              Median {formatCurrency(baseline.salary_band.median_annual)}
            </span>
            <span className="rounded-full bg-white px-3 py-1 font-medium text-[color:var(--ink)]">
              Max {formatCurrency(baseline.salary_band.max_annual)}
            </span>
          </div>
        </div>

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Baseline notes</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {baseline.notes.length ? (
              baseline.notes.map((note) => (
                <span key={note} className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                  {note}
                </span>
              ))
            ) : (
              <span className="text-sm text-slate-500">No special caveats on the baseline pass.</span>
            )}
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Extracted skills</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {baseline.extracted_skills.length ? (
              baseline.extracted_skills.map((skill) => (
                <span key={skill} className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700">
                  {skill}
                </span>
              ))
            ) : (
              <span className="text-sm text-slate-500">No extracted baseline skills surfaced.</span>
            )}
          </div>
        </div>

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Top skill gaps</p>
          <div className="mt-3 space-y-2">
            {baseline.top_skill_gaps.length ? (
              baseline.top_skill_gaps.map((gap) => (
                <div key={gap.name} className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 text-sm">
                  <span className="font-medium text-[color:var(--ink)]">{gap.name}</span>
                  <span className="text-slate-500">
                    {gap.job_count} jobs · {gap.share_pct.toFixed(0)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No recurring skill gaps surfaced in the baseline pool.</p>
            )}
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Top industries</p>
          <div className="mt-3 space-y-2">
            {baseline.top_industries.length ? (
              baseline.top_industries.map((industry) => (
                <div key={industry.name} className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 text-sm">
                  <span className="font-medium text-[color:var(--ink)]">{titleCaseFromKey(industry.name)}</span>
                  <span className="text-slate-500">
                    {industry.job_count} jobs · {industry.share_pct.toFixed(0)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">Industry concentration is still unclear for this profile.</p>
            )}
          </div>
        </div>

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Top companies</p>
          <div className="mt-3 space-y-2">
            {baseline.top_companies.length ? (
              baseline.top_companies.map((company) => (
                <div key={company.name} className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 text-sm">
                  <span className="font-medium text-[color:var(--ink)]">{company.name}</span>
                  <span className="text-slate-500">
                    {company.job_count} jobs · {company.share_pct.toFixed(0)}%
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No dominant employer cluster surfaced in the baseline pool.</p>
            )}
          </div>
        </div>
      </div>
    </article>
  )
}

function WhatIfScenarioPreview({ index, scenario }: { index: number; scenario: CareerDeltaScenarioSummary }) {
  const primarySignal = bestScenarioSignal(scenario.signals)
  const changeLines = describeScenarioChange(scenario.change)

  return (
    <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6 shadow-[0_16px_40px_rgba(15,23,42,0.05)]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-full bg-[color:var(--brand)]/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--brand)]">
            #{index + 1}
          </span>
          <span className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            {scenario.scenario_type.replaceAll('_', ' ')}
          </span>
          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800">
            {formatConfidence(scenario.confidence.score)} confidence
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {scenario.thin_market ? (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
              thin market
            </span>
          ) : null}
          {scenario.degraded ? (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
              degraded
            </span>
          ) : null}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-slate-600">
        <span>
          Market position:{' '}
          <span className="font-semibold capitalize text-[color:var(--ink)]">{scenario.market_position}</span>
        </span>
        {scenario.target_title ? (
          <span>
            Target title: <span className="font-semibold text-[color:var(--ink)]">{scenario.target_title}</span>
          </span>
        ) : null}
        {scenario.target_sector ? (
          <span>
            Sector: <span className="font-semibold text-[color:var(--ink)]">{titleCaseFromKey(scenario.target_sector)}</span>
          </span>
        ) : null}
      </div>

      <h3 className="mt-3 text-lg font-semibold text-[color:var(--ink)]">{scenario.title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-600">{scenario.summary}</p>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <SummaryMetric label="Salary delta" value={formatRatioPercent(scenario.expected_salary_delta_pct)} accent />
        <SummaryMetric
          label="Evidence coverage"
          value={`${(scenario.confidence.evidence_coverage * 100).toFixed(0)}%`}
        />
        <SummaryMetric label="Sample size" value={scenario.confidence.market_sample_size.toLocaleString()} />
        <SummaryMetric
          label="Pivot cost"
          value={scenario.score_breakdown ? scenario.score_breakdown.pivot_cost.toFixed(2) : 'n/a'}
        />
      </div>

      {scenario.score_breakdown ? (
        <div className="mt-5 rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Score breakdown</p>
          <div className="mt-3 grid gap-3 md:grid-cols-3 xl:grid-cols-6">
            <SummaryMetric label="Opportunity" value={scenario.score_breakdown.opportunity.toFixed(2)} />
            <SummaryMetric label="Quality" value={scenario.score_breakdown.quality.toFixed(2)} />
            <SummaryMetric label="Salary" value={scenario.score_breakdown.salary.toFixed(2)} />
            <SummaryMetric label="Momentum" value={scenario.score_breakdown.momentum.toFixed(2)} />
            <SummaryMetric label="Diversity" value={scenario.score_breakdown.diversity.toFixed(2)} />
            <SummaryMetric label="Final score" value={scenario.score_breakdown.final_score.toFixed(2)} accent />
          </div>
        </div>
      ) : null}

      <div className="mt-5 grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Recommended move</p>
          <div className="mt-3 space-y-2">
            {changeLines.length ? (
              changeLines.map((line) => (
                <p key={line} className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-700">
                  {line}
                </p>
              ))
            ) : (
              <p className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-500">
                The engine did not expose a structured change list for this scenario.
              </p>
            )}
          </div>
        </div>

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Top signal</p>
          {primarySignal ? (
            <div className="mt-3 space-y-2">
              <p className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-700">
                {primarySignal.supporting_jobs} supporting jobs · {primarySignal.supporting_share_pct.toFixed(0)}%
                share
              </p>
              <p className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-700">
                Demand momentum {formatPercent(primarySignal.market_momentum)} · Median salary{' '}
                {formatCurrency(primarySignal.market_salary_annual_median)}
              </p>
              {primarySignal.skill ? (
                <p className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-700">
                  Signal skill: <span className="font-medium text-[color:var(--ink)]">{primarySignal.skill}</span>
                </p>
              ) : null}
            </div>
          ) : (
            <p className="mt-3 rounded-2xl bg-white px-4 py-3 text-sm text-slate-500">
              No primary signal was attached to this recommendation.
            </p>
          )}
        </div>
      </div>

      <div className="mt-5">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Confidence notes</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {scenario.confidence.reasons.length ? (
            scenario.confidence.reasons.map((reason) => (
              <span key={reason} className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-slate-700">
                {reason}
              </span>
            ))
          ) : (
            <span className="text-sm text-slate-500">No extra confidence notes supplied.</span>
          )}
        </div>
      </div>
    </article>
  )
}

function FilteredScenariosPanel({ filtered }: { filtered: CareerDeltaFilteredScenario[] }) {
  return (
    <section className="rounded-[28px] border border-dashed border-[color:var(--border)] bg-white/75 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Filtered scenarios</p>
          <h3 className="mt-2 text-xl font-semibold text-[color:var(--ink)]">Rejected moves the engine considered</h3>
        </div>
        <span className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-semibold text-slate-600">
          {filtered.length} filtered
        </span>
      </div>

      <div className="mt-4 space-y-3">
        {filtered.map((scenario) => (
          <article key={scenario.scenario_id} className="rounded-[22px] border border-[color:var(--border)] bg-white px-5 py-4">
            <div className="flex flex-wrap items-center gap-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              <span>{scenario.scenario_type.replaceAll('_', ' ')}</span>
              <span>{scenario.reason_code.replaceAll('_', ' ')}</span>
              <span>{formatConfidence(scenario.confidence.score)} confidence</span>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">{scenario.explanation}</p>
          </article>
        ))}
      </div>
    </section>
  )
}

function WhatIfSummaryPanel({
  response,
  isPending,
  hasAttempted,
  onRetry,
}: {
  response: CareerDeltaAnalysisResponse | undefined
  isPending: boolean
  hasAttempted: boolean
  onRetry: () => void
}) {
  const baseline = response?.baseline
  const budgetExhausted = response?.filtered_scenarios.some((item) => item.reason_code === 'budget_exhausted') ?? false
  const noRankedScenarios = !!response && response.scenarios.length === 0

  if (isPending) {
    return (
      <div className="space-y-5">
        <StatePanel
          eyebrow="Computing"
          title="Running counterfactual analysis"
          message="The engine is building a baseline market position, scoring candidate moves, and filtering weak or unsupported scenarios. This can take longer than current-fit matching because it evaluates multiple bounded alternatives."
        />
        <div className="grid gap-4 md:grid-cols-2">
          {['Baseline market position', 'Recommendation ranking'].map((label) => (
            <div
              key={label}
              className="animate-pulse rounded-[28px] border border-[color:var(--border)] bg-white/70 p-6"
            >
              <div className="h-3 w-28 rounded-full bg-slate-200" />
              <div className="mt-4 h-8 w-2/3 rounded-full bg-slate-200" />
              <div className="mt-3 h-4 w-full rounded-full bg-slate-200" />
              <div className="mt-2 h-4 w-5/6 rounded-full bg-slate-200" />
              <div className="mt-5 grid gap-3 md:grid-cols-2">
                <div className="h-20 rounded-[20px] bg-slate-200" />
                <div className="h-20 rounded-[20px] bg-slate-200" />
              </div>
              <p className="mt-4 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</p>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!response && !hasAttempted) {
    return (
      <StatePanel
        eyebrow="What If"
        title="Counterfactual results will appear here"
        message="Run What If to see your current market position, the highest-confidence moves the engine found, and the filtered scenarios it rejected."
      />
    )
  }

  if (!response) {
    return null
  }

  const resolvedResponse = response

  return (
    <div className="space-y-5">
      {resolvedResponse.thin_market ? (
        <StatePanel
          eyebrow="Thin market"
          title="There is not much reliable market evidence for this profile"
          message="The engine found too little consistent demand in the reachable pool to make strong recommendations. Treat any surfaced moves as low-confidence signals rather than a broad market read."
          tone="warning"
          actionLabel="Try again with a broader title set"
          onAction={onRetry}
        />
      ) : null}

      {resolvedResponse.degraded ? (
        <StatePanel
          eyebrow={budgetExhausted ? 'Partial results' : 'Degraded retrieval'}
          title={
            budgetExhausted
              ? 'The engine returned a partial recommendation set'
              : 'The engine fell back to a weaker retrieval path'
          }
          message={
            budgetExhausted
              ? 'Some scenarios were not fully evaluated before the compute budget expired. Use the current list as a conservative partial view, not a complete ranking of every viable move.'
              : 'The backend could not use its preferred retrieval mode, so these results may miss opportunities or understate confidence compared with a healthy run.'
          }
          tone="warning"
          actionLabel="Retry analysis"
          onAction={onRetry}
        />
      ) : null}

      <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
        <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
          <span>
            Scenario count:{' '}
            <span className="font-semibold text-[color:var(--ink)]">{resolvedResponse.scenarios.length}</span>
          </span>
          <span>
            Filtered:{' '}
            <span className="font-semibold text-[color:var(--ink)]">{resolvedResponse.filtered_scenarios.length}</span>
          </span>
          <span>
            Analysis time:{' '}
            <span className="font-semibold text-[color:var(--ink)]">
              {resolvedResponse.analysis_time_ms?.toFixed(0) ?? '0'}ms
            </span>
          </span>
          {resolvedResponse.thin_market ? (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
              thin market
            </span>
          ) : null}
          {resolvedResponse.degraded ? (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
              degraded retrieval
            </span>
          ) : null}
        </div>
      </div>

      {baseline ? <BaselineInsightCard baseline={baseline} /> : null}

      {noRankedScenarios && !resolvedResponse.thin_market ? (
        <StatePanel
          eyebrow="No high-confidence move"
          title="The engine did not find a recommendation worth promoting"
          message="This is different from a system failure. The current profile appears reasonably positioned, or the available deltas were too weak, too costly, or too thin to surface as trustworthy recommendations."
          tone="neutral"
          actionLabel="Run again"
          onAction={onRetry}
        />
      ) : null}

      <section className="space-y-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Ranked scenarios</p>
          <h3 className="mt-2 text-2xl font-semibold text-[color:var(--ink)]">Recommended moves with market evidence</h3>
        </div>

        {resolvedResponse.scenarios.length ? (
          resolvedResponse.scenarios.map((scenario, index) => (
            <WhatIfScenarioPreview key={scenario.scenario_id} index={index} scenario={scenario} />
          ))
        ) : (
          <div className="rounded-[28px] border border-dashed border-[color:var(--border)] bg-white/70 p-10 text-center text-sm text-slate-500">
            {resolvedResponse.thin_market
              ? 'Recommendations are withheld because the reachable pool is too thin to support a trustworthy move.'
              : 'No recommendation cleared the quality bar for this profile. Use the baseline and filtered scenarios below to see why the engine stayed conservative.'}
          </div>
        )}
      </section>

      {resolvedResponse.filtered_scenarios.length ? (
        <FilteredScenariosPanel filtered={resolvedResponse.filtered_scenarios} />
      ) : null}
    </div>
  )
}

export default function MatchLabPage() {
  const [activeTab, setActiveTab] = useState<MatchLabTab>('match')
  const [inputs, setInputs] = useState<MatchLabSharedInputs>({
    profileText:
      'Senior data professional with Python, SQL, machine learning, experimentation, stakeholder management, and dashboarding experience. Looking for Singapore-based roles in AI, analytics, or applied ML.',
    targetTitles: 'Data Scientist, Machine Learning Engineer',
    salaryExpectation: '180000',
    employmentType: '',
    region: '',
  })

  const matchMutation = useMutation({
    mutationFn: () => matchProfile(buildProfileMatchRequest(inputs)),
  })

  const whatIfMutation = useMutation({
    mutationFn: () => analyzeCareerDelta(buildCareerDeltaAnalysisRequest(inputs)),
  })

  const inputsReady = inputs.profileText.trim().length >= 20
  const anyPending = matchMutation.isPending || whatIfMutation.isPending
  const whatIfHasAttempted = whatIfMutation.data !== undefined || whatIfMutation.error !== null

  const runCurrentMatch = () => {
    setActiveTab('match')
    matchMutation.mutate()
  }

  const runWhatIf = () => {
    setActiveTab('what-if')
    whatIfMutation.mutate()
  }

  return (
    <div className="space-y-8">
      <section className="rounded-[32px] border border-[color:var(--border)] bg-white/90 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-500">Match lab</p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight text-[color:var(--ink)]">
          Paste a profile and inspect both current fit and improvement paths.
        </h1>
        <p className="mt-3 max-w-3xl text-base leading-7 text-slate-600">
          Stay in one workflow: score what fits now, then switch into What If to see how the same
          profile could move toward stronger scenarios without re-entering the context.
        </p>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
          <label className="block text-sm font-semibold text-[color:var(--ink)]">
            Candidate profile or resume text
            <textarea
              value={inputs.profileText}
              onChange={(event) => setInputs((current) => ({ ...current, profileText: event.target.value }))}
              rows={12}
              className="mt-3 block w-full rounded-[24px] border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-4 text-sm leading-6 text-slate-700"
            />
          </label>

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <label className="text-sm text-slate-600">
              Target titles
              <input
                value={inputs.targetTitles}
                onChange={(event) => setInputs((current) => ({ ...current, targetTitles: event.target.value }))}
                placeholder="Data Scientist, ML Engineer"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-slate-600">
              Salary expectation (annual)
              <input
                value={inputs.salaryExpectation}
                onChange={(event) =>
                  setInputs((current) => ({ ...current, salaryExpectation: event.target.value }))
                }
                type="number"
                min={0}
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-slate-600">
              Employment type
              <input
                value={inputs.employmentType}
                onChange={(event) => setInputs((current) => ({ ...current, employmentType: event.target.value }))}
                placeholder="Full Time"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-slate-600">
              Region
              <input
                value={inputs.region}
                onChange={(event) => setInputs((current) => ({ ...current, region: event.target.value }))}
                placeholder="Central"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
          </div>

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={runCurrentMatch}
              disabled={anyPending || !inputsReady}
              className="rounded-full bg-[color:var(--brand)] px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-[color:var(--brand-strong)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {matchMutation.isPending ? 'Scoring profile...' : 'Run profile match'}
            </button>
            <button
              type="button"
              onClick={runWhatIf}
              disabled={anyPending || !inputsReady}
              className="rounded-full border border-[color:var(--brand)] bg-white px-5 py-3 text-sm font-semibold text-[color:var(--brand)] transition hover:bg-[color:var(--surface)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {whatIfMutation.isPending ? 'Running What If...' : 'Run What If'}
            </button>
          </div>
        </article>

        <article className="space-y-5">
          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Result shell</p>
                <h2 className="mt-2 text-2xl font-semibold text-[color:var(--ink)]">
                  One profile, two analysis modes.
                </h2>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setActiveTab('match')}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${tabButtonClass(activeTab === 'match')}`}
                >
                  Current Match
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('what-if')}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${tabButtonClass(activeTab === 'what-if')}`}
                >
                  What If
                </button>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-slate-600">
              <span>
                Shared profile context:{' '}
                <span className="font-semibold text-[color:var(--ink)]">
                  {inputsReady ? 'ready' : 'needs more detail'}
                </span>
              </span>
            </div>
          </div>

          {activeTab === 'match' ? (
            <>
              <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
                <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
                  <span>
                    Candidates scanned:{' '}
                    <span className="font-semibold text-[color:var(--ink)]">
                      {matchMutation.data?.total_candidates.toLocaleString() ?? 0}
                    </span>
                  </span>
                  <span>{matchMutation.data ? `${matchMutation.data.search_time_ms.toFixed(0)}ms` : null}</span>
                  {matchMutation.data?.degraded ? (
                    <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
                      degraded retrieval
                    </span>
                  ) : null}
                </div>

                <div className="mt-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Extracted skills</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {matchMutation.data?.extracted_skills.length ? (
                      matchMutation.data.extracted_skills.map((skill) => (
                        <span key={skill} className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-slate-700">
                          {skill}
                        </span>
                      ))
                    ) : (
                      <span className="text-sm text-slate-500">Run a profile match to inspect extracted skills.</span>
                    )}
                  </div>
                </div>
              </div>

              {matchMutation.error ? (
                <div className="rounded-[24px] border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-900">
                  {matchMutation.error instanceof Error ? matchMutation.error.message : 'Current match request failed.'}
                </div>
              ) : null}

              <div className="space-y-4">
                {matchMutation.data?.results.length ? (
                  matchMutation.data.results.map((job) => <JobCard key={job.uuid} job={job} />)
                ) : (
                  <div className="rounded-[28px] border border-dashed border-[color:var(--border)] bg-white/70 p-10 text-center text-sm text-slate-500">
                    Match results will appear here with score decomposition and missing-skill signals.
                  </div>
                )}
              </div>
            </>
          ) : (
            <>
              {whatIfMutation.error ? (
                <StatePanel
                  eyebrow="Request failed"
                  title="The What If analysis did not complete"
                  message={
                    whatIfMutation.error instanceof Error
                      ? whatIfMutation.error.message
                      : 'The request failed before the engine could return market evidence.'
                  }
                  tone="danger"
                  actionLabel="Retry analysis"
                  onAction={runWhatIf}
                />
              ) : null}

              <WhatIfSummaryPanel
                response={whatIfMutation.data}
                isPending={whatIfMutation.isPending}
                hasAttempted={whatIfHasAttempted}
                onRetry={runWhatIf}
              />
            </>
          )}
        </article>
      </section>
    </div>
  )
}

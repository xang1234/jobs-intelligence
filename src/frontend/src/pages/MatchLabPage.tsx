import { useEffect, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import JobCard from '@/components/JobCard'
import { toast } from '@/components/ui'
import { analyzeCareerDelta, getCareerDeltaScenarioDetail, matchProfile } from '@/services/api'
import { buildCareerDeltaAnalysisRequest, buildProfileMatchRequest } from '@/services/matchLab'
import type {
  CareerDeltaAnalysisResponse,
  CareerDeltaBaseline,
  CareerDeltaFilteredScenario,
  CareerDeltaScenarioChange,
  CareerDeltaScenarioDetail,
  CareerDeltaScenarioSignal,
  CareerDeltaScenarioSummary,
  MatchLabSharedInputs,
} from '@/types/api'

type MatchLabTab = 'match' | 'what-if'

type AppliedScenarioState = {
  scenarioId: string
  title: string
  previousInputs: MatchLabSharedInputs
  nextInputs: MatchLabSharedInputs
  changes: string[]
}

function tabButtonClass(isActive: boolean): string {
  return isActive
    ? 'bg-[color:var(--brand)] text-white shadow-lg'
    : 'bg-[color:var(--surface)] text-[color:var(--ink-muted)] hover:text-[color:var(--brand)]'
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

function areSharedInputsEqual(left: MatchLabSharedInputs, right: MatchLabSharedInputs): boolean {
  return (
    left.profileText === right.profileText
    && left.targetTitles === right.targetTitles
    && left.salaryExpectation === right.salaryExpectation
    && left.employmentType === right.employmentType
    && left.region === right.region
  )
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

function detailTradeoffs(detail: CareerDeltaScenarioDetail): string[] {
  const tradeoffs: string[] = []
  if (detail.missing_skills.length) {
    tradeoffs.push(`You still need to demonstrate ${compactList(detail.missing_skills, '')}.`)
  }
  if (detail.degraded) {
    tradeoffs.push('This detail came from a degraded retrieval path and may miss supporting evidence.')
  }
  if (detail.thin_market) {
    tradeoffs.push('The reachable pool is thin, so treat this move as directional rather than exhaustive.')
  }
  if (detail.change?.removed_skills.length) {
    tradeoffs.push(`Applying this move de-emphasizes ${compactList(detail.change.removed_skills, '')}.`)
  }
  return tradeoffs
}

function buildAppliedScenarioState(
  currentInputs: MatchLabSharedInputs,
  detail: CareerDeltaScenarioDetail,
): AppliedScenarioState {
  const changes: string[] = []

  if (detail.target_title) {
    changes.push(`Target roles shifted toward ${detail.target_title}.`)
  } else if (detail.change?.target_title_family && detail.change.target_title_family !== detail.change.source_title_family) {
    changes.push(`Target title family shifted toward ${titleCaseFromKey(detail.change.target_title_family)}.`)
  }

  if (detail.change?.added_skills.length) {
    changes.push(`Added skill emphasis: ${compactList(detail.change.added_skills, '')}.`)
  }
  if (detail.change?.replaced_skills.length) {
    changes.push(
      `Substitution focus: ${detail.change.replaced_skills
        .map((item) => `${item.from_skill} -> ${item.to_skill}`)
        .join(', ')}.`,
    )
  }
  if (detail.target_sector) {
    changes.push(`Sector focus shifted toward ${titleCaseFromKey(detail.target_sector)}.`)
  }

  if (!changes.length) {
    changes.push(`Applied scenario: ${detail.title}.`)
  }

  const nextInputs: MatchLabSharedInputs = {
    ...currentInputs,
    targetTitles: detail.target_title ?? currentInputs.targetTitles,
  }

  return {
    scenarioId: detail.scenario_id,
    title: detail.title,
    previousInputs: currentInputs,
    nextInputs,
    changes,
  }
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
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">{label}</p>
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
      ? 'border-[color:var(--color-danger-500)]/30 bg-[color:var(--danger-bg)] text-[color:var(--color-danger-900)]'
      : tone === 'warning'
        ? 'border-[color:var(--color-warning-500)]/30 bg-[color:var(--color-warning-50)] text-[color:var(--warning-fg)]'
        : 'border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] text-[color:var(--ink-muted)]'

  return (
    <section className={`rounded-[28px] border p-6 ${toneClass}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] opacity-70">{eyebrow}</p>
      <h3 className="mt-2 text-xl font-semibold">{title}</h3>
      <p className="mt-3 max-w-3xl text-sm leading-6 opacity-90">{message}</p>
      {actionLabel && onAction ? (
        <button
          type="button"
          onClick={onAction}
          className="mt-4 rounded-full border border-current px-4 py-2 text-sm font-semibold transition hover:bg-[color:var(--surface-1)]/40"
        >
          {actionLabel}
        </button>
      ) : null}
    </section>
  )
}

function BaselineInsightCard({ baseline }: { baseline: CareerDeltaBaseline }) {
  return (
    <article className="rounded-[28px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
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
              thin market
            </span>
          ) : null}
          {baseline.degraded ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
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

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
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
        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
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

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
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
        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
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

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
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

function WhatIfScenarioPreview({
  index,
  scenario,
  detail,
  detailLoading,
  detailError,
  expanded,
  applied,
  onToggleDetail,
  onRetryDetail,
  onApplyScenario,
}: {
  index: number
  scenario: CareerDeltaScenarioSummary
  detail: CareerDeltaScenarioDetail | null
  detailLoading: boolean
  detailError: string | null
  expanded: boolean
  applied: boolean
  onToggleDetail: () => void
  onRetryDetail: () => void
  onApplyScenario: (detail: CareerDeltaScenarioDetail) => void
}) {
  const primarySignal = bestScenarioSignal(scenario.signals)
  const changeLines = describeScenarioChange(scenario.change)

  return (
    <article className="rounded-[28px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6 shadow-[0_16px_40px_rgba(15,23,42,0.05)]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-full bg-[color:var(--brand)]/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--brand)]">
            #{index + 1}
          </span>
          <span className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
            {scenario.scenario_type.replaceAll('_', ' ')}
          </span>
          <span className="rounded-full bg-[color:var(--color-success-50)] px-3 py-1 text-xs font-semibold text-[color:var(--color-success-900)]">
            {formatConfidence(scenario.confidence.score)} confidence
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {scenario.thin_market ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              thin market
            </span>
          ) : null}
          {scenario.degraded ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              degraded
            </span>
          ) : null}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
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
      <p className="mt-2 text-sm leading-6 text-[color:var(--ink-muted)]">{scenario.summary}</p>

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
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Score breakdown</p>
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
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Recommended move</p>
          <div className="mt-3 space-y-2">
            {changeLines.length ? (
              changeLines.map((line) => (
                <p key={line} className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                  {line}
                </p>
              ))
            ) : (
              <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-subtle)]">
                The engine did not expose a structured change list for this scenario.
              </p>
            )}
          </div>
        </div>

        <div className="rounded-[24px] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Top signal</p>
          {primarySignal ? (
            <div className="mt-3 space-y-2">
              <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                {primarySignal.supporting_jobs} supporting jobs · {primarySignal.supporting_share_pct.toFixed(0)}%
                share
              </p>
              <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                Demand momentum {formatPercent(primarySignal.market_momentum)} · Median salary{' '}
                {formatCurrency(primarySignal.market_salary_annual_median)}
              </p>
              {primarySignal.skill ? (
                <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                  Signal skill: <span className="font-medium text-[color:var(--ink)]">{primarySignal.skill}</span>
                </p>
              ) : null}
            </div>
          ) : (
            <p className="mt-3 rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-subtle)]">
              No primary signal was attached to this recommendation.
            </p>
          )}
        </div>
      </div>

      <div className="mt-5">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Confidence notes</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {scenario.confidence.reasons.length ? (
            scenario.confidence.reasons.map((reason) => (
              <span key={reason} className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-[color:var(--ink-muted)]">
                {reason}
              </span>
            ))
          ) : (
            <span className="text-sm text-[color:var(--ink-subtle)]">No extra confidence notes supplied.</span>
          )}
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onToggleDetail}
          className="rounded-full border border-[color:var(--brand)] px-4 py-2 text-sm font-semibold text-[color:var(--brand)] transition hover:bg-[color:var(--surface)]"
        >
          {expanded ? 'Hide detail' : 'Inspect detail'}
        </button>
        {detail ? (
          <button
            type="button"
            onClick={() => onApplyScenario(detail)}
            className="rounded-full bg-[color:var(--brand)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--brand-strong)]"
          >
            {applied ? 'Applied to Match Lab' : 'Apply this scenario'}
          </button>
        ) : null}
      </div>

      {expanded ? (
        <div className="mt-5 rounded-[24px] border border-[color:var(--border)] bg-[color:var(--surface)] px-5 py-5">
          {detailLoading ? (
            <div className="space-y-3 animate-pulse">
              <div className="h-4 w-40 rounded-full bg-[color:var(--surface-3)]" />
              <div className="h-5 w-full rounded-full bg-[color:var(--surface-3)]" />
              <div className="h-5 w-5/6 rounded-full bg-[color:var(--surface-3)]" />
              <div className="grid gap-3 md:grid-cols-2">
                <div className="h-28 rounded-[20px] bg-[color:var(--surface-3)]" />
                <div className="h-28 rounded-[20px] bg-[color:var(--surface-3)]" />
              </div>
            </div>
          ) : detailError ? (
            <StatePanel
              eyebrow="Detail unavailable"
              title="The scenario detail could not be loaded"
              message={detailError}
              tone="danger"
              actionLabel="Retry detail"
              onAction={onRetryDetail}
            />
          ) : detail ? (
            <div className="space-y-5">
              <div className="grid gap-4 xl:grid-cols-2">
                <div className="rounded-[20px] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Current angle</p>
                  <p className="mt-3 text-sm leading-6 text-[color:var(--ink-muted)]">
                    {detail.change?.source_title_family || detail.change?.source_industry
                      ? `Current role signal: ${titleCaseFromKey(detail.change?.source_title_family)} in ${titleCaseFromKey(detail.change?.source_industry)}.`
                      : 'Current baseline is represented by your existing Match Lab inputs and market-position summary.'}
                  </p>
                </div>
                <div className="rounded-[20px] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Counterfactual angle</p>
                  <p className="mt-3 text-sm leading-6 text-[color:var(--ink-muted)]">{detail.narrative}</p>
                </div>
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                <div className="rounded-[20px] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Evidence</p>
                  <div className="mt-3 space-y-2">
                    {detail.evidence.length ? (
                      detail.evidence.map((item) => (
                        <p key={item} className="rounded-2xl bg-[color:var(--surface)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                          {item}
                        </p>
                      ))
                    ) : (
                      <p className="text-sm text-[color:var(--ink-subtle)]">No extra evidence was attached to this scenario.</p>
                    )}
                  </div>
                </div>

                <div className="rounded-[20px] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Signals and skill gaps</p>
                  <div className="mt-3 space-y-2">
                    {detail.signals.map((signal, signalIndex) => (
                      <p
                        key={`${detail.scenario_id}-${signalIndex}`}
                        className="rounded-2xl bg-[color:var(--surface)] px-4 py-3 text-sm text-[color:var(--ink-muted)]"
                      >
                        {signal.supporting_jobs} jobs support this move with {signal.supporting_share_pct.toFixed(0)}%
                        share, fit median {signal.fit_median != null ? `${(signal.fit_median * 100).toFixed(0)}%` : 'n/a'},
                        and salary {formatCurrency(signal.market_salary_annual_median)}.
                      </p>
                    ))}
                    {detail.missing_skills.length ? (
                      <p className="rounded-2xl bg-[color:var(--surface)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                        Missing skills to validate: {compactList(detail.missing_skills, '')}
                      </p>
                    ) : null}
                  </div>
                </div>
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                <div className="rounded-[20px] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Tradeoffs</p>
                  <div className="mt-3 space-y-2">
                    {detailTradeoffs(detail).length ? (
                      detailTradeoffs(detail).map((item) => (
                        <p key={item} className="rounded-2xl bg-[color:var(--surface)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                          {item}
                        </p>
                      ))
                    ) : (
                      <p className="rounded-2xl bg-[color:var(--surface)] px-4 py-3 text-sm text-[color:var(--ink-subtle)]">
                        No extra risks were attached beyond the baseline confidence notes.
                      </p>
                    )}
                  </div>
                </div>

                <div className="rounded-[20px] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Suggested search probes</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {detail.search_queries.length ? (
                      detail.search_queries.map((query) => (
                        <span key={query} className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-[color:var(--ink-muted)]">
                          {query}
                        </span>
                      ))
                    ) : (
                      <span className="text-sm text-[color:var(--ink-subtle)]">No extra search probes were generated.</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  )
}

function FilteredScenariosPanel({ filtered }: { filtered: CareerDeltaFilteredScenario[] }) {
  return (
    <section className="rounded-[28px] border border-dashed border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Filtered scenarios</p>
          <h3 className="mt-2 text-xl font-semibold text-[color:var(--ink)]">Rejected moves the engine considered</h3>
        </div>
        <span className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-semibold text-[color:var(--ink-muted)]">
          {filtered.length} filtered
        </span>
      </div>

      <div className="mt-4 space-y-3">
        {filtered.map((scenario) => (
          <article key={scenario.scenario_id} className="rounded-[22px] border border-[color:var(--border)] bg-[color:var(--surface-1)] px-5 py-4">
            <div className="flex flex-wrap items-center gap-3 text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
              <span>{scenario.scenario_type.replaceAll('_', ' ')}</span>
              <span>{scenario.reason_code.replaceAll('_', ' ')}</span>
              <span>{formatConfidence(scenario.confidence.score)} confidence</span>
            </div>
            <p className="mt-3 text-sm leading-6 text-[color:var(--ink-muted)]">{scenario.explanation}</p>
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
  expandedScenarioId,
  detailByScenarioId,
  detailErrorByScenarioId,
  detailLoadingId,
  appliedScenarioId,
  onToggleDetail,
  onRetryDetail,
  onApplyScenario,
}: {
  response: CareerDeltaAnalysisResponse | undefined
  isPending: boolean
  hasAttempted: boolean
  onRetry: () => void
  expandedScenarioId: string | null
  detailByScenarioId: Record<string, CareerDeltaScenarioDetail>
  detailErrorByScenarioId: Record<string, string>
  detailLoadingId: string | null
  appliedScenarioId: string | null
  onToggleDetail: (scenarioId: string) => void
  onRetryDetail: (scenarioId: string) => void
  onApplyScenario: (detail: CareerDeltaScenarioDetail) => void
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
              className="animate-pulse rounded-[28px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6"
            >
              <div className="h-3 w-28 rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-4 h-8 w-2/3 rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-3 h-4 w-full rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-2 h-4 w-5/6 rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-5 grid gap-3 md:grid-cols-2">
                <div className="h-20 rounded-[20px] bg-[color:var(--surface-3)]" />
                <div className="h-20 rounded-[20px] bg-[color:var(--surface-3)]" />
              </div>
              <p className="mt-4 text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">{label}</p>
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

      <div className="rounded-[28px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
        <div className="flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
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
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              thin market
            </span>
          ) : null}
          {resolvedResponse.degraded ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
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
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Ranked scenarios</p>
          <h3 className="mt-2 text-2xl font-semibold text-[color:var(--ink)]">Recommended moves with market evidence</h3>
        </div>

        {resolvedResponse.scenarios.length ? (
          resolvedResponse.scenarios.map((scenario, index) => (
            <WhatIfScenarioPreview
              key={scenario.scenario_id}
              index={index}
              scenario={scenario}
              detail={detailByScenarioId[scenario.scenario_id] ?? null}
              detailLoading={detailLoadingId === scenario.scenario_id}
              detailError={detailErrorByScenarioId[scenario.scenario_id] ?? null}
              expanded={expandedScenarioId === scenario.scenario_id}
              applied={appliedScenarioId === scenario.scenario_id}
              onToggleDetail={() => onToggleDetail(scenario.scenario_id)}
              onRetryDetail={() => onRetryDetail(scenario.scenario_id)}
              onApplyScenario={onApplyScenario}
            />
          ))
        ) : (
          <div className="rounded-[28px] border border-dashed border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-10 text-center text-sm text-[color:var(--ink-subtle)]">
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
  const [expandedScenarioId, setExpandedScenarioId] = useState<string | null>(null)
  const [scenarioDetails, setScenarioDetails] = useState<Record<string, CareerDeltaScenarioDetail>>({})
  const [detailErrors, setDetailErrors] = useState<Record<string, string>>({})
  const [detailLoadingId, setDetailLoadingId] = useState<string | null>(null)
  const [appliedScenario, setAppliedScenario] = useState<AppliedScenarioState | null>(null)
  const appliedScenarioRef = useRef<AppliedScenarioState | null>(null)
  useEffect(() => {
    appliedScenarioRef.current = appliedScenario
  }, [appliedScenario])

  const updateInputs = (updater: (current: MatchLabSharedInputs) => MatchLabSharedInputs) => {
    setInputs((current) => {
      const next = updater(current)
      if (appliedScenario && !areSharedInputsEqual(next, appliedScenario.nextInputs)) {
        setAppliedScenario(null)
      }
      return next
    })
  }

  const matchMutation = useMutation({
    mutationFn: (nextInputs: MatchLabSharedInputs) => matchProfile(buildProfileMatchRequest(nextInputs)),
  })

  const whatIfMutation = useMutation({
    mutationFn: (nextInputs: MatchLabSharedInputs) => analyzeCareerDelta(buildCareerDeltaAnalysisRequest(nextInputs)),
  })

  const detailMutation = useMutation({
    mutationFn: (scenarioId: string) => getCareerDeltaScenarioDetail(scenarioId),
    onSuccess: (detail) => {
      setScenarioDetails((current) => ({ ...current, [detail.scenario_id]: detail }))
      setDetailErrors((current) => {
        if (!(detail.scenario_id in current)) {
          return current
        }
        const next = { ...current }
        delete next[detail.scenario_id]
        return next
      })
    },
  })

  const inputsReady = inputs.profileText.trim().length >= 20
  const anyPending = matchMutation.isPending || whatIfMutation.isPending
  const whatIfHasAttempted = whatIfMutation.data !== undefined || whatIfMutation.error !== null

  const runCurrentMatch = (nextInputs = inputs) => {
    setActiveTab('match')
    matchMutation.mutate(nextInputs)
  }

  const runWhatIf = (nextInputs = inputs) => {
    setActiveTab('what-if')
    whatIfMutation.mutate(nextInputs)
  }

  const loadScenarioDetail = async (scenarioId: string) => {
    if (detailLoadingId === scenarioId) {
      return
    }

    setDetailLoadingId(scenarioId)
    setDetailErrors((current) => {
      if (!(scenarioId in current)) {
        return current
      }
      const next = { ...current }
      delete next[scenarioId]
      return next
    })

    try {
      await detailMutation.mutateAsync(scenarioId)
    } catch (error) {
      setDetailErrors((current) => ({
        ...current,
        [scenarioId]: error instanceof Error ? error.message : 'Scenario detail request failed.',
      }))
    } finally {
      setDetailLoadingId((current) => (current === scenarioId ? null : current))
    }
  }

  const toggleScenarioDetail = (scenarioId: string) => {
    if (expandedScenarioId === scenarioId) {
      setExpandedScenarioId(null)
      return
    }
    setExpandedScenarioId(scenarioId)
    void loadScenarioDetail(scenarioId)
  }

  const retryScenarioDetail = (scenarioId: string) => {
    setExpandedScenarioId(scenarioId)
    void loadScenarioDetail(scenarioId)
  }

  const applyScenario = (detail: CareerDeltaScenarioDetail) => {
    const applied = buildAppliedScenarioState(inputs, detail)
    setInputs(applied.nextInputs)
    setAppliedScenario(applied)
    runCurrentMatch(applied.nextInputs)
    toast.success('Scenario applied', {
      description: applied.title,
      action: {
        label: 'Undo',
        onClick: () => {
          if (appliedScenarioRef.current !== applied) return
          setInputs(applied.previousInputs)
          setAppliedScenario(null)
          runCurrentMatch(applied.previousInputs)
        },
      },
    })
  }

  const resetAppliedScenario = () => {
    if (!appliedScenario) {
      return
    }
    const previous = appliedScenario
    setInputs(previous.previousInputs)
    setAppliedScenario(null)
    runCurrentMatch(previous.previousInputs)
    toast('Scenario reverted', {
      description: previous.title,
    })
  }

  return (
    <div className="space-y-8">
      <section className="rounded-[32px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-8 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-[color:var(--ink-subtle)]">Match lab</p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight text-[color:var(--ink)]">
          Paste a profile and inspect both current fit and improvement paths.
        </h1>
        <p className="mt-3 max-w-3xl text-base leading-7 text-[color:var(--ink-muted)]">
          Stay in one workflow: score what fits now, then switch into What If to see how the same
          profile could move toward stronger scenarios without re-entering the context.
        </p>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <article className="rounded-[28px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
          <label className="block text-sm font-semibold text-[color:var(--ink)]">
            Candidate profile or resume text
            <textarea
              value={inputs.profileText}
              onChange={(event) => updateInputs((current) => ({ ...current, profileText: event.target.value }))}
              rows={12}
              className="mt-3 block w-full rounded-[24px] border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-4 text-sm leading-6 text-[color:var(--ink-muted)]"
            />
          </label>

          {appliedScenario ? (
            <div className="mt-5 rounded-[24px] border border-[color:var(--brand)]/30 bg-[color:var(--brand)]/5 px-5 py-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--brand)]">
                    Applied scenario
                  </p>
                  <h3 className="mt-2 text-lg font-semibold text-[color:var(--ink)]">{appliedScenario.title}</h3>
                </div>
                <button
                  type="button"
                  onClick={resetAppliedScenario}
                  className="rounded-full border border-[color:var(--brand)] px-4 py-2 text-sm font-semibold text-[color:var(--brand)] transition hover:bg-[color:var(--surface-1)]"
                >
                  Revert changes
                </button>
              </div>
              <div className="mt-3 space-y-2">
                {appliedScenario.changes.map((change) => (
                  <p key={change} className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                    {change}
                  </p>
                ))}
              </div>
            </div>
          ) : null}

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <label className="text-sm text-[color:var(--ink-muted)]">
              Target titles
              <input
                value={inputs.targetTitles}
                onChange={(event) => updateInputs((current) => ({ ...current, targetTitles: event.target.value }))}
                placeholder="Data Scientist, ML Engineer"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-[color:var(--ink-muted)]">
              Salary expectation (annual)
              <input
                value={inputs.salaryExpectation}
                onChange={(event) =>
                  updateInputs((current) => ({ ...current, salaryExpectation: event.target.value }))
                }
                type="number"
                min={0}
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-[color:var(--ink-muted)]">
              Employment type
              <input
                value={inputs.employmentType}
                onChange={(event) => updateInputs((current) => ({ ...current, employmentType: event.target.value }))}
                placeholder="Full Time"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-[color:var(--ink-muted)]">
              Region
              <input
                value={inputs.region}
                onChange={(event) => updateInputs((current) => ({ ...current, region: event.target.value }))}
                placeholder="Central"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
          </div>

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => runCurrentMatch()}
              disabled={anyPending || !inputsReady}
              className="rounded-full bg-[color:var(--brand)] px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-[color:var(--brand-strong)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {matchMutation.isPending ? 'Scoring profile...' : 'Run profile match'}
            </button>
            <button
              type="button"
              onClick={() => runWhatIf()}
              disabled={anyPending || !inputsReady}
              className="rounded-full border border-[color:var(--brand)] bg-[color:var(--surface-1)] px-5 py-3 text-sm font-semibold text-[color:var(--brand)] transition hover:bg-[color:var(--surface)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {whatIfMutation.isPending ? 'Running What If...' : 'Run What If'}
            </button>
          </div>
        </article>

        <article className="space-y-5">
          <div className="rounded-[28px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Result shell</p>
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

            <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
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
              <div className="rounded-[28px] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
                <div className="flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
                  <span>
                    Candidates scanned:{' '}
                    <span className="font-semibold text-[color:var(--ink)]">
                      {matchMutation.data?.total_candidates.toLocaleString() ?? 0}
                    </span>
                  </span>
                  <span>{matchMutation.data ? `${matchMutation.data.search_time_ms.toFixed(0)}ms` : null}</span>
                  {matchMutation.data?.degraded ? (
                    <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
                      degraded retrieval
                    </span>
                  ) : null}
                </div>

                <div className="mt-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Extracted skills</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {matchMutation.data?.extracted_skills.length ? (
                      matchMutation.data.extracted_skills.map((skill) => (
                        <span key={skill} className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-[color:var(--ink-muted)]">
                          {skill}
                        </span>
                      ))
                    ) : (
                      <span className="text-sm text-[color:var(--ink-subtle)]">Run a profile match to inspect extracted skills.</span>
                    )}
                  </div>
                </div>
              </div>

              {matchMutation.error ? (
                <div className="rounded-[24px] border border-[color:var(--color-danger-500)]/30 bg-[color:var(--danger-bg)] px-5 py-4 text-sm text-[color:var(--color-danger-900)]">
                  {matchMutation.error instanceof Error ? matchMutation.error.message : 'Current match request failed.'}
                </div>
              ) : null}

              <div className="space-y-4">
                {matchMutation.data?.results.length ? (
                  matchMutation.data.results.map((job) => <JobCard key={job.uuid} job={job} />)
                ) : (
                  <div className="rounded-[28px] border border-dashed border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-10 text-center text-sm text-[color:var(--ink-subtle)]">
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
                  onAction={() => runWhatIf()}
                />
              ) : null}

              {!whatIfMutation.error ? (
                <WhatIfSummaryPanel
                  response={whatIfMutation.data}
                  isPending={whatIfMutation.isPending}
                  hasAttempted={whatIfHasAttempted}
                  onRetry={runWhatIf}
                  expandedScenarioId={expandedScenarioId}
                  detailByScenarioId={scenarioDetails}
                  detailErrorByScenarioId={detailErrors}
                  detailLoadingId={detailLoadingId}
                  appliedScenarioId={appliedScenario?.scenarioId ?? null}
                  onToggleDetail={toggleScenarioDetail}
                  onRetryDetail={retryScenarioDetail}
                  onApplyScenario={applyScenario}
                />
              ) : null}
            </>
          )}
        </article>
      </section>
    </div>
  )
}

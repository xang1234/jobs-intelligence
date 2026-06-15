import { Collapsible } from '@/components/ui'
import type { CareerDeltaScenarioDetail, CareerDeltaScenarioSummary } from '@/types/api'
import {
  bestScenarioSignal,
  compactList,
  describeScenarioChange,
  detailTradeoffs,
  formatConfidence,
  formatCurrency,
  formatPercent,
  formatRatioPercent,
  titleCaseFromKey,
} from './helpers'
import { StatePanel, SummaryMetric } from './primitives'

export function WhatIfScenarioPreview({
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
    <article className="rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6 shadow-[var(--shadow-lg)]">
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
              limited data
            </span>
          ) : null}
          {scenario.degraded ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              rough estimate
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

      {/* The answer: what you'd gain and what to change. */}
      <div className="mt-5 grid gap-3 md:grid-cols-2">
        <SummaryMetric label="Expected salary change" value={formatRatioPercent(scenario.expected_salary_delta_pct)} accent />
        <SummaryMetric label="Roles supporting this move" value={scenario.confidence.market_sample_size.toLocaleString()} />
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">What to change</p>
          <div className="mt-3 space-y-2">
            {changeLines.length ? (
              changeLines.map((line) => (
                <p key={line} className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                  {line}
                </p>
              ))
            ) : (
              <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-subtle)]">
                No specific skill or title change was suggested for this move.
              </p>
            )}
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] bg-[color:var(--surface)] px-5 py-4">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Why this move</p>
          {primarySignal ? (
            <div className="mt-3 space-y-2">
              <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                {primarySignal.supporting_jobs} jobs back this move · {primarySignal.supporting_share_pct.toFixed(0)}%
                of the pool
              </p>
              <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                Demand {formatPercent(primarySignal.market_momentum)} · Median pay{' '}
                {formatCurrency(primarySignal.market_salary_annual_median)}
              </p>
              {primarySignal.skill ? (
                <p className="rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-muted)]">
                  Key skill: <span className="font-medium text-[color:var(--ink)]">{primarySignal.skill}</span>
                </p>
              ) : null}
            </div>
          ) : (
            <p className="mt-3 rounded-2xl bg-[color:var(--surface-1)] px-4 py-3 text-sm text-[color:var(--ink-subtle)]">
              No standout signal was attached to this recommendation.
            </p>
          )}
        </div>
      </div>

      {/* The math: collapsed by default — most people never need it. */}
      <Collapsible summary="How we scored this" variant="card" className="mt-5">
        <div className="grid gap-3 md:grid-cols-3">
          <SummaryMetric
            label="Market evidence"
            value={`${(scenario.confidence.evidence_coverage * 100).toFixed(0)}%`}
          />
          <SummaryMetric label="Jobs analysed" value={scenario.confidence.market_sample_size.toLocaleString()} />
          <SummaryMetric
            label="Effort to switch"
            value={scenario.score_breakdown ? scenario.score_breakdown.pivot_cost.toFixed(2) : 'n/a'}
          />
        </div>

        {scenario.score_breakdown ? (
          <div>
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

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Why we're confident</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {scenario.confidence.reasons.length ? (
              scenario.confidence.reasons.map((reason) => (
                <span key={reason} className="rounded-full bg-[color:var(--surface-1)] px-3 py-1 text-xs font-medium text-[color:var(--ink-muted)]">
                  {reason}
                </span>
              ))
            ) : (
              <span className="text-sm text-[color:var(--ink-subtle)]">No extra notes supplied.</span>
            )}
          </div>
        </div>
      </Collapsible>

      <div className="mt-5 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onToggleDetail}
          className="rounded-full border border-[color:var(--brand)] px-4 py-2 text-sm font-semibold text-[color:var(--brand)] transition hover:bg-[color:var(--surface)]"
        >
          {expanded ? 'Hide details' : 'See the full picture'}
        </button>
        {detail ? (
          <button
            type="button"
            onClick={() => onApplyScenario(detail)}
            className="rounded-full bg-[color:var(--brand)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--brand-strong)]"
          >
            {applied ? 'Applied ✓' : 'Use this move'}
          </button>
        ) : null}
      </div>

      {expanded ? (
        <div className="mt-5 rounded-[var(--radius-lg)] border border-[color:var(--border)] bg-[color:var(--surface)] px-5 py-5">
          {detailLoading ? (
            <div className="space-y-3 animate-pulse">
              <div className="h-4 w-40 rounded-full bg-[color:var(--surface-3)]" />
              <div className="h-5 w-full rounded-full bg-[color:var(--surface-3)]" />
              <div className="h-5 w-5/6 rounded-full bg-[color:var(--surface-3)]" />
              <div className="grid gap-3 md:grid-cols-2">
                <div className="h-28 rounded-[var(--radius-md)] bg-[color:var(--surface-3)]" />
                <div className="h-28 rounded-[var(--radius-md)] bg-[color:var(--surface-3)]" />
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
                <div className="rounded-[var(--radius-md)] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Current angle</p>
                  <p className="mt-3 text-sm leading-6 text-[color:var(--ink-muted)]">
                    {detail.change?.source_title_family || detail.change?.source_industry
                      ? `Current role signal: ${titleCaseFromKey(detail.change?.source_title_family)} in ${titleCaseFromKey(detail.change?.source_industry)}.`
                      : 'Current baseline is represented by your existing Match Lab inputs and market-position summary.'}
                  </p>
                </div>
                <div className="rounded-[var(--radius-md)] bg-[color:var(--surface-1)] px-5 py-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Counterfactual angle</p>
                  <p className="mt-3 text-sm leading-6 text-[color:var(--ink-muted)]">{detail.narrative}</p>
                </div>
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                <div className="rounded-[var(--radius-md)] bg-[color:var(--surface-1)] px-5 py-4">
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

                <div className="rounded-[var(--radius-md)] bg-[color:var(--surface-1)] px-5 py-4">
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
                <div className="rounded-[var(--radius-md)] bg-[color:var(--surface-1)] px-5 py-4">
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

                <div className="rounded-[var(--radius-md)] bg-[color:var(--surface-1)] px-5 py-4">
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

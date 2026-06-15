import { Collapsible } from '@/components/ui'
import type {
  CareerDeltaAnalysisResponse,
  CareerDeltaFilteredScenario,
  CareerDeltaScenarioDetail,
} from '@/types/api'
import { formatConfidence } from './helpers'
import { BaselineInsightCard } from './BaselineCard'
import { StatePanel } from './primitives'
import { WhatIfScenarioPreview } from './ScenarioCard'

function FilteredScenariosPanel({ filtered }: { filtered: CareerDeltaFilteredScenario[] }) {
  return (
    <section className="rounded-[var(--radius-xl)] border border-dashed border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Ruled out</p>
          <h3 className="mt-2 text-xl font-semibold text-[color:var(--ink)]">Moves we considered but ruled out</h3>
        </div>
        <span className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-semibold text-[color:var(--ink-muted)]">
          {filtered.length} ruled out
        </span>
      </div>

      <div className="mt-4 space-y-3">
        {filtered.map((scenario) => (
          <article key={scenario.scenario_id} className="rounded-[var(--radius-lg)] border border-[color:var(--border)] bg-[color:var(--surface-1)] px-5 py-4">
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

export function WhatIfSummaryPanel({
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
              className="animate-pulse rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6"
            >
              <div className="h-3 w-28 rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-4 h-8 w-2/3 rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-3 h-4 w-full rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-2 h-4 w-5/6 rounded-full bg-[color:var(--surface-3)]" />
              <div className="mt-5 grid gap-3 md:grid-cols-2">
                <div className="h-20 rounded-[var(--radius-md)] bg-[color:var(--surface-3)]" />
                <div className="h-20 rounded-[var(--radius-md)] bg-[color:var(--surface-3)]" />
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

      <div className="rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
        <div className="flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
          <span>
            Moves found:{' '}
            <span className="font-semibold text-[color:var(--ink)]">{resolvedResponse.scenarios.length}</span>
          </span>
          <span>
            Ruled out:{' '}
            <span className="font-semibold text-[color:var(--ink)]">{resolvedResponse.filtered_scenarios.length}</span>
          </span>
          {resolvedResponse.thin_market ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              limited data
            </span>
          ) : null}
          {resolvedResponse.degraded ? (
            <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
              rough estimate
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
          <div className="rounded-[var(--radius-xl)] border border-dashed border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-10 text-center text-sm text-[color:var(--ink-subtle)]">
            {resolvedResponse.thin_market
              ? 'Recommendations are withheld because the reachable pool is too thin to support a trustworthy move.'
              : 'No recommendation cleared the quality bar for this profile. Use the baseline and filtered scenarios below to see why the engine stayed conservative.'}
          </div>
        )}
      </section>

      {resolvedResponse.filtered_scenarios.length ? (
        <Collapsible
          summary="Why were some moves ruled out?"
          badge={`(${resolvedResponse.filtered_scenarios.length})`}
        >
          <FilteredScenariosPanel filtered={resolvedResponse.filtered_scenarios} />
        </Collapsible>
      ) : null}
    </div>
  )
}

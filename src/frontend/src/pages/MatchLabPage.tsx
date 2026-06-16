import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import JobCard from '@/components/JobCard'
import PageHero from '@/components/shell/PageHero'
import { toast } from '@/components/ui'
import { analyzeCareerDelta, getCareerDeltaScenarioDetail, matchProfile } from '@/services/api'
import { buildCareerDeltaAnalysisRequest, buildProfileMatchRequest } from '@/services/matchLab'
import type { CareerDeltaScenarioDetail, MatchLabSharedInputs } from '@/types/api'
import {
  areSharedInputsEqual,
  buildAppliedScenarioState,
  tabButtonClass,
  type AppliedScenarioState,
  type MatchLabTab,
} from './match-lab/helpers'
import { StatePanel } from './match-lab/primitives'

const loadWhatIfPanel = () =>
  import('./match-lab/WhatIfPanel').then(({ WhatIfSummaryPanel }) => ({
    default: WhatIfSummaryPanel,
  }))
const WhatIfSummaryPanel = lazy(loadWhatIfPanel)

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
  const profileMatchExtractedSkills = matchMutation.data?.extracted_skills ?? []

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
      <PageHero
        eyebrow="Match my CV"
        title="Paste your CV. See what fits now — and your best next move."
        subtitle="Get your strongest matches today, then see the moves that would open up better-paying roles — all from the same profile."
      />

      <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <article className="rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
          <label className="block text-sm font-semibold text-[color:var(--ink)]">
            Your CV or profile text
            <textarea
              value={inputs.profileText}
              onChange={(event) => updateInputs((current) => ({ ...current, profileText: event.target.value }))}
              rows={12}
              className="mt-3 block w-full rounded-[var(--radius-lg)] border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-4 text-sm leading-6 text-[color:var(--ink-muted)]"
            />
          </label>

          {appliedScenario ? (
            <div className="mt-5 rounded-[var(--radius-lg)] border border-[color:var(--brand)]/30 bg-[color:var(--brand)]/5 px-5 py-4">
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
              className="rounded-full bg-[color:var(--brand)] px-5 py-3 text-sm font-semibold text-white shadow-[var(--shadow-md)] transition hover:bg-[color:var(--brand-strong)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {matchMutation.isPending ? 'Finding matches…' : 'Find my matches'}
            </button>
            <button
              type="button"
              onClick={() => runWhatIf()}
              disabled={anyPending || !inputsReady}
              className="rounded-full border border-[color:var(--brand)] bg-[color:var(--surface-1)] px-5 py-3 text-sm font-semibold text-[color:var(--brand)] transition hover:bg-[color:var(--surface)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {whatIfMutation.isPending ? 'Looking for moves…' : 'Suggest better moves'}
            </button>
          </div>
        </article>

        <article className="space-y-5">
          <div className="rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Your results</p>
                <h2 className="mt-2 text-2xl font-semibold text-[color:var(--ink)]">
                  Your matches and your best next move.
                </h2>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setActiveTab('match')}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${tabButtonClass(activeTab === 'match')}`}
                >
                  My matches
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('what-if')}
                  onPointerEnter={() => void loadWhatIfPanel()}
                  onMouseEnter={() => void loadWhatIfPanel()}
                  onFocus={() => void loadWhatIfPanel()}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${tabButtonClass(activeTab === 'what-if')}`}
                >
                  Better moves
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
              <div className="rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6">
                <div className="flex flex-wrap items-center gap-3 text-sm text-[color:var(--ink-muted)]">
                  <span>
                    Jobs considered:{' '}
                    <span className="font-semibold text-[color:var(--ink)]">
                      {matchMutation.data?.total_candidates.toLocaleString() ?? 0}
                    </span>
                  </span>
                  {matchMutation.data?.degraded ? (
                    <span className="rounded-full bg-[color:var(--warning-bg)] px-3 py-1 text-xs font-semibold text-[color:var(--warning-fg)]">
                      limited data
                    </span>
                  ) : null}
                </div>

                <div className="mt-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">Skills we found in your CV</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {profileMatchExtractedSkills.length ? (
                      profileMatchExtractedSkills.map((skill) => (
                        <span key={skill} className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-[color:var(--ink-muted)]">
                          {skill}
                        </span>
                      ))
                    ) : (
                      <span className="text-sm text-[color:var(--ink-subtle)]">Run a match to see the skills we found in your CV.</span>
                    )}
                  </div>
                </div>
              </div>

              {matchMutation.error ? (
                <div className="rounded-[var(--radius-lg)] border border-[color:var(--color-danger-500)]/30 bg-[color:var(--danger-bg)] px-5 py-4 text-sm text-[color:var(--color-danger-900)]">
                  {matchMutation.error instanceof Error ? matchMutation.error.message : 'Current match request failed.'}
                </div>
              ) : null}

              <div className="space-y-4">
                {matchMutation.data?.results.length ? (
                  matchMutation.data.results.map((job) => <JobCard key={job.uuid} job={job} />)
                ) : (
                  <div className="rounded-[var(--radius-xl)] border border-dashed border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-10 text-center text-sm text-[color:var(--ink-subtle)]">
                    Your best-matching jobs will appear here, each with a fit score and the skills you're missing.
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
                <Suspense
                  fallback={
                    <StatePanel
                      eyebrow="Loading"
                      title="Preparing Better moves"
                      message="The counterfactual results panel is loading."
                    />
                  }
                >
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
                </Suspense>
              ) : null}
            </>
          )}
        </article>
      </section>
    </div>
  )
}

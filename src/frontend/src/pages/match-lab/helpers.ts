import type {
  CareerDeltaScenarioChange,
  CareerDeltaScenarioDetail,
  CareerDeltaScenarioSignal,
  MatchLabSharedInputs,
} from '@/types/api'

export type MatchLabTab = 'match' | 'what-if'

export type AppliedScenarioState = {
  scenarioId: string
  title: string
  previousInputs: MatchLabSharedInputs
  nextInputs: MatchLabSharedInputs
  changes: string[]
}

export function tabButtonClass(isActive: boolean): string {
  return isActive
    ? 'bg-[color:var(--brand)] text-white shadow-[var(--shadow-md)]'
    : 'bg-[color:var(--surface)] text-[color:var(--ink-muted)] hover:text-[color:var(--brand)]'
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return 'n/a'
  }
  return `${value > 0 ? '+' : ''}${value.toFixed(0)}%`
}

export function formatRatioPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return 'n/a'
  }
  return `${value > 0 ? '+' : ''}${(value * 100).toFixed(0)}%`
}

export function formatConfidence(score: number): string {
  return `${(score * 100).toFixed(0)}%`
}

export function formatCurrency(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return 'n/a'
  }
  return new Intl.NumberFormat('en-SG', {
    style: 'currency',
    currency: 'SGD',
    maximumFractionDigits: 0,
  }).format(value)
}

export function titleCaseFromKey(value: string | null | undefined): string {
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

export function compactList(items: string[], emptyLabel: string): string {
  return items.length ? items.join(', ') : emptyLabel
}

export function areSharedInputsEqual(left: MatchLabSharedInputs, right: MatchLabSharedInputs): boolean {
  return (
    left.profileText === right.profileText
    && left.targetTitles === right.targetTitles
    && left.salaryExpectation === right.salaryExpectation
    && left.employmentType === right.employmentType
    && left.region === right.region
  )
}

export function describeScenarioChange(change: CareerDeltaScenarioChange | null): string[] {
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

export function bestScenarioSignal(signals: CareerDeltaScenarioSignal[]): CareerDeltaScenarioSignal | null {
  if (!signals.length) {
    return null
  }
  return [...signals].sort((left, right) => right.supporting_jobs - left.supporting_jobs)[0]
}

export function detailTradeoffs(detail: CareerDeltaScenarioDetail): string[] {
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

export function buildAppliedScenarioState(
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

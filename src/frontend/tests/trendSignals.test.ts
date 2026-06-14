import assert from 'node:assert/strict'
import { test } from 'node:test'

import { getMomentumSignal, TREND_WINDOW_OPTIONS } from '../src/services/trendSignals.ts'
import type { TrendPoint } from '../src/types/api.ts'

function point(overrides: Partial<TrendPoint>): TrendPoint {
  return {
    month: '2026-03',
    job_count: 10,
    market_share: 1.2,
    median_salary_annual: 60000,
    momentum: 0,
    momentum_status: 'stable',
    momentum_label: 'Steady',
    ...overrides,
  }
}

test('labels zero-baseline growth as a new signal instead of plus one hundred percent', () => {
  const signal = getMomentumSignal(point({
    momentum: 100,
    momentum_status: 'new',
    momentum_label: 'New signal',
  }))

  assert.equal(signal.label, 'New signal')
  assert.equal(signal.detail, 'No prior baseline in the hosted window')
  assert.equal(signal.showPercent, false)
  assert.equal(signal.intent, 'info')
})

test('labels directional trends with percentages when a baseline exists', () => {
  const signal = getMomentumSignal(point({
    momentum: 24.6,
    momentum_status: 'up',
    momentum_label: 'Rising',
  }))

  assert.equal(signal.label, '+24.6%')
  assert.equal(signal.detail, 'Rising vs prior baseline')
  assert.equal(signal.showPercent, true)
  assert.equal(signal.intent, 'success')
})

test('restricts frontend trend windows to the hosted three-month data horizon', () => {
  assert.deepEqual(
    TREND_WINDOW_OPTIONS.map((option) => option.value),
    [1, 2, 3],
  )
})

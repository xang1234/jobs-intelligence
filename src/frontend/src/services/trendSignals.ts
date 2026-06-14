import type { TrendPoint } from '../types/api'

export type MomentumSignalIntent = 'neutral' | 'success' | 'warning' | 'danger' | 'info'

export interface MomentumSignal {
  label: string
  detail: string
  intent: MomentumSignalIntent
  showPercent: boolean
}

export const TREND_WINDOW_OPTIONS = [
  { value: 1, label: 'Latest month' },
  { value: 2, label: '2 months' },
  { value: 3, label: '3 months' },
] as const

function signedPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`
}

export function getMomentumSignal(point: TrendPoint | null | undefined): MomentumSignal {
  if (!point) {
    return {
      label: 'No signal',
      detail: 'No matching postings in this window',
      intent: 'neutral',
      showPercent: false,
    }
  }

  switch (point.momentum_status) {
    case 'new':
      return {
        label: point.momentum_label || 'New signal',
        detail: 'No prior baseline in the hosted window',
        intent: 'info',
        showPercent: false,
      }
    case 'insufficient_baseline':
      return {
        label: point.momentum_label || 'No baseline',
        detail: 'Not enough prior postings for a growth rate',
        intent: 'warning',
        showPercent: false,
      }
    case 'down':
      return {
        label: signedPercent(point.momentum),
        detail: `${point.momentum_label || 'Cooling'} vs prior baseline`,
        intent: 'danger',
        showPercent: true,
      }
    case 'up':
      return {
        label: signedPercent(point.momentum),
        detail: `${point.momentum_label || 'Rising'} vs prior baseline`,
        intent: 'success',
        showPercent: true,
      }
    case 'stable':
      return {
        label: signedPercent(point.momentum),
        detail: `${point.momentum_label || 'Steady'} vs prior baseline`,
        intent: 'neutral',
        showPercent: true,
      }
    default:
      if (point.momentum === 100 && point.job_count > 0) {
        return {
          label: 'New signal',
          detail: 'No prior baseline in the hosted window',
          intent: 'info',
          showPercent: false,
        }
      }
      return {
        label: signedPercent(point.momentum),
        detail: 'Compared with prior baseline',
        intent: point.momentum >= 15 ? 'success' : point.momentum <= -15 ? 'danger' : 'neutral',
        showPercent: true,
      }
  }
}

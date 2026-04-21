import type { ReactNode } from 'react'
import { ArrowDownRightIcon, ArrowUpRightIcon } from '@heroicons/react/20/solid'
import { Card, Chip, Skeleton } from '@/components/ui'

export interface MetricCardProps {
  label: string
  value: ReactNode
  hint?: ReactNode
  deltaPct?: number | null
  loading?: boolean
  elevation?: 1 | 2
  as?: 'article' | 'div'
}

function formatDelta(pct: number): string {
  const sign = pct >= 0 ? '+' : ''
  return `${sign}${pct.toFixed(1)}%`
}

export default function MetricCard({
  label,
  value,
  hint,
  deltaPct,
  loading,
  elevation = 1,
  as = 'article',
}: MetricCardProps) {
  const isUp = (deltaPct ?? 0) >= 0
  const valueClasses =
    elevation === 2
      ? 'text-4xl font-semibold tracking-tight text-[color:var(--ink)] sm:text-5xl'
      : 'text-3xl font-semibold tracking-tight text-[color:var(--ink)]'

  return (
    <Card as={as} radius="xl" elevation={elevation} className="p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
        {label}
      </p>
      <div className="mt-2 flex items-end justify-between gap-3">
        {loading ? (
          <Skeleton height={elevation === 2 ? 44 : 32} width="60%" />
        ) : (
          <p className={valueClasses}>{value}</p>
        )}
        {deltaPct != null && !loading && (
          <Chip
            intent={isUp ? 'success' : 'danger'}
            size="sm"
            leftIcon={
              isUp ? (
                <ArrowUpRightIcon className="h-3 w-3" />
              ) : (
                <ArrowDownRightIcon className="h-3 w-3" />
              )
            }
          >
            {formatDelta(deltaPct)}
          </Chip>
        )}
      </div>
      {hint && (
        <p className="mt-2 text-xs text-[color:var(--ink-subtle)]">{hint}</p>
      )}
    </Card>
  )
}

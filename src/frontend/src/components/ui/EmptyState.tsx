import type { ReactNode } from 'react'
import { cn } from './cn'

export interface EmptyStateProps {
  icon?: ReactNode
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
  compact?: boolean
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  compact,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 text-center',
        compact ? 'py-8' : 'py-14',
        className,
      )}
    >
      {icon && (
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[color:var(--surface-2)] text-[color:var(--brand)] [&_svg]:h-6 [&_svg]:w-6">
          {icon}
        </div>
      )}
      <h3 className="text-base font-semibold text-[color:var(--ink)]">{title}</h3>
      {description && (
        <p className="max-w-md text-sm text-[color:var(--ink-muted)]">
          {description}
        </p>
      )}
      {action && <div className="mt-1">{action}</div>}
    </div>
  )
}

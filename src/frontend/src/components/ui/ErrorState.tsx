import { ExclamationTriangleIcon } from '@heroicons/react/24/outline'
import type { ReactNode } from 'react'
import { Button } from './Button'
import { cn } from './cn'

export interface ErrorStateProps {
  title?: ReactNode
  description?: ReactNode
  retryLabel?: string
  onRetry?: () => void
  className?: string
  compact?: boolean
}

export function ErrorState({
  title = 'Something went wrong',
  description = 'Please try again. If the issue persists, refresh the page.',
  retryLabel = 'Try again',
  onRetry,
  className,
  compact,
}: ErrorStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 text-center',
        compact ? 'py-8' : 'py-14',
        className,
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[color:var(--danger-bg)] text-[color:var(--danger-fg)]">
        <ExclamationTriangleIcon className="h-6 w-6" />
      </div>
      <h3 className="text-base font-semibold text-[color:var(--ink)]">{title}</h3>
      {description && (
        <p className="max-w-md text-sm text-[color:var(--ink-muted)]">
          {description}
        </p>
      )}
      {onRetry && (
        <Button variant="secondary" size="sm" onClick={onRetry} className="mt-1">
          {retryLabel}
        </Button>
      )}
    </div>
  )
}

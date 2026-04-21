import { cn } from './cn'

export interface ProgressBarProps {
  value?: number // 0-100, omit for indeterminate
  className?: string
  intent?: 'brand' | 'success' | 'warning' | 'danger'
  size?: 'sm' | 'md'
  'aria-label'?: string
}

const COLOR: Record<NonNullable<ProgressBarProps['intent']>, string> = {
  brand: 'bg-[color:var(--brand)]',
  success: 'bg-[color:var(--color-success-500)]',
  warning: 'bg-[color:var(--color-warning-500)]',
  danger: 'bg-[color:var(--color-danger-500)]',
}

const HEIGHT: Record<NonNullable<ProgressBarProps['size']>, string> = {
  sm: 'h-1',
  md: 'h-2',
}

export function ProgressBar({
  value,
  intent = 'brand',
  size = 'sm',
  className,
  'aria-label': ariaLabel = 'Progress',
}: ProgressBarProps) {
  const indeterminate = value == null
  return (
    <div
      role="progressbar"
      aria-label={ariaLabel}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
      className={cn(
        'relative w-full overflow-hidden rounded-full bg-[color:var(--surface-2)]',
        HEIGHT[size],
        className,
      )}
    >
      {indeterminate ? (
        <span
          className={cn(
            'absolute inset-y-0 left-0 w-1/3 rounded-full motion-safe:animate-[progress_1.4s_ease-in-out_infinite]',
            COLOR[intent],
          )}
        />
      ) : (
        <span
          style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
          className={cn('block h-full rounded-full transition-[width] duration-300', COLOR[intent])}
        />
      )}
      <style>{`
        @keyframes progress {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
      `}</style>
    </div>
  )
}

import type { ReactNode } from 'react'
import { cn } from './cn'

export interface SegmentedOption<T extends string | number> {
  value: T
  label: ReactNode
  disabled?: boolean
}

export interface SegmentedControlProps<T extends string | number> {
  value: T
  onChange: (value: T) => void
  options: ReadonlyArray<SegmentedOption<T>>
  size?: 'sm' | 'md'
  'aria-label'?: string
  className?: string
}

export function SegmentedControl<T extends string | number>({
  value,
  onChange,
  options,
  size = 'md',
  'aria-label': ariaLabel,
  className,
}: SegmentedControlProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        'inline-flex items-center rounded-full border border-[color:var(--border)] bg-[color:var(--surface-1)] p-1 shadow-[var(--shadow-xs)]',
        className,
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value
        return (
          <button
            key={String(opt.value)}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={opt.disabled}
            onClick={() => onChange(opt.value)}
            className={cn(
              'rounded-full transition font-semibold focus-visible:outline-none',
              size === 'sm' ? 'px-3 py-1 text-xs' : 'px-4 py-1.5 text-sm',
              active
                ? 'bg-[color:var(--brand)] text-white shadow-[var(--shadow-sm)]'
                : 'text-[color:var(--ink-muted)] hover:text-[color:var(--ink)]',
              opt.disabled && 'cursor-not-allowed opacity-50',
            )}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

import { forwardRef, useId, type InputHTMLAttributes, type ReactNode } from 'react'
import { cn } from './cn'

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  hint?: string
  error?: string | null
  leftIcon?: ReactNode
  rightIcon?: ReactNode
  fullWidth?: boolean
}

const WRAPPER =
  'relative flex items-center rounded-[var(--radius-md)] border bg-[color:var(--surface-1)] transition ' +
  'focus-within:border-[color:var(--border-strong)] focus-within:shadow-[var(--ring)]'

const INPUT_BASE =
  'w-full bg-transparent px-3 py-2 text-sm text-[color:var(--ink)] placeholder:text-[color:var(--ink-subtle)] ' +
  'focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-60'

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { label, hint, error, leftIcon, rightIcon, fullWidth = true, id, className, ...rest },
  ref,
) {
  const reactId = useId()
  const inputId = id ?? reactId
  const hintId = `${inputId}-hint`
  const errorId = `${inputId}-error`

  return (
    <div className={cn('flex flex-col gap-1.5', fullWidth && 'w-full')}>
      {label && (
        <label
          htmlFor={inputId}
          className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-muted)]"
        >
          {label}
        </label>
      )}
      <div
        className={cn(
          WRAPPER,
          error
            ? 'border-[color:var(--color-danger-500)]'
            : 'border-[color:var(--border)]',
        )}
      >
        {leftIcon && (
          <span className="pointer-events-none pl-3 text-[color:var(--ink-subtle)] [&_svg]:h-4 [&_svg]:w-4">
            {leftIcon}
          </span>
        )}
        <input
          ref={ref}
          id={inputId}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? errorId : hint ? hintId : undefined}
          className={cn(
            INPUT_BASE,
            leftIcon && 'pl-2',
            rightIcon && 'pr-2',
            className,
          )}
          {...rest}
        />
        {rightIcon && (
          <span className="pr-2 text-[color:var(--ink-subtle)]">{rightIcon}</span>
        )}
      </div>
      {error ? (
        <p id={errorId} className="text-xs text-[color:var(--color-danger-600)]">
          {error}
        </p>
      ) : hint ? (
        <p id={hintId} className="text-xs text-[color:var(--ink-subtle)]">
          {hint}
        </p>
      ) : null}
    </div>
  )
})

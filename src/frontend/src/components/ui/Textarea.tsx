import { forwardRef, useId, type TextareaHTMLAttributes } from 'react'
import { cn } from './cn'

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  hint?: string
  error?: string | null
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  function Textarea({ label, hint, error, className, id, rows = 4, ...rest }, ref) {
    const reactId = useId()
    const inputId = id ?? reactId
    const hintId = `${inputId}-hint`
    const errorId = `${inputId}-error`

    return (
      <div className="flex w-full flex-col gap-1.5">
        {label && (
          <label
            htmlFor={inputId}
            className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-muted)]"
          >
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          id={inputId}
          rows={rows}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? errorId : hint ? hintId : undefined}
          className={cn(
            'w-full resize-y rounded-[var(--radius-md)] border bg-[color:var(--surface-1)] px-3 py-2.5 text-sm text-[color:var(--ink)]',
            'placeholder:text-[color:var(--ink-subtle)]',
            'focus-visible:outline-none focus-visible:shadow-[var(--ring)]',
            'disabled:cursor-not-allowed disabled:opacity-60 transition',
            error
              ? 'border-[color:var(--color-danger-500)]'
              : 'border-[color:var(--border)] focus-visible:border-[color:var(--border-strong)]',
            className,
          )}
          {...rest}
        />
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
  },
)

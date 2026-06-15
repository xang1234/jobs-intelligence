import type { ReactNode } from 'react'
import { cn } from './cn'

export interface CollapsibleProps {
  /** Left-aligned summary label. */
  summary: ReactNode
  /** Optional text shown before the Show/Hide affordance (e.g. a count). */
  badge?: ReactNode
  /** 'card' wraps the disclosure in a bordered surface; 'plain' is bare. */
  variant?: 'card' | 'plain'
  className?: string
  children: ReactNode
}

/**
 * Native `<details>` disclosure with a Show/Hide affordance. Zero-JS, accessible,
 * and keyboard-toggleable for free. Replaces the hand-rolled `group-open` markup
 * that was being copy-pasted per collapsible section.
 */
export function Collapsible({
  summary,
  badge,
  variant = 'plain',
  className,
  children,
}: CollapsibleProps) {
  return (
    <details
      className={cn(
        'group',
        variant === 'card' &&
          'rounded-[var(--radius-lg)] border border-[color:var(--border)] bg-[color:var(--surface)] px-5 py-4',
        className,
      )}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-sm font-semibold text-[color:var(--ink-muted)] hover:text-[color:var(--ink)]">
        <span>{summary}</span>
        <span className="inline-flex items-center gap-1 text-xs font-normal text-[color:var(--ink-subtle)]">
          {badge != null ? <span>{badge}</span> : null}
          <span className="group-open:hidden">Show</span>
          <span className="hidden group-open:inline">Hide</span>
        </span>
      </summary>
      <div className="mt-4 space-y-4">{children}</div>
    </details>
  )
}

import type { ReactNode } from 'react'
import { cn } from './cn'

export type BadgeIntent =
  | 'neutral'
  | 'brand'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info'

export interface BadgeProps {
  intent?: BadgeIntent
  children: ReactNode
  className?: string
}

const INTENT: Record<BadgeIntent, string> = {
  neutral: 'bg-[color:var(--surface-2)] text-[color:var(--ink-muted)]',
  brand: 'bg-[color:var(--brand-soft)] text-[color:var(--brand)]',
  success: 'bg-[color:var(--success-bg)] text-[color:var(--success-fg)]',
  warning: 'bg-[color:var(--warning-bg)] text-[color:var(--warning-fg)]',
  danger: 'bg-[color:var(--danger-bg)] text-[color:var(--danger-fg)]',
  info: 'bg-[color:var(--info-bg)] text-[color:var(--info-fg)]',
}

export function Badge({ intent = 'neutral', children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider',
        INTENT[intent],
        className,
      )}
    >
      {children}
    </span>
  )
}

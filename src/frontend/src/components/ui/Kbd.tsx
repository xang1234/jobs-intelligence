import type { ReactNode } from 'react'
import { cn } from './cn'

export interface KbdProps {
  children: ReactNode
  className?: string
}

export function Kbd({ children, className }: KbdProps) {
  return (
    <kbd
      className={cn(
        'inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded-[var(--radius-xs)] border border-[color:var(--border)] bg-[color:var(--surface-2)] px-1.5 font-mono text-[10px] font-semibold text-[color:var(--ink-muted)]',
        className,
      )}
    >
      {children}
    </kbd>
  )
}

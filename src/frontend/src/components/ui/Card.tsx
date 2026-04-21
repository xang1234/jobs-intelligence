import { forwardRef, type HTMLAttributes, type KeyboardEvent as ReactKeyboardEvent } from 'react'
import { cn } from './cn'

export type CardElevation = 0 | 1 | 2
export type CardRadius = 'md' | 'lg' | 'xl' | '2xl'
export type CardIntent = 'default' | 'warning' | 'danger' | 'info' | 'success'

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  elevation?: CardElevation
  radius?: CardRadius
  interactive?: boolean
  intent?: CardIntent
  as?: 'div' | 'article' | 'section'
}

const RADIUS: Record<CardRadius, string> = {
  md: 'rounded-[var(--radius-md)]',
  lg: 'rounded-[var(--radius-lg)]',
  xl: 'rounded-[var(--radius-xl)]',
  '2xl': 'rounded-[var(--radius-2xl)]',
}

const SHADOW: Record<CardElevation, string> = {
  0: '',
  1: 'shadow-[var(--shadow-xl)]',
  2: 'shadow-[var(--shadow-xl)]',
}

const INTENT: Record<CardIntent, string> = {
  default:
    'bg-[color:var(--surface-1-alpha)] border border-[color:var(--border)] text-[color:var(--ink)]',
  warning:
    'bg-[color:var(--warning-bg)] border border-[color:var(--color-warning-500)]/30 text-[color:var(--warning-fg)]',
  danger:
    'bg-[color:var(--danger-bg)] border border-[color:var(--color-danger-500)]/30 text-[color:var(--danger-fg)]',
  info:
    'bg-[color:var(--info-bg)] border border-[color:var(--color-info-500)]/30 text-[color:var(--info-fg)]',
  success:
    'bg-[color:var(--success-bg)] border border-[color:var(--color-success-500)]/30 text-[color:var(--success-fg)]',
}

export const Card = forwardRef<HTMLDivElement, CardProps>(function Card(
  {
    elevation = 1,
    radius = 'xl',
    interactive = false,
    intent = 'default',
    as = 'div',
    className,
    children,
    ...rest
  },
  ref,
) {
  const Tag = as as 'div'
  const hasClickHandler = typeof (rest as HTMLAttributes<HTMLElement>).onClick === 'function'
  const keyboardProps =
    interactive && hasClickHandler
      ? {
          tabIndex: 0,
          role: 'button' as const,
          onKeyDown: (e: ReactKeyboardEvent<HTMLElement>) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              e.currentTarget.click()
            }
          },
        }
      : {}
  return (
    <Tag
      ref={ref}
      className={cn(
        'backdrop-blur-sm transition',
        RADIUS[radius],
        SHADOW[elevation],
        INTENT[intent],
        interactive &&
          'cursor-pointer hover:shadow-[var(--shadow-card-hover)] motion-safe:hover:-translate-y-0.5',
        interactive && hasClickHandler && 'focus-visible:ring-2 focus-visible:ring-[color:var(--brand)] focus-visible:ring-offset-2 focus-visible:outline-none',
        className,
      )}
      {...keyboardProps}
      {...rest}
    >
      {children}
    </Tag>
  )
})

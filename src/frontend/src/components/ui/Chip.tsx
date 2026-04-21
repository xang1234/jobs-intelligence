import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { XMarkIcon } from '@heroicons/react/20/solid'
import { cn } from './cn'

export type ChipIntent =
  | 'neutral'
  | 'brand'
  | 'accent'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info'
export type ChipSize = 'sm' | 'md'

export interface ChipProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onClick'> {
  intent?: ChipIntent
  size?: ChipSize
  children: ReactNode
  leftIcon?: ReactNode
  onRemove?: () => void
  onClick?: ButtonHTMLAttributes<HTMLButtonElement>['onClick']
  active?: boolean
  as?: 'span' | 'button'
}

const SIZE: Record<ChipSize, string> = {
  sm: 'h-6 px-2.5 text-[11px] gap-1',
  md: 'h-7 px-3 text-xs gap-1.5',
}

const INTENT: Record<ChipIntent, string> = {
  neutral:
    'bg-[color:var(--surface-2)] text-[color:var(--ink-muted)] border border-[color:var(--border)]',
  brand:
    'bg-[color:var(--brand-soft)] text-[color:var(--brand)] border border-transparent',
  accent:
    'bg-[color:var(--color-accent-100)] text-[color:var(--color-accent-700)] border border-transparent',
  success:
    'bg-[color:var(--success-bg)] text-[color:var(--success-fg)] border border-transparent',
  warning:
    'bg-[color:var(--warning-bg)] text-[color:var(--warning-fg)] border border-transparent',
  danger:
    'bg-[color:var(--danger-bg)] text-[color:var(--danger-fg)] border border-transparent',
  info: 'bg-[color:var(--info-bg)] text-[color:var(--info-fg)] border border-transparent',
}

export const Chip = forwardRef<HTMLElement, ChipProps>(function Chip(
  {
    intent = 'neutral',
    size = 'md',
    leftIcon,
    onRemove,
    onClick,
    active,
    children,
    className,
    as,
    type,
    'aria-pressed': ariaPressed,
    ...rest
  },
  ref,
) {
  const hasInteractiveBody = Boolean(onClick || active !== undefined)
  const Tag: 'span' | 'button' = as ?? (hasInteractiveBody ? 'button' : 'span')
  const interactive = hasInteractiveBody || Boolean(onRemove)

  const classes = cn(
    'inline-flex items-center rounded-full font-medium transition whitespace-nowrap',
    SIZE[size],
    INTENT[intent],
    interactive && 'cursor-pointer hover:brightness-[0.96]',
    active && 'ring-2 ring-[color:var(--brand)]',
    className,
  )

  if (Tag === 'button' && onRemove) {
    return (
      <span className="inline-flex items-center">
        <button
          ref={ref as React.Ref<HTMLButtonElement>}
          type={type ?? 'button'}
          onClick={onClick}
          aria-pressed={ariaPressed}
          className={classes}
          {...rest}
        >
          {leftIcon}
          <span>{children}</span>
        </button>
        <button
          type="button"
          aria-label="Remove"
          onClick={(e) => {
            e.stopPropagation()
            onRemove()
          }}
          className="-mr-1 inline-flex h-4 w-4 items-center justify-center rounded-full transition hover:bg-[color:var(--border)]"
        >
          <XMarkIcon className="h-3 w-3" aria-hidden="true" />
        </button>
      </span>
    )
  }

  if (Tag === 'button') {
    return (
      <button
        ref={ref as React.Ref<HTMLButtonElement>}
        type={type ?? 'button'}
        onClick={onClick}
        aria-pressed={ariaPressed}
        className={classes}
        {...rest}
      >
        {leftIcon}
        <span>{children}</span>
      </button>
    )
  }

  return (
    <span ref={ref as React.Ref<HTMLSpanElement>} className={classes} {...(rest as object)}>
      {leftIcon}
      <span>{children}</span>
      {onRemove && (
        <button
          type="button"
          aria-label="Remove"
          onClick={(e) => {
            e.stopPropagation()
            onRemove()
          }}
          className="-mr-1 inline-flex h-4 w-4 items-center justify-center rounded-full transition hover:bg-[color:var(--border)]"
        >
          <XMarkIcon className="h-3 w-3" aria-hidden="true" />
        </button>
      )}
    </span>
  )
})

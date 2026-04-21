import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cn } from './cn'
import { Spinner } from './Spinner'

export type IconButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export type IconButtonSize = 'sm' | 'md' | 'lg'

export interface IconButtonProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  variant?: IconButtonVariant
  size?: IconButtonSize
  loading?: boolean
  icon: ReactNode
  'aria-label': string
}

const BASE =
  'inline-flex items-center justify-center rounded-full transition select-none ' +
  'disabled:cursor-not-allowed disabled:opacity-50 ' +
  'motion-safe:active:translate-y-px ' +
  'focus-visible:outline-none'

const SIZE: Record<IconButtonSize, string> = {
  sm: 'h-8 w-8 [&_svg]:h-4 [&_svg]:w-4',
  md: 'h-10 w-10 [&_svg]:h-5 [&_svg]:w-5',
  lg: 'h-12 w-12 [&_svg]:h-6 [&_svg]:w-6',
}

const VARIANT: Record<IconButtonVariant, string> = {
  primary:
    'bg-[color:var(--brand)] text-white shadow-[var(--shadow-md)] hover:bg-[color:var(--brand-strong)]',
  secondary:
    'bg-[color:var(--surface-1)] text-[color:var(--ink)] border border-[color:var(--border)] hover:border-[color:var(--border-strong)] hover:text-[color:var(--brand)]',
  ghost:
    'bg-transparent text-[color:var(--ink-muted)] hover:bg-[color:var(--surface-2)] hover:text-[color:var(--ink)]',
  danger:
    'bg-[color:var(--color-danger-600)] text-white hover:bg-[color:var(--color-danger-700)]',
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    { variant = 'ghost', size = 'md', icon, loading, className, disabled, type = 'button', ...rest },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled || loading}
        aria-busy={loading || undefined}
        className={cn(BASE, SIZE[size], VARIANT[variant], className)}
        {...rest}
      >
        {loading ? <Spinner size={size === 'lg' ? 'md' : 'sm'} /> : icon}
      </button>
    )
  },
)

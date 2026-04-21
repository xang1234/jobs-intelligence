import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cn } from './cn'
import { Spinner } from './Spinner'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'link'
export type ButtonSize = 'sm' | 'md' | 'lg'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  iconLeft?: ReactNode
  iconRight?: ReactNode
  fullWidth?: boolean
}

const BASE =
  'inline-flex items-center justify-center gap-2 rounded-full font-semibold select-none whitespace-nowrap transition ' +
  'disabled:cursor-not-allowed disabled:opacity-50 ' +
  'motion-safe:active:translate-y-px ' +
  'focus-visible:outline-none'

const SIZE: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-xs',
  md: 'h-10 px-4 text-sm',
  lg: 'h-12 px-6 text-base',
}

const VARIANT: Record<ButtonVariant, string> = {
  primary:
    'bg-[color:var(--brand)] text-white shadow-[var(--shadow-md)] ' +
    'hover:bg-[color:var(--brand-strong)] hover:shadow-[var(--shadow-lg)]',
  secondary:
    'bg-[color:var(--surface-1)] text-[color:var(--ink)] border border-[color:var(--border)] shadow-[var(--shadow-xs)] ' +
    'hover:border-[color:var(--border-strong)] hover:text-[color:var(--brand)]',
  ghost:
    'bg-transparent text-[color:var(--ink)] ' +
    'hover:bg-[color:var(--surface-2)]',
  danger:
    'bg-[color:var(--color-danger-600)] text-white shadow-[var(--shadow-md)] ' +
    'hover:bg-[color:var(--color-danger-700)]',
  link:
    'bg-transparent text-[color:var(--brand)] underline-offset-4 ' +
    'hover:text-[color:var(--brand-strong)] hover:underline',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = 'primary',
    size = 'md',
    loading = false,
    iconLeft,
    iconRight,
    fullWidth,
    className,
    disabled,
    children,
    type = 'button',
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        BASE,
        SIZE[size],
        VARIANT[variant],
        fullWidth && 'w-full',
        className,
      )}
      {...rest}
    >
      {loading ? <Spinner size={size === 'lg' ? 'md' : 'sm'} /> : iconLeft}
      <span className={cn(loading && 'opacity-80')}>{children}</span>
      {!loading && iconRight}
    </button>
  )
})

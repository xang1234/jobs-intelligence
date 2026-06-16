import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cn } from './cn'
import { Spinner } from './Spinner'
import { buttonClasses, type ButtonSize, type ButtonVariant } from './buttonClasses'

export type { ButtonVariant, ButtonSize }

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  iconLeft?: ReactNode
  iconRight?: ReactNode
  fullWidth?: boolean
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
  // A truly-disabled button looks muted (grey), not a dimmed copy of its active colour (issue #15).
  // Loading keeps the brand colour + spinner, so it still reads as "working".
  const muted = disabled && !loading
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={buttonClasses(
        variant,
        size,
        cn(
          fullWidth && 'w-full',
          muted &&
            'bg-[color:var(--surface-3)]! text-[color:var(--ink-subtle)]! border-transparent! shadow-none!',
          className,
        ),
      )}
      {...rest}
    >
      {loading ? <Spinner size={size === 'lg' ? 'md' : 'sm'} /> : iconLeft}
      <span className={cn(loading && 'opacity-80')}>{children}</span>
      {!loading && iconRight}
    </button>
  )
})

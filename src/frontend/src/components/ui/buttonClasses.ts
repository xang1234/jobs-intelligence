import { cn } from './cn'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'link'
export type ButtonSize = 'sm' | 'md' | 'lg'

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
  ghost: 'bg-transparent text-[color:var(--ink)] hover:bg-[color:var(--surface-2)]',
  danger:
    'bg-[color:var(--color-danger-600)] text-white shadow-[var(--shadow-md)] ' +
    'hover:bg-[color:var(--color-danger-700)]',
  link:
    'bg-transparent text-[color:var(--brand)] underline-offset-4 ' +
    'hover:text-[color:var(--brand-strong)] hover:underline',
}

/**
 * Button styling as a plain class string — the single source of truth for button
 * appearance. Lets a non-`<button>` element (e.g. an anchor that must look like a
 * primary button) match exactly without copying the utility classes.
 */
export function buttonClasses(
  variant: ButtonVariant = 'primary',
  size: ButtonSize = 'md',
  className?: string,
): string {
  return cn(BASE, SIZE[size], VARIANT[variant], className)
}

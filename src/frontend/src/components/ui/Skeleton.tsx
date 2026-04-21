import type { CSSProperties } from 'react'
import { cn } from './cn'

export interface SkeletonProps {
  className?: string
  width?: number | string
  height?: number | string
  rounded?: 'sm' | 'md' | 'lg' | 'full'
  'aria-label'?: string
}

const RADIUS: Record<NonNullable<SkeletonProps['rounded']>, string> = {
  sm: 'rounded-[var(--radius-sm)]',
  md: 'rounded-[var(--radius-md)]',
  lg: 'rounded-[var(--radius-lg)]',
  full: 'rounded-full',
}

export function Skeleton({
  className,
  width,
  height,
  rounded = 'md',
  'aria-label': ariaLabel = 'Loading',
}: SkeletonProps) {
  const style: CSSProperties = {
    width,
    height,
    backgroundImage:
      'linear-gradient(90deg, var(--surface-2) 0%, var(--surface-3) 50%, var(--surface-2) 100%)',
    backgroundSize: '200% 100%',
    animation: 'var(--animate-shimmer)',
  }
  return (
    <span
      role="status"
      aria-label={ariaLabel}
      style={style}
      className={cn('block', RADIUS[rounded], className)}
    />
  )
}

export function SkeletonText({
  lines = 3,
  className,
}: {
  lines?: number
  className?: string
}) {
  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          height={12}
          width={i === lines - 1 ? '65%' : '100%'}
          rounded="sm"
        />
      ))}
    </div>
  )
}

export function SkeletonCircle({
  size = 40,
  className,
}: {
  size?: number
  className?: string
}) {
  return (
    <Skeleton
      width={size}
      height={size}
      rounded="full"
      className={className}
    />
  )
}

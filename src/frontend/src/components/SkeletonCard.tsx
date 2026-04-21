import { Skeleton } from '@/components/ui'

export default function SkeletonCard() {
  return (
    <div className="rounded-[var(--radius-xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-6 shadow-[var(--shadow-xl)]">
      <Skeleton height={20} width="65%" />
      <div className="mt-3">
        <Skeleton height={14} width="40%" />
      </div>
      <div className="mt-4 flex gap-3">
        <Skeleton height={14} width={90} />
        <Skeleton height={14} width={120} />
        <Skeleton height={14} width={80} />
      </div>
      <div className="mt-5 flex gap-2">
        <Skeleton height={24} width={72} rounded="full" />
        <Skeleton height={24} width={96} rounded="full" />
        <Skeleton height={24} width={64} rounded="full" />
      </div>
    </div>
  )
}

import type { SkillCloudItem } from '@/types/api'

const CLUSTER_COLORS = [
  'text-[color:var(--color-info-600)]',
  'text-[color:var(--color-success-600)]',
  'text-[color:var(--color-accent-600)]',
  'text-[color:var(--color-brand-600)]',
  'text-[color:var(--color-danger-600)]',
  'text-[color:var(--color-brand-500)]',
  'text-[color:var(--color-warning-600)]',
  'text-[color:var(--color-info-700)]',
]

interface SkillCloudProps {
  items: SkillCloudItem[]
  onSkillClick: (skill: string) => void
  title?: string
}

export default function SkillCloud({
  items,
  onSkillClick,
  title = 'Skills',
}: SkillCloudProps) {
  if (items.length === 0) return null

  const maxCount = Math.max(...items.map((i) => i.job_count))
  const minCount = Math.min(...items.map((i) => i.job_count))
  const range = maxCount - minCount || 1

  function fontSize(count: number): string {
    const ratio = (count - minCount) / range
    const size = 0.8 + ratio * 0.75
    return `${size}rem`
  }

  function colorClass(clusterId: number | null): string {
    if (clusterId == null) return 'text-[color:var(--ink-muted)]'
    return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length]
  }

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-[color:var(--ink)]">{title}</h3>
      <div className="flex flex-wrap gap-x-2.5 gap-y-1.5">
        {items.map((item) => (
          <button
            key={item.skill}
            type="button"
            onClick={() => onSkillClick(item.skill)}
            style={{ fontSize: fontSize(item.job_count) }}
            className={`cursor-pointer font-medium leading-relaxed transition hover:underline focus-visible:outline-none ${colorClass(item.cluster_id)}`}
            title={`${item.job_count.toLocaleString()} jobs`}
          >
            {item.skill}
          </button>
        ))}
      </div>
    </div>
  )
}

import type { SkillCloudItem } from '@/types/api'
import { Chip } from '@/components/ui'

interface SkillCloudProps {
  items: SkillCloudItem[]
  onSkillClick: (skill: string) => void
  title?: string
}

// ponytail: uniform, scannable chips ranked by count — no variable-size word cloud (issue #7).
export default function SkillCloud({ items, onSkillClick, title = 'Skills' }: SkillCloudProps) {
  if (items.length === 0) return null

  const ranked = [...items].sort((a, b) => b.job_count - a.job_count)

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-[color:var(--ink)]">{title}</h3>
      <div className="flex flex-wrap gap-2">
        {ranked.map((item) => (
          <Chip
            key={item.skill}
            intent="neutral"
            size="md"
            onClick={() => onSkillClick(item.skill)}
            title={`${item.job_count.toLocaleString()} jobs`}
          >
            {item.skill} <span className="opacity-60">({item.job_count.toLocaleString()})</span>
          </Chip>
        ))}
      </div>
    </div>
  )
}

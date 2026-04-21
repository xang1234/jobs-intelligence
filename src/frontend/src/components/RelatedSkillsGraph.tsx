import type { RelatedSkillsResponse } from '@/types/api'
import { Button, Card } from '@/components/ui'

interface RelatedSkillsGraphProps {
  data?: RelatedSkillsResponse
  onSelectSkill: (skill: string) => void
}

export default function RelatedSkillsGraph({ data, onSelectSkill }: RelatedSkillsGraphProps) {
  if (!data) {
    return (
      <Card radius="xl" elevation={0} className="border-dashed p-6 text-sm text-[color:var(--ink-subtle)]">
        Select a skill from the cloud or run a skill query to inspect the embedding neighborhood.
      </Card>
    )
  }

  const orbit = data.related.slice(0, 8)

  return (
    <Card radius="xl" className="p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
            Related skill graph
          </p>
          <h3 className="mt-1 text-lg font-semibold text-[color:var(--ink)]">{data.skill}</h3>
        </div>
        <Button variant="secondary" size="sm" onClick={() => onSelectSkill(data.skill)}>
          Search this skill
        </Button>
      </div>

      <div className="mt-6 grid place-items-center">
        <div className="relative h-72 w-72 rounded-full bg-[radial-gradient(circle_at_center,var(--brand-soft),transparent_62%)]">
          <button
            type="button"
            onClick={() => onSelectSkill(data.skill)}
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[color:var(--brand)] px-5 py-2 text-sm font-semibold text-white shadow-[var(--shadow-md)] transition motion-safe:hover:scale-[1.03] focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--brand)]"
            aria-label={`Search ${data.skill}`}
          >
            {data.skill}
          </button>

          {orbit.map((skill, index) => {
            const angle = (Math.PI * 2 * index) / orbit.length
            const radius = 108
            const x = 50 + (Math.cos(angle) * radius) / 2.72
            const y = 50 + (Math.sin(angle) * radius) / 2.72

            return (
              <button
                key={skill.skill}
                type="button"
                onClick={() => onSelectSkill(skill.skill)}
                style={{
                  left: `${x}%`,
                  top: `${y}%`,
                  transition: 'left var(--duration-slow) var(--ease-standard), top var(--duration-slow) var(--ease-standard), transform var(--duration-base) var(--ease-standard)',
                }}
                className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full border px-3 py-1.5 text-xs font-semibold shadow-[var(--shadow-xs)] motion-safe:hover:scale-105 focus-visible:ring-2 focus-visible:ring-[color:var(--brand)] focus-visible:ring-offset-2 ${
                  skill.same_cluster
                    ? 'border-transparent bg-[color:var(--success-bg)] text-[color:var(--success-fg)]'
                    : 'border-[color:var(--border)] bg-[color:var(--surface-1)] text-[color:var(--ink-muted)]'
                }`}
                title={`${Math.round(skill.similarity * 100)}% similarity`}
              >
                {skill.skill}
              </button>
            )
          })}
        </div>
      </div>
    </Card>
  )
}

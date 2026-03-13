import type { RelatedSkillsResponse } from '@/types/api'

interface RelatedSkillsGraphProps {
  data?: RelatedSkillsResponse
  onSelectSkill: (skill: string) => void
}

export default function RelatedSkillsGraph({ data, onSelectSkill }: RelatedSkillsGraphProps) {
  if (!data) {
    return (
      <div className="rounded-[24px] border border-dashed border-[color:var(--border)] bg-white/70 p-6 text-sm text-slate-500">
        Select a skill from the cloud or run a skill query to inspect the embedding neighborhood.
      </div>
    )
  }

  const orbit = data.related.slice(0, 8)

  return (
    <div className="rounded-[24px] border border-[color:var(--border)] bg-white/90 p-5 shadow-[0_12px_36px_rgba(15,23,42,0.08)]">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Related skill graph</p>
          <h3 className="mt-1 text-lg font-semibold text-[color:var(--ink)]">{data.skill}</h3>
        </div>
        <button
          type="button"
          onClick={() => onSelectSkill(data.skill)}
          className="rounded-full border border-[color:var(--border)] px-3 py-1.5 text-xs font-semibold text-slate-600"
        >
          Search this skill
        </button>
      </div>

      <div className="mt-6 grid place-items-center">
        <div className="relative h-72 w-72 rounded-full bg-[radial-gradient(circle_at_center,rgba(15,118,110,0.14),rgba(255,255,255,0.9)_58%)]">
          <button
            type="button"
            onClick={() => onSelectSkill(data.skill)}
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[color:var(--brand)] px-5 py-2 text-sm font-semibold text-white shadow-lg"
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
                style={{ left: `${x}%`, top: `${y}%` }}
                className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full px-3 py-1.5 text-xs font-semibold shadow-sm transition hover:scale-105 ${
                  skill.same_cluster
                    ? 'bg-emerald-100 text-emerald-900'
                    : 'bg-white text-slate-700'
                }`}
                title={`${Math.round(skill.similarity * 100)}% similarity`}
              >
                {skill.skill}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

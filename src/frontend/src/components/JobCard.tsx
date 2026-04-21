import type { JobResult } from '@/types/api'
import {
  BriefcaseIcon,
  ChartBarSquareIcon,
  CurrencyDollarIcon,
  MapPinIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import { Badge, Button, Card, Chip } from '@/components/ui'

interface JobCardProps {
  job: JobResult
  onFindSimilar?: (uuid: string) => void
}

function formatSalary(min: number | null, max: number | null): string | null {
  if (min != null && max != null) {
    return `$${min.toLocaleString()} - $${max.toLocaleString()}`
  }
  if (min != null) return `From $${min.toLocaleString()}`
  if (max != null) return `Up to $${max.toLocaleString()}`
  return null
}

function fitIntent(score: number): 'success' | 'info' | 'neutral' {
  if (score >= 0.8) return 'success'
  if (score >= 0.6) return 'info'
  return 'neutral'
}

function pct(value: number | null | undefined): string | null {
  if (value == null) return null
  return `${Math.round(value * 100)}%`
}

export default function JobCard({ job, onFindSimilar }: JobCardProps) {
  const salary = formatSalary(job.salary_min, job.salary_max)
  const skills = job.skills
    ? job.skills.split(',').map((item) => item.trim()).filter(Boolean)
    : []
  const explanation = job.explanations

  return (
    <Card as="article" radius="xl" elevation={1} className="p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0 space-y-2">
          {job.job_url ? (
            <a
              href={job.job_url}
              target="_blank"
              rel="noopener noreferrer"
              className="block text-xl font-semibold tracking-tight text-[color:var(--ink)] hover:text-[color:var(--brand)] transition-colors"
            >
              {job.title}
            </a>
          ) : (
            <h3 className="text-xl font-semibold tracking-tight text-[color:var(--ink)]">
              {job.title}
            </h3>
          )}
          <div className="flex flex-wrap items-center gap-2 text-sm text-[color:var(--ink-muted)]">
            {job.company_name && <span className="font-medium">{job.company_name}</span>}
            {job.seniority && <Chip intent="neutral" size="sm">{job.seniority}</Chip>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge intent={fitIntent(job.similarity_score)}>
            {(job.similarity_score * 100).toFixed(0)}% fit
          </Badge>
          {job.semantic_score != null && (
            <Chip intent="neutral" size="sm">semantic {pct(job.semantic_score)}</Chip>
          )}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-sm text-[color:var(--ink-muted)]">
        {salary && (
          <span className="inline-flex items-center gap-1.5">
            <CurrencyDollarIcon className="h-4 w-4 text-[color:var(--brand)]" aria-hidden="true" />
            {salary}
          </span>
        )}
        {job.location && (
          <span className="inline-flex items-center gap-1.5">
            <MapPinIcon className="h-4 w-4 text-[color:var(--brand)]" aria-hidden="true" />
            {job.location}
          </span>
        )}
        {job.employment_type && (
          <span className="inline-flex items-center gap-1.5">
            <BriefcaseIcon className="h-4 w-4 text-[color:var(--brand)]" aria-hidden="true" />
            {job.employment_type}
          </span>
        )}
      </div>

      <p className="mt-4 text-sm leading-6 text-[color:var(--ink-muted)]">{job.description}</p>

      {skills.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {skills.slice(0, 8).map((skill) => (
            <Chip key={skill} intent="neutral" size="sm">{skill}</Chip>
          ))}
        </div>
      )}

      {explanation && (
        <div className="mt-5 grid gap-4 rounded-[var(--radius-lg)] bg-[color:var(--surface-3)] p-4 md:grid-cols-2 lg:grid-cols-[1.2fr_0.8fr]">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-[color:var(--ink)]">
              <SparklesIcon className="h-4 w-4 text-[color:var(--brand)]" aria-hidden="true" />
              Why this matched
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {explanation.query_terms.map((term) => (
                <Chip key={term} intent="neutral" size="sm">{term}</Chip>
              ))}
            </div>
            {explanation.matched_skills.length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
                  Matched skills
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {explanation.matched_skills.map((skill) => (
                    <Chip key={skill} intent="success" size="sm">{skill}</Chip>
                  ))}
                </div>
              </div>
            )}
            {explanation.missing_skills.length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">
                  Missing skills
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {explanation.missing_skills.slice(0, 6).map((skill) => (
                    <Chip key={skill} intent="warning" size="sm">{skill}</Chip>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-[color:var(--ink)]">
              <ChartBarSquareIcon className="h-4 w-4 text-[color:var(--brand)]" aria-hidden="true" />
              Score breakdown
            </div>
            <dl className="mt-3 space-y-2 text-sm text-[color:var(--ink-muted)]">
              <ScoreRow label="Overall" value={pct(explanation.overall_fit) ?? pct(job.similarity_score)} />
              {pct(explanation.semantic_score) && (
                <ScoreRow label="Semantic" value={pct(explanation.semantic_score)} />
              )}
              {pct(explanation.bm25_score) && (
                <ScoreRow label="Keyword" value={pct(explanation.bm25_score)} />
              )}
              {pct(explanation.skill_overlap_score) && (
                <ScoreRow label="Skill overlap" value={pct(explanation.skill_overlap_score)} />
              )}
              {pct(explanation.seniority_fit) && (
                <ScoreRow label="Seniority" value={pct(explanation.seniority_fit)} />
              )}
              {pct(explanation.salary_fit) && (
                <ScoreRow label="Salary fit" value={pct(explanation.salary_fit)} />
              )}
              {pct(explanation.freshness_score) && (
                <ScoreRow label="Freshness" value={pct(explanation.freshness_score)} />
              )}
            </dl>
          </div>
        </div>
      )}

      {onFindSimilar && (
        <div className="mt-5">
          <Button variant="secondary" size="sm" onClick={() => onFindSimilar(job.uuid)}>
            Explore similar roles
          </Button>
        </div>
      )}
    </Card>
  )
}

function ScoreRow({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="flex items-center justify-between rounded-[var(--radius-md)] bg-[color:var(--surface-1)] px-3 py-2">
      <dt>{label}</dt>
      <dd className="font-semibold text-[color:var(--ink)]">{value}</dd>
    </div>
  )
}

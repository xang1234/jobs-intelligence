import type { JobResult } from '@/types/api'
import {
  BriefcaseIcon,
  ChartBarSquareIcon,
  CurrencyDollarIcon,
  MapPinIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'

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

function scorePill(score: number): string {
  if (score >= 0.8) return 'bg-emerald-100 text-emerald-900'
  if (score >= 0.6) return 'bg-sky-100 text-sky-900'
  return 'bg-stone-100 text-stone-700'
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
    <article className="rounded-[28px] border border-[color:var(--border)] bg-white/95 p-6 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0 space-y-2">
          {job.job_url ? (
            <a
              href={job.job_url}
              target="_blank"
              rel="noopener noreferrer"
              className="block text-xl font-semibold tracking-tight text-[color:var(--ink)] hover:text-[color:var(--brand)]"
            >
              {job.title}
            </a>
          ) : (
            <h3 className="text-xl font-semibold tracking-tight text-[color:var(--ink)]">
              {job.title}
            </h3>
          )}
          <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600">
            {job.company_name && <span>{job.company_name}</span>}
            {job.seniority && (
              <span className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-slate-700">
                {job.seniority}
              </span>
            )}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className={`rounded-full px-3 py-1 text-sm font-semibold ${scorePill(job.similarity_score)}`}>
            {(job.similarity_score * 100).toFixed(0)}% fit
          </span>
          {job.semantic_score != null && (
            <span className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-slate-700">
              semantic {pct(job.semantic_score)}
            </span>
          )}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-sm text-slate-600">
        {salary && (
          <span className="inline-flex items-center gap-1.5">
            <CurrencyDollarIcon className="h-4 w-4 text-[color:var(--brand)]" />
            {salary}
          </span>
        )}
        {job.location && (
          <span className="inline-flex items-center gap-1.5">
            <MapPinIcon className="h-4 w-4 text-[color:var(--brand)]" />
            {job.location}
          </span>
        )}
        {job.employment_type && (
          <span className="inline-flex items-center gap-1.5">
            <BriefcaseIcon className="h-4 w-4 text-[color:var(--brand)]" />
            {job.employment_type}
          </span>
        )}
      </div>

      <p className="mt-4 text-sm leading-6 text-slate-600">{job.description}</p>

      {skills.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {skills.slice(0, 8).map((skill) => (
            <span
              key={skill}
              className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-slate-700"
            >
              {skill}
            </span>
          ))}
        </div>
      )}

      {explanation && (
        <div className="mt-5 grid gap-4 rounded-[22px] bg-[color:var(--surface-strong)] p-4 lg:grid-cols-[1.2fr_0.8fr]">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-[color:var(--ink)]">
              <SparklesIcon className="h-4 w-4 text-[color:var(--brand)]" />
              Why this matched
            </div>
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              {explanation.query_terms.map((term) => (
                <span
                  key={term}
                  className="rounded-full border border-[color:var(--border)] bg-white px-3 py-1 text-slate-600"
                >
                  {term}
                </span>
              ))}
            </div>
            {explanation.matched_skills.length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Matched skills</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {explanation.matched_skills.map((skill) => (
                    <span key={skill} className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-900">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {explanation.missing_skills.length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Missing skills</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {explanation.missing_skills.slice(0, 6).map((skill) => (
                    <span key={skill} className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-[color:var(--ink)]">
              <ChartBarSquareIcon className="h-4 w-4 text-[color:var(--brand)]" />
              Score breakdown
            </div>
            <dl className="mt-3 space-y-2 text-sm text-slate-600">
              <div className="flex items-center justify-between rounded-2xl bg-white px-3 py-2">
                <dt>Overall</dt>
                <dd className="font-semibold text-[color:var(--ink)]">{pct(explanation.overall_fit) ?? pct(job.similarity_score)}</dd>
              </div>
              {pct(explanation.semantic_score) && (
                <div className="flex items-center justify-between rounded-2xl bg-white px-3 py-2">
                  <dt>Semantic</dt>
                  <dd className="font-semibold text-[color:var(--ink)]">{pct(explanation.semantic_score)}</dd>
                </div>
              )}
              {pct(explanation.bm25_score) && (
                <div className="flex items-center justify-between rounded-2xl bg-white px-3 py-2">
                  <dt>Keyword</dt>
                  <dd className="font-semibold text-[color:var(--ink)]">{pct(explanation.bm25_score)}</dd>
                </div>
              )}
              {pct(explanation.skill_overlap_score) && (
                <div className="flex items-center justify-between rounded-2xl bg-white px-3 py-2">
                  <dt>Skill overlap</dt>
                  <dd className="font-semibold text-[color:var(--ink)]">{pct(explanation.skill_overlap_score)}</dd>
                </div>
              )}
              {pct(explanation.seniority_fit) && (
                <div className="flex items-center justify-between rounded-2xl bg-white px-3 py-2">
                  <dt>Seniority</dt>
                  <dd className="font-semibold text-[color:var(--ink)]">{pct(explanation.seniority_fit)}</dd>
                </div>
              )}
              {pct(explanation.salary_fit) && (
                <div className="flex items-center justify-between rounded-2xl bg-white px-3 py-2">
                  <dt>Salary fit</dt>
                  <dd className="font-semibold text-[color:var(--ink)]">{pct(explanation.salary_fit)}</dd>
                </div>
              )}
              {pct(explanation.freshness_score) && (
                <div className="flex items-center justify-between rounded-2xl bg-white px-3 py-2">
                  <dt>Freshness</dt>
                  <dd className="font-semibold text-[color:var(--ink)]">{pct(explanation.freshness_score)}</dd>
                </div>
              )}
            </dl>
          </div>
        </div>
      )}

      {onFindSimilar && (
        <div className="mt-5">
          <button
            type="button"
            onClick={() => onFindSimilar(job.uuid)}
            className="rounded-full border border-[color:var(--border-strong)] px-4 py-2 text-sm font-semibold text-[color:var(--ink)] transition hover:border-[color:var(--brand)] hover:text-[color:var(--brand)]"
          >
            Explore similar roles
          </button>
        </div>
      )}
    </article>
  )
}

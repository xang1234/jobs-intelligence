import { useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import JobCard from '@/components/JobCard'
import { matchProfile } from '@/services/api'

export default function MatchLabPage() {
  const [profileText, setProfileText] = useState(
    'Senior data professional with Python, SQL, machine learning, experimentation, stakeholder management, and dashboarding experience. Looking for Singapore-based roles in AI, analytics, or applied ML.',
  )
  const [targetTitles, setTargetTitles] = useState('Data Scientist, Machine Learning Engineer')
  const [salaryExpectation, setSalaryExpectation] = useState('180000')
  const [employmentType, setEmploymentType] = useState('')
  const [region, setRegion] = useState('')

  const titles = useMemo(
    () => targetTitles.split(',').map((item) => item.trim()).filter(Boolean),
    [targetTitles],
  )

  const mutation = useMutation({
    mutationFn: () => matchProfile({
      profile_text: profileText,
      target_titles: titles,
      salary_expectation_annual: salaryExpectation ? Number(salaryExpectation) : null,
      employment_type: employmentType || null,
      region: region || null,
      limit: 12,
    }),
  })

  return (
    <div className="space-y-8">
      <section className="rounded-[32px] border border-[color:var(--border)] bg-white/90 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-500">Match lab</p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight text-[color:var(--ink)]">
          Paste a profile and inspect the fit model behind ranked roles.
        </h1>
        <p className="mt-3 max-w-3xl text-base leading-7 text-slate-600">
          The ranking mixes profile semantics, explicit skill overlap, seniority alignment,
          and salary fit. It is deterministic and designed to make the NLP layer visible.
        </p>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <article className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
          <label className="block text-sm font-semibold text-[color:var(--ink)]">
            Candidate profile or resume text
            <textarea
              value={profileText}
              onChange={(event) => setProfileText(event.target.value)}
              rows={12}
              className="mt-3 block w-full rounded-[24px] border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-4 text-sm leading-6 text-slate-700"
            />
          </label>

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <label className="text-sm text-slate-600">
              Target titles
              <input
                value={targetTitles}
                onChange={(event) => setTargetTitles(event.target.value)}
                placeholder="Data Scientist, ML Engineer"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-slate-600">
              Salary expectation (annual)
              <input
                value={salaryExpectation}
                onChange={(event) => setSalaryExpectation(event.target.value)}
                type="number"
                min={0}
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-slate-600">
              Employment type
              <input
                value={employmentType}
                onChange={(event) => setEmploymentType(event.target.value)}
                placeholder="Full Time"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
            <label className="text-sm text-slate-600">
              Region
              <input
                value={region}
                onChange={(event) => setRegion(event.target.value)}
                placeholder="Central"
                className="mt-1 block w-full rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] px-4 py-3"
              />
            </label>
          </div>

          <button
            type="button"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || profileText.trim().length < 20}
            className="mt-6 rounded-full bg-[color:var(--brand)] px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-[color:var(--brand-strong)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {mutation.isPending ? 'Scoring profile...' : 'Run profile match'}
          </button>
        </article>

        <article className="space-y-5">
          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-6">
            <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
              <span>
                Candidates scanned:{' '}
                <span className="font-semibold text-[color:var(--ink)]">
                  {mutation.data?.total_candidates.toLocaleString() ?? 0}
                </span>
              </span>
              <span>{mutation.data ? `${mutation.data.search_time_ms.toFixed(0)}ms` : null}</span>
              {mutation.data?.degraded && (
                <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">
                  degraded retrieval
                </span>
              )}
            </div>

            <div className="mt-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Extracted skills</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {mutation.data?.extracted_skills.map((skill) => (
                  <span key={skill} className="rounded-full bg-[color:var(--surface)] px-3 py-1 text-xs font-medium text-slate-700">
                    {skill}
                  </span>
                )) ?? <span className="text-sm text-slate-500">Run a profile match to inspect extracted skills.</span>}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            {mutation.data?.results.map((job) => (
              <JobCard key={job.uuid} job={job} />
            )) ?? (
              <div className="rounded-[28px] border border-dashed border-[color:var(--border)] bg-white/70 p-10 text-center text-sm text-slate-500">
                Match results will appear here with score decomposition and missing-skill signals.
              </div>
            )}
          </div>
        </article>
      </section>
    </div>
  )
}

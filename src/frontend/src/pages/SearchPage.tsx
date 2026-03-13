import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import CompanySearch from '@/components/CompanySearch'
import DegradedBanner from '@/components/DegradedBanner'
import FilterPanel from '@/components/FilterPanel'
import JobList from '@/components/JobList'
import RelatedSkillsGraph from '@/components/RelatedSkillsGraph'
import SearchBar from '@/components/SearchBar'
import SearchStats from '@/components/SearchStats'
import SimilarJobsModal from '@/components/SimilarJobsModal'
import SkillCloud from '@/components/SkillCloud'
import {
  findSimilarCompanies,
  findSimilarJobs,
  getHealth,
  getRelatedSkills,
  getSkillCloud,
  searchJobs,
} from '@/services/api'
import type { CompanySimilarity, Filters, JobResult } from '@/types/api'

const emptyFilters: Filters = {
  salary_min: null,
  salary_max: null,
  employment_type: null,
  company: null,
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)
  const [filters, setFilters] = useState<Filters>(emptyFilters)
  const [similarJobs, setSimilarJobs] = useState<JobResult[] | undefined>()
  const [similarModalOpen, setSimilarModalOpen] = useState(false)
  const [companyResults, setCompanyResults] = useState<CompanySimilarity[] | undefined>()

  const searchResult = useQuery({
    queryKey: ['search', query, filters],
    queryFn: () => searchJobs({
      query,
      limit: 20,
      salary_min: filters.salary_min,
      salary_max: filters.salary_max,
      employment_type: filters.employment_type,
      company: filters.company,
      expand_query: true,
    }),
    enabled: query.length > 0,
  })

  const skillCloud = useQuery({
    queryKey: ['skillCloud'],
    queryFn: () => getSkillCloud(10, 80),
    staleTime: 10 * 60 * 1000,
  })

  const relatedSkills = useQuery({
    queryKey: ['relatedSkills', selectedSkill],
    queryFn: () => getRelatedSkills(selectedSkill!, 10),
    enabled: selectedSkill != null,
  })

  const health = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 60_000,
  })

  const similarMutation = useMutation({
    mutationFn: (uuid: string) => findSimilarJobs({
      job_uuid: uuid,
      limit: 8,
      exclude_same_company: true,
    }),
    onSuccess: (data) => {
      setSimilarJobs(data.results)
      setSimilarModalOpen(true)
    },
  })

  const companyMutation = useMutation({
    mutationFn: (company: string) => findSimilarCompanies({ company_name: company, limit: 8 }),
    onSuccess: (data) => setCompanyResults(data),
  })

  function handleSearch(nextQuery: string) {
    setQuery(nextQuery)
  }

  function handleSkillSelect(skill: string) {
    setSelectedSkill(skill)
    setQuery(skill)
  }

  return (
    <div className="space-y-8">
      <section className="rounded-[32px] border border-[color:var(--border)] bg-white/90 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
        <p className="text-sm font-semibold uppercase tracking-[0.24em] text-slate-500">Search and similarity</p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight text-[color:var(--ink)]">
          Hybrid retrieval with visible query expansion, skill neighborhoods, and match evidence.
        </h1>
        <p className="mt-3 max-w-3xl text-base leading-7 text-slate-600">
          This page keeps the technical NLP showcase intact while exposing the underlying
          semantic ranking and related-skill graph that drive the results.
        </p>
      </section>

      <DegradedBanner show={health.data?.degraded ?? false} />

      <section className="grid gap-6 xl:grid-cols-[0.75fr_1.25fr]">
        <aside className="space-y-6">
          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-5">
            <FilterPanel filters={filters} onChange={setFilters} />
          </div>

          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-5">
            {skillCloud.data && (
              <SkillCloud items={skillCloud.data.items} onSkillClick={handleSkillSelect} />
            )}
          </div>

          <RelatedSkillsGraph
            data={relatedSkills.data}
            onSelectSkill={handleSkillSelect}
          />

          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-5">
            <CompanySearch
              onSearch={(company) => companyMutation.mutate(company)}
              results={companyResults}
              isLoading={companyMutation.isPending}
            />
          </div>
        </aside>

        <main className="space-y-6">
          <div className="rounded-[28px] border border-[color:var(--border)] bg-white/90 p-5">
            <SearchBar onSearch={handleSearch} isLoading={searchResult.isFetching} />
          </div>

          {searchResult.data && <SearchStats data={searchResult.data} />}

          <JobList
            jobs={searchResult.data?.results}
            isLoading={searchResult.isFetching}
            hasSearched={query.length > 0}
            onFindSimilar={(uuid) => similarMutation.mutate(uuid)}
          />
        </main>
      </section>

      <SimilarJobsModal
        open={similarModalOpen}
        onClose={() => setSimilarModalOpen(false)}
        jobs={similarJobs}
        isLoading={similarMutation.isPending}
        onFindSimilar={(uuid) => similarMutation.mutate(uuid)}
      />
    </div>
  )
}

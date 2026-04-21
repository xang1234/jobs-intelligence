import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import ActiveFiltersBar from '@/components/ActiveFiltersBar'
import CompanySearch from '@/components/CompanySearch'
import DegradedBanner from '@/components/DegradedBanner'
import FilterPanel from '@/components/FilterPanel'
import JobList from '@/components/JobList'
import RelatedSkillsGraph from '@/components/RelatedSkillsGraph'
import SearchBar, { type SearchBarHandle } from '@/components/SearchBar'
import SearchStats from '@/components/SearchStats'
import SimilarJobsModal from '@/components/SimilarJobsModal'
import SkillCloud from '@/components/SkillCloud'
import { Card } from '@/components/ui'
import { useUrlFilters } from '@/hooks/useUrlFilters'
import {
  findSimilarCompanies,
  findSimilarJobs,
  getHealth,
  getRelatedSkills,
  getSkillCloud,
  searchJobs,
} from '@/services/api'
import type { CompanySimilarity, JobResult } from '@/types/api'

export default function SearchPage() {
  const { query, filters, setQuery, setFilters, removeKey, clearAll } = useUrlFilters()
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)
  const [similarJobs, setSimilarJobs] = useState<JobResult[] | undefined>()
  const [similarModalOpen, setSimilarModalOpen] = useState(false)
  const [companyResults, setCompanyResults] = useState<CompanySimilarity[] | undefined>()
  const searchBarRef = useRef<SearchBarHandle>(null)

  // Keep SearchBar text in sync when query changes via URL/skill clicks.
  useEffect(() => {
    searchBarRef.current?.setValue(query)
  }, [query])

  const searchResult = useQuery({
    queryKey: ['search', query, filters],
    queryFn: () =>
      searchJobs({
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
    mutationFn: (uuid: string) =>
      findSimilarJobs({ job_uuid: uuid, limit: 8, exclude_same_company: true }),
    onSuccess: (data) => {
      setSimilarJobs(data.results)
      setSimilarModalOpen(true)
    },
  })

  const companyMutation = useMutation({
    mutationFn: (company: string) => findSimilarCompanies({ company_name: company, limit: 8 }),
    onSuccess: (data) => setCompanyResults(data),
  })

  function handleSkillSelect(skill: string) {
    setSelectedSkill(skill)
    setQuery(skill)
  }

  return (
    <div className="space-y-8">
      <Card as="section" radius="2xl" className="p-8">
        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[color:var(--ink-subtle)]">
          Search and similarity
        </p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight text-[color:var(--ink)]">
          Hybrid retrieval with visible query expansion, skill neighborhoods, and match evidence.
        </h1>
        <p className="mt-3 max-w-3xl text-base leading-7 text-[color:var(--ink-muted)]">
          This page keeps the technical NLP showcase intact while exposing the underlying
          semantic ranking and related-skill graph that drive the results.
        </p>
      </Card>

      <DegradedBanner show={health.data?.degraded ?? false} />

      <section className="grid gap-6 xl:grid-cols-[0.75fr_1.25fr]">
        <aside className="space-y-6">
          <Card radius="xl" className="p-5">
            <FilterPanel filters={filters} onChange={setFilters} />
          </Card>

          <Card radius="xl" className="p-5">
            {skillCloud.data && (
              <SkillCloud items={skillCloud.data.items} onSkillClick={handleSkillSelect} />
            )}
          </Card>

          <RelatedSkillsGraph data={relatedSkills.data} onSelectSkill={handleSkillSelect} />

          <Card radius="xl" className="p-5">
            <CompanySearch
              onSearch={(company) => companyMutation.mutate(company)}
              results={companyResults}
              isLoading={companyMutation.isPending}
            />
          </Card>
        </aside>

        <main className="space-y-6">
          <Card radius="xl" className="p-5">
            <SearchBar
              ref={searchBarRef}
              onSearch={setQuery}
              isLoading={searchResult.isFetching}
              defaultValue={query}
            />
          </Card>

          <ActiveFiltersBar
            query={query}
            filters={filters}
            onClearQuery={() => setQuery('')}
            onRemoveFilter={(key) => removeKey(key)}
            onClearAll={clearAll}
          />

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

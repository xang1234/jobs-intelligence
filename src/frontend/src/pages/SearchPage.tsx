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
import PageHero from '@/components/shell/PageHero'
import { Card } from '@/components/ui'
import { useIdleEnabled } from '@/hooks/useIdleEnabled'
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
  const showSecondaryData = useIdleEnabled()
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
    enabled: showSecondaryData,
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
    <div className="space-y-6">
      <PageHero
        eyebrow="Find jobs"
        title="Find your next role"
        subtitle="Search across Singapore job listings — type a role in plain words and we expand your query and surface related skills automatically. No exact keywords needed."
      >
        <SearchBar
          ref={searchBarRef}
          onSearch={setQuery}
          isLoading={searchResult.isFetching}
          defaultValue={query}
          placeholder="Try: machine learning engineer, registered nurse, product manager…"
        />
        <div className="mt-3">
          <ActiveFiltersBar
            query={query}
            filters={filters}
            onClearQuery={() => setQuery('')}
            onRemoveFilter={(key) => removeKey(key)}
            onClearAll={clearAll}
          />
        </div>
      </PageHero>

      <DegradedBanner show={health.data?.degraded ?? false} />

      <section className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
        <main className="space-y-6">
          {searchResult.data && <SearchStats data={searchResult.data} />}

          <JobList
            jobs={searchResult.data?.results}
            isLoading={searchResult.isFetching}
            hasSearched={query.length > 0}
            onFindSimilar={(uuid) => similarMutation.mutate(uuid)}
          />
        </main>

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

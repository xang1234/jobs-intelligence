import { MagnifyingGlassIcon, NoSymbolIcon } from '@heroicons/react/24/outline'
import type { JobResult } from '@/types/api'
import JobCard from '@/components/JobCard'
import SkeletonCard from '@/components/SkeletonCard'
import { EmptyState } from '@/components/ui'

interface JobListProps {
  jobs: JobResult[] | undefined
  isLoading: boolean
  hasSearched: boolean
  onFindSimilar: (uuid: string) => void
  emptyTitle?: string
  emptyDescription?: string
}

export default function JobList({
  jobs,
  isLoading,
  hasSearched,
  onFindSimilar,
  emptyTitle = 'No jobs found',
  emptyDescription = 'Try a different search, loosen the filters, or pick a suggested skill.',
}: JobListProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    )
  }

  if (!hasSearched) {
    return (
      <EmptyState
        icon={<MagnifyingGlassIcon />}
        title="Start with a search"
        description="Enter a role, skill, or company to surface matching jobs with explanations."
      />
    )
  }

  if (!jobs || jobs.length === 0) {
    return (
      <EmptyState
        icon={<NoSymbolIcon />}
        title={emptyTitle}
        description={emptyDescription}
      />
    )
  }

  return (
    <ul role="list" className="space-y-4">
      {jobs.map((job) => (
        <li key={job.uuid}>
          <JobCard job={job} onFindSimilar={onFindSimilar} />
        </li>
      ))}
    </ul>
  )
}

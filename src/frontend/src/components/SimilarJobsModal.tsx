import type { JobResult } from '@/types/api'
import JobCard from '@/components/JobCard'
import SkeletonCard from '@/components/SkeletonCard'
import { EmptyState, Modal } from '@/components/ui'
import { SparklesIcon } from '@heroicons/react/24/outline'

interface SimilarJobsModalProps {
  open: boolean
  onClose: () => void
  jobs: JobResult[] | undefined
  isLoading: boolean
  onFindSimilar: (uuid: string) => void
}

export default function SimilarJobsModal({
  open,
  onClose,
  jobs,
  isLoading,
  onFindSimilar,
}: SimilarJobsModalProps) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Similar roles"
      description="Neighbors in the embedding space, excluding the same company."
      size="lg"
    >
      <div className="space-y-4">
        {isLoading ? (
          <>
            <SkeletonCard />
            <SkeletonCard />
          </>
        ) : jobs && jobs.length > 0 ? (
          jobs.map((job) => (
            <JobCard key={job.uuid} job={job} onFindSimilar={onFindSimilar} />
          ))
        ) : (
          <EmptyState
            icon={<SparklesIcon />}
            title="No similar roles"
            description="We couldn't find comparable roles right now. Try a different job or loosen the filters."
            compact
          />
        )}
      </div>
    </Modal>
  )
}

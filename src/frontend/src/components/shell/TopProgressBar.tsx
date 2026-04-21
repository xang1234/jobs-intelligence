import { useIsFetching, useIsMutating } from '@tanstack/react-query'
import { ProgressBar } from '@/components/ui'

export default function TopProgressBar() {
  const fetching = useIsFetching() > 0
  const mutating = useIsMutating() > 0
  const active = fetching || mutating

  return (
    <div
      aria-hidden={!active}
      className={`pointer-events-none fixed inset-x-0 top-0 z-[70] transition-opacity duration-200 ${
        active ? 'opacity-100' : 'opacity-0'
      }`}
    >
      <ProgressBar size="sm" aria-label="Loading" />
    </div>
  )
}

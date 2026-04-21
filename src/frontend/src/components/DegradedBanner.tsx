import { ExclamationTriangleIcon } from '@heroicons/react/24/outline'
import { Card } from '@/components/ui'

export default function DegradedBanner({ show }: { show: boolean }) {
  if (!show) return null

  return (
    <Card intent="warning" radius="lg" elevation={0} className="p-4">
      <div className="flex items-start gap-3">
        <ExclamationTriangleIcon className="h-5 w-5 shrink-0" aria-hidden="true" />
        <p className="text-sm leading-6">
          Search is running in degraded mode. Some features may be unavailable or results may be
          less accurate.
        </p>
      </div>
    </Card>
  )
}

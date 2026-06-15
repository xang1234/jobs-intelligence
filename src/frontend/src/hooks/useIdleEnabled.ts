import { useEffect, useState } from 'react'
import { scheduleIdleTask } from '@/services/idle'

export function useIdleEnabled(delayMs = 600): boolean {
  const [enabled, setEnabled] = useState(false)

  useEffect(() => {
    if (enabled) return undefined
    return scheduleIdleTask(() => setEnabled(true), { delayMs, timeoutMs: 1_500 })
  }, [delayMs, enabled])

  return enabled
}

type IdleWindow = Window & {
  requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number
  cancelIdleCallback?: (handle: number) => void
}

function shouldSkipIdleWork(): boolean {
  if (typeof navigator === 'undefined') return false
  const connection = (navigator as Navigator & {
    connection?: { saveData?: boolean; effectiveType?: string }
  }).connection

  return Boolean(connection?.saveData || connection?.effectiveType?.includes('2g'))
}

export function scheduleIdleTask(
  task: () => void,
  {
    delayMs = 1_500,
    timeoutMs = 2_500,
    skipOnConstrainedConnection = false,
  }: { delayMs?: number; timeoutMs?: number; skipOnConstrainedConnection?: boolean } = {},
): () => void {
  if (typeof window === 'undefined') return () => {}
  if (skipOnConstrainedConnection && shouldSkipIdleWork()) return () => {}

  const idleWindow = window as IdleWindow
  let idleHandle: number | undefined
  const delayHandle = window.setTimeout(() => {
    if (typeof idleWindow.requestIdleCallback === 'function') {
      idleHandle = idleWindow.requestIdleCallback(task, { timeout: timeoutMs })
      return
    }
    task()
  }, delayMs)

  return () => {
    window.clearTimeout(delayHandle)
    if (idleHandle !== undefined) {
      idleWindow.cancelIdleCallback?.(idleHandle)
    }
  }
}

export function scheduleIdleQueue(
  tasks: Array<() => void>,
  {
    delayMs = 1_500,
    timeoutMs = 2_500,
    fallbackGapMs = 250,
    skipOnConstrainedConnection = true,
  } = {},
): () => void {
  if (typeof window === 'undefined') return () => {}
  if (skipOnConstrainedConnection && shouldSkipIdleWork()) return () => {}

  const idleWindow = window as IdleWindow
  const timeoutHandles: number[] = []
  const idleHandles: number[] = []
  let cancelled = false

  const runNext = () => {
    if (cancelled) return
    const next = tasks.shift()
    next?.()
    if (tasks.length > 0) scheduleNext()
  }

  const scheduleNext = () => {
    if (typeof idleWindow.requestIdleCallback === 'function') {
      idleHandles.push(idleWindow.requestIdleCallback(runNext, { timeout: timeoutMs }))
      return
    }
    timeoutHandles.push(window.setTimeout(runNext, fallbackGapMs))
  }

  timeoutHandles.push(window.setTimeout(scheduleNext, delayMs))

  return () => {
    cancelled = true
    timeoutHandles.forEach((handle) => window.clearTimeout(handle))
    idleHandles.forEach((handle) => idleWindow.cancelIdleCallback?.(handle))
  }
}

// Daily-fresh aggregates are published as same-origin static JSON under
// /snapshots/*.json by CI. Serving them from the Pages edge means these
// surfaces render even when the API (HF Space) or Neon are cold.
// NOTE: relative URL on purpose — resolves against the Pages origin, NOT the
// cross-origin API base.

export async function loadSnapshot<T>(name: string): Promise<T | null> {
  try {
    // 'no-cache' = always revalidate against Pages (cheap 304 when unchanged).
    // The snapshot URL is stable but its content changes daily, so we must NOT
    // hard-cache it in the browser.
    const res = await fetch(`/snapshots/${name}.json`, { cache: 'no-cache' })
    if (!res.ok) return null
    return (await res.json()) as T
  } catch {
    return null
  }
}

export function snapshotFirst<T>(name: string, apiFn: () => Promise<T>): () => Promise<T> {
  return async () => (await loadSnapshot<T>(name)) ?? apiFn()
}

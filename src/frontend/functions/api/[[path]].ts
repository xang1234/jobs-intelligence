/**
 * Cloudflare Pages Function — reverse-proxy for /api/*.
 *
 * Why this exists: the upstream API runs on a US-hosted Hugging Face Space.
 * When the browser called it directly, every request paid a full TLS handshake
 * across the Pacific (~1s) before any work happened. Routing through this
 * function instead means the browser talks same-origin to Cloudflare's nearest
 * edge (Singapore for SG users): TLS terminates locally, the edge reuses a warm
 * connection to the origin, and CORS preflights disappear (same-origin).
 *
 * Idempotent GETs are cached at the edge per the origin's Cache-Control, so
 * repeat hits to public endpoints (overview, skills cloud, stats, trends) skip
 * the origin entirely.
 *
 * ponytail: per-colo Cache API cache (each edge location caches independently).
 * Fine for this read-mostly traffic; reach for a zone-wide cache rule only if
 * cross-colo hit-rate ever matters.
 */

interface Env {
  API_ORIGIN?: string
}

const DEFAULT_ORIGIN = 'https://xang1234-jobs-intelligence-api.hf.space'

// Cap how long we wait on the origin. A down/hung Space then yields a clean 502
// instead of a ~30s hang. Generous enough for the slowest warm endpoints
// (match / career-delta), since the goal is to catch true hangs, not slow work.
const UPSTREAM_TIMEOUT_MS = 20_000

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, env } = context
  const origin = (env.API_ORIGIN ?? DEFAULT_ORIGIN).replace(/\/+$/, '')
  const url = new URL(request.url)
  const upstreamUrl = origin + url.pathname + url.search

  const cacheable = request.method === 'GET'
  const cache = caches.default

  // The Workers Cache API honors s-maxage for storage TTL but does NOT implement
  // stale-while-revalidate: once an entry expires, this match misses and the
  // fetch below revalidates synchronously against the (kept-warm) origin. The
  // swr directive on the response is still honored by browser caches.
  if (cacheable) {
    const hit = await cache.match(request)
    if (hit) return hit
  }

  // The runtime routes the outbound fetch by the target URL's host, so the
  // origin Space receives the correct Host regardless of the inbound one.
  let response: Response
  try {
    response = await fetch(new Request(upstreamUrl, request), {
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    })
  } catch (err) {
    const detail = err instanceof Error && err.name === 'TimeoutError' ? 'timeout' : 'unreachable'
    return new Response(JSON.stringify({ error: 'upstream_unavailable', detail }), {
      status: 502,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // Only cache what the origin explicitly marked shareable. Endpoints that omit
  // a public Cache-Control (e.g. per-scenario career-delta detail) must not be
  // edge-cached, so we never fall back to the CDN's heuristic caching.
  if (cacheable && response.ok) {
    const cc = response.headers.get('Cache-Control') ?? ''
    const shareable = /\bs-maxage\b|\bpublic\b/.test(cc) && !/\bno-store\b|\bprivate\b/.test(cc)
    if (shareable) {
      context.waitUntil(cache.put(request, response.clone()))
    }
  }
  return response
}

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

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, env } = context
  const origin = (env.API_ORIGIN ?? DEFAULT_ORIGIN).replace(/\/+$/, '')
  const url = new URL(request.url)
  const upstreamUrl = origin + url.pathname + url.search

  const cacheable = request.method === 'GET'
  const cache = caches.default

  if (cacheable) {
    const hit = await cache.match(request)
    if (hit) return hit
  }

  // The runtime routes the outbound fetch by the target URL's host, so the
  // origin Space receives the correct Host regardless of the inbound one.
  const response = await fetch(new Request(upstreamUrl, request))

  if (cacheable && response.ok) {
    // Cache API honors the origin's Cache-Control (s-maxage) for the TTL.
    context.waitUntil(cache.put(request, response.clone()))
  }
  return response
}

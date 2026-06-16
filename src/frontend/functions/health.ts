/**
 * Cloudflare Pages Function — proxy for /health.
 *
 * The app polls /health (same-origin) for its status indicator. Forward it to
 * the upstream Space. Not cached — it must reflect live readiness.
 */

interface Env {
  API_ORIGIN?: string
}

const DEFAULT_ORIGIN = 'https://xang1234-jobs-intelligence-api.hf.space'
const HEALTH_TIMEOUT_MS = 10_000

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, env } = context
  const origin = (env.API_ORIGIN ?? DEFAULT_ORIGIN).replace(/\/+$/, '')
  try {
    return await fetch(new Request(origin + '/health', request), {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    })
  } catch {
    // Origin down or hung — report unhealthy with a controlled status rather
    // than throwing (which would surface a generic Cloudflare error page).
    return new Response(JSON.stringify({ status: 'unavailable' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}

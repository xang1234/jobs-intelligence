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

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, env } = context
  const origin = (env.API_ORIGIN ?? DEFAULT_ORIGIN).replace(/\/+$/, '')
  return fetch(new Request(origin + '/health', request))
}

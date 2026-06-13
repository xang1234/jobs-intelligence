type ApiBaseUrlEnv = {
  readonly VITE_API_BASE_URL?: string
}

export function resolveApiBaseUrl(env?: ApiBaseUrlEnv): string {
  const viteEnv = (import.meta as ImportMeta & { readonly env?: ApiBaseUrlEnv }).env
  const configured = (env?.VITE_API_BASE_URL ?? viteEnv?.VITE_API_BASE_URL)?.trim()
  if (!configured) {
    return '/'
  }
  return configured.length > 1 ? configured.replace(/\/+$/, '') : configured
}

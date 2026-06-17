import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  MutationCache,
  QueryCache,
  QueryClient,
} from '@tanstack/react-query'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary, toast, Toaster } from '@/components/ui'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { resolveInitialTheme } from '@/hooks/useTheme'

// Apply theme synchronously to avoid flash-of-unthemed-content.
if (typeof document !== 'undefined') {
  const initial = resolveInitialTheme()
  document.documentElement.classList.toggle('dark', initial === 'dark')
  document.documentElement.style.colorScheme = initial
}

function readErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  return 'An unexpected error occurred.'
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
  queryCache: new QueryCache({
    onError: (error, query) => {
      // Surface background errors (after first render) via toast.
      // Initial loading errors are surfaced inline by the requesting component.
      if (query.state.data !== undefined) {
        toast.error('Something went wrong', {
          description: readErrorMessage(error),
        })
      }
    },
  }),
  mutationCache: new MutationCache({
    onError: (error) => {
      toast.error('Action failed', {
        description: readErrorMessage(error),
      })
    },
  }),
})

// Canonical list of PUBLIC read-only query-key roots that are safe to persist to
// localStorage. Allowlist (not denylist) on purpose: it fails safe — a query
// that's missing here simply isn't persisted, it can never leak. Search
// ('search') and any future user-specific keys are excluded; Match Lab uses
// mutations and is never in the query cache. Add new public read-only roots here.
const PERSIST_ALLOWLIST = new Set([
  'stats', 'skillCloud', 'overview', 'popularQueries', 'performanceStats',
  'skillTrends', 'roleTrend', 'companyTrend', 'similarCompanies', 'relatedSkills', 'health',
])

const persister = createSyncStoragePersister({ storage: window.localStorage })

// Bump when any persisted query's response SHAPE changes (invalidates old caches).
const QUERY_CACHE_BUSTER = 'v1'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <PersistQueryClientProvider
        client={queryClient}
        persistOptions={{
          persister,
          maxAge: 24 * 60 * 60 * 1000,
          buster: QUERY_CACHE_BUSTER,
          dehydrateOptions: {
            shouldDehydrateQuery: (query) =>
              query.state.status === 'success' &&
              typeof query.queryKey[0] === 'string' &&
              PERSIST_ALLOWLIST.has(query.queryKey[0]),
          },
        }}
      >
        <BrowserRouter>
          <ThemeProvider>
            <App />
          </ThemeProvider>
        </BrowserRouter>
      </PersistQueryClientProvider>
      <Toaster />
    </ErrorBoundary>
  </StrictMode>,
)

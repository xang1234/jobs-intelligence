import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  MutationCache,
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from '@tanstack/react-query'
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

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <ThemeProvider>
            <App />
          </ThemeProvider>
        </BrowserRouter>
      </QueryClientProvider>
      <Toaster />
    </ErrorBoundary>
  </StrictMode>,
)

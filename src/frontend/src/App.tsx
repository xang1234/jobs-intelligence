import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { Bars3Icon, MagnifyingGlassIcon } from '@heroicons/react/24/outline'
import ThemeToggle from '@/components/shell/ThemeToggle'
import TopProgressBar from '@/components/shell/TopProgressBar'
import { IconButton, Kbd, Spinner } from '@/components/ui'
import { useHotkeys } from '@/hooks/useHotkeys'

type NavItem = { to: string; label: string; end?: boolean }

const loadOverviewPage = () => import('@/pages/OverviewPage')
const loadTrendsPage = () => import('@/pages/TrendsPage')
const loadMatchLabPage = () => import('@/pages/MatchLabPage')
const loadSearchPage = () => import('@/pages/SearchPage')
const loadCommandPalette = () => import('@/components/shell/CommandPalette')
const loadMobileNav = () => import('@/components/shell/MobileNav')

const OverviewPage = lazy(loadOverviewPage)
const TrendsPage = lazy(loadTrendsPage)
const MatchLabPage = lazy(loadMatchLabPage)
const SearchPage = lazy(loadSearchPage)
const CommandPalette = lazy(loadCommandPalette)
const MobileNav = lazy(loadMobileNav)
const SystemStatus = lazy(() => import('@/components/shell/SystemStatus'))

const NAV_ITEMS: ReadonlyArray<NavItem> = [
  { to: '/', label: 'Find jobs', end: true },
  { to: '/match-lab', label: 'Match my CV' },
  { to: '/trends', label: 'Trends' },
  { to: '/pulse', label: 'Market pulse' },
]

const ROUTE_LOADERS: Record<string, () => Promise<unknown>> = {
  '/': loadSearchPage,
  '/match-lab': loadMatchLabPage,
  '/trends': loadTrendsPage,
  '/pulse': loadOverviewPage,
}

const TREND_PREFETCH = {
  skills: ['Customer Service', 'Microsoft Excel', 'Communication Skills'],
  role: 'customer service',
  company: 'DBS BANK LTD.',
  months: 3,
} as const

function RouteLoading() {
  return (
    <div className="flex min-h-[420px] items-center justify-center">
      <Spinner size="lg" label="Loading page" className="text-[color:var(--brand)]" />
    </div>
  )
}

export default function App() {
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const location = useLocation()
  const queryClient = useQueryClient()

  const activeItem =
    NAV_ITEMS.find((item) =>
      item.end ? location.pathname === item.to : location.pathname.startsWith(item.to),
    ) ?? NAV_ITEMS[0]

  const prefetchRouteModule = useCallback((to: string) => {
    void ROUTE_LOADERS[to]?.()
  }, [])

  const prefetchPageData = useCallback(
    (to: string) => {
      void import('@/services/api').then(
        ({
          getCompanyTrend,
          getHealth,
          getOverview,
          getPerformanceStats,
          getPopularQueries,
          getRoleTrend,
          getSkillCloud,
          getSkillTrends,
          getStats,
        }) => {
          if (to === '/pulse') {
            void queryClient.prefetchQuery({
              queryKey: ['overview', 3],
              queryFn: () => getOverview(3),
            })
            void queryClient.prefetchQuery({ queryKey: ['stats'], queryFn: getStats })
            void queryClient.prefetchQuery({
              queryKey: ['popularQueries'],
              queryFn: () => getPopularQueries(30, 8),
            })
            void queryClient.prefetchQuery({
              queryKey: ['performanceStats'],
              queryFn: () => getPerformanceStats(30),
            })
          }

          if (to === '/trends') {
            void queryClient.prefetchQuery({
              queryKey: ['overview', TREND_PREFETCH.months],
              queryFn: () => getOverview(TREND_PREFETCH.months),
            })
            void queryClient.prefetchQuery({
              queryKey: ['skillTrends', TREND_PREFETCH.skills, TREND_PREFETCH.months, '', ''],
              queryFn: () =>
                getSkillTrends({
                  skills: [...TREND_PREFETCH.skills],
                  months: TREND_PREFETCH.months,
                  employment_type: null,
                  region: null,
                }),
            })
            void queryClient.prefetchQuery({
              queryKey: ['roleTrend', TREND_PREFETCH.role, TREND_PREFETCH.months, '', ''],
              queryFn: () =>
                getRoleTrend({
                  query: TREND_PREFETCH.role,
                  months: TREND_PREFETCH.months,
                  employment_type: null,
                  region: null,
                }),
            })
            void queryClient.prefetchQuery({
              queryKey: ['companyTrend', TREND_PREFETCH.company, TREND_PREFETCH.months],
              queryFn: () => getCompanyTrend(TREND_PREFETCH.company, TREND_PREFETCH.months, false),
            })
          }

          if (to === '/') {
            void queryClient.prefetchQuery({
              queryKey: ['skillCloud'],
              queryFn: () => getSkillCloud(10, 80),
              staleTime: 10 * 60 * 1000,
            })
            void queryClient.prefetchQuery({ queryKey: ['health'], queryFn: getHealth })
          }
        },
      )
    },
    [queryClient],
  )

  const prefetchRoute = useCallback(
    (to: string) => {
      prefetchRouteModule(to)
      prefetchPageData(to)
    },
    [prefetchPageData, prefetchRouteModule],
  )

  const openPalette = useCallback(() => {
    void loadCommandPalette()
    setPaletteOpen(true)
  }, [])
  const closePalette = useCallback(() => setPaletteOpen(false), [])

  useEffect(() => {
    const mql = window.matchMedia('(min-width: 1024px)')
    const close = (e: MediaQueryListEvent | MediaQueryList) => {
      if (e.matches) setMobileNavOpen(false)
    }
    close(mql)
    mql.addEventListener('change', close)
    return () => mql.removeEventListener('change', close)
  }, [])

  useEffect(() => {
    const prefetchIdleWork = () => {
      for (const item of NAV_ITEMS) {
        prefetchRouteModule(item.to)
      }
      prefetchPageData('/')
      prefetchPageData('/pulse')
    }

    const idleWindow = window as Window & {
      requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number
      cancelIdleCallback?: (handle: number) => void
    }
    if (typeof idleWindow.requestIdleCallback === 'function') {
      let idleHandle: number | undefined
      const delayHandle = window.setTimeout(() => {
        idleHandle = idleWindow.requestIdleCallback?.(prefetchIdleWork, { timeout: 2500 })
      }, 1500)
      return () => {
        window.clearTimeout(delayHandle)
        if (idleHandle !== undefined) {
          idleWindow.cancelIdleCallback?.(idleHandle)
        }
      }
    }

    const handle = window.setTimeout(prefetchIdleWork, 1500)
    return () => window.clearTimeout(handle)
  }, [prefetchPageData, prefetchRouteModule])

  useHotkeys({
    'mod+k': (e) => {
      e.preventDefault()
      setPaletteOpen((o) => !o)
    },
    Escape: () => setPaletteOpen(false),
  })

  return (
    <div className="min-h-screen bg-[color:var(--bg)] text-[color:var(--ink)]">
      <TopProgressBar />

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <header className="rounded-[var(--radius-2xl)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] p-5 shadow-[var(--shadow-xl)] backdrop-blur">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            {/* Compact app bar: brand + tagline only — each page owns its own <h1> (issue #12). */}
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-[0.26em] text-[color:var(--ink-subtle)]">
                MCF Intelligence
              </p>
              <p className="mt-1 text-sm font-medium text-[color:var(--ink-muted)]">
                Find roles that fit you — and see where Singapore is hiring
              </p>
            </div>

            <div className="flex items-center justify-between gap-3 lg:justify-end">
              <nav className="hidden flex-wrap gap-2 lg:flex" aria-label="Primary">
                {NAV_ITEMS.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.end}
                    onPointerEnter={() => prefetchRoute(item.to)}
                    onMouseEnter={() => prefetchRoute(item.to)}
                    onFocus={() => prefetchRoute(item.to)}
                    className={({ isActive }) =>
                      `rounded-full px-4 py-2 text-sm font-semibold transition ${
                        isActive
                          ? 'bg-[color:var(--brand)] text-white shadow-[var(--shadow-md)]'
                          : 'bg-[color:var(--surface-2)] text-[color:var(--ink-muted)] hover:text-[color:var(--brand)]'
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ))}
              </nav>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={openPalette}
                  onPointerEnter={() => void loadCommandPalette()}
                  onMouseEnter={() => void loadCommandPalette()}
                  onFocus={() => void loadCommandPalette()}
                  className="hidden items-center gap-2 rounded-full border border-[color:var(--border)] bg-[color:var(--surface-1)] px-3 py-1.5 text-xs font-medium text-[color:var(--ink-muted)] transition hover:border-[color:var(--border-strong)] hover:text-[color:var(--ink)] focus-visible:ring-2 focus-visible:ring-[color:var(--brand)] focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--bg)] sm:inline-flex"
                  aria-label="Open command palette"
                >
                  <MagnifyingGlassIcon className="h-4 w-4" aria-hidden="true" />
                  <span>Go to…</span>
                  <span className="inline-flex items-center gap-0.5">
                    <Kbd>⌘</Kbd>
                    <Kbd>K</Kbd>
                  </span>
                </button>
                <IconButton
                  aria-label="Open command palette"
                  icon={<MagnifyingGlassIcon />}
                  size="sm"
                  onClick={openPalette}
                  onPointerEnter={() => void loadCommandPalette()}
                  onMouseEnter={() => void loadCommandPalette()}
                  onFocus={() => void loadCommandPalette()}
                  className="sm:hidden"
                />
                <ThemeToggle />
                <IconButton
                  aria-label="Open navigation menu"
                  icon={<Bars3Icon />}
                  size="sm"
                  onClick={() => {
                    void loadMobileNav()
                    setMobileNavOpen(true)
                  }}
                  onPointerEnter={() => void loadMobileNav()}
                  onMouseEnter={() => void loadMobileNav()}
                  onFocus={() => void loadMobileNav()}
                  className="lg:hidden"
                />
              </div>
            </div>
          </div>

          <div className="mt-4 flex items-center gap-2 border-t border-[color:var(--border)] pt-4 text-xs text-[color:var(--ink-subtle)]">
            <span>MCF Intelligence</span>
            <span aria-hidden>›</span>
            <span className="font-semibold text-[color:var(--ink)]">{activeItem.label}</span>
          </div>
        </header>

        <main className="py-8">
          <Suspense fallback={<RouteLoading />}>
            <Routes>
              <Route path="/" element={<SearchPage />} />
              <Route path="/match-lab" element={<MatchLabPage />} />
              <Route path="/trends" element={<TrendsPage />} />
              <Route path="/pulse" element={<OverviewPage />} />
              {/* back-compat alias for the old Search route */}
              <Route path="/search" element={<SearchPage />} />
            </Routes>
          </Suspense>
        </main>

        <Suspense fallback={null}>
          <SystemStatus />
        </Suspense>
      </div>

      {paletteOpen ? (
        <Suspense fallback={null}>
          <CommandPalette open={paletteOpen} onClose={closePalette} />
        </Suspense>
      ) : null}
      {mobileNavOpen ? (
        <Suspense fallback={null}>
          <MobileNav
            open={mobileNavOpen}
            onClose={() => setMobileNavOpen(false)}
            items={NAV_ITEMS}
            onPrefetch={prefetchRoute}
          />
        </Suspense>
      ) : null}
    </div>
  )
}

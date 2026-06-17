import type { QueryClient } from '@tanstack/react-query'
import { scheduleIdleQueue } from '@/services/idle'
import { loadRouteModule, ROUTE_WARMUP_ORDER } from '@/services/routeModules'
import { snapshotFirst } from '@/services/snapshots'

const TREND_PREFETCH = {
  skills: ['Customer Service', 'Microsoft Excel', 'Communication Skills'],
  role: 'customer service',
  company: 'DBS BANK LTD.',
  months: 3,
} as const

let apiModulePromise: Promise<typeof import('@/services/api')> | null = null

function loadApiModule() {
  apiModulePromise ??= import('@/services/api').catch((error) => {
    apiModulePromise = null
    throw error
  })
  return apiModulePromise
}

export function prefetchRouteModule(to: string): void {
  void loadRouteModule(to)?.catch(() => {
    // Warmup is speculative; route render will surface real module failures.
  })
}

export function prefetchRouteData(queryClient: QueryClient, to: string): void {
  void loadApiModule().then(
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
          queryFn: snapshotFirst('overview', () => getOverview(3)),
        })
        void queryClient.prefetchQuery({ queryKey: ['stats'], queryFn: snapshotFirst('stats', getStats) })
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
          queryFn: snapshotFirst('skills_cloud', () => getSkillCloud(10, 80)),
          staleTime: 10 * 60 * 1000,
        })
        void queryClient.prefetchQuery({ queryKey: ['health'], queryFn: getHealth })
      }
    },
  ).catch(() => {
    // Warmup is speculative; page queries will surface real API module failures.
  })
}

export function prefetchRoute(queryClient: QueryClient, to: string): void {
  prefetchRouteModule(to)
  prefetchRouteData(queryClient, to)
}

export function scheduleRouteWarmup(queryClient: QueryClient): () => void {
  const tasks: Array<() => void> = [
    ...ROUTE_WARMUP_ORDER.map((to) => () => prefetchRouteModule(to)),
    () => prefetchRouteData(queryClient, '/'),
    () => prefetchRouteData(queryClient, '/pulse'),
  ]

  return scheduleIdleQueue(tasks)
}

import type { ComponentType } from 'react'

type RouteModule = { default: ComponentType }

export type RoutePath = '/' | '/match-lab' | '/trends' | '/pulse'

export const ROUTE_LOADERS: Record<RoutePath, () => Promise<RouteModule>> = {
  '/': () => import('@/pages/SearchPage'),
  '/match-lab': () => import('@/pages/MatchLabPage'),
  '/trends': () => import('@/pages/TrendsPage'),
  '/pulse': () => import('@/pages/OverviewPage'),
}

export const ROUTE_WARMUP_ORDER: readonly RoutePath[] = ['/', '/pulse', '/trends', '/match-lab']

export function loadRouteModule(to: string): Promise<RouteModule> | undefined {
  return ROUTE_LOADERS[to as RoutePath]?.()
}

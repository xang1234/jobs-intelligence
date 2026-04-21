import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { Filters } from '@/types/api'

type FilterKey = keyof Filters | 'q'

export interface UrlFiltersState {
  query: string
  filters: Filters
  setQuery: (value: string) => void
  setFilters: (filters: Filters) => void
  patchFilters: (patch: Partial<Filters>) => void
  clearFilters: () => void
  removeKey: (key: FilterKey) => void
  clearAll: () => void
}

function readNumber(value: string | null): number | null {
  if (value == null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

export function useUrlFilters(): UrlFiltersState {
  const [searchParams, setSearchParams] = useSearchParams()

  const query = searchParams.get('q') ?? ''

  const filters: Filters = useMemo(
    () => ({
      salary_min: readNumber(searchParams.get('salary_min')),
      salary_max: readNumber(searchParams.get('salary_max')),
      employment_type: searchParams.get('employment_type') || null,
      company: searchParams.get('company') || null,
    }),
    [searchParams],
  )

  const applyPatch = useCallback(
    (
      patch: Partial<{
        q: string
        salary_min: number | null
        salary_max: number | null
        employment_type: string | null
        company: string | null
      }>,
    ) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          for (const [key, value] of Object.entries(patch)) {
            if (value == null || value === '') next.delete(key)
            else next.set(key, String(value))
          }
          return next
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  const setQuery = useCallback((value: string) => applyPatch({ q: value }), [applyPatch])

  const setFilters = useCallback(
    (next: Filters) => applyPatch(next),
    [applyPatch],
  )

  const patchFilters = useCallback(
    (patch: Partial<Filters>) => applyPatch(patch),
    [applyPatch],
  )

  const clearFilters = useCallback(
    () =>
      applyPatch({
        salary_min: null,
        salary_max: null,
        employment_type: null,
        company: null,
      }),
    [applyPatch],
  )

  const removeKey = useCallback(
    (key: FilterKey) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          next.delete(key)
          return next
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  const clearAll = useCallback(
    () =>
      applyPatch({
        q: '',
        salary_min: null,
        salary_max: null,
        employment_type: null,
        company: null,
      }),
    [applyPatch],
  )

  return {
    query,
    filters,
    setQuery,
    setFilters,
    patchFilters,
    clearFilters,
    removeKey,
    clearAll,
  }
}

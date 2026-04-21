import type { Filters } from '@/types/api'
import { Button, Chip } from '@/components/ui'

interface ActiveFiltersBarProps {
  query: string
  filters: Filters
  onClearQuery?: () => void
  onRemoveFilter: (key: keyof Filters) => void
  onClearAll?: () => void
}

function money(value: number) {
  return `$${value.toLocaleString()}`
}

export default function ActiveFiltersBar({
  query,
  filters,
  onClearQuery,
  onRemoveFilter,
  onClearAll,
}: ActiveFiltersBarProps) {
  const chips: { key: string; label: string; onRemove: () => void }[] = []

  if (query && onClearQuery) {
    chips.push({ key: 'q', label: `Query: "${query}"`, onRemove: onClearQuery })
  }

  if (filters.salary_min != null) {
    chips.push({
      key: 'salary_min',
      label: `Min ${money(filters.salary_min)}`,
      onRemove: () => onRemoveFilter('salary_min'),
    })
  }
  if (filters.salary_max != null) {
    chips.push({
      key: 'salary_max',
      label: `Max ${money(filters.salary_max)}`,
      onRemove: () => onRemoveFilter('salary_max'),
    })
  }
  if (filters.employment_type) {
    chips.push({
      key: 'employment_type',
      label: filters.employment_type,
      onRemove: () => onRemoveFilter('employment_type'),
    })
  }
  if (filters.company) {
    chips.push({
      key: 'company',
      label: `Company: ${filters.company}`,
      onRemove: () => onRemoveFilter('company'),
    })
  }

  if (chips.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-[var(--radius-lg)] border border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] px-3 py-2 shadow-[var(--shadow-xs)]">
      <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-subtle)]">
        Active
      </span>
      {chips.map((chip) => (
        <Chip key={chip.key} intent="brand" size="sm" onRemove={chip.onRemove}>
          {chip.label}
        </Chip>
      ))}
      {chips.length > 1 && onClearAll && (
        <Button variant="link" size="sm" onClick={onClearAll} className="ml-auto">
          Clear all
        </Button>
      )}
    </div>
  )
}

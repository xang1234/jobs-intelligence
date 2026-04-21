import { useMemo } from 'react'
import type { Filters } from '@/types/api'
import { Button, Input, NumberInput, RangeSlider, Select } from '@/components/ui'
import type { SelectOption } from '@/components/ui'

interface FilterPanelProps {
  filters: Filters
  onChange: (filters: Filters) => void
  salaryMin?: number
  salaryMax?: number
}

const EMPLOYMENT_OPTIONS: ReadonlyArray<SelectOption<string>> = [
  { value: 'Full Time', label: 'Full time' },
  { value: 'Part Time', label: 'Part time' },
  { value: 'Contract', label: 'Contract' },
  { value: 'Temporary', label: 'Temporary' },
  { value: 'Freelance', label: 'Freelance' },
]

function formatMoney(value: number): string {
  return `$${value.toLocaleString()}`
}

export default function FilterPanel({
  filters,
  onChange,
  salaryMin = 0,
  salaryMax = 30000,
}: FilterPanelProps) {
  const update = (patch: Partial<Filters>) => onChange({ ...filters, ...patch })

  const hasFilters =
    filters.salary_min != null ||
    filters.salary_max != null ||
    filters.employment_type != null ||
    filters.company != null

  const salaryError = useMemo(() => {
    if (
      filters.salary_min != null &&
      filters.salary_max != null &&
      filters.salary_min > filters.salary_max
    ) {
      return 'Minimum must be less than maximum'
    }
    return null
  }, [filters.salary_min, filters.salary_max])

  const handleClear = () => {
    onChange({ salary_min: null, salary_max: null, employment_type: null, company: null })
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-[color:var(--ink)]">Filters</h3>
        {hasFilters && (
          <Button variant="link" size="sm" onClick={handleClear}>
            Clear all
          </Button>
        )}
      </div>

      <RangeSlider
        label="Salary range (monthly)"
        min={salaryMin}
        max={salaryMax}
        step={500}
        value={[filters.salary_min, filters.salary_max]}
        onChange={([lo, hi]) => update({ salary_min: lo, salary_max: hi })}
        formatValue={formatMoney}
      />

      <div className="grid grid-cols-2 gap-3">
        <NumberInput
          label="Min"
          min={salaryMin}
          step={500}
          placeholder="5,000"
          value={filters.salary_min}
          onValueChange={(v) => update({ salary_min: v })}
        />
        <NumberInput
          label="Max"
          min={salaryMin}
          step={500}
          placeholder="15,000"
          value={filters.salary_max}
          onValueChange={(v) => update({ salary_max: v })}
          error={salaryError}
        />
      </div>

      <Select<string>
        label="Employment type"
        placeholder="All types"
        value={filters.employment_type}
        onChange={(v) => update({ employment_type: v })}
        options={EMPLOYMENT_OPTIONS}
      />

      <Input
        label="Company"
        type="text"
        placeholder="Filter by company"
        value={filters.company ?? ''}
        onChange={(e) => update({ company: e.target.value || null })}
      />
    </div>
  )
}

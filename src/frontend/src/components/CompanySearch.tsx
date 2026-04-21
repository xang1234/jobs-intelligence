import { useState } from 'react'
import { BuildingOfficeIcon } from '@heroicons/react/24/outline'
import type { CompanySimilarity } from '@/types/api'
import { Button, Chip, Input, Spinner } from '@/components/ui'

interface CompanySearchProps {
  onSearch: (company: string) => void
  results: CompanySimilarity[] | undefined
  isLoading: boolean
}

export default function CompanySearch({ onSearch, results, isLoading }: CompanySearchProps) {
  const [input, setInput] = useState('')

  function handleSubmit() {
    const trimmed = input.trim()
    if (trimmed) onSearch(trimmed)
  }

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-[color:var(--ink)]">Company search</h3>

      <div className="flex gap-2">
        <div className="flex-1">
          <Input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSubmit()
            }}
            placeholder="Company name"
            aria-label="Company name"
          />
        </div>
        <Button
          variant="secondary"
          size="md"
          onClick={handleSubmit}
          disabled={!input.trim()}
          loading={isLoading}
        >
          Go
        </Button>
      </div>

      {isLoading && !results && (
        <div className="mt-3 flex items-center gap-2 text-sm text-[color:var(--ink-subtle)]">
          <Spinner size="sm" /> Searching…
        </div>
      )}

      {results && results.length > 0 && (
        <ul className="mt-3 space-y-2">
          {results.map((c) => (
            <li
              key={c.company_name}
              className="rounded-[var(--radius-md)] border border-[color:var(--border)] p-3 text-sm"
            >
              <div className="flex items-center gap-2">
                <BuildingOfficeIcon className="h-4 w-4 shrink-0 text-[color:var(--ink-subtle)]" />
                <span className="truncate font-medium text-[color:var(--ink)]">{c.company_name}</span>
                <span className="ml-auto shrink-0 text-xs text-[color:var(--ink-subtle)]">
                  {(c.similarity_score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-[color:var(--ink-subtle)]">
                <span>{c.job_count} jobs</span>
                {c.avg_salary != null && <span>avg ${c.avg_salary.toLocaleString()}</span>}
              </div>
              {c.top_skills.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {c.top_skills.slice(0, 5).map((skill) => (
                    <Chip key={skill} intent="neutral" size="sm">
                      {skill}
                    </Chip>
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {results && results.length === 0 && (
        <p className="mt-3 text-sm text-[color:var(--ink-subtle)]">No similar companies found.</p>
      )}
    </div>
  )
}

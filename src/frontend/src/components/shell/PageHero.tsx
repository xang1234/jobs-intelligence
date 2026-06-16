import type { ReactNode } from 'react'
import { Card } from '@/components/ui'

interface PageHeroProps {
  title: string
  eyebrow?: string
  subtitle?: string
  /** Inline controls shown to the right of the title on wide screens. */
  actions?: ReactNode
  /** Full-width content rendered below the title block (search bar, metric strip…). */
  children?: ReactNode
  /** "brand" paints a saturated gradient with white text — used for the Market pulse banner. */
  tone?: 'default' | 'brand'
}

// ponytail: one hero for all four views (issue #13). Layout differs only via `actions`/`children`.
export default function PageHero({
  title,
  eyebrow,
  subtitle,
  actions,
  children,
  tone = 'default',
}: PageHeroProps) {
  const brand = tone === 'brand'
  return (
    <Card
      as="section"
      radius="2xl"
      elevation={brand ? 2 : 1}
      className="p-6 sm:p-8"
      style={
        brand
          ? { background: 'linear-gradient(135deg, var(--color-brand-700) 0%, var(--color-brand-900) 100%)' }
          : undefined
      }
    >
      <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl">
          {eyebrow && (
            <p
              className={`text-xs font-semibold uppercase tracking-[0.24em] ${
                brand ? 'text-white/75' : 'text-[color:var(--ink-subtle)]'
              }`}
            >
              {eyebrow}
            </p>
          )}
          <h1
            className={`text-3xl font-semibold tracking-tight sm:text-4xl ${eyebrow ? 'mt-2' : ''} ${
              brand ? 'text-white' : 'text-[color:var(--ink)]'
            }`}
          >
            {title}
          </h1>
          {subtitle && (
            <p
              className={`mt-3 max-w-2xl text-sm leading-6 ${
                brand ? 'text-white/85' : 'text-[color:var(--ink-muted)]'
              }`}
            >
              {subtitle}
            </p>
          )}
        </div>
        {actions}
      </div>
      {children && <div className="mt-6">{children}</div>}
    </Card>
  )
}

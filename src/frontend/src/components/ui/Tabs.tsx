import type { ReactNode } from 'react'
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from '@headlessui/react'
import { cn } from './cn'

export type TabVariant = 'pill' | 'underline'

export interface TabItem {
  label: ReactNode
  content: ReactNode
  disabled?: boolean
}

export interface TabsProps {
  items: ReadonlyArray<TabItem>
  variant?: TabVariant
  selectedIndex?: number
  onChange?: (index: number) => void
  className?: string
  panelClassName?: string
}

export function Tabs({
  items,
  variant = 'pill',
  selectedIndex,
  onChange,
  className,
  panelClassName,
}: TabsProps) {
  return (
    <TabGroup
      selectedIndex={selectedIndex}
      onChange={onChange}
      className={className}
    >
      <TabList
        className={cn(
          'flex flex-wrap items-center gap-1',
          variant === 'underline' &&
            'gap-4 border-b border-[color:var(--border)]',
        )}
      >
        {items.map((item, i) => (
          <Tab
            key={i}
            disabled={item.disabled}
            className={({ selected }) =>
              cn(
                'transition focus-visible:outline-none',
                variant === 'pill' &&
                  cn(
                    'rounded-full px-3.5 py-1.5 text-sm font-semibold',
                    selected
                      ? 'bg-[color:var(--brand)] text-white shadow-[var(--shadow-sm)]'
                      : 'bg-[color:var(--surface-2)] text-[color:var(--ink-muted)] hover:text-[color:var(--ink)]',
                  ),
                variant === 'underline' &&
                  cn(
                    '-mb-px border-b-2 px-2 py-2 text-sm font-semibold',
                    selected
                      ? 'border-[color:var(--brand)] text-[color:var(--ink)]'
                      : 'border-transparent text-[color:var(--ink-muted)] hover:text-[color:var(--ink)]',
                  ),
                item.disabled && 'cursor-not-allowed opacity-50',
              )
            }
          >
            {item.label}
          </Tab>
        ))}
      </TabList>
      <TabPanels className={cn('mt-4', panelClassName)}>
        {items.map((item, i) => (
          <TabPanel key={i} className="focus-visible:outline-none">
            {item.content}
          </TabPanel>
        ))}
      </TabPanels>
    </TabGroup>
  )
}

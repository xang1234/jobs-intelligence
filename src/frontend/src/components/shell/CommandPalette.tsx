import { Fragment, useCallback, useMemo, useState, type ReactNode } from 'react'
import {
  Combobox,
  ComboboxInput,
  ComboboxOption,
  ComboboxOptions,
  Dialog,
  DialogPanel,
  DialogTitle,
  Transition,
  TransitionChild,
} from '@headlessui/react'
import {
  MagnifyingGlassIcon,
  BeakerIcon,
  ChartBarIcon,
  HomeIcon,
  MoonIcon,
  SunIcon,
  Squares2X2Icon,
} from '@heroicons/react/24/outline'
import { useNavigate } from 'react-router-dom'
import { Kbd } from '@/components/ui'
import { useTheme } from '@/hooks/useTheme'

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
}

interface Action {
  id: string
  label: string
  hint?: string
  keywords?: string
  icon: ReactNode
  run: () => void
}

export default function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const navigate = useNavigate()
  const { theme, toggle } = useTheme()

  const go = useCallback(
    (to: string) => {
      onClose()
      setTimeout(() => navigate(to), 0)
    },
    [navigate, onClose],
  )

  const actions: Action[] = useMemo(
    () => [
      {
        id: 'nav-overview',
        label: 'Go to Overview',
        hint: 'Home dashboard',
        keywords: 'home dashboard market',
        icon: <HomeIcon />,
        run: () => go('/'),
      },
      {
        id: 'nav-trends',
        label: 'Go to Trends Explorer',
        hint: 'Skills, roles, companies over time',
        keywords: 'trends charts time series',
        icon: <ChartBarIcon />,
        run: () => go('/trends'),
      },
      {
        id: 'nav-matchlab',
        label: 'Go to Match Lab',
        hint: 'Profile-to-role matching & what-ifs',
        keywords: 'match scenario what if career',
        icon: <BeakerIcon />,
        run: () => go('/match-lab'),
      },
      {
        id: 'nav-search',
        label: 'Go to Search & Similarity',
        hint: 'Hybrid retrieval',
        keywords: 'search jobs retrieval',
        icon: <Squares2X2Icon />,
        run: () => go('/search'),
      },
      {
        id: 'theme-toggle',
        label: theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
        keywords: 'theme dark light mode',
        icon: theme === 'dark' ? <SunIcon /> : <MoonIcon />,
        run: () => {
          toggle()
          onClose()
        },
      },
    ],
    [theme, toggle, onClose, go],
  )

  const trimmed = query.trim()
  const filtered = trimmed
    ? actions.filter((a) => {
        const hay = `${a.label} ${a.keywords ?? ''} ${a.hint ?? ''}`.toLowerCase()
        return hay.includes(trimmed.toLowerCase())
      })
    : actions

  // "Search jobs: X" virtual action when the query looks like free-text.
  const searchVirtual: Action | null = trimmed
    ? {
        id: 'virtual-search',
        label: `Search jobs: "${trimmed}"`,
        hint: 'Jump to Search & Similarity',
        icon: <MagnifyingGlassIcon />,
        run: () => go(`/search?q=${encodeURIComponent(trimmed)}`),
      }
    : null

  const items = searchVirtual ? [searchVirtual, ...filtered] : filtered

  return (
    <Transition show={open} as={Fragment} afterLeave={() => setQuery('')}>
      <Dialog onClose={onClose} className="relative z-[70]">
        <TransitionChild
          as={Fragment}
          enter="ease-out duration-150"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-100"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" aria-hidden="true" />
        </TransitionChild>

        <div className="fixed inset-0 overflow-y-auto p-4 pt-[10vh]">
          <div className="mx-auto max-w-xl">
            <TransitionChild
              as={Fragment}
              enter="ease-out duration-200"
              enterFrom="opacity-0 translate-y-2 scale-95"
              enterTo="opacity-100 translate-y-0 scale-100"
              leave="ease-in duration-100"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <DialogPanel className="overflow-hidden rounded-[var(--radius-2xl)] border border-[color:var(--border)] bg-[color:var(--surface-1)] shadow-[var(--shadow-xl)]">
                <DialogTitle className="sr-only">Command palette</DialogTitle>
                <Combobox
                  onChange={(action: Action | null) => action?.run()}
                >
                  <div className="flex items-center gap-3 border-b border-[color:var(--border)] px-4">
                    <MagnifyingGlassIcon
                      className="h-5 w-5 shrink-0 text-[color:var(--ink-subtle)]"
                      aria-hidden="true"
                    />
                    <ComboboxInput
                      autoFocus
                      placeholder="Jump to a page or search jobs…"
                      className="w-full bg-transparent py-3.5 text-sm text-[color:var(--ink)] placeholder:text-[color:var(--ink-subtle)] focus-visible:outline-none"
                      onChange={(e) => setQuery(e.target.value)}
                    />
                    <Kbd>Esc</Kbd>
                  </div>

                  <ComboboxOptions
                    static
                    className="max-h-80 overflow-y-auto p-1.5"
                  >
                    {items.length === 0 ? (
                      <p className="px-3 py-6 text-center text-sm text-[color:var(--ink-subtle)]">
                        No matches.
                      </p>
                    ) : (
                      items.map((action) => (
                        <ComboboxOption
                          key={action.id}
                          value={action}
                          className="group flex cursor-pointer items-center gap-3 rounded-[var(--radius-md)] px-3 py-2.5 data-[focus]:bg-[color:var(--surface-2)]"
                        >
                          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[color:var(--surface-2)] text-[color:var(--brand)] group-data-[focus]:bg-[color:var(--surface-1)] [&_svg]:h-4 [&_svg]:w-4">
                            {action.icon}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-medium text-[color:var(--ink)]">
                              {action.label}
                            </span>
                            {action.hint && (
                              <span className="block truncate text-xs text-[color:var(--ink-subtle)]">
                                {action.hint}
                              </span>
                            )}
                          </span>
                        </ComboboxOption>
                      ))
                    )}
                  </ComboboxOptions>

                  <div className="flex items-center justify-between border-t border-[color:var(--border)] px-4 py-2 text-[11px] text-[color:var(--ink-subtle)]">
                    <span className="inline-flex items-center gap-1">
                      <Kbd>↑</Kbd> <Kbd>↓</Kbd> navigate
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <Kbd>↵</Kbd> select
                    </span>
                  </div>
                </Combobox>
              </DialogPanel>
            </TransitionChild>
          </div>
        </div>
      </Dialog>
    </Transition>
  )
}

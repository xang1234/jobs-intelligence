import { Fragment, type ReactNode } from 'react'
import {
  Listbox,
  ListboxButton,
  ListboxOption,
  ListboxOptions,
  Transition,
} from '@headlessui/react'
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid'
import { cn } from './cn'

export interface SelectOption<T extends string | number> {
  value: T
  label: ReactNode
  disabled?: boolean
}

export interface SelectProps<T extends string | number> {
  label?: string
  hint?: string
  value: T | null
  onChange: (value: T | null) => void
  options: ReadonlyArray<SelectOption<T>>
  placeholder?: string
  disabled?: boolean
  clearable?: boolean
  className?: string
}

export function Select<T extends string | number>({
  label,
  hint,
  value,
  onChange,
  options,
  placeholder = 'Select…',
  disabled,
  clearable = true,
  className,
}: SelectProps<T>) {
  const active = options.find((opt) => opt.value === value) ?? null

  return (
    <div className={cn('flex w-full flex-col gap-1.5', className)}>
      {label && (
        <span className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-muted)]">
          {label}
        </span>
      )}
      <Listbox value={value} onChange={onChange} disabled={disabled}>
        <div className="relative">
          <ListboxButton
            className={cn(
              'flex w-full items-center justify-between gap-2 rounded-[var(--radius-md)] border border-[color:var(--border)] bg-[color:var(--surface-1)] px-3 py-2 text-left text-sm text-[color:var(--ink)] transition',
              'focus-visible:outline-none focus-visible:shadow-[var(--ring)] hover:border-[color:var(--border-strong)]',
              'disabled:cursor-not-allowed disabled:opacity-60',
            )}
          >
            <span className={cn(!active && 'text-[color:var(--ink-subtle)]')}>
              {active?.label ?? placeholder}
            </span>
            <ChevronUpDownIcon className="h-4 w-4 text-[color:var(--ink-subtle)]" aria-hidden="true" />
          </ListboxButton>
          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <ListboxOptions
              anchor="bottom start"
              className="z-[80] mt-2 max-h-64 w-[var(--button-width)] min-w-[10rem] origin-top overflow-auto rounded-[var(--radius-lg)] border border-[color:var(--border)] bg-[color:var(--surface-1)] p-1 shadow-[var(--shadow-lg)] focus-visible:outline-none"
            >
              {clearable && value != null && (
                <ListboxOption
                  value={null}
                  className="group flex cursor-pointer items-center gap-2 rounded-[var(--radius-sm)] px-2.5 py-1.5 text-sm text-[color:var(--ink-subtle)] data-[focus]:bg-[color:var(--surface-2)]"
                >
                  Clear
                </ListboxOption>
              )}
              {options.map((opt) => (
                <ListboxOption
                  key={String(opt.value)}
                  value={opt.value}
                  disabled={opt.disabled}
                  className="group flex cursor-pointer items-center justify-between gap-2 rounded-[var(--radius-sm)] px-2.5 py-1.5 text-sm text-[color:var(--ink)] data-[focus]:bg-[color:var(--surface-2)] data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50"
                >
                  <span>{opt.label}</span>
                  <CheckIcon className="h-4 w-4 text-[color:var(--brand)] opacity-0 group-data-[selected]:opacity-100" />
                </ListboxOption>
              ))}
            </ListboxOptions>
          </Transition>
        </div>
      </Listbox>
      {hint && <p className="text-xs text-[color:var(--ink-subtle)]">{hint}</p>}
    </div>
  )
}

import { Fragment } from 'react'
import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { NavLink } from 'react-router-dom'
import { IconButton } from '@/components/ui'

interface NavItem {
  to: string
  label: string
  end?: boolean
}

interface MobileNavProps {
  open: boolean
  onClose: () => void
  items: ReadonlyArray<NavItem>
}

export default function MobileNav({ open, onClose, items }: MobileNavProps) {
  return (
    <Transition show={open} as={Fragment}>
      <Dialog onClose={onClose} className="relative z-[65] lg:hidden">
        <TransitionChild
          as={Fragment}
          enter="ease-out duration-200"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-150"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" aria-hidden="true" />
        </TransitionChild>

        <div className="fixed inset-0 flex">
          <TransitionChild
            as={Fragment}
            enter="ease-out duration-200"
            enterFrom="translate-x-full"
            enterTo="translate-x-0"
            leave="ease-in duration-150"
            leaveFrom="translate-x-0"
            leaveTo="translate-x-full"
          >
            <DialogPanel className="ml-auto flex h-full w-full max-w-xs flex-col bg-[color:var(--surface-1)] shadow-[var(--shadow-xl)]">
              <div className="flex items-center justify-between border-b border-[color:var(--border)] px-5 py-4">
                <DialogTitle as="span" className="text-xs font-semibold uppercase tracking-[0.26em] text-[color:var(--ink-subtle)]">
                  MCF Intelligence
                </DialogTitle>
                <IconButton
                  aria-label="Close navigation"
                  icon={<XMarkIcon />}
                  size="sm"
                  onClick={onClose}
                />
              </div>
              <nav className="flex flex-col gap-1 p-3">
                {items.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.end}
                    onClick={onClose}
                    className={({ isActive }) =>
                      `rounded-[var(--radius-md)] px-4 py-3 text-sm font-semibold transition ${
                        isActive
                          ? 'bg-[color:var(--brand)] text-white shadow-[var(--shadow-sm)]'
                          : 'text-[color:var(--ink-muted)] hover:bg-[color:var(--surface-2)] hover:text-[color:var(--ink)]'
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ))}
              </nav>
            </DialogPanel>
          </TransitionChild>
        </div>
      </Dialog>
    </Transition>
  )
}

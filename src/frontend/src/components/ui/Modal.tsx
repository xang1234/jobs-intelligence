import { Fragment, type ReactNode } from 'react'
import { Dialog, DialogPanel, DialogTitle, Transition, TransitionChild } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { IconButton } from './IconButton'
import { cn } from './cn'

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl'

export interface ModalProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  description?: ReactNode
  children: ReactNode
  footer?: ReactNode
  size?: ModalSize
  closeOnOverlay?: boolean
  className?: string
}

const SIZE: Record<ModalSize, string> = {
  sm: 'max-w-md',
  md: 'max-w-xl',
  lg: 'max-w-3xl',
  xl: 'max-w-5xl',
}

export function Modal({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  size = 'md',
  closeOnOverlay = true,
  className,
}: ModalProps) {
  return (
    <Transition show={open} as={Fragment}>
      <Dialog
        onClose={closeOnOverlay ? onClose : () => undefined}
        className="relative z-[60]"
      >
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

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-start justify-center p-4 pt-14 sm:items-center sm:pt-4">
            <TransitionChild
              as={Fragment}
              enter="ease-out duration-200"
              enterFrom="opacity-0 scale-95 translate-y-2"
              enterTo="opacity-100 scale-100 translate-y-0"
              leave="ease-in duration-150"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <DialogPanel
                className={cn(
                  'relative w-full rounded-[var(--radius-2xl)] border border-[color:var(--border)] bg-[color:var(--surface-1)] shadow-[var(--shadow-xl)]',
                  SIZE[size],
                  className,
                )}
              >
                <IconButton
                  aria-label="Close dialog"
                  icon={<XMarkIcon />}
                  size="sm"
                  onClick={onClose}
                  className="absolute top-4 right-4"
                />
                {(title || description) && (
                  <div className="px-6 pt-6 pb-4 pr-14">
                    {title && (
                      <DialogTitle className="text-lg font-semibold tracking-tight text-[color:var(--ink)]">
                        {title}
                      </DialogTitle>
                    )}
                    {description && (
                      <p className="mt-1 text-sm text-[color:var(--ink-muted)]">
                        {description}
                      </p>
                    )}
                  </div>
                )}
                <div className="px-6 pb-6">{children}</div>
                {footer && (
                  <div className="flex items-center justify-end gap-2 border-t border-[color:var(--border)] px-6 py-4">
                    {footer}
                  </div>
                )}
              </DialogPanel>
            </TransitionChild>
          </div>
        </div>
      </Dialog>
    </Transition>
  )
}

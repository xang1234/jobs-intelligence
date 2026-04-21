import { Fragment, useEffect, useId, useRef, useState, type ReactElement, type ReactNode } from 'react'
import { Transition } from '@headlessui/react'
import { cn } from './cn'

export interface TooltipProps {
  content: ReactNode
  children: ReactElement
  side?: 'top' | 'bottom' | 'left' | 'right'
  delayMs?: number
  className?: string
}

const SIDE: Record<NonNullable<TooltipProps['side']>, string> = {
  top: 'bottom-full left-1/2 mb-2 -translate-x-1/2',
  bottom: 'top-full left-1/2 mt-2 -translate-x-1/2',
  left: 'right-full top-1/2 mr-2 -translate-y-1/2',
  right: 'left-full top-1/2 ml-2 -translate-y-1/2',
}

export function Tooltip({
  content,
  children,
  side = 'top',
  delayMs = 150,
  className,
}: TooltipProps) {
  const [open, setOpen] = useState(false)
  const timerRef = useRef<number | null>(null)
  const id = useId()

  const show = () => {
    if (timerRef.current) window.clearTimeout(timerRef.current)
    timerRef.current = window.setTimeout(() => setOpen(true), delayMs)
  }
  const hide = () => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
    setOpen(false)
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current)
    }
  }, [])

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      aria-describedby={open ? id : undefined}
    >
      {children}
      <Transition
        as={Fragment}
        show={open}
        enter="transition ease-out duration-150"
        enterFrom="opacity-0 scale-95"
        enterTo="opacity-100 scale-100"
        leave="transition ease-in duration-100"
        leaveFrom="opacity-100 scale-100"
        leaveTo="opacity-0 scale-95"
      >
        <span
          id={id}
          role="tooltip"
          className={cn(
            'pointer-events-none absolute z-50 whitespace-nowrap rounded-[var(--radius-sm)] bg-[color:var(--color-neutral-900)] px-2.5 py-1.5 text-xs font-medium text-[color:var(--color-neutral-50)] shadow-[var(--shadow-md)]',
            SIDE[side],
            className,
          )}
        >
          {content}
        </span>
      </Transition>
    </span>
  )
}

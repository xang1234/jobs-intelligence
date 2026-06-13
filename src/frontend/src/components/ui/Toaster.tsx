import { Toaster as SonnerToaster } from 'sonner'

export function Toaster() {
  return (
    <SonnerToaster
      position="bottom-right"
      duration={4000}
      closeButton
      richColors={false}
      toastOptions={{
        classNames: {
          toast:
            'group rounded-[var(--radius-lg)]! border! border-[color:var(--border)]! bg-[color:var(--surface-1)]! text-[color:var(--ink)]! shadow-[var(--shadow-lg)]!',
          title: 'text-sm! font-semibold! text-[color:var(--ink)]!',
          description: 'text-xs! text-[color:var(--ink-muted)]!',
          actionButton:
            'bg-[color:var(--brand)]! text-white! rounded-full! px-3! py-1! text-xs! font-semibold!',
          cancelButton:
            'bg-[color:var(--surface-2)]! text-[color:var(--ink-muted)]! rounded-full! px-3! py-1! text-xs! font-semibold!',
          success: 'border-[color:var(--color-success-500)]/30!',
          error: 'border-[color:var(--color-danger-500)]/30!',
          warning: 'border-[color:var(--color-warning-500)]/30!',
          info: 'border-[color:var(--color-info-500)]/30!',
        },
      }}
    />
  )
}

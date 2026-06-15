export function SummaryMetric({
  label,
  value,
  accent = false,
}: {
  label: string
  value: string
  accent?: boolean
}) {
  return (
    <div className="rounded-[var(--radius-md)] bg-[color:var(--surface)] px-4 py-3">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-subtle)]">{label}</p>
      <p className={`mt-2 text-lg font-semibold ${accent ? 'text-[color:var(--brand)]' : 'text-[color:var(--ink)]'}`}>
        {value}
      </p>
    </div>
  )
}

export function StatePanel({
  eyebrow,
  title,
  message,
  tone = 'neutral',
  actionLabel,
  onAction,
}: {
  eyebrow: string
  title: string
  message: string
  tone?: 'neutral' | 'warning' | 'danger'
  actionLabel?: string
  onAction?: () => void
}) {
  const toneClass =
    tone === 'danger'
      ? 'border-[color:var(--color-danger-500)]/30 bg-[color:var(--danger-bg)] text-[color:var(--color-danger-900)]'
      : tone === 'warning'
        ? 'border-[color:var(--color-warning-500)]/30 bg-[color:var(--color-warning-50)] text-[color:var(--warning-fg)]'
        : 'border-[color:var(--border)] bg-[color:var(--surface-1-alpha)] text-[color:var(--ink-muted)]'

  return (
    <section className={`rounded-[var(--radius-xl)] border p-6 ${toneClass}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] opacity-70">{eyebrow}</p>
      <h3 className="mt-2 text-xl font-semibold">{title}</h3>
      <p className="mt-3 max-w-3xl text-sm leading-6 opacity-90">{message}</p>
      {actionLabel && onAction ? (
        <button
          type="button"
          onClick={onAction}
          className="mt-4 rounded-full border border-current px-4 py-2 text-sm font-semibold transition hover:bg-[color:var(--surface-1)]/40"
        >
          {actionLabel}
        </button>
      ) : null}
    </section>
  )
}

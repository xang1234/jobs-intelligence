import { useCallback, useId, useMemo, useRef, useState, type PointerEvent } from 'react'
import { cn } from './cn'

export interface RangeSliderProps {
  min: number
  max: number
  step?: number
  value: [number | null, number | null]
  onChange: (value: [number | null, number | null]) => void
  label?: string
  formatValue?: (value: number) => string
  'aria-label'?: string
  className?: string
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v))
}

function round(v: number, step: number) {
  return Math.round(v / step) * step
}

export function RangeSlider({
  min,
  max,
  step = 1,
  value,
  onChange,
  label,
  formatValue = (v) => String(v),
  className,
}: RangeSliderProps) {
  const lo = value[0] ?? min
  const hi = value[1] ?? max
  const trackRef = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState<'lo' | 'hi' | null>(null)
  const id = useId()

  const percent = useCallback(
    (v: number) => ((v - min) / (max - min)) * 100,
    [min, max],
  )

  const updateFromPointer = useCallback(
    (clientX: number, which: 'lo' | 'hi') => {
      const track = trackRef.current
      if (!track) return
      const rect = track.getBoundingClientRect()
      const raw = ((clientX - rect.left) / rect.width) * (max - min) + min
      const stepped = round(clamp(raw, min, max), step)
      if (which === 'lo') {
        onChange([Math.min(stepped, hi), value[1]])
      } else {
        onChange([value[0], Math.max(stepped, lo)])
      }
    },
    [hi, lo, max, min, onChange, step, value],
  )

  const handlePointerDown = (which: 'lo' | 'hi') => (e: PointerEvent) => {
    e.preventDefault()
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    setDragging(which)
  }

  const handlePointerMove = (which: 'lo' | 'hi') => (e: PointerEvent) => {
    if (dragging !== which) return
    updateFromPointer(e.clientX, which)
  }

  const handlePointerUp = () => setDragging(null)

  const onKey =
    (which: 'lo' | 'hi') =>
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      let delta = 0
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') delta = -step
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') delta = step
      if (e.key === 'PageDown') delta = -step * 10
      if (e.key === 'PageUp') delta = step * 10
      if (delta === 0) return
      e.preventDefault()
      if (which === 'lo') {
        onChange([Math.min(clamp(lo + delta, min, max), hi), value[1]])
      } else {
        onChange([value[0], Math.max(clamp(hi + delta, min, max), lo)])
      }
    }

  const leftPct = useMemo(() => percent(lo), [percent, lo])
  const rightPct = useMemo(() => percent(hi), [percent, hi])

  return (
    <div className={cn('flex w-full flex-col gap-2', className)}>
      {label && (
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-muted)]">
            {label}
          </span>
          <span className="text-xs text-[color:var(--ink-muted)]">
            {formatValue(lo)} – {formatValue(hi)}
          </span>
        </div>
      )}
      <div
        ref={trackRef}
        className="relative h-8 w-full touch-none select-none"
        id={id}
      >
        <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-[color:var(--surface-3)]" />
        <div
          className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-[color:var(--brand)]"
          style={{
            left: `${leftPct}%`,
            width: `${Math.max(0, rightPct - leftPct)}%`,
          }}
        />
        {(['lo', 'hi'] as const).map((which) => {
          const pct = which === 'lo' ? leftPct : rightPct
          const v = which === 'lo' ? lo : hi
          return (
            <div
              key={which}
              role="slider"
              tabIndex={0}
              aria-valuemin={min}
              aria-valuemax={max}
              aria-valuenow={v}
              aria-valuetext={formatValue(v)}
              onPointerDown={handlePointerDown(which)}
              onPointerMove={handlePointerMove(which)}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              onKeyDown={onKey(which)}
              style={{ left: `${pct}%` }}
              className={cn(
                'absolute top-1/2 -translate-x-1/2 -translate-y-1/2 h-5 w-5 rounded-full border-2 border-[color:var(--brand)] bg-[color:var(--surface-1)] shadow-[var(--shadow-sm)] transition focus-visible:outline-none focus-visible:shadow-[var(--ring)]',
                dragging === which && 'scale-110',
              )}
            />
          )
        })}
      </div>
    </div>
  )
}

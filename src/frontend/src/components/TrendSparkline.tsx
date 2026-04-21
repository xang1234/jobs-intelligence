import { useId, useMemo, useRef, useState } from 'react'
import type { TrendPoint } from '@/types/api'

interface TrendSparklineProps {
  points: TrendPoint[]
  /** Tailwind stroke utility (e.g. "stroke-[color:var(--brand)]") */
  strokeClassName?: string
  /** When non-null shows value + month in a tooltip */
  valueFormatter?: (point: TrendPoint) => string
  ariaLabel?: string
}

const WIDTH = 220
const HEIGHT = 76
const PADDING = 6

export default function TrendSparkline({
  points,
  strokeClassName = 'stroke-[color:var(--brand)]',
  valueFormatter = (p) => `${p.job_count.toLocaleString()} jobs`,
  ariaLabel,
}: TrendSparklineProps) {
  const id = useId()
  const svgRef = useRef<SVGSVGElement>(null)
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

  const { coords, maxValue, minValue } = useMemo(() => {
    if (points.length === 0) {
      return { coords: [] as { x: number; y: number }[], maxValue: 0, minValue: 0 }
    }
    const counts = points.map((p) => p.job_count)
    const max = Math.max(...counts, 1)
    const min = Math.min(...counts, 0)
    const range = max - min || 1
    const xs = points.map((_, i) => {
      return PADDING + (i / Math.max(points.length - 1, 1)) * (WIDTH - PADDING * 2)
    })
    const ys = points.map(
      (p) => HEIGHT - PADDING - ((p.job_count - min) / range) * (HEIGHT - PADDING * 2),
    )
    return {
      coords: xs.map((x, i) => ({ x, y: ys[i] })),
      maxValue: max,
      minValue: min,
    }
  }, [points])

  if (points.length === 0) return null

  const polyPoints = coords.map((c) => `${c.x},${c.y}`).join(' ')
  const areaPoints = [
    `${PADDING},${HEIGHT - PADDING}`,
    ...coords.map((c) => `${c.x},${c.y}`),
    `${WIDTH - PADDING},${HEIGHT - PADDING}`,
  ].join(' ')

  const handleMove = (clientX: number) => {
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const relX = ((clientX - rect.left) / rect.width) * WIDTH
    let nearest = 0
    let minDist = Infinity
    coords.forEach((c, i) => {
      const d = Math.abs(c.x - relX)
      if (d < minDist) {
        minDist = d
        nearest = i
      }
    })
    setActiveIndex(nearest)
  }

  const active = activeIndex != null ? points[activeIndex] : null
  const activeCoord = activeIndex != null ? coords[activeIndex] : null

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-20 w-full text-[color:var(--brand)]"
        role="img"
        aria-label={
          ariaLabel ??
          `Trend: ${minValue.toLocaleString()} to ${maxValue.toLocaleString()} over ${points.length} months`
        }
        onMouseMove={(e) => handleMove(e.clientX)}
        onMouseLeave={() => setActiveIndex(null)}
      >
        <defs>
          <linearGradient id={`spark-fill-${id}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.28" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <polygon fill={`url(#spark-fill-${id})`} points={areaPoints} />
        <polyline
          fill="none"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={strokeClassName}
          points={polyPoints}
        />
        {activeCoord && (
          <>
            <line
              x1={activeCoord.x}
              x2={activeCoord.x}
              y1={PADDING}
              y2={HEIGHT - PADDING}
              className="stroke-[color:var(--border-default)]"
              strokeDasharray="2 2"
              strokeWidth="1"
            />
            <circle
              cx={activeCoord.x}
              cy={activeCoord.y}
              r={3.5}
              className="fill-[color:var(--surface-1)] stroke-[color:var(--brand)]"
              strokeWidth="2"
            />
          </>
        )}
      </svg>
      {active && activeCoord && (
        <div
          role="tooltip"
          style={{
            left: `${(activeCoord.x / WIDTH) * 100}%`,
            transform: 'translate(-50%, -100%)',
          }}
          className="pointer-events-none absolute -top-1 whitespace-nowrap rounded-[var(--radius-sm)] bg-[color:var(--color-neutral-900)] px-2 py-1 text-[11px] font-medium text-[color:var(--color-neutral-50)] shadow-[var(--shadow-md)]"
        >
          <div className="opacity-70">{active.month}</div>
          <div>{valueFormatter(active)}</div>
        </div>
      )}
    </div>
  )
}

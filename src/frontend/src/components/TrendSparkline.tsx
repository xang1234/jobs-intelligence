import type { TrendPoint } from '@/types/api'

interface TrendSparklineProps {
  points: TrendPoint[]
  strokeClassName?: string
}

export default function TrendSparkline({
  points,
  strokeClassName = 'stroke-[color:var(--brand)]',
}: TrendSparklineProps) {
  if (points.length === 0) return null

  const width = 220
  const height = 76
  const padding = 6
  const counts = points.map((point) => point.job_count)
  const maxValue = Math.max(...counts, 1)
  const minValue = Math.min(...counts, 0)
  const range = maxValue - minValue || 1

  const coordinates = points.map((point, index) => {
    const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2)
    const y = height - padding - ((point.job_count - minValue) / range) * (height - padding * 2)
    return `${x},${y}`
  })

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-20 w-full">
      <defs>
        <linearGradient id="spark-fill" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="rgba(15, 118, 110, 0.28)" />
          <stop offset="100%" stopColor="rgba(15, 118, 110, 0.02)" />
        </linearGradient>
      </defs>
      <polyline
        fill="none"
        strokeWidth="3"
        className={strokeClassName}
        points={coordinates.join(' ')}
      />
      <polygon
        fill="url(#spark-fill)"
        points={[
          `${padding},${height - padding}`,
          ...coordinates,
          `${width - padding},${height - padding}`,
        ].join(' ')}
      />
    </svg>
  )
}

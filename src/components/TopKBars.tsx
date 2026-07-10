import type { ClassScore } from '../lib/mobilenet'

interface Props {
  scores: ClassScore[] | null
  /** Class index to tint (e.g. the target class in a targeted attack). */
  highlightIndex?: number
}

/** Top-k class list with confidence bars — the 1000-class analog of ConfidenceBars. */
export default function TopKBars({ scores, highlightIndex }: Props) {
  return (
    <div className="flex w-full flex-col gap-1.5" role="list" aria-label="Top predictions">
      {(scores ?? []).map((s, rank) => {
        const isTop = rank === 0
        const isTarget = s.index === highlightIndex
        const barColor = isTarget ? 'bg-red-400' : isTop ? 'bg-accent' : 'bg-line'
        const textColor = isTarget ? 'text-red-400' : isTop ? 'text-accent' : 'text-muted'
        return (
          <div key={s.index} role="listitem" className="flex items-center gap-2 text-sm">
            <span className={`w-32 truncate text-right text-xs ${textColor}`} title={s.label}>
              {s.label}
            </span>
            <div className="bg-panel-2 h-4 flex-1 overflow-hidden rounded-sm">
              <div
                className={`h-full rounded-sm transition-[width] duration-150 ease-out ${barColor}`}
                style={{ width: `${(s.prob * 100).toFixed(1)}%` }}
              />
            </div>
            <span className={`w-12 text-right font-mono text-xs ${textColor}`}>
              {(s.prob * 100).toFixed(1)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

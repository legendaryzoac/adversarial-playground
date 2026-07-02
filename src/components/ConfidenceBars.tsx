interface Props {
  /** Softmax probabilities, length 10, or null when there's no prediction. */
  probs: number[] | null
}

/** Horizontal bar chart of the model's confidence for each digit class. */
export default function ConfidenceBars({ probs }: Props) {
  const top = probs ? probs.indexOf(Math.max(...probs)) : -1
  return (
    <div className="flex w-full flex-col gap-1.5" role="list" aria-label="Class confidences">
      {Array.from({ length: 10 }, (_, digit) => {
        const p = probs?.[digit] ?? 0
        const isTop = digit === top
        return (
          <div key={digit} role="listitem" className="flex items-center gap-2 text-sm">
            <span className={`w-4 text-right font-mono ${isTop ? 'text-accent font-bold' : 'text-muted'}`}>
              {digit}
            </span>
            <div className="bg-panel-2 h-4 flex-1 overflow-hidden rounded-sm">
              <div
                className={`h-full rounded-sm transition-[width] duration-150 ease-out ${
                  isTop ? 'bg-accent' : 'bg-line'
                }`}
                style={{ width: `${(p * 100).toFixed(1)}%` }}
              />
            </div>
            <span className={`w-12 text-right font-mono text-xs ${isTop ? 'text-accent' : 'text-muted'}`}>
              {probs ? `${(p * 100).toFixed(1)}%` : '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

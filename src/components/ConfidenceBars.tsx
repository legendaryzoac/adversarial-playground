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
            <span className={`w-4 text-right font-mono ${isTop ? 'font-bold text-cyan-300' : 'text-zinc-500'}`}>
              {digit}
            </span>
            <div className="h-4 flex-1 overflow-hidden rounded-sm bg-zinc-800">
              <div
                className={`h-full rounded-sm transition-[width] duration-150 ease-out ${
                  isTop ? 'bg-cyan-400' : 'bg-zinc-600'
                }`}
                style={{ width: `${(p * 100).toFixed(1)}%` }}
              />
            </div>
            <span className={`w-12 text-right font-mono text-xs ${isTop ? 'text-cyan-300' : 'text-zinc-500'}`}>
              {probs ? `${(p * 100).toFixed(1)}%` : '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

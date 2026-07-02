import { useEffect, useRef } from 'react'

export const CANVAS_SIZE = 280
const STROKE_WIDTH = 20

interface Props {
  /** Fires (rAF-throttled) while drawing; null means the canvas was cleared. */
  onDraw: (canvas: HTMLCanvasElement | null) => void
}

/**
 * 280x280 white-on-black drawing pad (10x the MNIST resolution, same palette
 * as the training data).
 */
export default function DrawingCanvas({ onDraw }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isDrawing = useRef(false)
  const lastPoint = useRef<{ x: number; y: number } | null>(null)
  const rafPending = useRef(false)

  useEffect(() => {
    clearCanvas(canvasRef.current!)
  }, [])

  const emit = () => {
    if (rafPending.current) return
    rafPending.current = true
    requestAnimationFrame(() => {
      rafPending.current = false
      if (canvasRef.current) onDraw(canvasRef.current)
    })
  }

  const toCanvasCoords = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (CANVAS_SIZE / rect.width),
      y: (e.clientY - rect.top) * (CANVAS_SIZE / rect.height),
    }
  }

  const handlePointerDown = (e: React.PointerEvent) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId)
    } catch {
      // Synthetic events (tests) have no active pointer to capture.
    }
    isDrawing.current = true
    const p = toCanvasCoords(e)
    lastPoint.current = p
    const ctx = canvasRef.current!.getContext('2d')!
    ctx.fillStyle = 'white'
    ctx.beginPath()
    ctx.arc(p.x, p.y, STROKE_WIDTH / 2, 0, Math.PI * 2)
    ctx.fill()
    emit()
  }

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDrawing.current || !lastPoint.current) return
    const p = toCanvasCoords(e)
    const ctx = canvasRef.current!.getContext('2d')!
    ctx.strokeStyle = 'white'
    ctx.lineWidth = STROKE_WIDTH
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(lastPoint.current.x, lastPoint.current.y)
    ctx.lineTo(p.x, p.y)
    ctx.stroke()
    lastPoint.current = p
    emit()
  }

  const handlePointerUp = () => {
    isDrawing.current = false
    lastPoint.current = null
  }

  const handleClear = () => {
    clearCanvas(canvasRef.current!)
    onDraw(null)
  }

  return (
    <div className="flex flex-col items-center gap-3">
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        className="border-line w-full max-w-[280px] cursor-crosshair touch-none rounded-lg border bg-black shadow-inner"
        aria-label="Drawing canvas — draw a digit from 0 to 9"
      />
      <button
        onClick={handleClear}
        className="border-line text-muted font-display hover:border-accent hover:text-fg rounded-md border px-4 py-1.5 text-sm transition-colors"
      >
        Clear
      </button>
    </div>
  )
}

function clearCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
}

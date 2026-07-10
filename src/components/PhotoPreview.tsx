import { useEffect, useRef } from 'react'
import { MOBILENET_SIZE } from '../lib/mobilenet'

interface Props {
  /** 224×224×3 pixels. For 'image': values in [-1,1]. For 'perturbation': signed delta. */
  pixels: Float32Array | null
  kind: 'image' | 'perturbation'
  /** For 'perturbation': the ε used, to normalize the amplified view. */
  epsilon?: number
  label?: string
  className?: string
}

/**
 * Renders a 224×224 RGB tensor. 'image' maps [-1,1] → [0,255]; 'perturbation'
 * maps the signed delta to gray-centered color, amplified by ~0.5/ε so a tiny
 * ±ε signal fills the visible range.
 */
export default function PhotoPreview({ pixels, kind, epsilon = 0.02, label, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const image = ctx.createImageData(MOBILENET_SIZE, MOBILENET_SIZE)
    const amp = epsilon > 0 ? 0.5 / epsilon : 0
    for (let i = 0; i < MOBILENET_SIZE * MOBILENET_SIZE; i++) {
      for (let c = 0; c < 3; c++) {
        const raw = pixels ? pixels[i * 3 + c] : 0
        const norm = kind === 'image' ? (raw + 1) / 2 : 0.5 + raw * amp
        image.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(norm * 255)))
      }
      image.data[i * 4 + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }, [pixels, kind, epsilon])

  return (
    <figure className="flex flex-col items-center gap-1.5">
      <canvas
        ref={canvasRef}
        width={MOBILENET_SIZE}
        height={MOBILENET_SIZE}
        className={`border-line rounded border bg-black ${className ?? 'size-32'}`}
        aria-label={label ?? 'Image'}
      />
      {label && <figcaption className="text-muted max-w-32 text-center font-mono text-[11px] leading-tight">{label}</figcaption>}
    </figure>
  )
}

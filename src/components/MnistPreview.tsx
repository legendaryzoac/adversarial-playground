import { useEffect, useRef } from 'react'
import { IMAGE_SIZE } from '../lib/mnist'

interface Props {
  /** 28x28 grayscale values, or null for an empty frame. */
  pixels: Float32Array | null
  label?: string
  /**
   * Multiplies values before display. 1 for real images; the perturbation
   * view uses ~0.45/ε so a tiny ±ε signal fills the visible range.
   */
  amplify?: number
  /**
   * Signed rendering for perturbations: 0 maps to mid-gray, negative values
   * darken, positive values lighten.
   */
  signed?: boolean
  /** Size override, e.g. "size-24" (defaults to size-20). */
  className?: string
}

/**
 * Renders a 28x28 model input as a chunky pixelated thumbnail — used for the
 * "what the model sees" view and the perturbation / adversarial panels.
 */
export default function MnistPreview({ pixels, label, amplify = 1, signed = false, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const image = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE)
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
      const raw = pixels ? pixels[i] * amplify + (signed ? 0.5 : 0) : signed ? 0.5 : 0
      const v = Math.max(0, Math.min(255, Math.round(raw * 255)))
      image.data[i * 4] = v
      image.data[i * 4 + 1] = v
      image.data[i * 4 + 2] = v
      image.data[i * 4 + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }, [pixels, amplify, signed])

  return (
    <figure className="flex flex-col items-center gap-1.5">
      <canvas
        ref={canvasRef}
        width={IMAGE_SIZE}
        height={IMAGE_SIZE}
        className={`border-line rounded border bg-black [image-rendering:pixelated] ${className ?? 'size-20'}`}
        aria-label={label ?? 'Model input'}
      />
      {label && <figcaption className="text-muted text-center font-mono text-xs">{label}</figcaption>}
    </figure>
  )
}

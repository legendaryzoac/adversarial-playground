import { useEffect, useRef } from 'react'
import { IMAGE_SIZE } from '../lib/mnist'

interface Props {
  /** 28x28 grayscale intensities in [0, 1], or null for an empty frame. */
  pixels: Float32Array | null
  label?: string
  /**
   * Multiplies intensities before display. 1 for real images; milestone 2
   * uses ~20x to make the (tiny) adversarial perturbation visible.
   */
  amplify?: number
}

/**
 * Renders a 28x28 model input as a chunky pixelated thumbnail. Used for the
 * "what the model sees" view now, and for the perturbation / adversarial
 * panels in milestone 2.
 */
export default function MnistPreview({ pixels, label, amplify = 1 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const image = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE)
    for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
      const v = pixels ? Math.max(0, Math.min(255, Math.round(pixels[i] * amplify * 255))) : 0
      image.data[i * 4] = v
      image.data[i * 4 + 1] = v
      image.data[i * 4 + 2] = v
      image.data[i * 4 + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }, [pixels, amplify])

  return (
    <figure className="flex flex-col items-center gap-1.5">
      <canvas
        ref={canvasRef}
        width={IMAGE_SIZE}
        height={IMAGE_SIZE}
        className="border-line size-20 rounded border bg-black [image-rendering:pixelated]"
        aria-label={label ?? 'Model input'}
      />
      {label && <figcaption className="text-muted text-xs">{label}</figcaption>}
    </figure>
  )
}

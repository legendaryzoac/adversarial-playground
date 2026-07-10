// MobileNet v1 (1.0, 224) — a real 1000-class ImageNet classifier, loaded as a
// LayersModel so we can compute input gradients for adversarial attacks (a
// GraphModel from tfhub could classify but not be attacked). Weights come from
// Google's CDN (~16 MB), so they cost us no S3 egress. The model outputs
// SOFTMAX PROBABILITIES and expects RGB pixels in [-1, 1].
import * as tf from '@tensorflow/tfjs'
import type { AttackSpec } from './attacks'
import { IMAGENET_CLASSES } from './imagenet-classes'

const MODEL_URL =
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'

export const MOBILENET_SIZE = 224
export const MOBILENET_CLASSES = 1000

/** Attack-engine spec for MobileNet. */
export const MOBILENET_SPEC: AttackSpec = {
  inputShape: [MOBILENET_SIZE, MOBILENET_SIZE, 3],
  numClasses: MOBILENET_CLASSES,
  clip: [-1, 1],
  outputsLogits: false,
}

/** Human-readable label for a class index (first synonym only, for brevity). */
export function classLabel(index: number): string {
  const full = IMAGENET_CLASSES[index] ?? `class ${index}`
  return full.split(',')[0]
}

/** All 1000 labels as an array, index-aligned with the model's outputs. */
export const LABELS: string[] = Array.from({ length: MOBILENET_CLASSES }, (_, i) => classLabel(i))

export interface ClassScore {
  index: number
  label: string
  prob: number
}

let modelPromise: Promise<tf.LayersModel> | null = null

/**
 * Lazy-loads MobileNet once (only when the user enters photo mode).
 * @param onProgress fraction in [0,1] during weight download.
 */
export function loadMobilenet(onProgress?: (fraction: number) => void): Promise<tf.LayersModel> {
  modelPromise ??= (async () => {
    const model = await tf.loadLayersModel(MODEL_URL, {
      onProgress: onProgress ? (f) => onProgress(f) : undefined,
    })
    // Warm up BOTH passes here, behind the loading bar: the forward shaders and
    // — more importantly — the backward shaders. Compiling the gradient path is
    // expensive (~seconds) and would otherwise stall the user's first attack.
    tf.tidy(() => {
      const x = tf.zeros([1, MOBILENET_SIZE, MOBILENET_SIZE, 3])
      model.predict(x)
      const grad = tf.grad((inp: tf.Tensor) =>
        (model.apply(inp) as tf.Tensor2D).slice([0, 0], [1, 1]).asScalar(),
      )(x)
      grad.dataSync() // force the backward pass to actually execute + compile
    })
    return model
  })()
  return modelPromise
}

/**
 * Center-crops an image to square, resizes to 224×224, and returns pixels
 * normalized to [-1, 1] (RGB, row-major) — the model's expected input.
 */
export function imageToPixels(source: CanvasImageSource, width: number, height: number): Float32Array {
  const side = Math.min(width, height)
  const sx = (width - side) / 2
  const sy = (height - side) / 2

  const canvas = document.createElement('canvas')
  canvas.width = MOBILENET_SIZE
  canvas.height = MOBILENET_SIZE
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(source, sx, sy, side, side, 0, 0, MOBILENET_SIZE, MOBILENET_SIZE)

  const data = ctx.getImageData(0, 0, MOBILENET_SIZE, MOBILENET_SIZE).data
  const pixels = new Float32Array(MOBILENET_SIZE * MOBILENET_SIZE * 3)
  for (let i = 0; i < MOBILENET_SIZE * MOBILENET_SIZE; i++) {
    // RGBA (4 ch) -> RGB (3 ch), 0..255 -> -1..1
    pixels[i * 3] = data[i * 4] / 127.5 - 1
    pixels[i * 3 + 1] = data[i * 4 + 1] / 127.5 - 1
    pixels[i * 3 + 2] = data[i * 4 + 2] / 127.5 - 1
  }
  return pixels
}

/** Top-k entries of a raw probability vector (length 1000). */
export function topKFromProbs(probs: ArrayLike<number>, k = 5): ClassScore[] {
  return Array.from(probs)
    .map((prob, index) => ({ index, prob, label: classLabel(index) }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k)
}

/** Top-k predictions for a normalized [-1,1] RGB image. */
export function predictTopK(model: tf.LayersModel, pixels: Float32Array, k = 5): ClassScore[] {
  const probs = tf.tidy(() => {
    const input = tf.tensor4d(pixels, [1, MOBILENET_SIZE, MOBILENET_SIZE, 3])
    return (model.predict(input) as tf.Tensor2D).dataSync() as Float32Array
  })
  return topKFromProbs(probs, k)
}

/** First-synonym label → class index, for the targeted-attack picker. */
export const LABEL_TO_INDEX: Map<string, number> = new Map(
  LABELS.map((label, index) => [label, index]).filter(
    ([, index]) => LABELS.indexOf(LABELS[index as number]) === index,
  ) as [string, number][],
)

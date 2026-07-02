import * as tf from '@tensorflow/tfjs'

export const NUM_CLASSES = 10
export const IMAGE_SIZE = 28

export interface Prediction {
  /** Softmax probabilities, length 10. */
  probs: number[]
  /** argmax of probs. */
  label: number
}

let modelPromise: Promise<tf.LayersModel> | null = null

/**
 * Loads the classifier once and caches it. The model outputs raw LOGITS
 * (no softmax layer) — softmax is applied here in {@link predict}. Attacks
 * (milestone 2) rely on having logits available to compute
 * grad(softmaxCrossEntropy) w.r.t. the input.
 */
export function loadModel(): Promise<tf.LayersModel> {
  modelPromise ??= (async () => {
    const model = await tf.loadLayersModel(`${import.meta.env.BASE_URL}model/model.json`)
    // Warm up so the first real prediction doesn't pay shader-compile cost.
    tf.tidy(() => model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])))
    return model
  })()
  return modelPromise
}

/** Runs the classifier on a 28x28 grayscale image (values in [0, 1]). */
export function predict(model: tf.LayersModel, pixels: Float32Array): Prediction {
  const probs = tf.tidy(() => {
    const input = tf.tensor4d(pixels, [1, IMAGE_SIZE, IMAGE_SIZE, 1])
    const logits = model.predict(input) as tf.Tensor2D
    return tf.softmax(logits).dataSync()
  })
  const probsArr = Array.from(probs)
  let label = 0
  for (let i = 1; i < probsArr.length; i++) {
    if (probsArr[i] > probsArr[label]) label = i
  }
  return { probs: probsArr, label }
}

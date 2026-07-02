// Adversarial attacks — milestone 2.
//
// The pieces the rest of the app already provides for this:
//  - the model outputs LOGITS (softmax lives in mnist.ts), so the loss here
//    is tf.losses.softmaxCrossEntropy(oneHot(label), logits)
//  - predictions flow through App state as plain Float32Array pixels, so an
//    attack just maps pixels -> perturbed pixels and re-renders/re-predicts
import * as tf from '@tensorflow/tfjs'

export type AttackKind = 'fgsm' | 'pgd'

export interface AttackConfig {
  /** Max L∞ perturbation, in [0, 1] pixel units. Interesting range ~0.05–0.3. */
  epsilon: number
  /** PGD only: number of gradient steps. */
  steps?: number
  /** PGD only: step size per iteration (defaults to epsilon / steps * 2.5). */
  stepSize?: number
  /** If set, targeted attack: minimize loss toward this class instead. */
  targetClass?: number
}

export interface AttackResult {
  /** Perturbed image, same shape as the input, clipped to [0, 1]. */
  adversarial: Float32Array
  /** adversarial - original (for the amplified perturbation view). */
  perturbation: Float32Array
}

/**
 * Fast Gradient Sign Method (Goodfellow et al. 2014):
 *   x_adv = clip(x + ε · sign(∇ₓ L(f(x), y)), 0, 1)
 *
 * Implementation sketch:
 *   const lossFn = (x: tf.Tensor) => tf.losses.softmaxCrossEntropy(
 *     tf.oneHot([label], 10), model.predict(x) as tf.Tensor2D)
 *   const grad = tf.grad(lossFn)(input)
 *   const adv = input.add(grad.sign().mul(epsilon)).clipByValue(0, 1)
 */
export function fgsm(
  _model: tf.LayersModel,
  _pixels: Float32Array,
  _label: number,
  _config: AttackConfig,
): AttackResult {
  throw new Error('Not implemented until milestone 2')
}

/**
 * Projected Gradient Descent (Madry et al. 2017): iterated FGSM with the
 * perturbation projected back into the ε-ball after each step.
 */
export function pgd(
  _model: tf.LayersModel,
  _pixels: Float32Array,
  _label: number,
  _config: AttackConfig,
): AttackResult {
  throw new Error('Not implemented until milestone 2')
}

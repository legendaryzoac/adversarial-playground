// Adversarial attacks, computed in-browser with real input gradients.
//
// The model outputs LOGITS (softmax lives in mnist.ts), so the loss here is
// softmaxCrossEntropy(oneHot(label), logits) — numerically clean to
// differentiate. Attacks map pixels -> perturbed pixels; App re-predicts on
// the result.
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
 * One gradient step in the direction that most increases the loss for the
 * model's current prediction. `label` should be the class to attack away
 * from (normally the model's own prediction).
 */
export function fgsm(
  model: tf.LayersModel,
  pixels: Float32Array,
  label: number,
  config: AttackConfig,
): AttackResult {
  return tf.tidy(() => {
    const input = tf.tensor4d(pixels, [1, 28, 28, 1])
    const oneHot = tf.oneHot([label], 10)
    const lossFromInput = (x: tf.Tensor): tf.Scalar => {
      // model.apply (not .predict) so ops are recorded for tf.grad.
      const logits = model.apply(x) as tf.Tensor2D
      return tf.losses.softmaxCrossEntropy(oneHot, logits).asScalar()
    }
    const grad = tf.grad(lossFromInput)(input)
    const adversarial = input.add(grad.sign().mul(config.epsilon)).clipByValue(0, 1)
    // Report the *effective* perturbation (post-clip), not ε·sign itself.
    const perturbation = adversarial.sub(input)
    return {
      adversarial: adversarial.dataSync() as Float32Array,
      perturbation: perturbation.dataSync() as Float32Array,
    }
  })
}

/**
 * Projected Gradient Descent (Madry et al. 2017): iterated FGSM with the
 * perturbation projected back into the ε-ball after each step. Milestone 3.
 */
export function pgd(
  _model: tf.LayersModel,
  _pixels: Float32Array,
  _label: number,
  _config: AttackConfig,
): AttackResult {
  throw new Error('Not implemented until milestone 3')
}

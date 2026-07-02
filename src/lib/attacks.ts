// Adversarial attacks, computed in-browser with real input gradients.
//
// The model outputs LOGITS (softmax lives in mnist.ts), so the loss here is
// softmaxCrossEntropy(oneHot(label), logits) — numerically clean to
// differentiate. Attacks map pixels -> perturbed pixels; the frame's own
// probs come from the same forward pass so the UI can plot them per step.
import * as tf from '@tensorflow/tfjs'
import { IMAGE_SIZE, NUM_CLASSES } from './mnist'

export type AttackKind = 'fgsm' | 'pgd'

export interface AttackConfig {
  kind: AttackKind
  /** Max L∞ perturbation, in [0, 1] pixel units. Interesting range ~0.05–0.3. */
  epsilon: number
  /** PGD only: number of gradient steps (default 10). */
  steps?: number
  /** PGD only: per-step size (default 2.5·ε/steps, the Madry et al. rule). */
  stepSize?: number
  /** If set, targeted attack: drive the prediction toward this class. */
  targetClass?: number
}

export interface AttackFrame {
  /** Perturbed image, clipped to [0, 1]. */
  adversarial: Float32Array
  /** adversarial - original (for the amplified perturbation view). */
  perturbation: Float32Array
  /** Softmax probabilities at this step. */
  probs: number[]
  /** argmax of probs. */
  label: number
}

function argmax(a: ArrayLike<number>): number {
  let m = 0
  for (let i = 1; i < a.length; i++) if (a[i] > a[m]) m = i
  return m
}

/**
 * Runs FGSM or PGD and returns one frame per gradient step (FGSM = 1 frame,
 * PGD = `steps` frames) so the UI can animate the iterations.
 *
 * Untargeted (no targetClass): ascend the loss for `sourceLabel`, pushing the
 * model away from the right answer —  x ← x + α·sign(∇ₓL).
 * Targeted: descend the loss toward `targetClass`, pulling the model to a
 * chosen wrong answer —  x ← x − α·sign(∇ₓL).
 *
 * PGD (Madry et al. 2017) is iterated FGSM with the running perturbation
 * projected back into the L∞ ε-ball after every step. Starts from the clean
 * image (no random init) so the animation reads as a smooth, reproducible
 * descent.
 */
export function runAttack(
  model: tf.LayersModel,
  pixels: Float32Array,
  sourceLabel: number,
  config: AttackConfig,
): AttackFrame[] {
  const targeted = config.targetClass !== undefined
  const attackLabel = targeted ? config.targetClass! : sourceLabel
  const direction = targeted ? -1 : 1
  const steps = config.kind === 'fgsm' ? 1 : config.steps ?? 10
  const stepSize =
    config.kind === 'fgsm' ? config.epsilon : config.stepSize ?? (2.5 * config.epsilon) / steps

  const frames: AttackFrame[] = []
  let current = pixels
  for (let i = 0; i < steps; i++) {
    const out = tf.tidy(() => {
      const orig = tf.tensor4d(pixels, [1, IMAGE_SIZE, IMAGE_SIZE, 1])
      const adv = tf.tensor4d(current, [1, IMAGE_SIZE, IMAGE_SIZE, 1])
      const oneHot = tf.oneHot([attackLabel], NUM_CLASSES)
      const lossFromInput = (x: tf.Tensor): tf.Scalar =>
        tf.losses.softmaxCrossEntropy(oneHot, model.apply(x) as tf.Tensor2D).asScalar()
      const grad = tf.grad(lossFromInput)(adv)
      const stepped = adv.add(grad.sign().mul(stepSize * direction))
      // Project back into the ε-ball around the original, then to valid pixels.
      const delta = stepped.sub(orig).clipByValue(-config.epsilon, config.epsilon)
      const next = orig.add(delta).clipByValue(0, 1)
      return {
        probs: Array.from(tf.softmax(model.apply(next) as tf.Tensor2D).dataSync() as Float32Array),
        perturbation: next.sub(orig).dataSync() as Float32Array,
        adversarial: next.dataSync() as Float32Array,
      }
    })
    current = out.adversarial
    frames.push({
      adversarial: out.adversarial,
      perturbation: out.perturbation,
      probs: out.probs,
      label: argmax(out.probs),
    })
  }
  return frames
}

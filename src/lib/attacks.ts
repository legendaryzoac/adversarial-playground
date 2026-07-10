// Adversarial attacks, computed in-browser with real input gradients.
//
// Works for any LayersModel that records ops through model.apply(), across two
// very different classifiers:
//   - MNIST  : [28,28,1], 10 classes, logits output, pixels in [0,1]
//   - MobileNet: [224,224,3], 1000 classes, SOFTMAX output, pixels in [-1,1]
// so the model/pixel specifics live in an AttackSpec rather than being baked in.
import * as tf from '@tensorflow/tfjs'

export type AttackKind = 'fgsm' | 'pgd'

/** Describes the input/output space of the model under attack. */
export interface AttackSpec {
  /** [height, width, channels]. */
  inputShape: [number, number, number]
  numClasses: number
  /** Valid pixel range: [0,1] for MNIST, [-1,1] for MobileNet. */
  clip: [number, number]
  /** true → model emits logits (apply softmax); false → already probabilities. */
  outputsLogits: boolean
}

export interface AttackConfig {
  kind: AttackKind
  /** Max L∞ perturbation, in the model's pixel units. */
  epsilon: number
  /** PGD only: number of gradient steps (default 10). */
  steps?: number
  /** PGD only: per-step size (default 2.5·ε/steps, the Madry et al. rule). */
  stepSize?: number
  /** If set, targeted attack: drive the prediction toward this class. */
  targetClass?: number
}

export interface AttackFrame {
  /** Perturbed image, clipped to the model's valid range. */
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
 * Categorical cross-entropy loss on `label`, differentiable w.r.t. the input,
 * whether the model emits logits or probabilities. Equivalent either way to
 * -log(p_label); we just avoid double-softmaxing a probability output.
 */
function classLoss(
  model: tf.LayersModel,
  x: tf.Tensor,
  oneHot: tf.Tensor2D,
  spec: AttackSpec,
): tf.Scalar {
  const raw = model.apply(x) as tf.Tensor2D
  const probs = spec.outputsLogits ? tf.softmax(raw) : raw
  const picked = probs.mul(oneHot).sum().add(1e-7)
  return tf.neg(tf.log(picked)).asScalar()
}

interface StepParams {
  attackLabel: number
  direction: 1 | -1
  epsilon: number
  stepSize: number
}

/** One gradient step; returns the new adversarial state and its prediction. */
function attackStep(
  model: tf.LayersModel,
  origPixels: Float32Array,
  currentPixels: Float32Array,
  spec: AttackSpec,
  p: StepParams,
): AttackFrame {
  const shape: [number, number, number, number] = [1, ...spec.inputShape]
  const out = tf.tidy(() => {
    const orig = tf.tensor4d(origPixels, shape)
    const adv = tf.tensor4d(currentPixels, shape)
    const oneHot = tf.oneHot([p.attackLabel], spec.numClasses) as tf.Tensor2D
    const grad = tf.grad((x: tf.Tensor) => classLoss(model, x, oneHot, spec))(adv)
    const stepped = adv.add(grad.sign().mul(p.stepSize * p.direction))
    // Project back into the ε-ball around the original, then to valid pixels.
    const delta = stepped.sub(orig).clipByValue(-p.epsilon, p.epsilon)
    const next = orig.add(delta).clipByValue(spec.clip[0], spec.clip[1])
    const raw = model.apply(next) as tf.Tensor2D
    const probs = spec.outputsLogits ? tf.softmax(raw) : raw
    return {
      probs: Array.from(probs.dataSync() as Float32Array),
      perturbation: next.sub(orig).dataSync() as Float32Array,
      adversarial: next.dataSync() as Float32Array,
    }
  })
  return { ...out, label: argmax(out.probs) }
}

function planSteps(config: AttackConfig): { steps: number; stepSize: number } {
  const steps = config.kind === 'fgsm' ? 1 : config.steps ?? 10
  const stepSize =
    config.kind === 'fgsm' ? config.epsilon : config.stepSize ?? (2.5 * config.epsilon) / steps
  return { steps, stepSize }
}

/** Resolves the per-run attack parameters shared by both runners. */
function planParams(config: AttackConfig, sourceLabel: number): StepParams & { steps: number } {
  const { steps, stepSize } = planSteps(config)
  const targeted = config.targetClass !== undefined
  return {
    attackLabel: targeted ? config.targetClass! : sourceLabel,
    direction: (targeted ? -1 : 1) as 1 | -1,
    epsilon: config.epsilon,
    stepSize,
    steps,
  }
}

/**
 * Runs FGSM or PGD synchronously and returns one frame per gradient step.
 * Use this for small/fast models (MNIST). For heavy models that would block
 * the UI, use {@link runAttackProgressive}.
 *
 * Untargeted (no targetClass): ascend the loss for `sourceLabel`, pushing the
 * model away from the right answer. Targeted: descend the loss toward
 * `targetClass`. PGD (Madry et al. 2017) projects the running perturbation
 * back into the L∞ ε-ball each step, starting from the clean image.
 */
export function runAttack(
  model: tf.LayersModel,
  pixels: Float32Array,
  sourceLabel: number,
  config: AttackConfig,
  spec: AttackSpec,
): AttackFrame[] {
  const { steps, ...params } = planParams(config, sourceLabel)
  const frames: AttackFrame[] = []
  let current = pixels
  for (let i = 0; i < steps; i++) {
    const frame = attackStep(model, pixels, current, spec, params)
    current = frame.adversarial
    frames.push(frame)
  }
  return frames
}

/**
 * Async variant that yields to the event loop between steps and reports each
 * frame as it's computed — so a heavy model (MobileNet) animates its PGD
 * descent live instead of freezing the tab. `shouldContinue` lets the caller
 * abort a stale run (e.g. the user changed the image).
 */
export async function runAttackProgressive(
  model: tf.LayersModel,
  pixels: Float32Array,
  sourceLabel: number,
  config: AttackConfig,
  spec: AttackSpec,
  onFrame: (frame: AttackFrame, index: number, total: number) => void,
  shouldContinue: () => boolean = () => true,
): Promise<AttackFrame[]> {
  const { steps, ...params } = planParams(config, sourceLabel)
  const frames: AttackFrame[] = []
  let current = pixels
  for (let i = 0; i < steps; i++) {
    // Yield to the event loop so React can paint the previous frame. We use
    // setTimeout, not rAF/tf.nextFrame — rAF is paused in backgrounded/hidden
    // tabs, which would hang the whole attack.
    await new Promise((resolve) => setTimeout(resolve, 24))
    if (!shouldContinue()) break
    const frame = attackStep(model, pixels, current, spec, params)
    current = frame.adversarial
    frames.push(frame)
    onFrame(frame, i, steps)
  }
  return frames
}

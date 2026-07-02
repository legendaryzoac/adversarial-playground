import { useCallback, useEffect, useMemo, useState } from 'react'
import type * as tf from '@tensorflow/tfjs'
import SiteNav from './components/SiteNav'
import SiteFooter from './components/SiteFooter'
import DrawingCanvas from './components/DrawingCanvas'
import ConfidenceBars from './components/ConfidenceBars'
import MnistPreview from './components/MnistPreview'
import { loadModel, predict, type Prediction } from './lib/mnist'
import { canvasToMnist } from './lib/preprocess'
import { runAttack, type AttackKind } from './lib/attacks'

type ModelStatus = 'loading' | 'ready' | 'error'

const PGD_STEPS = 10
/** ms between animated PGD frames. */
const FRAME_MS = 90

interface InputState {
  /** The 28x28 input the model actually saw — attacks start from this. */
  pixels: Float32Array
  prediction: Prediction
}

export default function App() {
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [status, setStatus] = useState<ModelStatus>('loading')
  const [input, setInput] = useState<InputState | null>(null)

  // Attack controls.
  const [kind, setKind] = useState<AttackKind>('fgsm')
  const [epsilon, setEpsilon] = useState(0.15)
  const [targeted, setTargeted] = useState(false)
  const [targetClass, setTargetClass] = useState(0)

  // Animation state (PGD steps through frames; FGSM has a single frame).
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    loadModel()
      .then((m) => {
        setModel(m)
        setStatus('ready')
      })
      .catch((err) => {
        console.error('Failed to load model', err)
        setStatus('error')
      })
  }, [])

  const handleDraw = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      if (!canvas || !model) {
        setInput(null)
        return
      }
      const pixels = canvasToMnist(canvas)
      setInput(pixels ? { pixels, prediction: predict(model, pixels) } : null)
    },
    [model],
  )

  // The attack recomputes live on every input / control change. The model is
  // small enough that even 10 PGD gradient passes are a few ms on WebGL.
  const frames = useMemo(() => {
    if (!model || !input) return null
    return runAttack(model, input.pixels, input.prediction.label, {
      kind,
      epsilon,
      steps: PGD_STEPS,
      targetClass: targeted ? targetClass : undefined,
    })
  }, [model, input, kind, epsilon, targeted, targetClass])

  // When a fresh attack is computed, jump to its final state — unless we're
  // mid-animation, in which case the play effect drives the frame index.
  useEffect(() => {
    if (frames && !playing) setFrame(frames.length - 1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frames])

  // Advance the animation one frame at a time while playing.
  useEffect(() => {
    if (!playing || !frames) return
    if (frame >= frames.length - 1) {
      setPlaying(false)
      return
    }
    const id = setTimeout(() => setFrame((f) => f + 1), FRAME_MS)
    return () => clearTimeout(id)
  }, [playing, frame, frames])

  const play = () => {
    setFrame(0)
    setPlaying(true)
  }
  const chooseKind = (k: AttackKind) => {
    setKind(k)
    if (k === 'pgd') play()
    else setPlaying(false)
  }
  const replayIfPgd = () => {
    if (kind === 'pgd') play()
  }

  const shown = frames ? frames[Math.min(frame, frames.length - 1)] : null
  const flipped = !!(input && shown && shown.label !== input.prediction.label)
  const success = targeted ? !!(shown && shown.label === targetClass) : flipped
  const statusText = !shown
    ? ''
    : targeted
      ? success
        ? `hit target "${targetClass}"`
        : `aiming for "${targetClass}" — raise ε${kind === 'fgsm' ? ' or use PGD' : ''}`
      : success
        ? `prediction flipped ${input!.prediction.label} → ${shown.label}`
        : `still correct — raise ε${kind === 'fgsm' ? ' or use PGD' : ''}`

  return (
    <div className="min-h-screen">
      <SiteNav />
      <div className="mx-auto max-w-5xl px-4 pt-[60px] pb-4">
        <header className="mt-10 mb-8">
          <p className="text-accent mb-2 font-mono text-xs tracking-widest uppercase">
            // AI Safety Lab
          </p>
          <h1 className="font-display text-3xl font-bold tracking-tight">
            Adversarial Playground
          </h1>
          <p className="text-muted mt-3 max-w-2xl text-sm leading-relaxed">
            A convolutional neural network runs live in your browser — no server, no API.
            Draw a digit, watch it classify, then attack it: a gradient-crafted perturbation
            you can barely see flips the prediction while the digit still looks the same to you.
          </p>
        </header>

        {status === 'error' && (
          <div className="mb-6 rounded-lg border border-red-800 bg-red-950/40 p-4 text-sm text-red-300">
            Failed to load the model. If you're running locally, make sure you've trained it
            first: <code className="font-mono">npm run train</code>
          </div>
        )}

        <main className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <section className="border-line bg-panel rounded-xl border p-5">
            <h2 className="font-display mb-1 text-sm font-semibold">1 · Draw a digit</h2>
            <p className="text-muted mb-4 text-xs">Any digit, 0–9. Prediction updates as you draw.</p>
            <DrawingCanvas onDraw={handleDraw} />
          </section>

          <section className="border-line bg-panel rounded-xl border p-5">
            <h2 className="font-display mb-1 text-sm font-semibold">2 · Model prediction</h2>
            <p className="text-muted mb-4 text-xs">
              {status === 'loading' ? 'Loading model…' : 'Softmax confidence per class.'}
            </p>
            <div className="mb-4 flex items-center justify-between gap-4">
              <div>
                <div className="text-accent font-mono text-5xl font-bold">
                  {input ? input.prediction.label : '·'}
                </div>
                <div className="text-muted mt-1 text-xs">
                  {input
                    ? `${(input.prediction.probs[input.prediction.label] * 100).toFixed(1)}% confident`
                    : 'waiting for a drawing'}
                </div>
              </div>
              <MnistPreview pixels={input?.pixels ?? null} label="model input (28×28)" />
            </div>
            <ConfidenceBars probs={input?.prediction.probs ?? null} />
          </section>

          <section className="border-line bg-panel rounded-xl border p-5 md:col-span-2 lg:col-span-1">
            <h2 className="font-display mb-3 text-sm font-semibold">3 · Attack the model</h2>

            <div className={input ? 'space-y-4' : 'space-y-4 opacity-40'}>
              {/* Attack type */}
              <div>
                <div className="text-muted mb-1.5 font-mono text-[10px] tracking-wide uppercase">
                  Method
                </div>
                <div className="border-line flex overflow-hidden rounded-md border text-sm">
                  {(['fgsm', 'pgd'] as const).map((k) => (
                    <button
                      key={k}
                      onClick={() => chooseKind(k)}
                      disabled={!input}
                      className={`flex-1 px-3 py-1.5 font-medium transition-colors ${
                        kind === k
                          ? 'bg-accent text-site'
                          : 'text-muted hover:text-fg'
                      }`}
                    >
                      {k === 'fgsm' ? 'FGSM' : `PGD ×${PGD_STEPS}`}
                    </button>
                  ))}
                </div>
              </div>

              {/* Targeted */}
              <div>
                <label className="flex cursor-pointer items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={targeted}
                    onChange={(e) => {
                      setTargeted(e.target.checked)
                      replayIfPgd()
                    }}
                    disabled={!input}
                    className="accent-accent"
                  />
                  <span className="text-fg font-medium">Targeted</span>
                  <span className="text-muted">— force a specific wrong answer</span>
                </label>
                {targeted && (
                  <div className="mt-2 flex items-center gap-2 text-xs">
                    <span className="text-muted">Make it say</span>
                    <select
                      value={targetClass}
                      onChange={(e) => {
                        setTargetClass(Number(e.target.value))
                        replayIfPgd()
                      }}
                      className="border-line bg-panel-2 text-fg rounded border px-2 py-1 font-mono"
                    >
                      {Array.from({ length: 10 }, (_, d) => (
                        <option key={d} value={d}>
                          {d}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {/* Epsilon */}
              <div>
                <div className="text-muted mb-1 flex justify-between text-xs">
                  <span>Perturbation strength ε</span>
                  <span className="text-fg font-mono">{epsilon.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={0.35}
                  step={0.005}
                  value={epsilon}
                  onChange={(e) => setEpsilon(Number(e.target.value))}
                  disabled={!input}
                  className="accent-accent w-full"
                  aria-label="Perturbation strength epsilon"
                />
                <div className="text-muted mt-1 flex justify-between font-mono text-[10px]">
                  <span>0 · no attack</span>
                  <span>0.35 · obvious noise</span>
                </div>
              </div>

              {kind === 'pgd' && (
                <button
                  onClick={play}
                  disabled={!input}
                  className="border-accent text-accent hover:bg-accent hover:text-site w-full rounded-md border py-1.5 text-sm font-medium transition-colors"
                >
                  ▶ Replay {PGD_STEPS} iterations
                </button>
              )}
            </div>

            {!input && (
              <p className="text-muted mt-4 text-xs italic">
                Draw a digit first — the attack runs automatically once there's something to attack.
              </p>
            )}

            <details className="border-line mt-5 border-t pt-3">
              <summary className="text-muted hover:text-fg cursor-pointer text-xs font-medium">
                How does this work?
              </summary>
              <p className="text-muted mt-2 text-xs leading-relaxed">
                <strong className="text-fg">FGSM</strong> (Goodfellow et al., 2014) takes the
                gradient of the model's loss with respect to <em>every pixel</em> — "which way
                should each pixel move to make the model most wrong?" — and steps once by ε in
                that direction: x′ = x + ε·sign(∇ₓL).{' '}
                <strong className="text-fg">PGD</strong> (Madry et al., 2017) repeats that in
                small steps, projecting back into the ε-ball each time — a stronger attack that
                often flips the label at an ε where FGSM can't.{' '}
                <strong className="text-fg">Targeted</strong> mode descends the loss toward a
                class you pick instead of just away from the truth. Every gradient is computed
                right here in your browser. This is why robustness matters for ML in the real
                world: stop signs, face recognition, content filters.
              </p>
            </details>
          </section>

          {input && shown && (
            <section className="border-line bg-panel rounded-xl border p-5 md:col-span-2 lg:col-span-3">
              <div className="mb-1 flex items-center gap-2">
                <h2 className="font-display text-sm font-semibold">4 · Result</h2>
                <span className="border-accent/60 text-accent rounded-full border px-2 py-0.5 font-mono text-[10px] font-medium tracking-wide uppercase">
                  {kind === 'fgsm' ? 'FGSM' : `PGD · step ${Math.min(frame, frames!.length - 1) + 1}/${frames!.length}`}
                </span>
                {targeted && (
                  <span className="border-line text-muted rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-wide uppercase">
                    targeted → {targetClass}
                  </span>
                )}
              </div>
              <p className="text-muted mb-5 text-xs">
                Left to right: what you drew, the adversarial perturbation (amplified so you can
                see it), and their sum — what the model was actually shown.
              </p>
              <div className="flex flex-wrap items-start justify-between gap-8">
                <div className="flex items-center gap-2 sm:gap-4">
                  <MnistPreview
                    pixels={input.pixels}
                    label={`original · "${input.prediction.label}"`}
                    className="size-20 sm:size-28"
                  />
                  <span className="text-muted -mt-6 font-mono text-xl">+</span>
                  <MnistPreview
                    pixels={shown.perturbation}
                    signed
                    amplify={epsilon > 0 ? 0.45 / epsilon : 0}
                    label="perturbation · amplified"
                    className="size-20 sm:size-28"
                  />
                  <span className="text-muted -mt-6 font-mono text-xl">=</span>
                  <MnistPreview
                    pixels={shown.adversarial}
                    label={`adversarial · "${shown.label}"`}
                    className="size-20 sm:size-28"
                  />
                </div>
                <div className="min-w-60 flex-1">
                  <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <span
                      className={`font-mono text-4xl font-bold ${success ? 'text-red-400' : 'text-accent'}`}
                    >
                      {shown.label}
                    </span>
                    <span className="text-muted text-xs">
                      {(shown.probs[shown.label] * 100).toFixed(1)}% confident
                    </span>
                    <span
                      className={`rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-wide uppercase ${
                        success
                          ? 'border-red-800 bg-red-950/40 text-red-400'
                          : 'border-line text-muted'
                      }`}
                    >
                      {statusText}
                    </span>
                  </div>
                  <ConfidenceBars probs={shown.probs} />
                </div>
              </div>
            </section>
          )}
        </main>

        <SiteFooter
          note={`Runs entirely client-side with TensorFlow.js · trained on MNIST · ${
            model ? `${model.countParams().toLocaleString()} parameters` : 'model loading…'
          }`}
        />
      </div>
    </div>
  )
}

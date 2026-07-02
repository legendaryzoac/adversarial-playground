import { useCallback, useEffect, useMemo, useState } from 'react'
import type * as tf from '@tensorflow/tfjs'
import SiteNav from './components/SiteNav'
import SiteFooter from './components/SiteFooter'
import DrawingCanvas from './components/DrawingCanvas'
import ConfidenceBars from './components/ConfidenceBars'
import MnistPreview from './components/MnistPreview'
import { loadModel, predict, type Prediction } from './lib/mnist'
import { canvasToMnist } from './lib/preprocess'
import { fgsm } from './lib/attacks'

type ModelStatus = 'loading' | 'ready' | 'error'

interface InputState {
  /** The 28x28 input the model actually saw — attacks start from this. */
  pixels: Float32Array
  prediction: Prediction
}

export default function App() {
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [status, setStatus] = useState<ModelStatus>('loading')
  const [input, setInput] = useState<InputState | null>(null)
  const [epsilon, setEpsilon] = useState(0.15)

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

  // FGSM recomputes live on every draw stroke and slider move — the model is
  // small enough that a full gradient pass is a few milliseconds on WebGL.
  const attacked = useMemo(() => {
    if (!model || !input) return null
    const result = fgsm(model, input.pixels, input.prediction.label, { epsilon })
    return { ...result, prediction: predict(model, result.adversarial) }
  }, [model, input, epsilon])

  const flipped = input && attacked && attacked.prediction.label !== input.prediction.label

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
            <h2 className="font-display mb-1 flex items-center gap-2 text-sm font-semibold">
              3 · Attack the model
              <span className="border-accent/60 text-accent rounded-full border px-2 py-0.5 font-mono text-[10px] font-medium tracking-wide uppercase">
                FGSM
              </span>
            </h2>
            <p className="text-muted mb-4 text-xs">
              One gradient step against the model's loss, recomputed live as you drag.
            </p>
            <div className={input ? '' : 'pointer-events-none opacity-40 select-none'}>
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
                FGSM (Goodfellow et al., 2014) computes the gradient of the model's loss with
                respect to <em>every pixel</em> — asking "which direction should each pixel move
                to make the model most wrong?" — then nudges each pixel by ε in that direction:
                x′ = x + ε·sign(∇ₓL). The gradient is computed right here in your browser.
                Attacks like this are why robustness matters for ML systems that face the real
                world: stop signs, face recognition, content filters.
              </p>
            </details>
          </section>

          {input && attacked && (
            <section className="border-line bg-panel rounded-xl border p-5 md:col-span-2 lg:col-span-3">
              <h2 className="font-display mb-1 text-sm font-semibold">4 · Result</h2>
              <p className="text-muted mb-5 text-xs">
                Left to right: what you drew, the adversarial perturbation (amplified so you can
                see it), and their sum — what the model was actually shown.
              </p>
              <div className="flex flex-wrap items-start justify-between gap-8">
                <div className="flex items-center gap-3 sm:gap-4">
                  <MnistPreview
                    pixels={input.pixels}
                    label={`original · "${input.prediction.label}"`}
                    className="size-24 sm:size-28"
                  />
                  <span className="text-muted -mt-6 font-mono text-xl">+</span>
                  <MnistPreview
                    pixels={attacked.perturbation}
                    signed
                    amplify={epsilon > 0 ? 0.45 / epsilon : 0}
                    label="ε·sign(∇ₓL) · amplified"
                    className="size-24 sm:size-28"
                  />
                  <span className="text-muted -mt-6 font-mono text-xl">=</span>
                  <MnistPreview
                    pixels={attacked.adversarial}
                    label={`adversarial · "${attacked.prediction.label}"`}
                    className="size-24 sm:size-28"
                  />
                </div>
                <div className="min-w-60 flex-1">
                  <div className="mb-3 flex items-baseline gap-3">
                    <span
                      className={`font-mono text-4xl font-bold ${flipped ? 'text-red-400' : 'text-accent'}`}
                    >
                      {attacked.prediction.label}
                    </span>
                    <span className="text-muted text-xs">
                      {(attacked.prediction.probs[attacked.prediction.label] * 100).toFixed(1)}%
                      confident
                    </span>
                    {flipped ? (
                      <span className="rounded-full border border-red-800 bg-red-950/40 px-2 py-0.5 font-mono text-[10px] tracking-wide text-red-400 uppercase">
                        prediction flipped {input.prediction.label} → {attacked.prediction.label}
                      </span>
                    ) : (
                      <span className="border-line text-muted rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-wide uppercase">
                        still correct — raise ε
                      </span>
                    )}
                  </div>
                  <ConfidenceBars probs={attacked.prediction.probs} />
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

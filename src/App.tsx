import { useCallback, useEffect, useState } from 'react'
import type * as tf from '@tensorflow/tfjs'
import SiteNav from './components/SiteNav'
import SiteFooter from './components/SiteFooter'
import DrawingCanvas from './components/DrawingCanvas'
import ConfidenceBars from './components/ConfidenceBars'
import MnistPreview from './components/MnistPreview'
import { loadModel, predict, type Prediction } from './lib/mnist'
import { canvasToMnist } from './lib/preprocess'

type ModelStatus = 'loading' | 'ready' | 'error'

interface InputState {
  /** The 28x28 input the model actually saw — milestone 2 attacks start from this. */
  pixels: Float32Array
  prediction: Prediction
}

export default function App() {
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [status, setStatus] = useState<ModelStatus>('loading')
  const [input, setInput] = useState<InputState | null>(null)

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
            Draw a digit and watch it classify. Soon: attack it with adversarial
            perturbations that are invisible to you but devastating to the model.
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

          <section className="border-line bg-panel/40 rounded-xl border border-dashed p-5 md:col-span-2 lg:col-span-1">
            <h2 className="font-display text-muted mb-1 flex items-center gap-2 text-sm font-semibold">
              3 · Attack the model
              <span className="rounded-full border border-amber-700/60 bg-amber-950/40 px-2 py-0.5 font-mono text-[10px] font-medium tracking-wide text-amber-400 uppercase">
                coming soon
              </span>
            </h2>
            <p className="text-muted mb-4 text-xs">
              FGSM &amp; PGD attacks computed live with in-browser gradients. A perturbation you
              can barely see will flip the prediction.
            </p>
            <div className="space-y-4 opacity-40 select-none" aria-hidden="true">
              <div>
                <div className="text-muted mb-1 flex justify-between text-xs">
                  <span>Perturbation strength ε</span>
                  <span className="font-mono">0.15</span>
                </div>
                <input type="range" disabled className="accent-accent w-full" />
              </div>
              <button
                disabled
                className="border-line text-muted font-display w-full rounded-md border py-2 text-sm"
              >
                Run FGSM attack
              </button>
            </div>
          </section>
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

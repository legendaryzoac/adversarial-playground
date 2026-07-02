import { useCallback, useEffect, useState } from 'react'
import type * as tf from '@tensorflow/tfjs'
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
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-5xl px-4 py-10">
        <header className="mb-8">
          <p className="mb-1 font-mono text-xs tracking-widest text-cyan-400 uppercase">
            AI Safety Lab
          </p>
          <h1 className="text-3xl font-bold tracking-tight">Adversarial Playground</h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-400">
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
          <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
            <h2 className="mb-1 text-sm font-semibold text-zinc-200">1 · Draw a digit</h2>
            <p className="mb-4 text-xs text-zinc-500">Any digit, 0–9. Prediction updates as you draw.</p>
            <DrawingCanvas onDraw={handleDraw} />
          </section>

          <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
            <h2 className="mb-1 text-sm font-semibold text-zinc-200">2 · Model prediction</h2>
            <p className="mb-4 text-xs text-zinc-500">
              {status === 'loading' ? 'Loading model…' : 'Softmax confidence per class.'}
            </p>
            <div className="mb-4 flex items-center justify-between gap-4">
              <div>
                <div className="font-mono text-5xl font-bold text-cyan-300">
                  {input ? input.prediction.label : '·'}
                </div>
                <div className="mt-1 text-xs text-zinc-500">
                  {input
                    ? `${(input.prediction.probs[input.prediction.label] * 100).toFixed(1)}% confident`
                    : 'waiting for a drawing'}
                </div>
              </div>
              <MnistPreview pixels={input?.pixels ?? null} label="model input (28×28)" />
            </div>
            <ConfidenceBars probs={input?.prediction.probs ?? null} />
          </section>

          <section className="rounded-xl border border-dashed border-zinc-800 bg-zinc-900/30 p-5 md:col-span-2 lg:col-span-1">
            <h2 className="mb-1 flex items-center gap-2 text-sm font-semibold text-zinc-400">
              3 · Attack the model
              <span className="rounded-full border border-amber-700/60 bg-amber-950/40 px-2 py-0.5 text-[10px] font-medium tracking-wide text-amber-400 uppercase">
                coming soon
              </span>
            </h2>
            <p className="mb-4 text-xs text-zinc-500">
              FGSM &amp; PGD attacks computed live with in-browser gradients. A perturbation you
              can barely see will flip the prediction.
            </p>
            <div className="space-y-4 opacity-40 select-none" aria-hidden="true">
              <div>
                <div className="mb-1 flex justify-between text-xs text-zinc-400">
                  <span>Perturbation strength ε</span>
                  <span className="font-mono">0.15</span>
                </div>
                <input type="range" disabled className="w-full accent-cyan-400" />
              </div>
              <button
                disabled
                className="w-full rounded-md border border-zinc-700 py-2 text-sm text-zinc-400"
              >
                Run FGSM attack
              </button>
            </div>
          </section>
        </main>

        <footer className="mt-10 border-t border-zinc-800/60 pt-4 text-xs text-zinc-600">
          Runs entirely client-side with TensorFlow.js · trained on MNIST ·{' '}
          {model ? `${model.countParams().toLocaleString()} parameters` : 'model loading…'}
        </footer>
      </div>
    </div>
  )
}

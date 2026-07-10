import { useEffect, useRef, useState } from 'react'
import type * as tf from '@tensorflow/tfjs'
import PhotoPreview from './PhotoPreview'
import TopKBars from './TopKBars'
import {
  loadMobilenet,
  imageToPixels,
  predictTopK,
  topKFromProbs,
  LABELS,
  LABEL_TO_INDEX,
  MOBILENET_SPEC,
  classLabel,
  type ClassScore,
} from '../lib/mobilenet'
import { runAttackProgressive, type AttackFrame, type AttackKind } from '../lib/attacks'

type ModelStatus = 'loading' | 'ready' | 'error'

const PGD_STEPS = 10
const DATALIST_ID = 'imagenet-labels'
/** Fun targeted-attack presets (resolved to indices at render). */
const QUICK_TARGETS = ['toaster', 'banana', 'zebra', 'guacamole', 'traffic light']
const SAMPLES = [
  { url: `${import.meta.env.BASE_URL}samples/dog.jpg`, name: 'Dog' },
  { url: `${import.meta.env.BASE_URL}samples/pug.jpg`, name: 'Pug' },
  { url: `${import.meta.env.BASE_URL}samples/hopper.jpg`, name: 'Portrait' },
]

interface OrigState {
  pixels: Float32Array
  top: ClassScore[]
}

export default function PhotoMode() {
  const [status, setStatus] = useState<ModelStatus>('loading')
  const [progress, setProgress] = useState(0)
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [orig, setOrig] = useState<OrigState | null>(null)

  const [kind, setKind] = useState<AttackKind>('pgd')
  const [epsilon, setEpsilon] = useState(0.03)
  const [targeted, setTargeted] = useState(false)
  const [targetText, setTargetText] = useState('')
  const [targetIndex, setTargetIndex] = useState<number | null>(null)

  const [shown, setShown] = useState<AttackFrame | null>(null)
  const [step, setStep] = useState<{ i: number; n: number } | null>(null)
  const [computing, setComputing] = useState(false)
  const runId = useRef(0)

  useEffect(() => {
    loadMobilenet(setProgress)
      .then((m) => {
        setModel(m)
        setStatus('ready')
      })
      .catch((err) => {
        console.error('Failed to load MobileNet', err)
        setStatus('error')
      })
  }, [])

  const loadImage = (source: CanvasImageSource, w: number, h: number) => {
    if (!model) return
    runId.current++ // cancel any in-flight attack
    const pixels = imageToPixels(source, w, h)
    setOrig({ pixels, top: predictTopK(model, pixels, 5) })
    setShown(null)
    setStep(null)
    setComputing(false)
  }

  const onFile = (file: File | undefined) => {
    if (!file) return
    const img = new Image()
    img.onload = () => {
      loadImage(img, img.naturalWidth, img.naturalHeight)
      URL.revokeObjectURL(img.src)
    }
    img.src = URL.createObjectURL(file)
  }

  const onSample = (url: string) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => loadImage(img, img.naturalWidth, img.naturalHeight)
    img.onerror = () => console.warn('sample failed to load', url)
    img.src = url
  }

  const setTarget = (text: string) => {
    setTargetText(text)
    setTargetIndex(LABEL_TO_INDEX.has(text) ? LABEL_TO_INDEX.get(text)! : null)
  }

  const runAttack = async () => {
    if (!model || !orig) return
    const id = ++runId.current
    setComputing(true)
    setShown(null)
    setStep(null)
    const useTarget = targeted && targetIndex !== null
    await runAttackProgressive(
      model,
      orig.pixels,
      orig.top[0].index,
      { kind, epsilon, steps: PGD_STEPS, targetClass: useTarget ? targetIndex! : undefined },
      MOBILENET_SPEC,
      (frame, i, n) => {
        if (runId.current !== id) return
        setShown(frame)
        setStep({ i: i + 1, n })
      },
      () => runId.current === id,
    )
    if (runId.current === id) setComputing(false)
  }

  const advTop = shown ? topKFromProbs(shown.probs, 5) : null
  const flipped = !!(orig && shown && shown.label !== orig.top[0].index)
  const success = targeted && targetIndex !== null ? !!(shown && shown.label === targetIndex) : flipped

  if (status === 'error') {
    return (
      <div className="rounded-lg border border-red-800 bg-red-950/40 p-4 text-sm text-red-300">
        Failed to load MobileNet from Google's CDN. Check your connection and reload.
      </div>
    )
  }

  if (status === 'loading') {
    return (
      <div className="border-line bg-panel rounded-xl border p-8 text-center">
        <p className="text-fg mb-3 text-sm font-medium">Loading MobileNet — a real 1000-class ImageNet model</p>
        <p className="text-muted mb-4 text-xs">~16 MB, from Google's CDN. One-time download, then it runs offline.</p>
        <div className="bg-panel-2 mx-auto h-2 max-w-sm overflow-hidden rounded-full">
          <div className="bg-accent h-full transition-[width] duration-200" style={{ width: `${(progress * 100).toFixed(0)}%` }} />
        </div>
        <p className="text-muted mt-2 font-mono text-xs">{(progress * 100).toFixed(0)}%</p>
      </div>
    )
  }

  return (
    <main className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <datalist id={DATALIST_ID}>
        {LABELS.map((l, i) => (
          <option key={i} value={l} />
        ))}
      </datalist>

      {/* 1 · Choose a photo */}
      <section className="border-line bg-panel rounded-xl border p-5">
        <h2 className="font-display mb-1 text-sm font-semibold">1 · Choose a photo</h2>
        <p className="text-muted mb-4 text-xs">Upload an image, or start from a sample.</p>
        <label
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            onFile(e.dataTransfer.files[0])
          }}
          className="border-line hover:border-accent flex aspect-square w-full max-w-[240px] cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed text-center transition-colors"
        >
          {orig ? (
            <PhotoPreview pixels={orig.pixels} kind="image" className="size-[200px]" />
          ) : (
            <span className="text-muted px-4 text-xs">
              Drop an image here
              <br />or click to browse
            </span>
          )}
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </label>
        <div className="mt-3 flex flex-wrap gap-2">
          {SAMPLES.map((s) => (
            <button
              key={s.url}
              onClick={() => onSample(s.url)}
              className="border-line text-muted hover:border-accent hover:text-fg rounded-md border px-3 py-1 text-xs transition-colors"
            >
              {s.name}
            </button>
          ))}
        </div>
      </section>

      {/* 2 · Model prediction */}
      <section className="border-line bg-panel rounded-xl border p-5">
        <h2 className="font-display mb-1 text-sm font-semibold">2 · Model prediction</h2>
        <p className="text-muted mb-4 text-xs">MobileNet's top-5 guesses for the original image.</p>
        {orig ? (
          <>
            <div className="mb-4">
              <div className="text-accent font-mono text-2xl font-bold">{orig.top[0].label}</div>
              <div className="text-muted mt-1 text-xs">{(orig.top[0].prob * 100).toFixed(1)}% confident</div>
            </div>
            <TopKBars scores={orig.top} />
          </>
        ) : (
          <p className="text-muted text-xs italic">Choose a photo to see predictions.</p>
        )}
      </section>

      {/* 3 · Attack the model */}
      <section className="border-line bg-panel rounded-xl border p-5 md:col-span-2 lg:col-span-1">
        <h2 className="font-display mb-3 text-sm font-semibold">3 · Attack the model</h2>
        <div className={orig ? 'space-y-4' : 'space-y-4 opacity-40'}>
          <div>
            <div className="text-muted mb-1.5 font-mono text-[10px] tracking-wide uppercase">Method</div>
            <div className="border-line flex overflow-hidden rounded-md border text-sm">
              {(['fgsm', 'pgd'] as const).map((k) => (
                <button
                  key={k}
                  onClick={() => setKind(k)}
                  disabled={!orig}
                  className={`flex-1 px-3 py-1.5 font-medium transition-colors ${
                    kind === k ? 'bg-accent text-site' : 'text-muted hover:text-fg'
                  }`}
                >
                  {k === 'fgsm' ? 'FGSM' : `PGD ×${PGD_STEPS}`}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="flex cursor-pointer items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={targeted}
                onChange={(e) => setTargeted(e.target.checked)}
                disabled={!orig}
                className="accent-accent"
              />
              <span className="text-fg font-medium">Targeted</span>
              <span className="text-muted">— pick the wrong answer</span>
            </label>
            {targeted && (
              <div className="mt-2 space-y-2">
                <input
                  type="text"
                  list={DATALIST_ID}
                  value={targetText}
                  onChange={(e) => setTarget(e.target.value)}
                  placeholder="type a class, e.g. toaster"
                  className={`border-line bg-panel-2 w-full rounded border px-2 py-1 text-xs ${
                    targetText && targetIndex === null ? 'text-red-400' : 'text-fg'
                  }`}
                />
                <div className="flex flex-wrap gap-1.5">
                  {QUICK_TARGETS.filter((t) => LABEL_TO_INDEX.has(t)).map((t) => (
                    <button
                      key={t}
                      onClick={() => setTarget(t)}
                      className="border-line text-muted hover:border-accent hover:text-fg rounded border px-2 py-0.5 text-[11px] transition-colors"
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="text-muted mb-1 flex justify-between text-xs">
              <span>Perturbation strength ε</span>
              <span className="text-fg font-mono">{epsilon.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={0.12}
              step={0.004}
              value={epsilon}
              onChange={(e) => setEpsilon(Number(e.target.value))}
              disabled={!orig}
              className="accent-accent w-full"
              aria-label="Perturbation strength epsilon"
            />
            <div className="text-muted mt-1 flex justify-between font-mono text-[10px]">
              <span>0 · none</span>
              <span>0.12 · visible</span>
            </div>
          </div>

          <button
            onClick={runAttack}
            disabled={!orig || computing}
            className="bg-accent text-site hover:bg-accent-dim w-full rounded-md py-2 text-sm font-semibold transition-colors disabled:opacity-50"
          >
            {computing
              ? step
                ? `Attacking… step ${step.i}/${step.n}`
                : 'Attacking…'
              : kind === 'fgsm'
                ? 'Run FGSM attack'
                : `Run PGD attack (${PGD_STEPS} steps)`}
          </button>
        </div>

        <details className="border-line mt-5 border-t pt-3">
          <summary className="text-muted hover:text-fg cursor-pointer text-xs font-medium">How does this work?</summary>
          <p className="text-muted mt-2 text-xs leading-relaxed">
            This is <strong className="text-fg">MobileNet</strong>, the same class of model that
            powers real photo tagging, running entirely in your browser. The attack computes the
            gradient of its loss with respect to every pixel and nudges them by a tiny amount ε —
            far too small for you to notice, but enough to push the image across the model's
            decision boundary. <strong className="text-fg">PGD</strong> takes {PGD_STEPS} small
            steps for a stronger push; <strong className="text-fg">targeted</strong> mode steers
            toward a class you choose. This fragility is exactly why adversarial robustness matters
            for vision systems in the wild.
          </p>
        </details>
      </section>

      {/* 4 · Result */}
      {orig && shown && (
        <section className="border-line bg-panel rounded-xl border p-5 md:col-span-2 lg:col-span-3">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <h2 className="font-display text-sm font-semibold">4 · Result</h2>
            <span className="border-accent/60 text-accent rounded-full border px-2 py-0.5 font-mono text-[10px] font-medium tracking-wide uppercase">
              {kind === 'fgsm' ? 'FGSM' : `PGD · step ${step?.i ?? PGD_STEPS}/${step?.n ?? PGD_STEPS}`}
            </span>
            {targeted && targetIndex !== null && (
              <span className="border-line text-muted rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-wide uppercase">
                targeted → {classLabel(targetIndex)}
              </span>
            )}
          </div>
          <p className="text-muted mb-5 text-xs">
            Original + an imperceptible perturbation (amplified here so it's visible) = the
            adversarial image the model actually sees.
          </p>
          <div className="flex flex-wrap items-start justify-between gap-8">
            <div className="flex items-center gap-3 sm:gap-4">
              <PhotoPreview pixels={orig.pixels} kind="image" label={`original · ${orig.top[0].label}`} className="size-28 sm:size-32" />
              <span className="text-muted -mt-6 font-mono text-xl">+</span>
              <PhotoPreview pixels={shown.perturbation} kind="perturbation" epsilon={epsilon} label="perturbation · amplified" className="size-28 sm:size-32" />
              <span className="text-muted -mt-6 font-mono text-xl">=</span>
              <PhotoPreview pixels={shown.adversarial} kind="image" label={`adversarial · ${classLabel(shown.label)}`} className="size-28 sm:size-32" />
            </div>
            <div className="min-w-64 flex-1">
              <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className={`font-mono text-2xl font-bold ${success ? 'text-red-400' : 'text-accent'}`}>
                  {classLabel(shown.label)}
                </span>
                <span className="text-muted text-xs">{(shown.probs[shown.label] * 100).toFixed(1)}% confident</span>
                <span
                  className={`rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-wide uppercase ${
                    success ? 'border-red-800 bg-red-950/40 text-red-400' : 'border-line text-muted'
                  }`}
                >
                  {targeted && targetIndex !== null
                    ? success
                      ? `hit target`
                      : `aiming for ${classLabel(targetIndex)}`
                    : flipped
                      ? `fooled: ${orig.top[0].label} → ${classLabel(shown.label)}`
                      : 'still correct — raise ε'}
                </span>
              </div>
              <TopKBars scores={advTop} highlightIndex={targeted && targetIndex !== null ? targetIndex : undefined} />
            </div>
          </div>
        </section>
      )}
    </main>
  )
}

import { useState } from 'react'
import SiteNav from './components/SiteNav'
import SiteFooter from './components/SiteFooter'
import DrawMode from './components/DrawMode'
import PhotoMode from './components/PhotoMode'

type Mode = 'draw' | 'photo'

export default function App() {
  const [mode, setMode] = useState<Mode>('draw')

  return (
    <div className="min-h-screen">
      <SiteNav />
      <div className="mx-auto max-w-5xl px-4 pt-[60px] pb-4">
        <header className="mt-10 mb-6">
          <p className="text-accent mb-2 font-mono text-xs tracking-widest uppercase">
            // AI Safety Lab
          </p>
          <h1 className="font-display text-3xl font-bold tracking-tight">Adversarial Playground</h1>
          <p className="text-muted mt-3 max-w-2xl text-sm leading-relaxed">
            Neural networks run live in your browser — no server, no API. Classify an image, then
            attack it: a gradient-crafted perturbation you can barely see flips the prediction while
            the image still looks the same to you.
          </p>
        </header>

        {/* Mode switcher */}
        <div className="mb-6 inline-flex gap-1 rounded-lg border border-line bg-panel p-1">
          {([
            { id: 'draw', label: 'Draw a digit', sub: 'MNIST · tiny CNN' },
            { id: 'photo', label: 'Attack a photo', sub: 'MobileNet · ImageNet' },
          ] as const).map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`rounded-md px-4 py-2 text-left transition-colors ${
                mode === m.id ? 'bg-accent text-site' : 'text-muted hover:text-fg'
              }`}
            >
              <div className="text-sm font-semibold">{m.label}</div>
              <div className={`font-mono text-[10px] ${mode === m.id ? 'text-site/70' : 'text-muted'}`}>
                {m.sub}
              </div>
            </button>
          ))}
        </div>

        {mode === 'draw' ? <DrawMode /> : <PhotoMode />}

        <SiteFooter note="Runs entirely client-side with TensorFlow.js · MNIST CNN + MobileNet, both attacked with in-browser gradients" />
      </div>
    </div>
  )
}

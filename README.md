# Adversarial Playground

An interactive AI-safety demo: a convolutional neural network runs **entirely in your
browser** with [TensorFlow.js](https://www.tensorflow.org/js) — no server, no API keys.
Draw a digit, watch the model classify it live, then (milestone 2) attack it with
adversarial perturbations (FGSM / PGD) computed client-side with real input gradients.

**Stack:** Vite · React 19 · TypeScript · Tailwind CSS 4 · TensorFlow.js

## Run locally

```bash
npm install
npm run dev
```

The trained model artifacts live in `public/model/` (checked in — they're ~25 KB).

## Retrain the model

Two options, both producing `model.json` + `weights.bin` for `public/model/`:

- **In the browser (fast):** `npm run dev`, open `/train.html`, click *Start training*.
  Trains on the WebGL backend in under a minute and downloads the artifacts.
- **In Node (slow, zero extra deps):** `npm run train`. Uses the pure-JS CPU backend,
  so expect it to take a while. Writes straight to `public/model/`.

The architecture (defined in both `training/train.mjs` and `training/browser-train.ts`)
is a small CNN — two conv/pool blocks and a linear head, ~5.3K parameters. The final
layer intentionally outputs **logits** (no softmax): the app applies `tf.softmax` for
display, and the adversarial attacks compute `∇ₓ softmaxCrossEntropy(y, logits)`
directly from the logits.

## Roadmap

1. ~~**Milestone 1** — draw a digit, live in-browser classification with confidence bars~~
2. **Milestone 2** — FGSM attack: ε slider, amplified-perturbation view, adversarial prediction panel (see `src/lib/attacks.ts`)
3. **Milestone 3** — PGD (animated iterations), targeted attacks, explainer content
4. **Stretch** — MobileNetV2 photo mode, webcam input, preprocessing-defense toggle

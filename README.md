# Adversarial Playground

An interactive AI-safety demo: neural networks run **entirely in your browser** with
[TensorFlow.js](https://www.tensorflow.org/js) — no server, no API keys. Two modes:

- **Draw a digit** — a tiny MNIST CNN classifies your drawing live, then you attack it.
- **Attack a photo** — MobileNet (a real 1000-class ImageNet model) classifies an
  uploaded image, then a gradient-crafted perturbation you can barely see flips it
  ("Labrador retriever" → "doormat").

Both attacks (FGSM / PGD, targeted or untargeted) are computed client-side with real
input gradients — the whole point is that the model is differentiable w.r.t. its pixels.

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

## How the photo mode works

`src/lib/attacks.ts` is model-agnostic: an `AttackSpec` describes the input shape,
class count, pixel range, and whether the model emits logits or probabilities, so the
same FGSM/PGD code drives both the MNIST CNN (`[28,28,1]`, logits, `[0,1]`) and MobileNet
(`[224,224,3]`, softmax, `[-1,1]`). MobileNet is
[MobileNet v1 (1.0, 224)](https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json),
loaded as a **LayersModel** so `tf.grad` can flow to the input (a GraphModel could
classify but not be attacked). Its ~16 MB of weights come from Google's CDN, so they
cost the S3 bucket no egress; the backward-pass shaders are compiled during the loading
bar so the first attack is snappy. Heavy attacks run through `runAttackProgressive`,
which yields between gradient steps so the PGD descent animates instead of freezing.

## Roadmap

1. ~~**Milestone 1** — draw a digit, live in-browser MNIST classification~~
2. ~~**Milestone 2** — FGSM attack: ε slider, amplified-perturbation view, adversarial prediction panel~~
3. ~~**Milestone 3** — PGD (animated iterations), targeted attacks, explainer content~~
4. ~~**Milestone 5 (stretch)** — MobileNet photo mode: upload/sample images, top-5 predictions, FGSM/PGD/targeted attacks on a real ImageNet classifier~~
5. **Later** — webcam input; a preprocessing-defense toggle (blur/JPEG) to show why defenses are hard

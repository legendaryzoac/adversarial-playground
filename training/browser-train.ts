// Browser-based training for the demo CNN (see /train.html).
//
// Runs on the WebGL backend, so it finishes in well under a minute — the
// pure-CPU Node path (training/train.mjs) works too but is very slow, and
// native tfjs-node has no prebuilt Windows binaries anymore. Data comes from
// the TF.js team's CORS-enabled MNIST sprite.
//
// Exposes window.__TRAIN__ = { status, message, result } so the training run
// can also be driven headlessly (that's how the checked-in model was made).
import * as tf from '@tensorflow/tfjs'

const SPRITE_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png'
const LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'
const NUM_DATASET = 65000
const NUM_TRAIN = 20000
const NUM_TEST = 5000
const IMG_LEN = 784
const EPOCHS = 8
const BATCH_SIZE = 256

interface TrainHandle {
  status: 'idle' | 'running' | 'done' | 'error'
  message: string
  result: { modelJson: string; weightsB64: string } | null
}

declare global {
  interface Window { __TRAIN__: TrainHandle }
}

window.__TRAIN__ = { status: 'idle', message: 'not started', result: null }

const logEl = document.getElementById('log')!
const startBtn = document.getElementById('start') as HTMLButtonElement
const downloadBtn = document.getElementById('download') as HTMLButtonElement

function log(msg: string) {
  window.__TRAIN__.message = msg
  logEl.textContent += `${msg}\n`
  console.log(msg)
}

/**
 * The MNIST sprite is a 784xN PNG: each ROW is one flattened 28x28 image.
 * Decoded in chunks because a 65000px-tall canvas exceeds browser limits.
 */
async function loadData() {
  log('Downloading MNIST sprite (~10 MB)…')
  const img = new Image()
  img.crossOrigin = 'anonymous'
  await new Promise((resolve, reject) => {
    img.onload = resolve
    img.onerror = reject
    img.src = SPRITE_URL
  })

  const pixels = new Float32Array(NUM_DATASET * IMG_LEN)
  const chunk = 5000
  const canvas = document.createElement('canvas')
  canvas.width = IMG_LEN
  canvas.height = chunk
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  for (let offset = 0; offset < NUM_DATASET; offset += chunk) {
    const rows = Math.min(chunk, NUM_DATASET - offset)
    ctx.drawImage(img, 0, offset, IMG_LEN, rows, 0, 0, IMG_LEN, rows)
    const data = ctx.getImageData(0, 0, IMG_LEN, rows).data
    for (let i = 0; i < rows * IMG_LEN; i++) {
      pixels[offset * IMG_LEN + i] = data[i * 4] / 255
    }
  }

  log('Downloading labels…')
  const labelsBuf = new Uint8Array(await (await fetch(LABELS_URL)).arrayBuffer()) // one-hot, N x 10

  const xTrain = tf.tensor4d(pixels.subarray(0, NUM_TRAIN * IMG_LEN), [NUM_TRAIN, 28, 28, 1])
  const yTrain = tf.tensor2d(labelsBuf.subarray(0, NUM_TRAIN * 10), [NUM_TRAIN, 10])
  const testStart = NUM_DATASET - NUM_TEST
  const xTest = tf.tensor4d(pixels.subarray(testStart * IMG_LEN), [NUM_TEST, 28, 28, 1])
  const yTest = tf.tensor2d(labelsBuf.subarray(testStart * 10), [NUM_TEST, 10])
  return { xTrain, yTrain, xTest, yTest }
}

function buildModel() {
  const model = tf.sequential()
  model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 8, kernelSize: 3, activation: 'relu' }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))
  model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu' }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 10 })) // logits — the app (and FGSM) applies softmax
  return model
}

async function train() {
  window.__TRAIN__.status = 'running'
  startBtn.disabled = true
  try {
    await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'))
    await tf.ready()
    log(`Backend: ${tf.getBackend()}`)

    const { xTrain, yTrain, xTest, yTest } = await loadData()
    const model = buildModel()
    model.compile({
      optimizer: tf.train.adam(1e-3),
      loss: (yTrue, yPred) => tf.losses.softmaxCrossEntropy(yTrue, yPred),
      metrics: ['accuracy'],
    })

    const t0 = Date.now()
    await model.fit(xTrain, yTrain, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationData: [xTest, yTest],
      callbacks: {
        onEpochEnd: (epoch, logs) =>
          log(`Epoch ${epoch + 1}/${EPOCHS}: loss=${logs!.loss!.toFixed(4)} acc=${(logs!.acc as number).toFixed(4)} val_acc=${(logs!.val_acc as number).toFixed(4)}`),
      },
    })
    log(`Training took ${((Date.now() - t0) / 1000).toFixed(1)}s`)

    await model.save(tf.io.withSaveHandler(async (artifacts) => {
      const weightData = Array.isArray(artifacts.weightData)
        ? artifacts.weightData[0]
        : (artifacts.weightData as ArrayBuffer)
      const modelJson = JSON.stringify({
        modelTopology: artifacts.modelTopology,
        format: 'layers-model',
        generatedBy: 'adversarial-playground train.html',
        convertedBy: null,
        weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }],
      })
      let binary = ''
      const bytes = new Uint8Array(weightData)
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
      window.__TRAIN__.result = { modelJson, weightsB64: btoa(binary) }
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } }
    }))

    // In dev, save straight into public/model/ via the Vite middleware
    // (see vite.config.ts). Falls back to the manual download button.
    if (import.meta.env.DEV) {
      const res = await fetch('/__save-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(window.__TRAIN__.result),
      }).catch(() => null)
      if (res?.ok) log('Artifacts written to public/model/.')
    }

    window.__TRAIN__.status = 'done'
    downloadBtn.disabled = false
    log('Done. Click "Download model" to save the artifacts.')
  } catch (err) {
    window.__TRAIN__.status = 'error'
    log(`ERROR: ${err}`)
    throw err
  }
}

function downloadResult() {
  const result = window.__TRAIN__.result
  if (!result) return
  const jsonBlob = new Blob([result.modelJson], { type: 'application/json' })
  const bytes = Uint8Array.from(atob(result.weightsB64), (c) => c.charCodeAt(0))
  const binBlob = new Blob([bytes], { type: 'application/octet-stream' })
  for (const [name, blob] of [['model.json', jsonBlob], ['weights.bin', binBlob]] as const) {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
    URL.revokeObjectURL(a.href)
  }
}

startBtn.addEventListener('click', train)
downloadBtn.addEventListener('click', downloadResult)

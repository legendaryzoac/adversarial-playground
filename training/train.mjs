// Trains the MNIST CNN used by the demo and writes TF.js Layers artifacts
// to public/model/ (model.json + weights.bin).
//
// Runs on the pure-JS CPU backend so there are no native/Python dependencies:
//   node training/train.mjs
//
// The final Dense layer is LINEAR (logits, no softmax). The app applies
// tf.softmax itself. This matters for milestone 2: FGSM/PGD compute
// grad(softmaxCrossEntropy(labels, logits)) w.r.t. the input, which is
// cleaner and more numerically stable from logits.
import * as tf from '@tensorflow/tfjs';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadMnist } from './data.mjs';

const TRAIN_SAMPLES = Number(process.env.TRAIN_SAMPLES ?? 16000);
const TEST_SAMPLES = Number(process.env.TEST_SAMPLES ?? 5000);
const EPOCHS = Number(process.env.EPOCHS ?? 8);
const BATCH_SIZE = 128;

const OUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public', 'model');

function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1], filters: 8, kernelSize: 3, activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10 })); // logits — softmax applied in the app
  return model;
}

async function main() {
  console.log(`Backend: ${tf.getBackend()} | train=${TRAIN_SAMPLES} test=${TEST_SAMPLES} epochs=${EPOCHS}`);
  const { train, test } = await loadMnist({ trainCount: TRAIN_SAMPLES, testCount: TEST_SAMPLES });

  const xTrain = tf.tensor4d(train.pixels, [train.count, 28, 28, 1]);
  const yTrain = tf.oneHot(tf.tensor1d(train.labels, 'int32'), 10);
  const xTest = tf.tensor4d(test.pixels, [test.count, 28, 28, 1]);
  const yTest = tf.oneHot(tf.tensor1d(test.labels, 'int32'), 10);

  const model = buildModel();
  model.summary();
  model.compile({
    optimizer: tf.train.adam(1e-3),
    // Cross-entropy from logits (see header comment).
    loss: (yTrue, yPred) => tf.losses.softmaxCrossEntropy(yTrue, yPred),
    metrics: ['accuracy'],
  });

  const t0 = Date.now();
  await model.fit(xTrain, yTrain, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [xTest, yTest],
    callbacks: {
      onBatchEnd: (batch, logs) => {
        if (batch % 20 === 0) {
          console.log(`  batch ${batch}: loss=${logs.loss.toFixed(4)} acc=${logs.acc?.toFixed(4)}`);
        }
      },
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}/${EPOCHS}: loss=${logs.loss.toFixed(4)} acc=${logs.acc.toFixed(4)} val_acc=${logs.val_acc.toFixed(4)}`);
      },
    },
  });
  console.log(`Training took ${((Date.now() - t0) / 1000).toFixed(0)}s`);

  await mkdir(OUT_DIR, { recursive: true });
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const weightData = Array.isArray(artifacts.weightData)
      ? Buffer.concat(artifacts.weightData.map((b) => Buffer.from(b)))
      : Buffer.from(artifacts.weightData);
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      format: 'layers-model',
      generatedBy: 'adversarial-playground training/train.mjs',
      convertedBy: null,
      weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }],
    };
    await writeFile(path.join(OUT_DIR, 'model.json'), JSON.stringify(modelJson));
    await writeFile(path.join(OUT_DIR, 'weights.bin'), weightData);
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));
  console.log(`Saved model to ${OUT_DIR}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

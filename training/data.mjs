// MNIST download + parsing (IDX format), cached locally under training/.cache/.
import { gunzipSync } from 'node:zlib';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MIRROR = 'https://storage.googleapis.com/cvdf-datasets/mnist';
const CACHE_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '.cache');

const FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz',
};

async function fetchCached(name) {
  const cachePath = path.join(CACHE_DIR, name);
  if (existsSync(cachePath)) {
    return readFile(cachePath);
  }
  const url = `${MIRROR}/${name}`;
  console.log(`Downloading ${url} ...`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${url}: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(CACHE_DIR, { recursive: true });
  await writeFile(cachePath, buf);
  return buf;
}

function parseImages(gzipped, limit) {
  const buf = gunzipSync(gzipped);
  const magic = buf.readUInt32BE(0);
  if (magic !== 2051) throw new Error(`Bad magic in image file: ${magic}`);
  const count = Math.min(buf.readUInt32BE(4), limit ?? Infinity);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const pixels = new Float32Array(count * rows * cols);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = buf[16 + i] / 255;
  }
  return { pixels, count, rows, cols };
}

function parseLabels(gzipped, limit) {
  const buf = gunzipSync(gzipped);
  const magic = buf.readUInt32BE(0);
  if (magic !== 2049) throw new Error(`Bad magic in label file: ${magic}`);
  const count = Math.min(buf.readUInt32BE(4), limit ?? Infinity);
  return new Uint8Array(buf.subarray(8, 8 + count));
}

/**
 * Loads MNIST as plain typed arrays.
 * @returns {{ train: {pixels, labels, count}, test: {pixels, labels, count} }}
 */
export async function loadMnist({ trainCount, testCount } = {}) {
  const [trImg, trLbl, teImg, teLbl] = await Promise.all([
    fetchCached(FILES.trainImages),
    fetchCached(FILES.trainLabels),
    fetchCached(FILES.testImages),
    fetchCached(FILES.testLabels),
  ]);
  const train = parseImages(trImg, trainCount);
  const test = parseImages(teImg, testCount);
  return {
    train: { pixels: train.pixels, labels: parseLabels(trLbl, trainCount), count: train.count },
    test: { pixels: test.pixels, labels: parseLabels(teLbl, testCount), count: test.count },
  };
}

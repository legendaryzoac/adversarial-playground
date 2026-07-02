import { IMAGE_SIZE } from './mnist'

/**
 * Converts the drawing canvas into a 28x28 MNIST-style input.
 *
 * MNIST digits were preprocessed by scaling the glyph to fit a 20x20 box and
 * centering it in the 28x28 frame by center of mass. Reproducing that here
 * matters a lot for accuracy — a digit drawn in a corner of the canvas would
 * otherwise be far outside the training distribution.
 *
 * Returns intensities in [0, 1] (white-on-black, like MNIST), or null if the
 * canvas is blank.
 */
export function canvasToMnist(canvas: HTMLCanvasElement): Float32Array | null {
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return null
  const { width, height } = canvas
  const data = ctx.getImageData(0, 0, width, height).data

  // Bounding box of drawn pixels (drawing is white on black; read red channel).
  let minX = width, minY = height, maxX = -1, maxY = -1
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4] > 16) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }
  if (maxX < 0) return null

  const boxW = maxX - minX + 1
  const boxH = maxY - minY + 1
  const scale = 20 / Math.max(boxW, boxH)
  const scaledW = Math.max(1, Math.round(boxW * scale))
  const scaledH = Math.max(1, Math.round(boxH * scale))

  // Scale the cropped glyph down to fit a 20x20 box.
  const glyph = document.createElement('canvas')
  glyph.width = scaledW
  glyph.height = scaledH
  const glyphCtx = glyph.getContext('2d')!
  glyphCtx.imageSmoothingEnabled = true
  glyphCtx.imageSmoothingQuality = 'high'
  glyphCtx.drawImage(canvas, minX, minY, boxW, boxH, 0, 0, scaledW, scaledH)

  // Center of mass of the scaled glyph.
  const glyphData = glyphCtx.getImageData(0, 0, scaledW, scaledH).data
  let mass = 0, comX = 0, comY = 0
  for (let y = 0; y < scaledH; y++) {
    for (let x = 0; x < scaledW; x++) {
      const v = glyphData[(y * scaledW + x) * 4]
      mass += v
      comX += x * v
      comY += y * v
    }
  }
  comX /= mass
  comY /= mass

  // Place the glyph so its center of mass lands at (14, 14).
  const out = document.createElement('canvas')
  out.width = IMAGE_SIZE
  out.height = IMAGE_SIZE
  const outCtx = out.getContext('2d')!
  outCtx.fillStyle = 'black'
  outCtx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE)
  outCtx.drawImage(glyph, Math.round(IMAGE_SIZE / 2 - comX), Math.round(IMAGE_SIZE / 2 - comY))

  const outData = outCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data
  const pixels = new Float32Array(IMAGE_SIZE * IMAGE_SIZE)
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = outData[i * 4] / 255
  }
  return pixels
}

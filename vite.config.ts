import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

/**
 * Dev-only endpoint for /train.html: POST { modelJson, weightsB64 } to
 * /__save-model and the trained artifacts land in public/model/ without a
 * manual download-and-copy step.
 */
function saveModelEndpoint(): Plugin {
  return {
    name: 'save-model-endpoint',
    configureServer(server) {
      server.middlewares.use('/__save-model', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405
          res.end()
          return
        }
        const chunks: Buffer[] = []
        req.on('data', (c) => chunks.push(c))
        req.on('end', async () => {
          try {
            const { modelJson, weightsB64 } = JSON.parse(Buffer.concat(chunks).toString('utf8'))
            const outDir = path.join(server.config.root, 'public', 'model')
            await mkdir(outDir, { recursive: true })
            await writeFile(path.join(outDir, 'model.json'), modelJson)
            await writeFile(path.join(outDir, 'weights.bin'), Buffer.from(weightsB64, 'base64'))
            res.statusCode = 200
            res.end('saved')
          } catch (err) {
            res.statusCode = 500
            res.end(String(err))
          }
        })
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), saveModelEndpoint()],
})

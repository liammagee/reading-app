import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Vite plugin that proxies Google Docs export URLs to avoid CORS.
 * Handles POST /api/fetch-url with { url } body.
 */
function googleDocsProxy(): Plugin {
  return {
    name: 'google-docs-proxy',
    configureServer(server) {
      server.middlewares.use('/api/fetch-url', async (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405
          res.end(JSON.stringify({ error: 'Method not allowed' }))
          return
        }

        const chunks: Buffer[] = []
        for await (const chunk of req) {
          chunks.push(chunk as Buffer)
        }
        const body = JSON.parse(Buffer.concat(chunks).toString())
        const { url } = body as { url: string }

        if (
          !url ||
          !url.startsWith('https://docs.google.com/document/d/')
        ) {
          res.statusCode = 400
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ error: 'Invalid Google Docs URL' }))
          return
        }

        try {
          const response = await fetch(url, {
            headers: { 'Accept': 'text/html' },
            redirect: 'follow',
          })

          if (!response.ok) {
            const status = response.status
            const message =
              status === 404
                ? 'Document not found. Check the URL.'
                : status === 401 || status === 403
                  ? 'Document is not publicly accessible. Make sure sharing is set to "Anyone with the link".'
                  : `Google Docs returned status ${status}`
            res.statusCode = status
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: message }))
            return
          }

          const html = await response.text()

          // Extract title from <title> tag if present
          const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i)
          const title = titleMatch?.[1]?.trim() || undefined

          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ html, title }))
        } catch (err) {
          res.statusCode = 502
          res.setHeader('Content-Type', 'application/json')
          res.end(
            JSON.stringify({
              error:
                err instanceof Error
                  ? err.message
                  : 'Failed to fetch document',
            }),
          )
        }
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  base: '/reading/',
  server: {
    port: 3001,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  plugins: [googleDocsProxy(), react()],
})

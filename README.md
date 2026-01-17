# Focus Reader

Speed reading sandbox with memory, parallel text alignment, and a hermeneutics companion stub.

## Quick Start

```bash
cd reading-app
npm install
npm run dev -- --port 5174
```

Then in the main site:

- Set `READING_APP_URL=http://localhost:5174` before running `node server.js`
- Open `/#/reading`

If you build the reading app separately, set `READING_APP_DIST` to the built `dist` folder path.

Voice playback calls `/api/tts/speak` on the main server, so ensure `LEMONFOX_API_KEY` is set there.

### Environment Variables

Vite automatically loads `.env` files. Any variable prefixed with `VITE_` is exposed to the client.

Example (`.env`):

```bash
VITE_API_BASE_URL=http://localhost:8080
```

`VITE_API_BASE_URL` overrides the API base for TTS calls (defaults to same-origin).

## Features in This Skeleton

- Word-by-word playback with WPM + chunk size controls
- Local progress + bookmarks (localStorage)
- Parallel text panel (rough proportional alignment)
- PDF import with page ranges + TOC navigation
- OCR option for scanned/image-only PDFs
- Lemonfox voice playback with voice + speed controls
- Conventional excerpt view
- Hermeneutics companion placeholder

## Roadmap Notes

- Kindle: only DRM-free exports or personal documents (no DRM bypass)
- EPUB parsing: add `epubjs`
- Parallel alignment: add segment-level mapping data and a manual alignment UI
- AI discussion: wire to `/api/tutor/quick-chat` with the current excerpt

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Focus Reader (Pacer) — a speed reading SPA built with React 19 + TypeScript + Vite. Supports PDF, EPUB, DOCX, HTML, and plain text import. Features RSVP-style word display with customizable playback, cross-device sync, and optional camera-based eye/hand tracking.

## Commands

```bash
npm run dev        # Vite dev server on port 5174, proxies /api → localhost:8080
npm run build      # tsc -b && vite build
npm run lint       # eslint .
npm run test       # vitest run (node environment)
npm run preview    # serve production build locally
```

## Architecture

### Monolithic App Component

`src/App.tsx` (~5400 lines) contains virtually all application logic. It manages 60+ useState hooks, extensive useEffect/useCallback/useMemo chains, and direct IndexedDB access. There is no state management library — everything is React hooks + context.

Key state objects defined at the top of App.tsx:
- `SavedSession` — current reading position, WPM, chunk size, granularity, bookmarks
- `StoredDocument` — full text + metadata in IndexedDB (`reader-documents`)
- `UserPreferences` — UI settings, view modes, display options

### Data Storage

- **IndexedDB `reader-documents`** — document text, PDF page data, metadata
- **IndexedDB `reader-sync-queue`** — offline sync queue for push actions
- **localStorage keys** — all prefixed `reader:` (auth_token, auth_user, preferences, streak, conventionalMode)

### Web Workers

- `src/textWorker.ts` — tokenization, segmentation, search indexing (messages: analyze, segment, find)
- `src/pdfWorker.ts` — PDF text extraction per page, outline extraction, progress reporting

### Auth (`src/auth/`)

Passwordless magic link flow. AuthProvider wraps the app, stores JWT in localStorage. Supports iframe embedding via `postMessage` token exchange with a 2-second timeout fallback.

### Sync (`src/sync/`)

`SyncService` is a singleton that pushes documents, sessions, and preferences to the server. Offline actions queue in IndexedDB and flush on authentication. Session updates debounce at 600ms, preferences at 1000ms. Polls every 60s for new documents from other devices.

### Text Processing (`src/textUtils.ts`)

Tokenizer splits on whitespace and merges bracketed tokens. Granularity modes: word, bigram, trigram, sentence, tweet (280-char chunks). Exported functions: `tokenize`, `segmentTokens`, `segmentTextBySentence`, `segmentTextByTweet`.

### Bookmarks (`src/bookmarkUtils.ts`)

Encodes reading positions as Base64 URL fragments (`#/reading?bm=[encoded]`). Includes version, index, page info, and document hash for shareability.

## API Endpoints

Backend runs separately (proxied at `/api` in dev). Key routes:
- `POST /api/auth/magic-link` — request magic link
- `POST /api/auth/magic-link/verify` — verify magic link token
- `GET /api/reader-docs` — list documents; `POST` — upload document file
- `PUT /api/reader-docs/{id}/session` — save reading session
- `GET /api/reader-docs/{id}/file` — download document file
- `PUT /api/reader-docs/preferences` — save preferences

## Deployment

Vite base path is `/reading/` — all routes are under this prefix. The app works standalone or embedded in an iframe. Environment variable `VITE_API_BASE_URL` overrides the API origin (defaults to same-origin).

## Styling

CSS custom properties in `src/index.css`. Key design tokens: `--display-scale`, `--fit-scale`. Fonts: Space Grotesk (body), Orbitron (display), Rajdhani (deck), Literata (serif). View modes: deck, mobile, focus, focus-text.

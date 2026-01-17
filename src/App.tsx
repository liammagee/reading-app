import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { buildApiUrl } from './env';
import './App.css';

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

type Bookmark = {
  id: string;
  index: number;
  label: string;
  createdAt: number;
  pageNumber?: number;
  pageOffset?: number;
};

type SavedSession = {
  index: number;
  wpm: number;
  chunkSize: number;
  sentencePauseMs?: number;
  sessionElapsedMs?: number;
  title: string;
  bookmarks: Bookmark[];
  furthestRead?: FurthestRead | null;
  updatedAt: number;
};

type LastDocument = {
  sourceText: string;
  parallelText?: string;
  title?: string;
  sourceKind?: 'text' | 'pdf';
  primaryFileMeta?: FileMeta | null;
};

type UserPreferences = {
  showConventional?: boolean;
  showParallel?: boolean;
  conventionalSeekEnabled?: boolean;
  autoFollowConventional?: boolean;
  cameraEnabled?: boolean;
  cameraHandEnabled?: boolean;
  cameraEyeEnabled?: boolean;
  cameraPreview?: boolean;
};

type Notice = {
  kind: 'info' | 'success' | 'error';
  message: string;
};

type PageRange = {
  start: number;
  end: number;
};

type FileMeta = {
  name: string;
  size: number;
  lastModified: number;
};

type PdfOutlineItem = {
  title: string;
  pageNumber: number | null;
  url?: string;
  items: PdfOutlineItem[];
};

type PdfState = {
  fileName: string;
  pageCount: number;
  pageTexts: string[];
  pageTokenCounts: number[];
  outline: PdfOutlineItem[];
};

type FurthestRead = {
  index: number;
  updatedAt: number;
  pageNumber?: number;
  pageOffset?: number;
};

type ResolvedBookmark = Bookmark & {
  resolvedIndex: number | null;
  outOfRange: boolean;
};

type ResolvedFurthest = FurthestRead & {
  resolvedIndex: number | null;
  outOfRange: boolean;
};

type NormalizedLandmark = {
  x: number;
  y: number;
  z?: number;
};

const DEFAULT_TEXT = `Paste a passage to begin.

Focus Reader flashes one word at a time so you can increase speed while staying attentive. Try adjusting the WPM slider or chunk size, then hit Start.
`;
const CONVENTIONAL_MODE_KEY = 'reader:conventionalMode';
const LAST_DOCUMENT_KEY = 'reader:lastDocument';
const LAST_DOCUMENT_DB = 'reader-documents';
const LAST_DOCUMENT_STORE = 'documents';
const LAST_DOCUMENT_ID = 'last';
const USER_PREFERENCES_KEY = 'reader:preferences';
const BRACKET_PAIRS: Array<[string, string]> = [
  ['[', ']'],
  ['(', ')'],
  ['{', '}'],
  ['<', '>'],
];
const LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144] as const;
const RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380] as const;

const tokenize = (text: string) => {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) return [];
  return mergeBracketedTokens(normalized.split(' '));
};

const hashText = (text: string) => {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash << 5) - hash + text.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash).toString(36);
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const getPivotIndex = (word: string) => {
  if (!word) return 0;
  const leadingMatch = word.match(/^[^\p{L}\p{N}]*/u);
  const trailingMatch = word.match(/[^\p{L}\p{N}]*$/u);
  const leading = leadingMatch ? leadingMatch[0].length : 0;
  const trailing = trailingMatch ? trailingMatch[0].length : 0;
  const coreLength = Math.max(0, word.length - leading - trailing);
  if (coreLength <= 1) return clamp(leading, 0, Math.max(0, word.length - 1));
  const pivot = Math.floor((coreLength + 2) / 4);
  const boundedPivot = clamp(pivot, 0, coreLength - 1);
  return clamp(leading + boundedPivot, 0, Math.max(0, word.length - 1));
};

const buildSnippet = (tokens: string[], activeIndex: number, radius: number) => {
  const start = clamp(activeIndex - radius, 0, Math.max(0, tokens.length - 1));
  const end = clamp(activeIndex + radius + 1, 0, tokens.length);
  return tokens.slice(start, end).map((token, offset) => ({
    text: token,
    isActive: start + offset === activeIndex,
  }));
};

const mergeBracketedTokens = (tokens: string[]) => {
  const merged: string[] = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    let handled = false;
    for (const [open, close] of BRACKET_PAIRS) {
      if (token === open) {
        const next = tokens[i + 1];
        const nextNext = tokens[i + 2];
        if (next && nextNext === close) {
          merged.push(`${open}${next}${close}`);
          i += 2;
          handled = true;
          break;
        }
        if (next && next.endsWith(close)) {
          merged.push(`${open}${next}`);
          i += 1;
          handled = true;
          break;
        }
      }
      if (token.startsWith(open) && token !== open && !token.endsWith(close)) {
        const next = tokens[i + 1];
        if (next === close) {
          merged.push(`${token}${close}`);
          i += 1;
          handled = true;
          break;
        }
      }
    }
    if (!handled) {
      merged.push(token);
    }
  }
  return merged;
};

const distance = (a: NormalizedLandmark, b: NormalizedLandmark) =>
  Math.hypot(a.x - b.x, a.y - b.y);

const computeEyeAspectRatio = (landmarks: NormalizedLandmark[], indices: readonly number[]) => {
  const points = indices.map((index) => landmarks[index]);
  if (points.some((point) => !point)) return null;
  const [p1, p2, p3, p4, p5, p6] = points as NormalizedLandmark[];
  const horizontal = distance(p1, p4);
  if (horizontal === 0) return null;
  const vertical = distance(p2, p6) + distance(p3, p5);
  return vertical / (2 * horizontal);
};

const getBlinkRatio = (landmarks: NormalizedLandmark[]) => {
  const left = computeEyeAspectRatio(landmarks, LEFT_EYE_LANDMARKS);
  const right = computeEyeAspectRatio(landmarks, RIGHT_EYE_LANDMARKS);
  if (left === null || right === null) return null;
  return (left + right) / 2;
};

const formatDuration = (ms: number) => {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

const getSentencePauseMs = (word: string, basePauseMs: number) => {
  const trimmed = word.trim();
  if (!trimmed) return 0;
  const base = Math.max(0, basePauseMs);
  const ellipsisPause = Math.round(base * 1.1);
  const minorPause = Math.round(base * 0.6);
  if (/\.{3}$/.test(trimmed)) return ellipsisPause;
  if (/[.!?]["')\]]*$/.test(trimmed)) return base;
  if (/[;:]["')\]]*$/.test(trimmed)) return minorPause;
  return 0;
};

const openLastDocumentDb = () =>
  new Promise<IDBDatabase>((resolve, reject) => {
    const request = window.indexedDB.open(LAST_DOCUMENT_DB, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(LAST_DOCUMENT_STORE)) {
        db.createObjectStore(LAST_DOCUMENT_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

const readLastDocumentFromDb = async (): Promise<LastDocument | null> => {
  if (typeof window === 'undefined' || !('indexedDB' in window)) return null;
  try {
    const db = await openLastDocumentDb();
    return await new Promise<LastDocument | null>((resolve, reject) => {
      const transaction = db.transaction(LAST_DOCUMENT_STORE, 'readonly');
      const store = transaction.objectStore(LAST_DOCUMENT_STORE);
      const request = store.get(LAST_DOCUMENT_ID);
      request.onsuccess = () => resolve((request.result as LastDocument) ?? null);
      request.onerror = () => reject(request.error);
      transaction.oncomplete = () => db.close();
      transaction.onerror = () => {
        db.close();
        reject(transaction.error);
      };
    });
  } catch (error) {
    console.warn('Failed to read last document from IndexedDB.', error);
    return null;
  }
};

const writeLastDocumentToDb = async (payload: LastDocument) => {
  if (typeof window === 'undefined' || !('indexedDB' in window)) return;
  try {
    const db = await openLastDocumentDb();
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(LAST_DOCUMENT_STORE, 'readwrite');
      const store = transaction.objectStore(LAST_DOCUMENT_STORE);
      store.put(payload, LAST_DOCUMENT_ID);
      transaction.oncomplete = () => {
        db.close();
        resolve();
      };
      transaction.onerror = () => {
        db.close();
        reject(transaction.error);
      };
    });
  } catch (error) {
    console.warn('Failed to persist last document to IndexedDB.', error);
  }
};

const isPdfFile = (file: File) => {
  const name = file.name.toLowerCase();
  return file.type === 'application/pdf' || name.endsWith('.pdf');
};

const normalizeRange = (start: number, end: number, pageCount: number): PageRange => {
  const safeStart = clamp(Math.floor(start || 1), 1, Math.max(1, pageCount));
  const safeEnd = clamp(Math.floor(end || pageCount), safeStart, Math.max(safeStart, pageCount));
  return { start: safeStart, end: safeEnd };
};

const buildRangeText = (
  pageTexts: string[],
  pageTokenCounts: number[],
  start: number,
  end: number,
) => {
  const startIndex = Math.max(0, start - 1);
  const endIndex = Math.min(pageTexts.length - 1, end - 1);
  const offsets: number[] = [];
  let offset = 0;
  for (let i = startIndex; i <= endIndex; i += 1) {
    offsets.push(offset);
    offset += pageTokenCounts[i] || 0;
  }
  return {
    text: pageTexts.slice(startIndex, endIndex + 1).join('\n\n'),
    offsets,
  };
};

const extractPdfData = async (
  data: Uint8Array,
  onProgress?: (current: number, total: number) => void,
) => {
  const loadingTask = getDocument({ data });
  try {
    const pdf = await loadingTask.promise;
    const pageTexts: string[] = [];
    const pageTokenCounts: number[] = [];

    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum += 1) {
      onProgress?.(pageNum, pdf.numPages);
      const page = await pdf.getPage(pageNum);
      const content = await page.getTextContent();
      const pageText = content.items
        .map((item) => ('str' in item ? item.str : ''))
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim();
      pageTexts.push(pageText);
      pageTokenCounts.push(tokenize(pageText).length);
    }

    const outline = await (async () => {
      const rawOutline = await pdf.getOutline();
      if (!rawOutline) return [] as PdfOutlineItem[];

      const resolveDest = async (dest: unknown) => {
        if (!dest) return null;
        const explicitDest = typeof dest === 'string' ? await pdf.getDestination(dest) : dest;
        if (!explicitDest || !Array.isArray(explicitDest)) return null;
        try {
          const pageIndex = await pdf.getPageIndex(explicitDest[0]);
          return pageIndex + 1;
        } catch {
          return null;
        }
      };

      const mapItems = async (items: any[]): Promise<PdfOutlineItem[]> => {
        const mapped: PdfOutlineItem[] = [];
        for (const item of items) {
          const pageNumber = await resolveDest(item.dest);
          const children = item.items ? await mapItems(item.items) : [];
          mapped.push({
            title: item.title || 'Untitled',
            pageNumber,
            url: item.url || undefined,
            items: children,
          });
        }
        return mapped;
      };

      return mapItems(rawOutline);
    })();

    return {
      pageTexts,
      pageTokenCounts,
      outline,
      pageCount: pdf.numPages,
    };
  } finally {
    await loadingTask.destroy();
  }
};

function App() {
  const [title, setTitle] = useState('Untitled Session');
  const [sourceText, setSourceText] = useState(DEFAULT_TEXT);
  const [parallelText, setParallelText] = useState('');
  const [primaryFileMeta, setPrimaryFileMeta] = useState<FileMeta | null>(null);
  const [wpm, setWpm] = useState(320);
  const [chunkSize, setChunkSize] = useState(1);
  const [sentencePauseMs, setSentencePauseMs] = useState(200);
  const [sessionElapsedMs, setSessionElapsedMs] = useState(0);
  const [wordIndex, setWordIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showConventional, setShowConventional] = useState(true);
  const [conventionalMode, setConventionalMode] = useState<'excerpt' | 'rendered'>(() => {
    if (typeof window === 'undefined') return 'rendered';
    const stored = window.localStorage.getItem(CONVENTIONAL_MODE_KEY);
    return stored === 'excerpt' || stored === 'rendered' ? stored : 'rendered';
  });
  const [showParallel, setShowParallel] = useState(true);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [furthestRead, setFurthestRead] = useState<FurthestRead | null>(null);
  const [sourceKind, setSourceKind] = useState<'text' | 'pdf'>('text');
  const [pdfState, setPdfState] = useState<PdfState | null>(null);
  const [pageRange, setPageRange] = useState<PageRange>({ start: 1, end: 1 });
  const [pageRangeDraft, setPageRangeDraft] = useState<PageRange>({ start: 1, end: 1 });
  const [pageOffsets, setPageOffsets] = useState<number[]>([]);
  const [isPdfBusy, setIsPdfBusy] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [ttsAvailable, setTtsAvailable] = useState<boolean | null>(null);
  const [ttsVoices, setTtsVoices] = useState<string[]>([]);
  const [ttsVoice, setTtsVoice] = useState('sarah');
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [ttsLanguage, setTtsLanguage] = useState('en-us');
  const [ttsChunkWords, setTtsChunkWords] = useState(26);
  const [ttsNotice, setTtsNotice] = useState<Notice | null>(null);
  const [ttsChecking, setTtsChecking] = useState(false);
  const [conventionalSeekEnabled, setConventionalSeekEnabled] = useState(true);
  const [autoFollowConventional, setAutoFollowConventional] = useState(true);
  const [primaryNotice, setPrimaryNotice] = useState<Notice | null>(null);
  const [parallelNotice, setParallelNotice] = useState<Notice | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraHandEnabled, setCameraHandEnabled] = useState(true);
  const [cameraEyeEnabled, setCameraEyeEnabled] = useState(true);
  const [cameraPreview, setCameraPreview] = useState(false);
  const [cameraNotice, setCameraNotice] = useState<Notice | null>(null);
  const [companionInput, setCompanionInput] = useState('');
  const [companionLog, setCompanionLog] = useState<string[]>([
    'Hermeneutics companion will live here. Connect to /api/tutor/quick-chat when ready.',
  ]);

  const tokens = useMemo(() => tokenize(sourceText), [sourceText]);
  const parallelTokens = useMemo(() => tokenize(parallelText), [parallelText]);
  const docKey = useMemo(() => {
    if (sourceKind === 'pdf' && primaryFileMeta) {
      return `reader:pdf:${primaryFileMeta.name}:${primaryFileMeta.size}:${primaryFileMeta.lastModified}`;
    }
    if (primaryFileMeta) {
      return `reader:text:${primaryFileMeta.name}:${primaryFileMeta.size}:${primaryFileMeta.lastModified}`;
    }
    if (!sourceText.trim()) return '';
    return `reader:text:${hashText(sourceText)}`;
  }, [primaryFileMeta, sourceKind, sourceText]);

  const saveTimerRef = useRef<number | null>(null);
  const pdfDataRef = useRef<Uint8Array | null>(null);
  const ttsAbortRef = useRef<AbortController | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const ttsAudioUrlRef = useRef<string | null>(null);
  const ttsIntervalRef = useRef<number | null>(null);
  const ttsEnabledRef = useRef(false);
  const ttsPlayingRef = useRef(false);
  const ttsConfigRef = useRef({
    voice: 'sarah',
    speed: 1.0,
    language: 'en-us',
    chunkWords: 26,
  });
  const tokensRef = useRef<string[]>([]);
  const conventionalRef = useRef<HTMLDivElement | null>(null);
  const scrollRafRef = useRef<number | null>(null);
  const scrollLockRef = useRef(false);
  const scrollEndTimeoutRef = useRef<number | null>(null);
  const autoFollowRafRef = useRef<number | null>(null);
  const prevActiveWordRef = useRef<number | null>(null);
  const isPlayingRef = useRef(false);
  const conventionalSeekEnabledRef = useRef(true);
  const wordIndexRef = useRef(0);
  const chunkSizeRef = useRef(chunkSize);
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const cameraRafRef = useRef<number | null>(null);
  const handLandmarkerRef = useRef<any>(null);
  const faceLandmarkerRef = useRef<any>(null);
  const handleSeekRef = useRef<(index: number, options?: { pause?: boolean }) => void>(() => {});
  const handHistoryRef = useRef<Array<{ x: number; t: number }>>([]);
  const blinkStartRef = useRef<number | null>(null);
  const blinkLockedRef = useRef(false);
  const lastGestureAtRef = useRef(0);
  const cameraHandEnabledRef = useRef(true);
  const cameraEyeEnabledRef = useRef(true);
  const sessionTimerRef = useRef<number | null>(null);
  const sessionStartRef = useRef<number | null>(null);
  const sessionAccumulatedRef = useRef(0);
  const postPauseDelayRef = useRef(0);
  const lastDocSaveTimerRef = useRef<number | null>(null);
  const didHydrateRef = useRef(false);
  const prefsSaveTimerRef = useRef<number | null>(null);
  const didPrefHydrateRef = useRef(false);

  useEffect(() => {
    ttsEnabledRef.current = ttsEnabled;
  }, [ttsEnabled]);

  useEffect(() => {
    ttsPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    conventionalSeekEnabledRef.current = conventionalSeekEnabled;
  }, [conventionalSeekEnabled]);

  useEffect(() => {
    wordIndexRef.current = wordIndex;
  }, [wordIndex]);

  useEffect(() => {
    chunkSizeRef.current = chunkSize;
  }, [chunkSize]);

  useEffect(() => {
    cameraHandEnabledRef.current = cameraHandEnabled;
  }, [cameraHandEnabled]);

  useEffect(() => {
    cameraEyeEnabledRef.current = cameraEyeEnabled;
  }, [cameraEyeEnabled]);

  useEffect(() => {
    ttsConfigRef.current = {
      voice: ttsVoice,
      speed: ttsSpeed,
      language: ttsLanguage,
      chunkWords: ttsChunkWords,
    };
  }, [ttsVoice, ttsSpeed, ttsLanguage, ttsChunkWords, tokens.length]);

  useEffect(() => {
    tokensRef.current = tokens;
    prevActiveWordRef.current = null;
  }, [tokens]);

  useEffect(() => {
    let cancelled = false;
    const hydrate = async () => {
      if (typeof window === 'undefined') return;
      let parsed: LastDocument | null = null;
      const saved = window.localStorage.getItem(LAST_DOCUMENT_KEY);
      if (saved) {
        try {
          parsed = JSON.parse(saved) as LastDocument;
        } catch {
          window.localStorage.removeItem(LAST_DOCUMENT_KEY);
        }
      }
      if (!parsed) {
        parsed = await readLastDocumentFromDb();
      }
      if (!parsed || cancelled) return;
      if (typeof parsed.sourceText === 'string') {
        setSourceText(parsed.sourceText);
      }
      if (typeof parsed.parallelText === 'string') {
        setParallelText(parsed.parallelText);
      }
      if (typeof parsed.title === 'string' && parsed.title.trim()) {
        setTitle(parsed.title);
      }
      if (parsed.sourceKind === 'pdf' || parsed.sourceKind === 'text') {
        setSourceKind(parsed.sourceKind);
      }
      if (
        parsed.primaryFileMeta &&
        typeof parsed.primaryFileMeta.name === 'string' &&
        typeof parsed.primaryFileMeta.size === 'number' &&
        typeof parsed.primaryFileMeta.lastModified === 'number'
      ) {
        setPrimaryFileMeta(parsed.primaryFileMeta);
      }
    };
    hydrate()
      .catch((error) => {
        console.warn('Failed to hydrate last document snapshot.', error);
      })
      .finally(() => {
        if (!cancelled) {
          didHydrateRef.current = true;
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const saved = window.localStorage.getItem(USER_PREFERENCES_KEY);
    if (!saved) {
      didPrefHydrateRef.current = true;
      return;
    }
    try {
      const parsed = JSON.parse(saved) as UserPreferences;
      if (typeof parsed.showConventional === 'boolean') {
        setShowConventional(parsed.showConventional);
      }
      if (typeof parsed.showParallel === 'boolean') {
        setShowParallel(parsed.showParallel);
      }
      if (typeof parsed.conventionalSeekEnabled === 'boolean') {
        setConventionalSeekEnabled(parsed.conventionalSeekEnabled);
      }
      if (typeof parsed.autoFollowConventional === 'boolean') {
        setAutoFollowConventional(parsed.autoFollowConventional);
      }
      if (typeof parsed.cameraEnabled === 'boolean') {
        setCameraEnabled(parsed.cameraEnabled);
      }
      if (typeof parsed.cameraHandEnabled === 'boolean') {
        setCameraHandEnabled(parsed.cameraHandEnabled);
      }
      if (typeof parsed.cameraEyeEnabled === 'boolean') {
        setCameraEyeEnabled(parsed.cameraEyeEnabled);
      }
      if (typeof parsed.cameraPreview === 'boolean') {
        setCameraPreview(parsed.cameraPreview);
      }
    } catch {
      // Ignore invalid preference data.
    } finally {
      didPrefHydrateRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!didPrefHydrateRef.current) return;
    if (prefsSaveTimerRef.current) {
      window.clearTimeout(prefsSaveTimerRef.current);
    }
    prefsSaveTimerRef.current = window.setTimeout(() => {
      const payload: UserPreferences = {
        showConventional,
        showParallel,
        conventionalSeekEnabled,
        autoFollowConventional,
        cameraEnabled,
        cameraHandEnabled,
        cameraEyeEnabled,
        cameraPreview,
      };
      try {
        window.localStorage.setItem(USER_PREFERENCES_KEY, JSON.stringify(payload));
      } catch (error) {
        console.warn('Failed to persist user preferences.', error);
      }
    }, 250);
    return () => {
      if (prefsSaveTimerRef.current) {
        window.clearTimeout(prefsSaveTimerRef.current);
      }
    };
  }, [
    autoFollowConventional,
    cameraEnabled,
    cameraEyeEnabled,
    cameraHandEnabled,
    cameraPreview,
    conventionalSeekEnabled,
    showConventional,
    showParallel,
  ]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!didHydrateRef.current) return;
    if (lastDocSaveTimerRef.current) {
      window.clearTimeout(lastDocSaveTimerRef.current);
    }
    lastDocSaveTimerRef.current = window.setTimeout(() => {
      const payload: LastDocument = {
        sourceText,
        parallelText,
        title,
        sourceKind,
        primaryFileMeta,
      };
      let storedLocally = false;
      try {
        window.localStorage.setItem(LAST_DOCUMENT_KEY, JSON.stringify(payload));
        storedLocally = true;
      } catch (error) {
        console.warn('Failed to persist last document snapshot.', error);
        try {
          window.localStorage.removeItem(LAST_DOCUMENT_KEY);
        } catch {
          // Ignore storage cleanup errors.
        }
      }
      if (!storedLocally) {
        void writeLastDocumentToDb(payload);
      }
    }, 400);
    return () => {
      if (lastDocSaveTimerRef.current) {
        window.clearTimeout(lastDocSaveTimerRef.current);
      }
    };
  }, [parallelText, primaryFileMeta, sourceKind, sourceText, title]);

  useEffect(() => {
    if (!cameraEnabled) {
      setCameraNotice(null);
      if (cameraRafRef.current) {
        window.cancelAnimationFrame(cameraRafRef.current);
        cameraRafRef.current = null;
      }
      if (handLandmarkerRef.current?.close) {
        handLandmarkerRef.current.close();
      }
      if (faceLandmarkerRef.current?.close) {
        faceLandmarkerRef.current.close();
      }
      handLandmarkerRef.current = null;
      faceLandmarkerRef.current = null;
      handHistoryRef.current = [];
      blinkStartRef.current = null;
      blinkLockedRef.current = false;
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((track) => track.stop());
        cameraStreamRef.current = null;
      }
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = null;
      }
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraNotice({ kind: 'error', message: 'Camera access is unavailable in this browser.' });
      setCameraEnabled(false);
      return;
    }

    let cancelled = false;
    const gestureCooldownMs = 900;
    const swipeWindowMs = 320;
    const swipeThreshold = 0.28;
    const blinkHoldMs = 240;
    const blinkThreshold = 0.19;

    const cleanup = () => {
      if (cameraRafRef.current) {
        window.cancelAnimationFrame(cameraRafRef.current);
        cameraRafRef.current = null;
      }
      if (handLandmarkerRef.current?.close) {
        handLandmarkerRef.current.close();
      }
      if (faceLandmarkerRef.current?.close) {
        faceLandmarkerRef.current.close();
      }
      handLandmarkerRef.current = null;
      faceLandmarkerRef.current = null;
      handHistoryRef.current = [];
      blinkStartRef.current = null;
      blinkLockedRef.current = false;
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((track) => track.stop());
        cameraStreamRef.current = null;
      }
      if (cameraVideoRef.current) {
        cameraVideoRef.current.srcObject = null;
      }
    };

    const triggerSeek = (direction: 'back' | 'forward') => {
      if (!tokensRef.current.length) return;
      const delta = direction === 'back' ? -chunkSizeRef.current : chunkSizeRef.current;
      handleSeekRef.current(wordIndexRef.current + delta);
    };

    const triggerPauseToggle = () => {
      if (!tokensRef.current.length) return;
      setIsPlaying((prev) => !prev);
    };

    const startCamera = async () => {
      setCameraNotice({ kind: 'info', message: 'Starting camera...' });
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        cameraStreamRef.current = stream;
        const video = cameraVideoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play();
        }

        const visionModule = await import('@mediapipe/tasks-vision');
        const vision = await visionModule.FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
        );
        const handLandmarker = await visionModule.HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
        });
        const faceLandmarker = await visionModule.FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });

        if (cancelled) {
          handLandmarker.close();
          faceLandmarker.close();
          return;
        }

        handLandmarkerRef.current = handLandmarker;
        faceLandmarkerRef.current = faceLandmarker;
        setCameraNotice({ kind: 'success', message: 'Camera active.' });

        const loop = () => {
          if (cancelled) return;
          const video = cameraVideoRef.current;
          if (!video || video.readyState < 2) {
            cameraRafRef.current = window.requestAnimationFrame(loop);
            return;
          }
          const now = performance.now();

          if (cameraHandEnabledRef.current && handLandmarkerRef.current) {
            const result = handLandmarkerRef.current.detectForVideo(video, now);
            const hand = result?.landmarks?.[0] as NormalizedLandmark[] | undefined;
            if (hand && hand[0]) {
              const history = handHistoryRef.current;
              history.push({ x: hand[0].x, t: now });
              while (history.length && now - history[0].t > swipeWindowMs) {
                history.shift();
              }
              const first = history[0];
              const last = history[history.length - 1];
              if (
                first &&
                last &&
                Math.abs(last.x - first.x) > swipeThreshold &&
                now - lastGestureAtRef.current > gestureCooldownMs
              ) {
                triggerSeek(last.x > first.x ? 'forward' : 'back');
                lastGestureAtRef.current = now;
                history.length = 0;
              }
            } else {
              handHistoryRef.current = [];
            }
          } else {
            handHistoryRef.current = [];
          }

          if (cameraEyeEnabledRef.current && faceLandmarkerRef.current) {
            const result = faceLandmarkerRef.current.detectForVideo(video, now);
            const face = result?.faceLandmarks?.[0] as NormalizedLandmark[] | undefined;
            if (face && face.length) {
              const ratio = getBlinkRatio(face);
              if (ratio !== null && ratio < blinkThreshold) {
                if (blinkLockedRef.current) {
                  // Wait for eyes to reopen before allowing another toggle.
                } else if (!blinkStartRef.current) {
                  blinkStartRef.current = now;
                } else if (
                  now - blinkStartRef.current > blinkHoldMs &&
                  now - lastGestureAtRef.current > gestureCooldownMs
                ) {
                  triggerPauseToggle();
                  lastGestureAtRef.current = now;
                  blinkStartRef.current = null;
                  blinkLockedRef.current = true;
                }
              } else {
                blinkStartRef.current = null;
                blinkLockedRef.current = false;
              }
            }
          } else {
            blinkStartRef.current = null;
            blinkLockedRef.current = false;
          }

          cameraRafRef.current = window.requestAnimationFrame(loop);
        };

        cameraRafRef.current = window.requestAnimationFrame(loop);
      } catch (error) {
        if (!cancelled) {
          console.error('Camera setup failed:', error);
          setCameraNotice({ kind: 'error', message: 'Unable to access the camera.' });
          setCameraEnabled(false);
        }
      }
    };

    startCamera();
    return () => {
      cancelled = true;
      cleanup();
    };
  }, [cameraEnabled]);

  useEffect(() => {
    if (!isPlaying || tokens.length === 0) {
      if (sessionTimerRef.current) {
        window.clearInterval(sessionTimerRef.current);
        sessionTimerRef.current = null;
      }
      if (sessionStartRef.current !== null) {
        sessionAccumulatedRef.current += performance.now() - sessionStartRef.current;
        sessionStartRef.current = null;
      }
      setSessionElapsedMs(sessionAccumulatedRef.current);
      return;
    }

    if (sessionStartRef.current === null) {
      sessionStartRef.current = performance.now();
    }
    if (sessionTimerRef.current) {
      window.clearInterval(sessionTimerRef.current);
    }
    sessionTimerRef.current = window.setInterval(() => {
      if (sessionStartRef.current === null) return;
      const now = performance.now();
      setSessionElapsedMs(sessionAccumulatedRef.current + (now - sessionStartRef.current));
    }, 1000);

    return () => {
      if (sessionTimerRef.current) {
        window.clearInterval(sessionTimerRef.current);
        sessionTimerRef.current = null;
      }
    };
  }, [isPlaying, tokens.length]);

  useEffect(() => {
    if (!conventionalRef.current) return;
    conventionalRef.current.scrollTo({ top: 0 });
  }, [sourceText]);

  useEffect(() => {
    if (!showConventional || conventionalMode !== 'excerpt') return;
    const container = conventionalRef.current;
    if (!container) return;
    const prevIndex = prevActiveWordRef.current;
    if (prevIndex !== null) {
      const prevEl = container.querySelector(`[data-word-index="${prevIndex}"]`) as HTMLElement | null;
      prevEl?.classList.remove('is-active');
    }
    const nextEl = container.querySelector(`[data-word-index="${wordIndex}"]`) as HTMLElement | null;
    nextEl?.classList.add('is-active');
    prevActiveWordRef.current = wordIndex;
  }, [conventionalMode, wordIndex, showConventional, tokens.length]);

  useEffect(() => {
    if (!autoFollowConventional || !showConventional || !isPlaying || conventionalMode !== 'excerpt') return;
    const container = conventionalRef.current;
    if (!container) return;
    const activeWord = container.querySelector(`[data-word-index="${wordIndex}"]`) as HTMLElement | null;
    if (!activeWord) return;
    if (autoFollowRafRef.current) return;
    autoFollowRafRef.current = window.requestAnimationFrame(() => {
      autoFollowRafRef.current = null;
      const rect = container.getBoundingClientRect();
      const wordRect = activeWord.getBoundingClientRect();
      const margin = Math.min(80, rect.height * 0.2);
      if (wordRect.top < rect.top + margin || wordRect.bottom > rect.bottom - margin) {
        const targetTop = container.scrollTop + (wordRect.top - rect.top) - rect.height / 2 + wordRect.height / 2;
        scrollLockRef.current = true;
        container.scrollTo({ top: targetTop, behavior: 'smooth' });
        window.setTimeout(() => {
          scrollLockRef.current = false;
        }, 120);
      }
    });
  }, [autoFollowConventional, conventionalMode, showConventional, isPlaying, wordIndex]);

  useEffect(() => {
    if (ttsAvailable === false && ttsEnabled) {
      setTtsEnabled(false);
    }
  }, [ttsAvailable, ttsEnabled]);

  useEffect(() => {
    if (!docKey) return;
    const saved = localStorage.getItem(docKey);
    if (!saved) {
      setBookmarks([]);
      setFurthestRead(null);
      setSessionElapsedMs(0);
      sessionAccumulatedRef.current = 0;
      sessionStartRef.current = null;
      return;
    }
    try {
      const parsed = JSON.parse(saved) as SavedSession;
      setWordIndex(clamp(parsed.index ?? 0, 0, Math.max(0, tokens.length - 1)));
      setWpm(parsed.wpm ?? 320);
      setChunkSize(parsed.chunkSize ?? 1);
      setSentencePauseMs(
        typeof parsed.sentencePauseMs === 'number' ? Math.max(0, parsed.sentencePauseMs) : 200,
      );
      const nextElapsed = typeof parsed.sessionElapsedMs === 'number' ? Math.max(0, parsed.sessionElapsedMs) : 0;
      setSessionElapsedMs(nextElapsed);
      sessionAccumulatedRef.current = nextElapsed;
      sessionStartRef.current = null;
      setTitle(parsed.title || 'Untitled Session');
      setBookmarks(parsed.bookmarks ?? []);
      setFurthestRead(parsed.furthestRead ?? null);
    } catch {
      // Ignore invalid session data.
    }
  }, [docKey, tokens.length]);

  useEffect(() => {
    localStorage.setItem(CONVENTIONAL_MODE_KEY, conventionalMode);
  }, [conventionalMode]);

  useEffect(() => {
    if (!docKey) return;
    if (saveTimerRef.current) {
      window.clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = window.setTimeout(() => {
      const payload: SavedSession = {
        index: wordIndex,
        wpm,
        chunkSize,
        sentencePauseMs,
        sessionElapsedMs,
        title,
        bookmarks,
        furthestRead,
        updatedAt: Date.now(),
      };
      localStorage.setItem(docKey, JSON.stringify(payload));
    }, 300);
    return () => {
      if (saveTimerRef.current) {
        window.clearTimeout(saveTimerRef.current);
      }
    };
  }, [docKey, wordIndex, wpm, chunkSize, sentencePauseMs, sessionElapsedMs, title, bookmarks, furthestRead]);

  useEffect(() => {
    if (!tokens.length) {
      setIsPlaying(false);
      setWordIndex(0);
      return;
    }
    setWordIndex((prev) => clamp(prev, 0, Math.max(0, tokens.length - 1)));
  }, [tokens.length]);

  useEffect(() => {
    if (!isPlaying) {
      postPauseDelayRef.current = 0;
    }
  }, [isPlaying]);

  useEffect(() => {
    if (!isPlaying || tokens.length === 0 || ttsEnabled) return;
    const intervalMs = Math.max(80, (60000 / Math.max(60, wpm)) * chunkSize);
    const lastIndex = Math.min(wordIndex + chunkSize - 1, Math.max(0, tokens.length - 1));
    const resumeDelayMs = postPauseDelayRef.current;
    const pauseMs = getSentencePauseMs(tokens[lastIndex] || '', sentencePauseMs);
    postPauseDelayRef.current = pauseMs > 0 ? Math.round(Math.min(160, pauseMs * 0.5)) : 0;
    const timerId = window.setTimeout(() => {
      setWordIndex((prev) => {
        const next = prev + chunkSize;
        if (next >= tokens.length) {
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }, intervalMs + pauseMs + resumeDelayMs);
    return () => window.clearTimeout(timerId);
  }, [chunkSize, isPlaying, sentencePauseMs, tokens, tokens.length, ttsEnabled, wordIndex, wpm]);

  const currentChunk = tokens.slice(wordIndex, wordIndex + chunkSize);
  const conventionalTokens = useMemo(() => tokens, [tokens]);
  const conventionalNodes = useMemo(() => {
    if (conventionalMode !== 'excerpt') return null;
    return conventionalTokens.map((token, index) => (
      <span
        key={`${index}-${token}`}
        data-word-index={index}
        className="word"
      >
        {token}{' '}
      </span>
    ));
  }, [conventionalMode, conventionalTokens]);
  const parallelIndex = tokens.length
    ? Math.min(parallelTokens.length - 1, Math.round((wordIndex / tokens.length) * parallelTokens.length))
    : 0;
  const parallelSnippet = useMemo(
    () => buildSnippet(parallelTokens, Math.max(0, parallelIndex), 40),
    [parallelTokens, parallelIndex],
  );
  const resolvedFurthest = useMemo<ResolvedFurthest | null>(() => {
    if (!furthestRead || tokens.length === 0) return null;
    if (sourceKind !== 'pdf') {
      const clampedIndex = clamp(furthestRead.index, 0, Math.max(0, tokens.length - 1));
      return {
        ...furthestRead,
        resolvedIndex: clampedIndex,
        outOfRange: clampedIndex < 0 || clampedIndex >= tokens.length,
      };
    }
    if (!pdfState || pageOffsets.length === 0) {
      const clampedIndex = clamp(furthestRead.index, 0, Math.max(0, tokens.length - 1));
      return {
        ...furthestRead,
        resolvedIndex: clampedIndex,
        outOfRange: clampedIndex < 0 || clampedIndex >= tokens.length,
      };
    }
    if (furthestRead.pageNumber == null || furthestRead.pageOffset == null) {
      const clampedIndex = clamp(furthestRead.index, 0, Math.max(0, tokens.length - 1));
      return {
        ...furthestRead,
        resolvedIndex: clampedIndex,
        outOfRange: clampedIndex < 0 || clampedIndex >= tokens.length,
      };
    }
    if (furthestRead.pageNumber < pageRange.start || furthestRead.pageNumber > pageRange.end) {
      return {
        ...furthestRead,
        resolvedIndex: null,
        outOfRange: true,
      };
    }
    const pageOffset = pageOffsets[furthestRead.pageNumber - pageRange.start];
    if (typeof pageOffset !== 'number') {
      return {
        ...furthestRead,
        resolvedIndex: null,
        outOfRange: true,
      };
    }
    const resolvedIndex = pageOffset + furthestRead.pageOffset;
    return {
      ...furthestRead,
      resolvedIndex,
      outOfRange: resolvedIndex < 0 || resolvedIndex >= tokens.length,
    };
  }, [furthestRead, pageOffsets, pageRange.end, pageRange.start, pdfState, sourceKind, tokens.length]);
  const resolvedBookmarks = useMemo<ResolvedBookmark[]>(() => {
    if (sourceKind !== 'pdf') {
      return bookmarks.map((bookmark) => ({
        ...bookmark,
        resolvedIndex: bookmark.index,
        outOfRange: bookmark.index < 0 || bookmark.index >= tokens.length,
      }));
    }
    if (!pdfState || pageOffsets.length === 0) {
      return bookmarks.map((bookmark) => ({
        ...bookmark,
        resolvedIndex: bookmark.index,
        outOfRange: bookmark.index < 0 || bookmark.index >= tokens.length,
      }));
    }
    return bookmarks.map((bookmark) => {
      if (bookmark.pageNumber == null || bookmark.pageOffset == null) {
        return {
          ...bookmark,
          resolvedIndex: bookmark.index,
          outOfRange: bookmark.index < 0 || bookmark.index >= tokens.length,
        };
      }
      if (bookmark.pageNumber < pageRange.start || bookmark.pageNumber > pageRange.end) {
        return {
          ...bookmark,
          resolvedIndex: null,
          outOfRange: true,
        };
      }
      const pageOffset = pageOffsets[bookmark.pageNumber - pageRange.start];
      if (typeof pageOffset !== 'number') {
        return {
          ...bookmark,
          resolvedIndex: null,
          outOfRange: true,
        };
      }
      const resolvedIndex = pageOffset + bookmark.pageOffset;
      return {
        ...bookmark,
        resolvedIndex,
        outOfRange: resolvedIndex < 0 || resolvedIndex >= tokens.length,
      };
    });
  }, [bookmarks, pageOffsets, pageRange.end, pageRange.start, pdfState, sourceKind, tokens.length]);

  const activePageIndex = useMemo(() => {
    if (sourceKind !== 'pdf' || pageOffsets.length === 0) return null;
    let index = 0;
    for (let i = 0; i < pageOffsets.length; i += 1) {
      if (pageOffsets[i] <= wordIndex) {
        index = i;
      } else {
        break;
      }
    }
    return index;
  }, [pageOffsets, sourceKind, wordIndex]);

  const activePageNumber = useMemo(() => {
    if (activePageIndex === null) return null;
    return pageRange.start + activePageIndex;
  }, [activePageIndex, pageRange.start]);

  const wordsRead = useMemo(() => {
    if (!tokens.length) return 0;
    const resolvedIndex =
      resolvedFurthest?.resolvedIndex != null ? resolvedFurthest.resolvedIndex : wordIndex;
    const hasStarted = sessionElapsedMs > 0 || resolvedIndex > 0;
    if (!hasStarted) return 0;
    return clamp(resolvedIndex + 1, 0, tokens.length);
  }, [resolvedFurthest, sessionElapsedMs, tokens.length, wordIndex]);

  const sessionPercent = tokens.length ? Math.round((wordsRead / tokens.length) * 100) : 0;
  const sessionDurationLabel = useMemo(() => formatDuration(sessionElapsedMs), [sessionElapsedMs]);

  useEffect(() => {
    if (!tokens.length) {
      setFurthestRead(null);
      return;
    }
    const current: FurthestRead = {
      index: wordIndex,
      updatedAt: Date.now(),
    };
    if (sourceKind === 'pdf' && activePageNumber !== null && activePageIndex !== null) {
      current.pageNumber = activePageNumber;
      const pageStart = pageOffsets[activePageIndex] ?? 0;
      current.pageOffset = Math.max(0, wordIndex - pageStart);
    }
    setFurthestRead((prev) => {
      if (!prev) return current;
      if (sourceKind === 'pdf' && prev.pageNumber != null && current.pageNumber != null) {
        if (current.pageNumber > prev.pageNumber) return current;
        if (
          current.pageNumber === prev.pageNumber &&
          (current.pageOffset ?? 0) > (prev.pageOffset ?? 0)
        ) {
          return current;
        }
        return prev;
      }
      return current.index > prev.index ? current : prev;
    });
  }, [activePageIndex, activePageNumber, pageOffsets, sourceKind, tokens.length, wordIndex]);

  const pdfStats = useMemo(() => {
    if (!pdfState) return null;
    const emptyPages = pdfState.pageTokenCounts.filter((count) => count < 5).length;
    return {
      pageCount: pdfState.pageCount,
      emptyPages,
    };
  }, [pdfState]);

  const progressPercent = tokens.length ? Math.round((wordIndex / tokens.length) * 100) : 0;
  const wordsRemaining = Math.max(0, tokens.length - wordIndex);
  const minutesRemaining = wpm ? Math.max(0, wordsRemaining / wpm) : 0;

  const applyPdfRange = (
    start: number,
    end: number,
    options?: { focusPage?: number; pdfOverride?: PdfState },
  ) => {
    const activePdf = options?.pdfOverride ?? pdfState;
    if (!activePdf) return;
    const range = normalizeRange(start, end, activePdf.pageCount);
    const { text, offsets } = buildRangeText(
      activePdf.pageTexts,
      activePdf.pageTokenCounts,
      range.start,
      range.end,
    );
    setSourceText(text);
    setPageOffsets(offsets);
    setPageRange(range);
    setPageRangeDraft(range);
    setIsPlaying(false);
    if (options?.focusPage) {
      const focus = clamp(options.focusPage, range.start, range.end);
      const localIndex = focus - range.start;
      setWordIndex(offsets[localIndex] ?? 0);
    } else {
      setWordIndex(0);
    }
  };

  const buildTtsChunk = (startIndex: number) => {
    const sourceTokens = tokensRef.current;
    if (startIndex >= sourceTokens.length) return null;
    const { chunkWords } = ttsConfigRef.current;
    const maxChars = 900;
    const minWords = Math.max(8, Math.floor(chunkWords / 2));
    let endIndex = startIndex;
    let charCount = 0;
    let wordsCount = 0;

    while (endIndex < sourceTokens.length && wordsCount < chunkWords && charCount < maxChars) {
      const word = sourceTokens[endIndex];
      charCount += word.length + 1;
      endIndex += 1;
      wordsCount += 1;
      if (wordsCount >= minWords && /[.!?]["')]*$/.test(word)) {
        break;
      }
    }

    if (endIndex <= startIndex) return null;
    return {
      startIndex,
      endIndex,
      text: sourceTokens.slice(startIndex, endIndex).join(' '),
      wordsCount,
    };
  };

  const refreshTtsStatus = async () => {
    setTtsChecking(true);
    setTtsNotice({ kind: 'info', message: 'Checking Lemonfox status...' });
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 4000);

    try {
      const response = await fetch(buildApiUrl('/api/tts/status'), {
        signal: controller.signal,
        cache: 'no-store',
      });
      window.clearTimeout(timeoutId);
      if (!response.ok) {
        setTtsAvailable(false);
        setTtsNotice({ kind: 'error', message: 'Unable to reach TTS status endpoint.' });
        return false;
      }
      const status = await response.json();
      const lemonfox = status?.lemonfox;
      const available = Boolean(lemonfox?.available);
      const voices = Array.isArray(lemonfox?.voices) ? lemonfox.voices : [];
      setTtsAvailable(available);
      setTtsVoices(voices);
      if (voices.length > 0 && !voices.includes(ttsVoice)) {
        setTtsVoice(voices[0]);
      }
      if (available) {
        setTtsNotice({ kind: 'success', message: 'Lemonfox ready.' });
      } else {
        setTtsNotice({ kind: 'info', message: 'Lemonfox not configured. Set LEMONFOX_API_KEY.' });
      }
      return available;
    } catch (error) {
      console.error('TTS status error:', error);
      setTtsAvailable(false);
      setTtsNotice({ kind: 'error', message: 'Unable to reach TTS status endpoint.' });
      return false;
    } finally {
      window.clearTimeout(timeoutId);
      setTtsChecking(false);
    }
  };

  const handleTtsToggle = async (nextValue: boolean) => {
    if (!nextValue) {
      setTtsEnabled(false);
      stopTtsPlayback({ clearNotice: false });
      return;
    }
    const available = await refreshTtsStatus();
    if (!available) {
      setTtsEnabled(false);
      return;
    }
    setTtsEnabled(true);
  };

  const stopTtsPlayback = (options?: { clearNotice?: boolean }) => {
    if (ttsAbortRef.current) {
      ttsAbortRef.current.abort();
      ttsAbortRef.current = null;
    }
    if (ttsIntervalRef.current) {
      window.clearInterval(ttsIntervalRef.current);
      ttsIntervalRef.current = null;
    }
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause();
      ttsAudioRef.current.src = '';
      ttsAudioRef.current = null;
    }
    if (ttsAudioUrlRef.current) {
      URL.revokeObjectURL(ttsAudioUrlRef.current);
      ttsAudioUrlRef.current = null;
    }
    if (options?.clearNotice) {
      setTtsNotice(null);
    }
  };

  const startTtsFromIndex = async (startIndex: number) => {
    stopTtsPlayback();
    if (ttsAvailable === false) {
      setTtsNotice({ kind: 'error', message: 'Lemonfox unavailable. Check server config.' });
      setIsPlaying(false);
      setTtsEnabled(false);
      return;
    }
    if (!ttsEnabledRef.current || !ttsPlayingRef.current) return;
    const chunk = buildTtsChunk(startIndex);
    if (!chunk || !chunk.text.trim()) {
      setIsPlaying(false);
      setTtsNotice({ kind: 'info', message: 'Reached end of text.' });
      return;
    }

    const { voice, speed, language } = ttsConfigRef.current;
    const controller = new AbortController();
    ttsAbortRef.current = controller;
    setTtsNotice({ kind: 'info', message: `Generating audio (${voice})...` });

    let response: Response;
    try {
      response = await fetch(buildApiUrl('/api/tts/speak'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: chunk.text,
          voice,
          speed,
          language,
        }),
        signal: controller.signal,
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        console.error('TTS fetch error:', error);
        setTtsNotice({ kind: 'error', message: 'TTS request failed.' });
        setIsPlaying(false);
      }
      return;
    }

    if (!response.ok) {
      const fallbackMessage = response.status === 503
        ? 'Lemonfox not configured. Set LEMONFOX_API_KEY.'
        : 'TTS generation failed.';
      let details = '';
      try {
        details = await response.text();
      } catch {
        details = '';
      }
      console.error('TTS response error:', response.status, details);
      setTtsAvailable(false);
      setTtsEnabled(false);
      setTtsNotice({ kind: 'error', message: fallbackMessage });
      setIsPlaying(false);
      return;
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    ttsAudioRef.current = audio;
    ttsAudioUrlRef.current = audioUrl;

    audio.onloadedmetadata = () => {
      const durationMs = Math.max(200, audio.duration * 1000);
      const stepMs = durationMs / Math.max(1, chunk.wordsCount);
      let stepIndex = 0;
      setWordIndex(chunk.startIndex);

      if (ttsIntervalRef.current) {
        window.clearInterval(ttsIntervalRef.current);
      }

      ttsIntervalRef.current = window.setInterval(() => {
        if (!ttsEnabledRef.current || !ttsPlayingRef.current) return;
        stepIndex += 1;
        const nextIndex = chunk.startIndex + stepIndex;
        if (nextIndex < chunk.endIndex) {
          setWordIndex(nextIndex);
        } else if (ttsIntervalRef.current) {
          window.clearInterval(ttsIntervalRef.current);
          ttsIntervalRef.current = null;
        }
      }, stepMs);
    };

    audio.onended = () => {
      if (ttsIntervalRef.current) {
        window.clearInterval(ttsIntervalRef.current);
        ttsIntervalRef.current = null;
      }
      if (ttsAudioUrlRef.current) {
        URL.revokeObjectURL(ttsAudioUrlRef.current);
        ttsAudioUrlRef.current = null;
      }
      if (ttsEnabledRef.current && ttsPlayingRef.current) {
        setWordIndex(chunk.endIndex);
        startTtsFromIndex(chunk.endIndex);
      }
    };

    audio.onerror = () => {
      if (ttsIntervalRef.current) {
        window.clearInterval(ttsIntervalRef.current);
        ttsIntervalRef.current = null;
      }
      if (ttsAudioUrlRef.current) {
        URL.revokeObjectURL(ttsAudioUrlRef.current);
        ttsAudioUrlRef.current = null;
      }
      setTtsNotice({ kind: 'error', message: 'Audio playback failed.' });
      setIsPlaying(false);
    };

    try {
      await audio.play();
      setTtsNotice({ kind: 'success', message: `Speaking (${voice})` });
    } catch (error) {
      console.error('Audio play error:', error);
      setTtsNotice({ kind: 'error', message: 'Audio playback blocked.' });
      setIsPlaying(false);
    }
  };

  const handleSeek = (nextIndex: number, options?: { pause?: boolean }) => {
    const total = Math.max(0, tokensRef.current.length - 1);
    const clampedIndex = clamp(nextIndex, 0, total);
    if (options?.pause) {
      setIsPlaying(false);
      stopTtsPlayback();
    }
    setWordIndex(clampedIndex);
    if (!options?.pause && ttsEnabledRef.current && ttsPlayingRef.current) {
      startTtsFromIndex(clampedIndex);
    }
  };

  useEffect(() => {
    handleSeekRef.current = handleSeek;
  }, [handleSeek]);

  const handleConventionalScroll = () => {
    if (conventionalMode !== 'excerpt') return;
    if (!conventionalSeekEnabledRef.current || scrollLockRef.current) return;
    if (scrollEndTimeoutRef.current) {
      window.clearTimeout(scrollEndTimeoutRef.current);
    }
    scrollEndTimeoutRef.current = window.setTimeout(() => {
      scrollEndTimeoutRef.current = null;
      const container = conventionalRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const pickX = rect.left + 24;
      const pickY = rect.top + Math.min(48, rect.height * 0.3);
      let target = document.elementFromPoint(pickX, pickY);
      let wordEl = target?.closest('[data-word-index]') as HTMLElement | null;
      if (!wordEl) {
        target = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
        wordEl = target?.closest('[data-word-index]') as HTMLElement | null;
      }
      if (!wordEl) return;
      const index = Number(wordEl.dataset.wordIndex);
      if (Number.isNaN(index)) return;
      if (isPlayingRef.current) {
        setIsPlaying(false);
      }
      if (index !== wordIndexRef.current) {
        handleSeek(index, { pause: true });
      }
    }, 120);
  };

  const handleConventionalClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (conventionalMode !== 'excerpt') return;
    const target = (event.target as HTMLElement).closest('[data-word-index]') as HTMLElement | null;
    if (!target) return;
    const index = Number(target.dataset.wordIndex);
    if (Number.isNaN(index)) return;
    handleSeek(index, { pause: true });
  };

  useEffect(() => {
    return () => {
      if (scrollRafRef.current) {
        window.cancelAnimationFrame(scrollRafRef.current);
        scrollRafRef.current = null;
      }
      if (scrollEndTimeoutRef.current) {
        window.clearTimeout(scrollEndTimeoutRef.current);
        scrollEndTimeoutRef.current = null;
      }
      if (autoFollowRafRef.current) {
        window.cancelAnimationFrame(autoFollowRafRef.current);
        autoFollowRafRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (ttsEnabled && isPlaying && tokens.length > 0) {
      startTtsFromIndex(wordIndex);
      return () => stopTtsPlayback();
    }
    stopTtsPlayback();
    return undefined;
  }, [ttsEnabled, isPlaying, tokens.length]);

  useEffect(() => {
    if (ttsEnabled && isPlaying && tokens.length > 0) {
      startTtsFromIndex(wordIndex);
    }
  }, [ttsVoice, ttsSpeed, ttsLanguage, ttsChunkWords]);

  const loadTextFile = async (
    file: File,
    setNotice: React.Dispatch<React.SetStateAction<Notice | null>>,
  ) => {
    if (isPdfFile(file)) {
      setNotice({ kind: 'info', message: 'Reading PDF...' });
      try {
        const data = new Uint8Array(await file.arrayBuffer());
        const pdfData = await extractPdfData(data, (current, total) => {
          setNotice({ kind: 'info', message: `Reading PDF... ${current}/${total}` });
        });
        const text = pdfData.pageTexts.join('\n\n');
        if (!text.trim()) {
          setNotice({ kind: 'error', message: 'No text found. Is this a scanned PDF?' });
          return '';
        }
        const wordCount = pdfData.pageTokenCounts.reduce((sum, count) => sum + count, 0);
        setNotice({ kind: 'success', message: `PDF loaded (${wordCount.toLocaleString()} words)` });
        return text;
      } catch (error) {
        console.error('Failed to read PDF:', error);
        setNotice({ kind: 'error', message: 'Failed to read PDF.' });
        return '';
      }
    }

    const text = await file.text();
    const wordCount = tokenize(text).length;
    setNotice({ kind: 'success', message: `Text loaded (${wordCount.toLocaleString()} words)` });
    return text;
  };

  const loadPrimaryPdf = async (file: File) => {
    setIsPdfBusy(true);
    setPrimaryNotice({ kind: 'info', message: 'Reading PDF...' });
    try {
      const data = new Uint8Array(await file.arrayBuffer());
      pdfDataRef.current = data;
      const pdfData = await extractPdfData(data, (current, total) => {
        setPrimaryNotice({ kind: 'info', message: `Reading PDF... ${current}/${total}` });
      });
      const combinedText = pdfData.pageTexts.join('\n\n');
      if (!combinedText.trim()) {
        const nextPdfState: PdfState = {
          fileName: file.name,
          pageCount: pdfData.pageCount,
          pageTexts: pdfData.pageTexts,
          pageTokenCounts: pdfData.pageTokenCounts,
          outline: pdfData.outline,
        };
        setPrimaryNotice({ kind: 'error', message: 'No text found. Is this a scanned PDF?' });
        setSourceKind('pdf');
        setPdfState(nextPdfState);
        setTitle(file.name.replace(/\.[^/.]+$/, ''));
        applyPdfRange(1, nextPdfState.pageCount, { pdfOverride: nextPdfState });
        return;
      }
      const nextPdfState: PdfState = {
        fileName: file.name,
        pageCount: pdfData.pageCount,
        pageTexts: pdfData.pageTexts,
        pageTokenCounts: pdfData.pageTokenCounts,
        outline: pdfData.outline,
      };
      const wordCount = pdfData.pageTokenCounts.reduce((sum, count) => sum + count, 0);
      setPdfState(nextPdfState);
      setSourceKind('pdf');
      setTitle(file.name.replace(/\.[^/.]+$/, ''));
      applyPdfRange(1, nextPdfState.pageCount, { pdfOverride: nextPdfState });
      setPrimaryNotice({ kind: 'success', message: `PDF loaded (${wordCount.toLocaleString()} words)` });
    } catch (error) {
      console.error('Failed to read PDF:', error);
      setPrimaryNotice({ kind: 'error', message: 'Failed to read PDF.' });
    } finally {
      setIsPdfBusy(false);
    }
  };

  const handlePageRangeApply = () => {
    if (!pdfState) return;
    const range = normalizeRange(pageRangeDraft.start, pageRangeDraft.end, pdfState.pageCount);
    applyPdfRange(range.start, range.end);
  };

  const jumpToPage = (pageNumber: number) => {
    if (!pdfState) return;
    const safePage = clamp(pageNumber, 1, pdfState.pageCount);
    if (safePage < pageRange.start || safePage > pageRange.end) {
      const nextRange = normalizeRange(
        Math.min(pageRange.start, safePage),
        Math.max(pageRange.end, safePage),
        pdfState.pageCount,
      );
      applyPdfRange(nextRange.start, nextRange.end, { focusPage: safePage });
      return;
    }
    const localIndex = safePage - pageRange.start;
    handleSeek(pageOffsets[localIndex] ?? 0);
  };

  const runOcrForRange = async () => {
    if (!pdfState || !pdfDataRef.current || isPdfBusy) return;
    const range = normalizeRange(pageRange.start, pageRange.end, pdfState.pageCount);
    setIsPdfBusy(true);
    setPrimaryNotice({ kind: 'info', message: `Running OCR for pages ${range.start}-${range.end}` });

    let worker: any = null;
    const loadingTask = getDocument({ data: pdfDataRef.current });
    try {
      const { createWorker } = await import('tesseract.js');
      worker = await createWorker('eng');
      const pdf = await loadingTask.promise;
      const updatedTexts = [...pdfState.pageTexts];

      for (let pageNum = range.start; pageNum <= range.end; pageNum += 1) {
        setPrimaryNotice({ kind: 'info', message: `OCR page ${pageNum}/${pdf.numPages}` });
        const page = await pdf.getPage(pageNum);
        const viewport = page.getViewport({ scale: 2 });
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!context) continue;
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: context, viewport, canvas }).promise;
        const result = await worker.recognize(canvas);
        const cleaned = result.data.text.replace(/\s+/g, ' ').trim();
        if (cleaned) {
          updatedTexts[pageNum - 1] = cleaned;
        }
      }

      const updatedCounts = updatedTexts.map((text) => tokenize(text).length);
      const nextPdfState: PdfState = {
        ...pdfState,
        pageTexts: updatedTexts,
        pageTokenCounts: updatedCounts,
      };
      setPdfState(nextPdfState);
      applyPdfRange(range.start, range.end, {
        pdfOverride: nextPdfState,
        focusPage: activePageNumber ?? range.start,
      });
      const totalWords = updatedCounts.reduce((sum, count) => sum + count, 0);
      setPrimaryNotice({ kind: 'success', message: `OCR complete (${totalWords.toLocaleString()} words)` });
    } catch (error) {
      console.error('OCR failed:', error);
      setPrimaryNotice({ kind: 'error', message: 'OCR failed.' });
    } finally {
      if (worker) {
        await worker.terminate();
      }
      await loadingTask.destroy();
      setIsPdfBusy(false);
    }
  };

  const handlePrimaryFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setPrimaryFileMeta({
      name: file.name,
      size: file.size,
      lastModified: file.lastModified,
    });
    if (isPdfFile(file)) {
      await loadPrimaryPdf(file);
    } else {
      const text = await loadTextFile(file, setPrimaryNotice);
      if (text) {
        setSourceKind('text');
        setPdfState(null);
        setPageOffsets([]);
        setPageRange({ start: 1, end: 1 });
        setPageRangeDraft({ start: 1, end: 1 });
        setSourceText(text);
        setTitle(file.name.replace(/\.[^/.]+$/, ''));
        setWordIndex(0);
        setIsPlaying(false);
      }
    }
    event.target.value = '';
  };

  const handleParallelFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await loadTextFile(file, setParallelNotice);
    if (text) {
      setParallelText(text);
    }
    event.target.value = '';
  };

  const handleBookmark = () => {
    if (!tokens.length) return;
    let pageNumber: number | undefined;
    let pageOffset: number | undefined;
    if (sourceKind === 'pdf' && activePageNumber !== null && activePageIndex !== null) {
      pageNumber = activePageNumber;
      const pageStart = pageOffsets[activePageIndex] ?? 0;
      pageOffset = Math.max(0, wordIndex - pageStart);
    }
    const label = tokens[wordIndex]
      ? `${tokens[wordIndex].slice(0, 28)}${tokens[wordIndex].length > 28 ? '...' : ''}`
      : `Word ${wordIndex + 1}`;
    const nextBookmark: Bookmark = {
      id: `${Date.now()}-${wordIndex}`,
      index: wordIndex,
      label,
      createdAt: Date.now(),
      pageNumber,
      pageOffset,
    };
    setBookmarks((prev) => [nextBookmark, ...prev].slice(0, 24));
  };

  const handleDeleteBookmark = (id: string) => {
    setBookmarks((prev) => prev.filter((bookmark) => bookmark.id !== id));
  };

  const handleCompanionSend = () => {
    if (!companionInput.trim()) return;
    setCompanionLog((prev) => [`You: ${companionInput.trim()}`, ...prev]);
    setCompanionInput('');
  };

  const renderDisplay = () => {
    if (!currentChunk.length) {
      return <span className="display-placeholder">Load a passage to begin.</span>;
    }
    if (chunkSize !== 1) {
      return <span className="display-chunk">{currentChunk.join(' ')}</span>;
    }
    const word = currentChunk[0];
    const pivotIndex = getPivotIndex(word);
    const left = word.slice(0, pivotIndex);
    const pivot = word[pivotIndex] || '';
    const right = word.slice(pivotIndex + 1);
    return (
      <span className="display-word" aria-live="polite">
        <span className="word-left">{left}</span>
        <span className="word-pivot">{pivot}</span>
        <span className="word-right">{right}</span>
      </span>
    );
  };

  const renderOutlineItems = (items: PdfOutlineItem[], depth = 0) =>
    items.map((item, index) => (
      <div key={`${item.title}-${index}`} className="toc-item" style={{ paddingLeft: depth * 12 }}>
        <div className="toc-row">
          {item.pageNumber ? (
            <button type="button" className="toc-link" onClick={() => jumpToPage(item.pageNumber!)}>
              {item.title}
            </button>
          ) : item.url ? (
            <a className="toc-link" href={item.url} target="_blank" rel="noreferrer">
              {item.title}
            </a>
          ) : (
            <span className="toc-text">{item.title}</span>
          )}
          {item.pageNumber && <span className="toc-page">p. {item.pageNumber}</span>}
        </div>
        {item.items.length > 0 && <div className="toc-children">{renderOutlineItems(item.items, depth + 1)}</div>}
      </div>
    ));

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="kicker">Reading Lab</p>
          <h1>Focus Reader</h1>
          <p className="sub">
            Speed reading with memory, parallel texts, and a future hermeneutics companion.
          </p>
        </div>
        <div className="hero-actions">
          <span className="status-pill">Local session</span>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              handleSeek(0, { pause: true });
            }}
          >
            Reset
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="panel source-panel">
          <div className="panel-header">
            <h2>Source</h2>
            <p>Load a text, then tune the session speed.</p>
          </div>

          <label className="field">
            <span>Session title</span>
            <input
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Untitled Session"
            />
          </label>

          <label className="field">
            <span>{conventionalMode === 'rendered' ? 'Primary text (raw)' : 'Primary text'}</span>
            <textarea
              rows={8}
              value={sourceText}
              onChange={(event) => setSourceText(event.target.value)}
              placeholder="Paste or import a passage."
            />
          </label>

          <div className="field-row">
            <label className="field file">
              <span>Import .txt/.md/.pdf</span>
              <input type="file" accept=".txt,.md,.text,.pdf,application/pdf" onChange={handlePrimaryFile} />
            </label>
            <label className="field file">
              <span>Parallel text</span>
              <input type="file" accept=".txt,.md,.text,.pdf,application/pdf" onChange={handleParallelFile} />
            </label>
          </div>
          {(primaryNotice || parallelNotice) && (
            <div className="notice-row">
              {primaryNotice && (
                <p className={`notice ${primaryNotice.kind}`}>Primary: {primaryNotice.message}</p>
              )}
              {parallelNotice && (
                <p className={`notice ${parallelNotice.kind}`}>Parallel: {parallelNotice.message}</p>
              )}
            </div>
          )}

          {sourceKind === 'pdf' && pdfState && (
            <div className="pdf-controls">
              <div className="pdf-range">
                <div className="range-inputs">
                  <label>
                    Start
                    <input
                      type="number"
                      min={1}
                      max={pdfState.pageCount}
                      value={pageRangeDraft.start}
                      onChange={(event) =>
                        setPageRangeDraft((prev) => ({
                          ...prev,
                          start: Number(event.target.value),
                        }))
                      }
                    />
                  </label>
                  <label>
                    End
                    <input
                      type="number"
                      min={1}
                      max={pdfState.pageCount}
                      value={pageRangeDraft.end}
                      onChange={(event) =>
                        setPageRangeDraft((prev) => ({
                          ...prev,
                          end: Number(event.target.value),
                        }))
                      }
                    />
                  </label>
                </div>
                <button type="button" className="ghost" onClick={handlePageRangeApply} disabled={isPdfBusy}>
                  Apply range
                </button>
              </div>
              <div className="pdf-actions">
                <button type="button" className="ghost" onClick={runOcrForRange} disabled={isPdfBusy}>
                  Run OCR for selected pages
                </button>
                {pdfStats?.emptyPages ? (
                  <p className="hint">
                    {pdfStats.emptyPages} page{pdfStats.emptyPages > 1 ? 's' : ''} look empty. OCR can
                    recover scanned text.
                  </p>
                ) : (
                  <p className="hint">OCR is useful for scanned or image-only PDFs.</p>
                )}
              </div>
              {pdfState.outline.length > 0 && (
                <details className="toc">
                  <summary>Table of contents</summary>
                  <div className="toc-list">
                    {renderOutlineItems(pdfState.outline)}
                  </div>
                </details>
              )}
            </div>
          )}

          <label className="field">
            <span>Parallel text (optional)</span>
            <textarea
              rows={6}
              value={parallelText}
              onChange={(event) => setParallelText(event.target.value)}
              placeholder="Paste a translation or commentary to align."
            />
          </label>

          <div className="meta-grid">
            <div>
              <span className="meta-label">Words</span>
              <strong>{tokens.length.toLocaleString()}</strong>
            </div>
            <div>
              <span className="meta-label">Remaining</span>
              <strong>{minutesRemaining.toFixed(1)} min</strong>
            </div>
            <div>
              <span className="meta-label">Progress</span>
              <strong>{progressPercent}%</strong>
            </div>
            {sourceKind === 'pdf' && pdfState && (
              <>
                <div>
                  <span className="meta-label">Pages</span>
                  <strong>{pdfState.pageCount}</strong>
                </div>
                <div>
                  <span className="meta-label">Current</span>
                  <strong>{activePageNumber ?? pageRange.start}</strong>
                </div>
              </>
            )}
          </div>

          <p className="hint">
            Kindle support requires DRM-free exports. PDF text extraction is best on text-based PDFs.
          </p>
        </section>

        <section className="panel reader-panel">
          <div className="reader-display">
            <div className="focus-rail" aria-hidden="true" />
            {renderDisplay()}
          </div>

          <div className="controls">
            <button
              type="button"
              onClick={() => {
                setIsPlaying((prev) => !prev);
              }}
              disabled={!tokens.length}
            >
              {isPlaying ? 'Pause' : 'Start'}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => handleSeek(wordIndex - chunkSize)}
              disabled={!tokens.length}
            >
              Back
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => handleSeek(wordIndex + chunkSize)}
              disabled={!tokens.length}
            >
              Forward
            </button>
            <button type="button" className="ghost" onClick={handleBookmark} disabled={!tokens.length}>
              Bookmark
            </button>
          </div>

          <div className="slider-group">
            <label className="slider">
              <span>Words per minute</span>
              <input
                type="range"
                min={120}
                max={900}
                value={wpm}
                onChange={(event) => setWpm(Number(event.target.value))}
              />
              <strong>{wpm} wpm</strong>
            </label>
            <label className="slider">
              <span>Chunk size</span>
              <input
                type="range"
                min={1}
                max={5}
                value={chunkSize}
                onChange={(event) => setChunkSize(Number(event.target.value))}
              />
              <strong>{chunkSize} word{chunkSize > 1 ? 's' : ''}</strong>
            </label>
            <label className="slider">
              <span>Sentence pause</span>
              <input
                type="range"
                min={0}
                max={400}
                step={20}
                value={sentencePauseMs}
                onChange={(event) => setSentencePauseMs(Number(event.target.value))}
              />
              <strong>{sentencePauseMs} ms</strong>
            </label>
          </div>

          <div className="session-panel">
            <div className="session-header">
              <h3>Session Stats</h3>
            </div>
            <div className="session-metrics">
              <div>
                <span className="meta-label">Time</span>
                <strong>{sessionDurationLabel}</strong>
              </div>
              <div>
                <span className="meta-label">Words read</span>
                <strong>{wordsRead.toLocaleString()}</strong>
              </div>
              <div>
                <span className="meta-label">Completion</span>
                <strong>{sessionPercent}%</strong>
              </div>
            </div>
          </div>

          <div className="tts-panel">
            <div className="tts-header">
              <h3>Voice Playback</h3>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={ttsEnabled}
                  onChange={(event) => {
                    void handleTtsToggle(event.target.checked);
                  }}
                  disabled={ttsChecking}
                />
                <span>
                  {ttsChecking ? 'Checking' : ttsAvailable === false ? 'Unavailable' : 'Enable'}
                </span>
              </label>
              <button type="button" className="ghost" onClick={() => void refreshTtsStatus()} disabled={ttsChecking}>
                Refresh
              </button>
            </div>
            {ttsAvailable === false && (
              <p className="hint">Lemonfox is unavailable. Set LEMONFOX_API_KEY on the server.</p>
            )}
            {ttsAvailable === null && !ttsNotice && (
              <p className="hint">Check Lemonfox status to enable voice playback.</p>
            )}
            {ttsEnabled && (
              <div className="tts-grid">
                <label className="field compact">
                  <span>Voice</span>
                  <select value={ttsVoice} onChange={(event) => setTtsVoice(event.target.value)}>
                    {(ttsVoices.length > 0 ? ttsVoices : ['sarah', 'john', 'emily', 'michael', 'alice']).map(
                      (voice) => (
                        <option key={voice} value={voice}>
                          {voice}
                        </option>
                      ),
                    )}
                  </select>
                </label>
                <label className="field compact">
                  <span>Language</span>
                  <input
                    value={ttsLanguage}
                    onChange={(event) => setTtsLanguage(event.target.value)}
                    placeholder="en-us"
                  />
                </label>
                <label className="slider">
                  <span>Voice speed</span>
                  <input
                    type="range"
                    min={0.6}
                    max={1.4}
                    step={0.05}
                    value={ttsSpeed}
                    onChange={(event) => setTtsSpeed(Number(event.target.value))}
                  />
                  <strong>{ttsSpeed.toFixed(2)}x</strong>
                </label>
                <label className="slider">
                  <span>Voice chunk</span>
                  <input
                    type="range"
                    min={10}
                    max={60}
                    step={1}
                    value={ttsChunkWords}
                    onChange={(event) => setTtsChunkWords(Number(event.target.value))}
                  />
                  <strong>{ttsChunkWords} words</strong>
                </label>
              </div>
            )}
            {ttsNotice && <p className={`notice ${ttsNotice.kind}`}>{ttsNotice.message}</p>}
          </div>

          <div className="camera-panel">
            <div className="camera-header">
              <h3>Camera Controls</h3>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={cameraEnabled}
                  onChange={(event) => setCameraEnabled(event.target.checked)}
                />
                <span>Enable</span>
              </label>
            </div>
            <div className="camera-toggles">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={cameraHandEnabled}
                  onChange={(event) => setCameraHandEnabled(event.target.checked)}
                  disabled={!cameraEnabled}
                />
                <span>Hand gestures</span>
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={cameraEyeEnabled}
                  onChange={(event) => setCameraEyeEnabled(event.target.checked)}
                  disabled={!cameraEnabled}
                />
                <span>Eye pause</span>
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={cameraPreview}
                  onChange={(event) => setCameraPreview(event.target.checked)}
                  disabled={!cameraEnabled}
                />
                <span>Preview</span>
              </label>
            </div>
            <div className={`camera-preview ${cameraPreview ? 'show' : 'hidden'}`}>
              <video
                ref={cameraVideoRef}
                className="camera-video"
                muted
                playsInline
                autoPlay
              />
            </div>
            <p className="hint">
              Swipe left/right to move back/forward. Long blink toggles pause. Video stays on-device.
            </p>
            {cameraNotice && <p className={`notice ${cameraNotice.kind}`}>{cameraNotice.message}</p>}
          </div>

          <label className="progress">
            <span>Navigate</span>
            <input
              type="range"
              min={0}
              max={Math.max(0, tokens.length - 1)}
              value={Math.min(wordIndex, Math.max(0, tokens.length - 1))}
              onChange={(event) => handleSeek(Number(event.target.value))}
              disabled={!tokens.length}
            />
          </label>

          {sourceKind === 'pdf' && pdfState && (
            <div className="page-nav">
              <button
                type="button"
                className="ghost"
                onClick={() => jumpToPage((activePageNumber ?? pageRange.start) - 1)}
                disabled={(activePageNumber ?? pageRange.start) <= pageRange.start}
              >
                Prev Page
              </button>
              <div className="page-status">
                Page {activePageNumber ?? pageRange.start} / {pdfState.pageCount}{' '}
                <span className="page-range">({pageRange.start}-{pageRange.end})</span>
              </div>
              <button
                type="button"
                className="ghost"
                onClick={() => jumpToPage((activePageNumber ?? pageRange.start) + 1)}
                disabled={(activePageNumber ?? pageRange.start) >= pageRange.end}
              >
                Next Page
              </button>
            </div>
          )}

          <div className="bookmarks">
            <div className="bookmarks-header">
              <h3>Bookmarks</h3>
              <button type="button" className="ghost" onClick={() => setBookmarks([])}>
                Clear
              </button>
            </div>
            {!resolvedFurthest && resolvedBookmarks.length === 0 ? (
              <p className="hint">Save anchor points to return to later.</p>
            ) : (
              <div className="bookmark-list">
                {resolvedFurthest && (() => {
                  const pageTag = resolvedFurthest.pageNumber ? `p. ${resolvedFurthest.pageNumber}` : null;
                  const progress = resolvedFurthest.resolvedIndex !== null && tokens.length
                    ? Math.round((resolvedFurthest.resolvedIndex / tokens.length) * 100)
                    : null;
                  const labelParts = ['Furthest read'];
                  if (pageTag) labelParts.push(pageTag);
                  labelParts.push(progress !== null ? `${progress}%` : 'out of range');
                  const resolvedIndex = resolvedFurthest.resolvedIndex ?? null;
                  return (
                    <div key="furthest-read" className="bookmark-row auto">
                      <button
                        type="button"
                        className="bookmark-jump"
                        onClick={() => {
                          if (resolvedIndex !== null) {
                            handleSeek(resolvedIndex);
                          }
                        }}
                        disabled={resolvedFurthest.outOfRange || resolvedIndex === null}
                      >
                        {labelParts.join(' · ')}
                      </button>
                    </div>
                  );
                })()}
                {resolvedBookmarks.map((bookmark) => {
                  const pageTag = bookmark.pageNumber ? `p. ${bookmark.pageNumber}` : null;
                  const progress = bookmark.resolvedIndex !== null && tokens.length
                    ? Math.round((bookmark.resolvedIndex / tokens.length) * 100)
                    : null;
                  const labelParts = [bookmark.label];
                  if (pageTag) labelParts.push(pageTag);
                  labelParts.push(progress !== null ? `${progress}%` : 'out of range');
                  const resolvedIndex = bookmark.resolvedIndex ?? null;
                  return (
                    <div key={bookmark.id} className="bookmark-row">
                      <button
                        type="button"
                        className="bookmark-jump"
                        onClick={() => {
                          if (resolvedIndex !== null) {
                            handleSeek(resolvedIndex);
                          }
                        }}
                        disabled={bookmark.outOfRange || resolvedIndex === null}
                      >
                        {labelParts.join(' · ')}
                      </button>
                      <button
                        type="button"
                        className="bookmark-remove ghost"
                        onClick={() => handleDeleteBookmark(bookmark.id)}
                      >
                        Delete
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        <section className="panel companion-panel">
          <div className="panel-header">
            <h2>Conventional View</h2>
            <div className="toggle-row">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={showConventional}
                  onChange={(event) => setShowConventional(event.target.checked)}
                />
                <span>Show view</span>
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={conventionalMode === 'rendered'}
                  onChange={(event) => setConventionalMode(event.target.checked ? 'rendered' : 'excerpt')}
                  disabled={!showConventional}
                />
                <span>Rendered markdown</span>
              </label>
              <button
                type="button"
                className="ghost small"
                onClick={() => {
                  setConventionalMode('rendered');
                  setShowConventional(true);
                }}
                disabled={conventionalMode === 'rendered'}
              >
                Reset view mode
              </button>
              {conventionalMode === 'excerpt' && (
                <>
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={conventionalSeekEnabled}
                      onChange={(event) => setConventionalSeekEnabled(event.target.checked)}
                      disabled={!showConventional}
                    />
                    <span>Scroll to seek</span>
                  </label>
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={autoFollowConventional}
                      onChange={(event) => setAutoFollowConventional(event.target.checked)}
                      disabled={!showConventional}
                    />
                    <span>Auto-follow</span>
                  </label>
                </>
              )}
            </div>
            {showConventional && conventionalMode === 'excerpt' && (
              <p className="hint">Scrolling the excerpt moves the reader to the word near the top edge.</p>
            )}
            {showConventional && conventionalMode === 'rendered' && (
              <p className="hint">Rendered view is read-only. Switch to Excerpt to scroll/seek.</p>
            )}
          </div>

          {showConventional && conventionalMode === 'excerpt' && (
            <div
              className="snippet full"
              ref={conventionalRef}
              onScroll={handleConventionalScroll}
              onClick={handleConventionalClick}
            >
              {conventionalNodes}
            </div>
          )}
          {showConventional && conventionalMode === 'rendered' && (
            <div className="snippet full rendered" ref={conventionalRef}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {sourceText || 'No text loaded.'}
              </ReactMarkdown>
            </div>
          )}

          <div className="panel-header">
            <h2>Parallel Text</h2>
            <label className="toggle">
              <input
                type="checkbox"
                checked={showParallel}
                onChange={(event) => setShowParallel(event.target.checked)}
              />
              <span>Sync with primary</span>
            </label>
          </div>

          {showParallel && parallelTokens.length > 0 && (
            <div className="snippet secondary">
              {parallelSnippet.map((token, index) => (
                <span key={`${index}-${token.text}`} className={token.isActive ? 'active' : ''}>
                  {token.text}{' '}
                </span>
              ))}
            </div>
          )}

          <div className="panel-header">
            <h2>Hermeneutics</h2>
            <p>Ask questions about the passage as you read.</p>
          </div>
          <div className="companion">
            <textarea
              rows={3}
              value={companionInput}
              onChange={(event) => setCompanionInput(event.target.value)}
              placeholder="Ask about the argument, context, or translation."
            />
            <button type="button" onClick={handleCompanionSend}>
              Send
            </button>
            <div className="companion-log">
              {companionLog.map((entry, index) => (
                <div key={`${index}-${entry}`}>{entry}</div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

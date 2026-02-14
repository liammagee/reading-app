import {
  forwardRef,
  memo,
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from 'react';
import type { FaceLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import JSZip from 'jszip';
import { buildApiUrl } from './env';
import { BrandSigil } from './BrandSigil';
import {
  BOOKMARK_VERSION,
  type BookmarkPayload,
  buildBookmarkUrl,
  hashText,
  parseBookmarkValue,
  readBookmarkParam,
} from './bookmarkUtils';
import {
  segmentTextBySentence,
  segmentTextByTweet,
  segmentTokens,
  tokenize,
  type Granularity,
  type Segment,
} from './textUtils';
import { useAuth } from './auth/useAuth';
import { syncService } from './sync/SyncService';
import { useSyncStatus } from './sync/useSyncStatus';
import { LoginPanel } from './auth/LoginPanel';
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
  granularity?: Granularity;
  contextRadius?: number;
  minWordMs?: number;
  sentencePauseMs?: number;
  sessionElapsedMs?: number;
  title: string;
  bookmarks: Bookmark[];
  furthestRead?: FurthestRead | null;
  updatedAt: number;
};

type LastDocument = {
  sourceText: string;
  title?: string;
  sourceKind?: 'text' | 'pdf' | 'epub';
  primaryFileMeta?: FileMeta | null;
};

type UserPreferences = {
  showConventional?: boolean;
  showReaderDisplay?: boolean;
  conventionalSeekEnabled?: boolean;
  autoFollowConventional?: boolean;
  audioCuesEnabled?: boolean;
  viewMode?: ViewMode;
  cameraEnabled?: boolean;
  cameraHandEnabled?: boolean;
  cameraEyeEnabled?: boolean;
  cameraPreview?: boolean;
  displayScale?: number;
  longWordSlowdown?: number;
};

type Notice = {
  kind: 'info' | 'success' | 'error';
  message: string;
};

type PageRange = {
  start: number;
  end: number;
};

type ViewMode = 'deck' | 'mobile' | 'focus' | 'focus-text';

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

type PdfOutlineNode = {
  title?: string;
  dest?: unknown;
  url?: string;
  items?: PdfOutlineNode[];
};

type PdfState = {
  fileName: string;
  pageCount: number;
  pageTexts: string[];
  pageTokenCounts: number[];
  outline: PdfOutlineItem[];
};

type StoredDocument = {
  id: string;
  sourceText: string;
  title?: string;
  sourceKind: 'text' | 'pdf' | 'epub';
  primaryFileMeta?: FileMeta | null;
  wordCount: number;
  updatedAt: number;
  pageRange?: PageRange;
  pdfState?: PdfState | null;
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

type AwardKind = 'words' | 'time' | 'progress';

type AwardDefinition = {
  id: string;
  label: string;
  detail: string;
  kind: AwardKind;
  threshold: number;
};

type AwardStatus = AwardDefinition & {
  value: number;
  earned: boolean;
};

type StreakStats = {
  dayKey: string;
  dailyWords: number;
  streak: number;
  bestStreak: number;
  lastCompletedDay?: string;
};

type TesseractWorker = import('tesseract.js').Worker;

const DEFAULT_TEXT = `Paste a passage to begin.

Pacer flashes one word at a time so you can increase speed while staying attentive. Try adjusting the WPM slider or chunk size, then hit Start.
`;
const CONVENTIONAL_MODE_KEY = 'reader:conventionalMode';
const LAST_DOCUMENT_KEY = 'reader:lastDocument';
const LAST_DOCUMENT_DB = 'reader-documents';
const LAST_DOCUMENT_STORE = 'documents';
const LAST_DOCUMENT_ID = 'last';
const USER_PREFERENCES_KEY = 'reader:preferences';
const STREAK_STORAGE_KEY = 'reader:streak';
const DEFAULT_CONVENTIONAL_CHUNK_SIZE = 120;
const CONVENTIONAL_BUFFER_CHUNKS = 2;
const DEFAULT_CONVENTIONAL_CHUNK_HEIGHT = 240;
const DEFAULT_DAILY_GOAL_WORDS = 2000;
const LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144] as const;
const RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380] as const;

const AWARD_DEFINITIONS: AwardDefinition[] = [
  { id: 'warmup', label: 'Warm-up', detail: '250 words', kind: 'words', threshold: 250 },
  { id: 'stride', label: 'Stride', detail: '1k words', kind: 'words', threshold: 1000 },
  { id: 'deep', label: 'Deep read', detail: '3k words', kind: 'words', threshold: 3000 },
  { id: 'focus', label: 'Focus', detail: '10 min', kind: 'time', threshold: 10 * 60 * 1000 },
  { id: 'endurance', label: 'Endurance', detail: '30 min', kind: 'time', threshold: 30 * 60 * 1000 },
  { id: 'halfway', label: 'Halfway', detail: '50%', kind: 'progress', threshold: 50 },
  { id: 'finish', label: 'Finish', detail: '100%', kind: 'progress', threshold: 100 },
];

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

const normalizeTokenForSearch = (value: string) =>
  value.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');

const getSegmentIndexForWordIndex = (wordIndex: number, segmentStarts: number[]) => {
  if (!segmentStarts.length) return 0;
  let low = 0;
  let high = segmentStarts.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high + 1) / 2);
    if (segmentStarts[mid] <= wordIndex) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return low;
};

const getConventionalChunkSize = (tokenCount: number) => {
  if (tokenCount > 60000) return 60;
  if (tokenCount > 30000) return 80;
  if (tokenCount > 15000) return 100;
  return DEFAULT_CONVENTIONAL_CHUNK_SIZE;
};

const getConventionalBufferChunks = (chunkSize: number) => (chunkSize <= 80 ? 3 : CONVENTIONAL_BUFFER_CHUNKS);

const buildDocumentKey = (
  sourceKind: 'text' | 'pdf' | 'epub',
  primaryFileMeta: FileMeta | null,
  sourceText: string,
) => {
  if (sourceKind === 'pdf' && primaryFileMeta) {
    return `reader:pdf:${primaryFileMeta.name}:${primaryFileMeta.size}:${primaryFileMeta.lastModified}`;
  }
  if (primaryFileMeta) {
    const kind = sourceKind === 'epub' ? 'epub' : 'text';
    return `reader:${kind}:${primaryFileMeta.name}:${primaryFileMeta.size}:${primaryFileMeta.lastModified}`;
  }
  if (!sourceText.trim()) return '';
  const kind = sourceKind === 'epub' ? 'epub' : 'text';
  return `reader:${kind}:${hashText(sourceText)}`;
};

const getSteppedWordIndex = (
  currentIndex: number,
  direction: 'back' | 'next',
  stepSize: number,
  segmentStarts: number[],
  segments: Segment[],
) => {
  if (!segments.length) return currentIndex;
  const segmentIndex = getSegmentIndexForWordIndex(currentIndex, segmentStarts);
  const delta = direction === 'back' ? -stepSize : stepSize;
  const nextSegmentIndex = clamp(segmentIndex + delta, 0, Math.max(0, segments.length - 1));
  const nextSegment = segments[nextSegmentIndex];
  return nextSegment ? nextSegment.startIndex : currentIndex;
};

const isEditableTarget = (target: EventTarget | null) => {
  if (!target || !(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tagName = target.tagName;
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT';
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

const getLocalDayKey = (date = new Date()) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const parseDayKey = (dayKey: string) => {
  const [year, month, day] = dayKey.split('-').map((part) => Number(part));
  if (!year || !month || !day) return null;
  return new Date(year, month - 1, day);
};

const isPreviousDay = (previousKey: string, currentKey: string) => {
  const previousDate = parseDayKey(previousKey);
  const currentDate = parseDayKey(currentKey);
  if (!previousDate || !currentDate) return false;
  const diffMs = currentDate.getTime() - previousDate.getTime();
  return diffMs > 0 && diffMs <= 24 * 60 * 60 * 1000;
};

const readStreakStorage = () => {
  if (typeof window === 'undefined') return null;
  const raw = window.localStorage.getItem(STREAK_STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as {
      dailyGoalWords?: number;
      stats?: Partial<StreakStats>;
    };
  } catch {
    return null;
  }
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

const getReadingComplexityMultiplier = (text: string) => {
  if (!text.trim()) return 1;
  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length) return 1;
  let maxLength = 0;
  let hyphenatedCount = 0;
  const hyphenPattern = /\p{L}[-\u2010-\u2014]\p{L}/u;
  for (const raw of words) {
    const cleaned = raw.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
    if (!cleaned) continue;
    const plain = cleaned.replace(/[^-\p{L}\p{N}]/gu, '');
    const length = plain.replace(/[-\u2010-\u2014]/g, '').length;
    if (length > maxLength) maxLength = length;
    if (hyphenPattern.test(cleaned)) hyphenatedCount += 1;
  }
  const extraLength = Math.max(0, maxLength - 6);
  const lengthBoost = Math.min(0.3, extraLength * 0.03);
  const hyphenBoost = Math.min(0.18, hyphenatedCount * 0.06);
  return 1 + lengthBoost + hyphenBoost;
};

const getMaxWordLength = (text: string) => {
  if (!text.trim()) return 0;
  const words = text.split(/\s+/).filter(Boolean);
  let maxLength = 0;
  for (const raw of words) {
    const cleaned = raw.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
    if (!cleaned) continue;
    const length = cleaned.replace(/[-\u2010-\u2014]/g, '').length;
    if (length > maxLength) maxLength = length;
  }
  return maxLength;
};

type ReaderDisplayProps = {
  currentSegments: Segment[];
  currentSegmentText: string;
  singleWordMode: boolean;
  contextRadius: number;
  tokens: string[];
  wordIndex: number;
};

const ReaderDisplay = memo(
  ({ currentSegments, currentSegmentText, singleWordMode, contextRadius, tokens, wordIndex }: ReaderDisplayProps) => {
    if (!currentSegments.length) {
      return <span className="display-placeholder">Load a passage to begin.</span>;
    }
    if (!singleWordMode) {
      return <span className="display-chunk">{currentSegmentText}</span>;
    }
    const word = currentSegments[0]?.text ?? '';
    const pivotIndex = getPivotIndex(word);
    const left = word.slice(0, pivotIndex);
    const pivot = word[pivotIndex] || '';
    const right = word.slice(pivotIndex + 1);
    const showContext = contextRadius > 0;
    const before = showContext
      ? tokens.slice(Math.max(0, wordIndex - contextRadius), wordIndex).join(' ')
      : '';
    const after = showContext
      ? tokens.slice(wordIndex + 1, wordIndex + 1 + contextRadius).join(' ')
      : '';
    return (
      <div className={`display-stack${showContext ? ' with-context' : ''}`}>
        <span className="display-word" aria-live="polite"><span className="word-left">{left}</span><span className="word-pivot">{pivot}</span><span className="word-right">{right}</span></span>
        {showContext && (
          <span className="display-context" aria-hidden="true">
            {before && <span className="context-before">{before} </span>}
            <span className="context-current">{word}</span>
            {after && <span className="context-after"> {after}</span>}
          </span>
        )}
      </div>
    );
  },
);
ReaderDisplay.displayName = 'ReaderDisplay';

type ConventionalExcerptProps = {
  nodes: React.ReactNode;
  spacerHeights: { top: number; bottom: number };
  onScroll: () => void;
  onClick: (event: React.MouseEvent<HTMLDivElement>) => void;
};

const ConventionalExcerpt = memo(
  forwardRef<HTMLDivElement, ConventionalExcerptProps>(({ nodes, spacerHeights, onScroll, onClick }, ref) => (
    <div className="snippet full" ref={ref} onScroll={onScroll} onClick={onClick}>
      {spacerHeights.top > 0 && <div className="excerpt-spacer" style={{ height: spacerHeights.top }} />}
      {nodes}
      {spacerHeights.bottom > 0 && <div className="excerpt-spacer" style={{ height: spacerHeights.bottom }} />}
    </div>
  )),
);
ConventionalExcerpt.displayName = 'ConventionalExcerpt';

type ConventionalRenderedProps = {
  content: React.ReactNode;
};

const ConventionalRendered = memo(
  forwardRef<HTMLDivElement, ConventionalRenderedProps>(({ content }, ref) => (
    <div className="snippet full rendered" ref={ref}>
      {content}
    </div>
  )),
);
ConventionalRendered.displayName = 'ConventionalRendered';

type BookmarksPanelProps = {
  onClear: () => void;
  bookmarkRows: React.ReactNode;
  notice: Notice | null;
};

const BookmarksPanel = memo(({ onClear, bookmarkRows, notice }: BookmarksPanelProps) => (
  <div className="bookmarks">
    <div className="bookmarks-header">
      <h3>Bookmarks</h3>
      <button type="button" className="ghost" onClick={onClear}>
        Clear
      </button>
    </div>
    {bookmarkRows ? bookmarkRows : <p className="hint">Save anchor points to return to later.</p>}
    {notice && <p className={`notice ${notice.kind}`}>{notice.message}</p>}
  </div>
));
BookmarksPanel.displayName = 'BookmarksPanel';

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

const readStoredDocumentsFromDb = async (): Promise<StoredDocument[]> => {
  if (typeof window === 'undefined' || !('indexedDB' in window)) return [];
  try {
    const db = await openLastDocumentDb();
    return await new Promise<StoredDocument[]>((resolve, reject) => {
      const transaction = db.transaction(LAST_DOCUMENT_STORE, 'readonly');
      const store = transaction.objectStore(LAST_DOCUMENT_STORE);
      const request = store.getAll();
      request.onsuccess = () => {
        const results = (request.result as StoredDocument[]).filter(
          (item) => item && typeof item.id === 'string',
        );
        resolve(results);
      };
      request.onerror = () => reject(request.error);
      transaction.oncomplete = () => db.close();
      transaction.onerror = () => {
        db.close();
        reject(transaction.error);
      };
    });
  } catch (error) {
    console.warn('Failed to read stored documents from IndexedDB.', error);
    return [];
  }
};

const writeStoredDocumentToDb = async (payload: StoredDocument) => {
  if (typeof window === 'undefined' || !('indexedDB' in window)) return;
  try {
    const db = await openLastDocumentDb();
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(LAST_DOCUMENT_STORE, 'readwrite');
      const store = transaction.objectStore(LAST_DOCUMENT_STORE);
      store.put(payload, payload.id);
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
    console.warn('Failed to persist stored document to IndexedDB.', error);
  }
};

const clearStoredDocumentsFromDb = async () => {
  if (typeof window === 'undefined' || !('indexedDB' in window)) return;
  try {
    const db = await openLastDocumentDb();
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(LAST_DOCUMENT_STORE, 'readwrite');
      const store = transaction.objectStore(LAST_DOCUMENT_STORE);
      const request = store.openCursor();
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) return;
        if (cursor.key !== LAST_DOCUMENT_ID) {
          cursor.delete();
        }
        cursor.continue();
      };
      request.onerror = () => reject(request.error);
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
    console.warn('Failed to clear stored documents from IndexedDB.', error);
  }
};

const isPdfFile = (file: File) => {
  const name = file.name.toLowerCase();
  return file.type === 'application/pdf' || name.endsWith('.pdf');
};

const isEpubFile = (file: File) => {
  const name = file.name.toLowerCase();
  return file.type === 'application/epub+zip' || name.endsWith('.epub');
};

const isHtmlFile = (file: File) => {
  const name = file.name.toLowerCase();
  return file.type === 'text/html' || name.endsWith('.html') || name.endsWith('.htm');
};

const isDocxFile = (file: File) => {
  const name = file.name.toLowerCase();
  return (
    file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    name.endsWith('.docx')
  );
};

const resolveEpubPath = (basePath: string, href: string) => {
  const cleaned = href.split('#')[0]?.split('?')[0]?.trim();
  if (!cleaned || /^[a-z]+:\/\//i.test(cleaned)) return null;
  const combined = cleaned.startsWith('/') ? cleaned.slice(1) : `${basePath}${cleaned}`;
  const segments = combined.split('/').filter(Boolean);
  const resolved: string[] = [];
  for (const segment of segments) {
    if (segment === '.') continue;
    if (segment === '..') {
      resolved.pop();
      continue;
    }
    resolved.push(segment);
  }
  return resolved.join('/');
};

const extractEpubHtmlText = (html: string) => {
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const root = doc.body ?? doc.documentElement;
  if (!root) return '';
  doc.querySelectorAll('script, style, svg, nav, header, footer, aside, noscript').forEach((node) => {
    node.remove();
  });
  const blocks = Array.from(root.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, blockquote, pre'));
  const parts = blocks
    .map((node) => node.textContent?.replace(/\s+/g, ' ').trim())
    .filter((value): value is string => Boolean(value));
  const raw = parts.length ? parts.join('\n\n') : root.textContent || '';
  return raw.replace(/\s+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
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

const scheduleIdle = (callback: () => void, timeout = 1200) => {
  if (typeof window === 'undefined') return null;
  const { requestIdleCallback } = window;
  if (typeof requestIdleCallback === 'function') {
    return requestIdleCallback(() => callback(), { timeout });
  }
  return window.setTimeout(callback, Math.min(timeout, 600));
};

const cancelIdle = (id: number | null) => {
  if (typeof window === 'undefined' || id === null) return;
  const { cancelIdleCallback } = window;
  if (typeof cancelIdleCallback === 'function') {
    cancelIdleCallback(id);
    return;
  }
  window.clearTimeout(id);
};

const extractPdfDataMainThread = async (
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
      const rawOutline = (await pdf.getOutline()) as PdfOutlineNode[] | null;
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

      const mapItems = async (items: PdfOutlineNode[]): Promise<PdfOutlineItem[]> => {
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

const SyncStatusPill = memo(() => {
  const { isAuthenticated, user, logout } = useAuth();
  const syncStatus = useSyncStatus();
  const [showLogin, setShowLogin] = useState(false);
  const [showMenu, setShowMenu] = useState(false);

  if (showLogin) {
    return (
      <div className="sync-login-dropdown">
        <LoginPanel onClose={() => setShowLogin(false)} />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <button type="button" className="status-pill status-pill--local" onClick={() => setShowLogin(true)} title="Sign in to sync across devices">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
        Sign in
      </button>
    );
  }

  const statusLabel = syncStatus === 'syncing' ? 'Syncing...' : syncStatus === 'error' ? 'Sync error' : 'Synced';
  const statusClass = `status-pill status-pill--${syncStatus === 'syncing' ? 'syncing' : syncStatus === 'error' ? 'error' : 'synced'}`;

  return (
    <span className="sync-status-wrapper">
      <button type="button" className={statusClass} onClick={() => setShowMenu((v) => !v)} title={`${user?.name || user?.email || 'User'} — ${statusLabel}`}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          {syncStatus === 'syncing' ? (
            <><path d="M21 12a9 9 0 1 1-6.22-8.56"/><polyline points="21 3 21 9 15 9"/></>
          ) : (
            <><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></>
          )}
        </svg>
        {statusLabel}
      </button>
      {showMenu && (
        <div className="sync-menu">
          <p className="sync-menu-user">{user?.name || user?.email}</p>
          <button type="button" className="ghost" onClick={() => { logout(); setShowMenu(false); }}>Sign out</button>
        </div>
      )}
    </span>
  );
});
SyncStatusPill.displayName = 'SyncStatusPill';

function App() {
  const { token: authToken, isAuthenticated } = useAuth();

  // Wire auth token to sync service
  useEffect(() => {
    syncService.setToken(authToken);
  }, [authToken]);

  const [title, setTitle] = useState('Untitled Session');
  const [sourceText, setSourceText] = useState(DEFAULT_TEXT);
  const [primaryFileMeta, setPrimaryFileMeta] = useState<FileMeta | null>(null);
  const [savedDocuments, setSavedDocuments] = useState<StoredDocument[]>([]);
  const [wpm, setWpm] = useState(320);
  const [chunkSize, setChunkSize] = useState(1);
  const [granularity, setGranularity] = useState<Granularity>('word');
  const [sentenceMode, setSentenceMode] = useState(false);
  const [contextRadius, setContextRadius] = useState(0);
  const [minWordMs, setMinWordMs] = useState(160);
  const [sentencePauseMs, setSentencePauseMs] = useState(200);
  const [displayScale, setDisplayScale] = useState(1);
  const [fitScale, setFitScale] = useState(1);
  const [readerDisplayWidth, setReaderDisplayWidth] = useState(0);
  const [longWordSlowdown, setLongWordSlowdown] = useState(40);
  const [sessionElapsedMs, setSessionElapsedMs] = useState(0);
  const [dailyGoalWords, setDailyGoalWords] = useState(() => {
    const stored = readStreakStorage();
    const goal = stored?.dailyGoalWords;
    return typeof goal === 'number' && goal > 0 ? goal : DEFAULT_DAILY_GOAL_WORDS;
  });
  const [streakStats, setStreakStats] = useState<StreakStats>(() => {
    const todayKey = getLocalDayKey();
    const stored = readStreakStorage();
    const stats = stored?.stats;
    const dayKey = typeof stats?.dayKey === 'string' ? stats.dayKey : todayKey;
    return {
      dayKey,
      dailyWords: typeof stats?.dailyWords === 'number' ? stats.dailyWords : 0,
      streak: typeof stats?.streak === 'number' ? stats.streak : 0,
      bestStreak: typeof stats?.bestStreak === 'number' ? stats.bestStreak : 0,
      lastCompletedDay: typeof stats?.lastCompletedDay === 'string' ? stats.lastCompletedDay : undefined,
    };
  });
  const [showSessionRecap, setShowSessionRecap] = useState(false);
  const [awardToast, setAwardToast] = useState<AwardStatus | null>(null);
  const [highlightAwardId, setHighlightAwardId] = useState<string | null>(null);
  const [wordIndex, setWordIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showHotkeys, setShowHotkeys] = useState(false);
  const [showLibrary, setShowLibrary] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('deck');
  const [showConventional, setShowConventional] = useState(true);
  const [showReaderDisplay, setShowReaderDisplay] = useState(true);
  const [conventionalMode, setConventionalMode] = useState<'excerpt' | 'rendered'>(() => {
    if (typeof window === 'undefined') return 'rendered';
    const stored = window.localStorage.getItem(CONVENTIONAL_MODE_KEY);
    return stored === 'excerpt' || stored === 'rendered' ? stored : 'rendered';
  });
  const [conventionalWindow, setConventionalWindow] = useState<{ start: number; end: number }>({
    start: 0,
    end: 0,
  });
  const [conventionalLayoutVersion, setConventionalLayoutVersion] = useState(0);
  const [conventionalBufferBoost, setConventionalBufferBoost] = useState(0);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [bookmarkNotice, setBookmarkNotice] = useState<Notice | null>(null);
  const [furthestRead, setFurthestRead] = useState<FurthestRead | null>(null);
  const [sourceKind, setSourceKind] = useState<'text' | 'pdf' | 'epub'>('text');
  const [pdfState, setPdfState] = useState<PdfState | null>(null);
  const [pageRange, setPageRange] = useState<PageRange>({ start: 1, end: 1 });
  const [pageRangeDraft, setPageRangeDraft] = useState<PageRange>({ start: 1, end: 1 });
  const [jumpPage, setJumpPage] = useState('');
  const [jumpPercent, setJumpPercent] = useState('');
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
  const [audioCuesEnabled, setAudioCuesEnabled] = useState(true);
  const [primaryNotice, setPrimaryNotice] = useState<Notice | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraHandEnabled, setCameraHandEnabled] = useState(true);
  const [cameraEyeEnabled, setCameraEyeEnabled] = useState(true);
  const [cameraPreview, setCameraPreview] = useState(false);
  const [cameraNotice, setCameraNotice] = useState<Notice | null>(null);
  const [companionInput, setCompanionInput] = useState('');
  const [companionLog, setCompanionLog] = useState<string[]>([
    'Hermeneutics companion will live here. Connect to /api/tutor/quick-chat when ready.',
  ]);
  const [findQuery, setFindQuery] = useState('');
  const [findCursor, setFindCursor] = useState(0);
  const [tokens, setTokens] = useState<string[]>([]);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [findMatches, setFindMatches] = useState<number[]>([]);
  const [workerReady, setWorkerReady] = useState(false);
  const [workerNonce, setWorkerNonce] = useState(0);
  const [workerFallbackNonce, setWorkerFallbackNonce] = useState(0);
  const [workerNotice, setWorkerNotice] = useState<Notice | null>(null);
  const [isIndexingPrimary, setIsIndexingPrimary] = useState(false);
  const [isFinding, setIsFinding] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const deferredSourceText = useDeferredValue(sourceText);
  const deferredFindQuery = useDeferredValue(findQuery);
  const renderedSourceText = deferredSourceText;
  const canUseWorker = typeof Worker !== 'undefined';
  const renderedMarkdown = useMemo(
    () => <ReactMarkdown remarkPlugins={[remarkGfm]}>{renderedSourceText}</ReactMarkdown>,
    [renderedSourceText],
  );
  const segmentStartIndices = useMemo(
    () => segments.map((segment) => segment.startIndex),
    [segments],
  );
  const sentenceSegments = useMemo(() => {
    if (!deferredSourceText.trim()) return [];
    return granularity === 'sentence' ? segments : segmentTextBySentence(deferredSourceText);
  }, [deferredSourceText, granularity, segments]);
  const sentenceStartIndices = useMemo(
    () => sentenceSegments.map((segment) => segment.startIndex),
    [sentenceSegments],
  );
  const segmentIndex = useMemo(
    () => getSegmentIndexForWordIndex(wordIndex, segmentStartIndices),
    [segmentStartIndices, wordIndex],
  );
  const currentSegments = useMemo(
    () => segments.slice(segmentIndex, segmentIndex + chunkSize),
    [segments, segmentIndex, chunkSize],
  );
  const currentSegmentWordCount = useMemo(
    () => currentSegments.reduce((sum, segment) => sum + segment.wordCount, 0),
    [currentSegments],
  );
  const currentSegmentText = useMemo(
    () => currentSegments.map((segment) => segment.text).join(' '),
    [currentSegments],
  );
  const singleWordMode = granularity === 'word' && chunkSize === 1;
  const longestWordLength = useMemo(() => getMaxWordLength(currentSegmentText), [currentSegmentText]);
  const fallbackFitScale = useMemo(() => {
    if (!singleWordMode) return 1;
    if (!longestWordLength) return 1;
    if (longestWordLength <= 12) return 1;
    return clamp(12 / longestWordLength, 0.35, 1);
  }, [longestWordLength, singleWordMode]);
  const readerDisplayStyle = useMemo(
    () => ({ '--display-scale': displayScale, '--fit-scale': fitScale } as CSSProperties),
    [displayScale, fitScale],
  );
  const activeSegmentRange = useMemo(() => {
    if (!currentSegments.length) return null;
    const start = currentSegments[0].startIndex;
    const last = currentSegments[currentSegments.length - 1];
    const end = last.startIndex + Math.max(1, last.wordCount) - 1;
    return { start, end };
  }, [currentSegments]);
  const docKey = useMemo(
    () => buildDocumentKey(sourceKind, primaryFileMeta, sourceText),
    [primaryFileMeta, sourceKind, sourceText],
  );

  const saveTimerRef = useRef<number | null>(null);
  const saveIdleRef = useRef<number | null>(null);
  const pendingBookmarkRef = useRef<BookmarkPayload | null>(null);
  const pendingBookmarkMismatchRef = useRef(false);
  const textWorkerRef = useRef<Worker | null>(null);
  const textWorkerIdRef = useRef(0);
  const primaryDocIdRef = useRef(0);
  const primaryDocReadyRef = useRef(false);
  const pendingPrimarySegmentRef = useRef(false);
  const findQueryRef = useRef('');
  const granularityRef = useRef<Granularity>('word');
  const workerRestartTimerRef = useRef<number | null>(null);
  const workerNoticeTimerRef = useRef<number | null>(null);
  const pdfDataRef = useRef<Uint8Array | null>(null);
  const pdfWorkerRef = useRef<Worker | null>(null);
  const pdfWorkerCallbacksRef = useRef<
    Map<
      number,
      {
        resolve: (value: {
          pageTexts: string[];
          pageTokenCounts: number[];
          outline: PdfOutlineItem[];
          pageCount: number;
        }) => void;
        reject: (error: Error) => void;
        onProgress?: (current: number, total: number) => void;
      }
    >
  >(new Map());
  const pdfWorkerIdRef = useRef(0);
  const ttsAbortRef = useRef<AbortController | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const ttsAudioUrlRef = useRef<string | null>(null);
  const ttsIntervalRef = useRef<number | null>(null);
  const ttsEnabledRef = useRef(false);
  const ttsPlayingRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioCuesEnabledRef = useRef(true);
  const audioCueLastAtRef = useRef(0);
  const ttsConfigRef = useRef({
    voice: 'sarah',
    speed: 1.0,
    language: 'en-us',
    chunkWords: 26,
  });
  const tokensRef = useRef<string[]>([]);
  const segmentsRef = useRef<Segment[]>([]);
  const segmentStartIndicesRef = useRef<number[]>([]);
  const sourcePanelRef = useRef<HTMLElement | null>(null);
  const readerDisplayRef = useRef<HTMLDivElement | null>(null);
  const measureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fitScaleRef = useRef(1);
  const conventionalRef = useRef<HTMLDivElement | null>(null);
  const conventionalChunkHeightsRef = useRef<number[]>([]);
  const conventionalAvgChunkHeightRef = useRef(DEFAULT_CONVENTIONAL_CHUNK_HEIGHT);
  const pendingConventionalScrollRef = useRef<{ index: number; behavior: ScrollBehavior } | null>(null);
  const scrollRafRef = useRef<number | null>(null);
  const scrollLockRef = useRef(false);
  const scrollEndTimeoutRef = useRef<number | null>(null);
  const autoFollowRafRef = useRef<number | null>(null);
  const lastConventionalScrollTopRef = useRef(0);
  const lastConventionalScrollAtRef = useRef(0);
  const conventionalScrollDirectionRef = useRef(1);
  const conventionalBoostTimeoutRef = useRef<number | null>(null);
  const conventionalBufferBoostRef = useRef(0);
  const prevActiveRangeRef = useRef<{ start: number; end: number } | null>(null);
  const prevChunkClassRef = useRef<number>(-1);
  const isPlayingRef = useRef(false);
  const conventionalSeekEnabledRef = useRef(true);
  const wordIndexRef = useRef(0);
  const chunkSizeRef = useRef(chunkSize);
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const cameraRafRef = useRef<number | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handleSeekRef = useRef<
    (index: number, options?: { pause?: boolean; focusConventional?: boolean; scrollBehavior?: ScrollBehavior }) => void
  >(() => {});
  const handleBookmarkRef = useRef<() => void>(() => {});
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
  const lastDocIdleRef = useRef<number | null>(null);
  const didHydrateRef = useRef(false);
  const prefsSaveTimerRef = useRef<number | null>(null);
  const prefsIdleRef = useRef<number | null>(null);
  const didPrefHydrateRef = useRef(false);
  const hasSessionActivityRef = useRef(false);
  const lastCountedRef = useRef<{ dayKey: string; docKey: string; words: number }>({
    dayKey: getLocalDayKey(),
    docKey: '',
    words: 0,
  });
  const awardToastTimerRef = useRef<number | null>(null);
  const awardHighlightTimerRef = useRef<number | null>(null);
  const earnedAwardIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    ttsEnabledRef.current = ttsEnabled;
  }, [ttsEnabled]);

  useEffect(() => {
    ttsPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    audioCuesEnabledRef.current = audioCuesEnabled;
  }, [audioCuesEnabled]);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    fitScaleRef.current = fitScale;
  }, [fitScale]);

  useLayoutEffect(() => {
    if (typeof window === 'undefined') return;
    const element = readerDisplayRef.current;
    if (!element) return;
    const updateSize = () => {
      setReaderDisplayWidth(element.clientWidth);
    };
    updateSize();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updateSize);
      return () => {
        window.removeEventListener('resize', updateSize);
      };
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      setReaderDisplayWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    if (!singleWordMode) {
      if (fitScaleRef.current !== 1) {
        setFitScale(1);
      }
      return;
    }
    if (typeof window === 'undefined') {
      if (fitScaleRef.current !== fallbackFitScale) {
        setFitScale(fallbackFitScale);
      }
      return;
    }
    const element = readerDisplayRef.current;
    if (!element || !currentSegmentText.trim()) {
      if (fitScaleRef.current !== fallbackFitScale) {
        setFitScale(fallbackFitScale);
      }
      return;
    }
    const styles = window.getComputedStyle(element);
    const paddingLeft = parseFloat(styles.paddingLeft) || 0;
    const paddingRight = parseFloat(styles.paddingRight) || 0;
    const availableWidth = element.clientWidth - paddingLeft - paddingRight;
    if (availableWidth <= 0) {
      if (fitScaleRef.current !== fallbackFitScale) {
        setFitScale(fallbackFitScale);
      }
      return;
    }
    const fontSize = parseFloat(styles.fontSize) || 16;
    const baseFontSize = fontSize / Math.max(0.1, fitScaleRef.current);
    const font = `${styles.fontWeight} ${baseFontSize}px ${styles.fontFamily}`;
    const canvas = measureCanvasRef.current ?? document.createElement('canvas');
    measureCanvasRef.current = canvas;
    const context = canvas.getContext('2d');
    if (!context) {
      if (fitScaleRef.current !== fallbackFitScale) {
        setFitScale(fallbackFitScale);
      }
      return;
    }
    context.font = font;
    const textWidth = context.measureText(currentSegmentText).width;
    if (!textWidth) {
      if (fitScaleRef.current !== fallbackFitScale) {
        setFitScale(fallbackFitScale);
      }
      return;
    }
    const nextScale = clamp(availableWidth / textWidth, 0.35, 1);
    if (Math.abs(nextScale - fitScaleRef.current) > 0.01) {
      setFitScale(nextScale);
    }
  }, [currentSegmentText, displayScale, fallbackFitScale, readerDisplayWidth, singleWordMode]);

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
    conventionalBufferBoostRef.current = conventionalBufferBoost;
  }, [conventionalBufferBoost]);

  useEffect(() => {
    granularityRef.current = granularity;
  }, [granularity]);

  useEffect(() => {
    if ((granularity === 'sentence' || granularity === 'tweet') && chunkSize !== 1) {
      setChunkSize(1);
    }
  }, [chunkSize, granularity]);

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
    segmentsRef.current = segments;
    segmentStartIndicesRef.current = segmentStartIndices;
    prevActiveRangeRef.current = null;
  }, [granularity, segmentStartIndices, segments, tokens]);

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const handleChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', handleChange);
    handleChange();
    return () => {
      document.removeEventListener('fullscreenchange', handleChange);
    };
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const encoded = readBookmarkParam(window.location.href);
    if (!encoded) return;
    const payload = parseBookmarkValue(encoded);
    if (!payload) return;
    pendingBookmarkRef.current = payload;
  }, []);

  useEffect(() => {
    findQueryRef.current = deferredFindQuery;
  }, [deferredFindQuery]);

  useEffect(() => {
    if (workerReady && workerNotice) {
      setWorkerNotice(null);
    }
  }, [workerNotice, workerReady]);

  useEffect(() => {
    return () => {
      if (workerRestartTimerRef.current) {
        window.clearTimeout(workerRestartTimerRef.current);
      }
      if (workerNoticeTimerRef.current) {
        window.clearTimeout(workerNoticeTimerRef.current);
      }
      if (conventionalBoostTimeoutRef.current) {
        window.clearTimeout(conventionalBoostTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!canUseWorker) return;
    const worker = new Worker(new URL('./textWorker.ts', import.meta.url), { type: 'module' });
    textWorkerRef.current = worker;
    setWorkerReady(true);

    const handleMessage = (event: MessageEvent) => {
      const data = event.data as { type?: string };
      if (!data || typeof data !== 'object') return;
      if (data.type === 'analyze-result') {
        const payload = data as {
          docId: number;
          kind: 'primary';
          tokens: string[];
          segments: Segment[];
        };
        if (payload.docId !== primaryDocIdRef.current) return;
        primaryDocReadyRef.current = true;
        setIsIndexingPrimary(false);
        setTokens(payload.tokens);
        if (pendingPrimarySegmentRef.current) {
          pendingPrimarySegmentRef.current = false;
          const requestId = ++textWorkerIdRef.current;
          setIsIndexingPrimary(true);
          worker.postMessage({
            type: 'segment',
            id: requestId,
            docId: payload.docId,
            kind: 'primary',
            granularity: granularityRef.current,
          });
        } else {
          setSegments(payload.segments);
        }
        return;
      }
      if (data.type === 'segment-result') {
        const payload = data as {
          docId: number;
          kind: 'primary';
          segments: Segment[];
        };
        if (payload.docId !== primaryDocIdRef.current) return;
        setIsIndexingPrimary(false);
        setSegments(payload.segments);
        return;
      }
      if (data.type === 'find-result') {
        const payload = data as { docId: number; query: string; matches: number[] };
        if (payload.docId !== primaryDocIdRef.current) return;
        if (payload.query !== findQueryRef.current) return;
        setIsFinding(false);
        setFindMatches(payload.matches);
      }
    };

    const handleWorkerError = () => {
      if (workerNoticeTimerRef.current) {
        window.clearTimeout(workerNoticeTimerRef.current);
      }
      setWorkerNotice({ kind: 'info', message: 'Background indexing paused. Retrying shortly.' });
      setWorkerReady(false);
      setWorkerFallbackNonce((value) => value + 1);
      if (workerRestartTimerRef.current) {
        window.clearTimeout(workerRestartTimerRef.current);
      }
      workerRestartTimerRef.current = window.setTimeout(() => {
        setWorkerNonce((value) => value + 1);
      }, 600);
      workerNoticeTimerRef.current = window.setTimeout(() => {
        setWorkerNotice(null);
      }, 4000);
      worker.terminate();
      if (textWorkerRef.current === worker) {
        textWorkerRef.current = null;
      }
    };

    worker.addEventListener('message', handleMessage);
    worker.addEventListener('error', handleWorkerError);
    worker.addEventListener('messageerror', handleWorkerError);
    return () => {
      worker.removeEventListener('message', handleMessage);
      worker.removeEventListener('error', handleWorkerError);
      worker.removeEventListener('messageerror', handleWorkerError);
      worker.terminate();
      if (textWorkerRef.current === worker) {
        textWorkerRef.current = null;
      }
      setWorkerReady(false);
    };
  }, [canUseWorker, workerNonce]);

  useEffect(() => {
    setFindCursor((prev) => {
      if (!findMatches.length) return 0;
      return prev >= findMatches.length ? 0 : prev;
    });
  }, [deferredFindQuery, findMatches.length]);

  useEffect(() => {
    primaryDocReadyRef.current = false;
    pendingPrimarySegmentRef.current = false;
    primaryDocIdRef.current += 1;
  }, [deferredSourceText]);

  useEffect(() => {
    const docId = primaryDocIdRef.current;
    if (!deferredSourceText.trim()) {
      setIsIndexingPrimary(false);
      setTokens([]);
      setSegments([]);
      setFindMatches([]);
      return;
    }
    if (!canUseWorker || !workerReady || !textWorkerRef.current) {
      setIsIndexingPrimary(true);
      const nextTokens = tokenize(deferredSourceText);
      setTokens(nextTokens);
      const nextSegments =
        granularityRef.current === 'sentence'
          ? segmentTextBySentence(deferredSourceText)
          : granularityRef.current === 'tweet'
            ? segmentTextByTweet(deferredSourceText)
            : segmentTokens(nextTokens, granularityRef.current);
      setSegments(nextSegments);
      setIsIndexingPrimary(false);
      return;
    }
    setIsIndexingPrimary(true);
    const requestId = ++textWorkerIdRef.current;
    textWorkerRef.current.postMessage({
      type: 'analyze',
      id: requestId,
      docId,
      kind: 'primary',
      text: deferredSourceText,
      granularity: granularityRef.current,
    });
  }, [canUseWorker, deferredSourceText, workerFallbackNonce, workerReady]);

  useEffect(() => {
    if (!deferredSourceText.trim()) {
      setSegments([]);
      return;
    }
    if (canUseWorker && workerReady && textWorkerRef.current) {
      if (primaryDocReadyRef.current) {
        const requestId = ++textWorkerIdRef.current;
        setIsIndexingPrimary(true);
        textWorkerRef.current.postMessage({
          type: 'segment',
          id: requestId,
          docId: primaryDocIdRef.current,
          kind: 'primary',
          granularity,
        });
        return;
      }
      pendingPrimarySegmentRef.current = true;
    }
    if (!tokens.length) return;
    setIsIndexingPrimary(true);
    const nextSegments =
      granularity === 'sentence'
        ? segmentTextBySentence(deferredSourceText)
        : granularity === 'tweet'
          ? segmentTextByTweet(deferredSourceText)
          : segmentTokens(tokens, granularity);
    setSegments(nextSegments);
    setIsIndexingPrimary(false);
  }, [canUseWorker, deferredSourceText, granularity, tokens, workerReady]);


  useEffect(() => {
    if (!deferredFindQuery.trim() || !tokens.length) {
      setIsFinding(false);
      setFindMatches([]);
      return;
    }
    if (canUseWorker && workerReady && textWorkerRef.current && primaryDocReadyRef.current) {
      setIsFinding(true);
      const requestId = ++textWorkerIdRef.current;
      textWorkerRef.current.postMessage({
        type: 'find',
        id: requestId,
        docId: primaryDocIdRef.current,
        query: deferredFindQuery,
      });
      return;
    }
    setIsFinding(true);
    const queryTokens = tokenize(deferredFindQuery)
      .map((token) => normalizeTokenForSearch(token))
      .filter(Boolean);
    const searchTokens = tokens.map((token) => normalizeTokenForSearch(token));
    if (!queryTokens.length || !searchTokens.length) {
      setIsFinding(false);
      setFindMatches([]);
      return;
    }
    const matches: number[] = [];
    const maxStart = searchTokens.length - queryTokens.length;
    for (let i = 0; i <= maxStart; i += 1) {
      let isMatch = true;
      for (let j = 0; j < queryTokens.length; j += 1) {
        const token = searchTokens[i + j];
        if (!token || !token.includes(queryTokens[j])) {
          isMatch = false;
          break;
        }
      }
      if (isMatch) {
        matches.push(i);
      }
    }
    setFindMatches(matches);
    setIsFinding(false);
  }, [canUseWorker, deferredFindQuery, tokens, workerReady]);

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
      if (typeof parsed.title === 'string' && parsed.title.trim()) {
        setTitle(parsed.title);
      }
      if (parsed.sourceKind === 'pdf' || parsed.sourceKind === 'text' || parsed.sourceKind === 'epub') {
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
    let cancelled = false;
    const hydrateHistory = async () => {
      const docs = await readStoredDocumentsFromDb();
      if (cancelled) return;
      const sorted = [...docs].sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0));
      setSavedDocuments(sorted);
    };
    hydrateHistory().catch((error) => {
      console.warn('Failed to hydrate stored document history.', error);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Sync: pull documents and preferences from server when authenticated
  useEffect(() => {
    if (!isAuthenticated) return;
    let cancelled = false;
    (async () => {
      // Pull document list from server and merge with local IndexedDB
      const serverDocs = await syncService.pullDocumentList();
      if (cancelled) return;
      const localDocs = await readStoredDocumentsFromDb();
      if (cancelled) return;

      // Build a set of local file names for matching (case-insensitive)
      const localFileNames = new Set(
        localDocs.map((d) => (d.primaryFileMeta?.name || d.title || '').toLowerCase()),
      );
      // Build a set of server file names for matching
      const serverFileNames = new Set(
        (serverDocs || []).map((d) => (d.fileName || d.title || '').toLowerCase()),
      );

      // Pull: download server docs missing locally
      if (serverDocs?.length) {
        for (const serverDoc of serverDocs) {
          const serverName = (serverDoc.fileName || serverDoc.title || '').toLowerCase();
          if (localFileNames.has(serverName)) continue;
          const blob = await syncService.pullFile(serverDoc.id);
          if (cancelled) return;
          if (blob) {
            const text = await blob.text();
            const localId = `reader:${serverDoc.sourceKind}:${serverDoc.fileName}:${serverDoc.fileSize}:0`;
            const newDoc: StoredDocument = {
              id: localId,
              sourceText: text,
              title: serverDoc.title,
              sourceKind: serverDoc.sourceKind as StoredDocument['sourceKind'],
              primaryFileMeta: {
                name: serverDoc.fileName,
                size: serverDoc.fileSize,
                lastModified: 0,
              },
              wordCount: serverDoc.wordCount,
              updatedAt: new Date(serverDoc.updatedAt).getTime(),
            };
            await writeStoredDocumentToDb(newDoc);
            if (cancelled) return;
            setSavedDocuments((prev) => {
              if (prev.some((d) => d.id === localId)) return prev;
              return [newDoc, ...prev];
            });
          }
        }
      }

      // Push: upload local docs missing from server (content-hash dedup on server prevents real duplication)
      for (const localDoc of localDocs) {
        if (cancelled) return;
        if (!localDoc.sourceText) continue;
        const localName = (localDoc.primaryFileMeta?.name || localDoc.title || '').toLowerCase();
        if (serverFileNames.has(localName)) continue;
        const blob = new Blob([localDoc.sourceText], { type: 'text/plain' });
        const fileName = localDoc.primaryFileMeta?.name || `${localDoc.title || 'untitled'}.txt`;
        const file = new File([blob], fileName, { type: 'text/plain' });
        await syncService.pushDocument(file, localDoc.title || 'Untitled', localDoc.sourceKind, localDoc.wordCount);
      }

      // Pull preferences from server and merge
      const serverPrefs = await syncService.pullPreferences();
      if (cancelled) return;
      if (serverPrefs?.preferences && typeof serverPrefs.preferences === 'object') {
        const prefs = serverPrefs.preferences as Record<string, unknown>;
        // Merge server preferences with local (server wins if newer)
        const localRaw = window.localStorage.getItem(USER_PREFERENCES_KEY);
        if (!localRaw) {
          window.localStorage.setItem(USER_PREFERENCES_KEY, JSON.stringify(prefs));
        }
      }
      if (serverPrefs?.streak && typeof serverPrefs.streak === 'object') {
        const localRaw = window.localStorage.getItem(STREAK_STORAGE_KEY);
        if (!localRaw) {
          window.localStorage.setItem(STREAK_STORAGE_KEY, JSON.stringify(serverPrefs.streak));
        }
      }
    })().catch((error) => {
      console.warn('Failed to pull sync data from server.', error);
    });
    return () => { cancelled = true; };
  }, [isAuthenticated]);

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
      if (typeof parsed.showReaderDisplay === 'boolean') {
        setShowReaderDisplay(parsed.showReaderDisplay);
      }
      if (typeof parsed.conventionalSeekEnabled === 'boolean') {
        setConventionalSeekEnabled(parsed.conventionalSeekEnabled);
      }
      if (typeof parsed.autoFollowConventional === 'boolean') {
        setAutoFollowConventional(parsed.autoFollowConventional);
      }
      if (typeof parsed.audioCuesEnabled === 'boolean') {
        setAudioCuesEnabled(parsed.audioCuesEnabled);
      }
      if (
        parsed.viewMode === 'deck' ||
        parsed.viewMode === 'mobile' ||
        parsed.viewMode === 'focus' ||
        parsed.viewMode === 'focus-text'
      ) {
        setViewMode(parsed.viewMode);
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
      if (typeof parsed.displayScale === 'number') {
        setDisplayScale(clamp(parsed.displayScale, 0.8, 1.35));
      }
      if (typeof parsed.longWordSlowdown === 'number') {
        setLongWordSlowdown(clamp(parsed.longWordSlowdown, 0, 80));
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
    if (prefsIdleRef.current) {
      cancelIdle(prefsIdleRef.current);
      prefsIdleRef.current = null;
    }
    prefsSaveTimerRef.current = window.setTimeout(() => {
      prefsIdleRef.current = scheduleIdle(() => {
        const payload: UserPreferences = {
          showConventional,
          showReaderDisplay,
          conventionalSeekEnabled,
          autoFollowConventional,
          audioCuesEnabled,
          viewMode,
          cameraEnabled,
          cameraHandEnabled,
          cameraEyeEnabled,
          cameraPreview,
          displayScale,
          longWordSlowdown,
        };
        try {
          window.localStorage.setItem(USER_PREFERENCES_KEY, JSON.stringify(payload));
          // Sync preferences to server
          if (syncService.isAuthenticated()) {
            syncService.pushPreferences({ preferences: payload });
          }
        } catch (error) {
          console.warn('Failed to persist user preferences.', error);
        }
      }, 1200);
    }, 250);
    return () => {
      if (prefsSaveTimerRef.current) {
        window.clearTimeout(prefsSaveTimerRef.current);
      }
      if (prefsIdleRef.current) {
        cancelIdle(prefsIdleRef.current);
        prefsIdleRef.current = null;
      }
    };
  }, [
    autoFollowConventional,
    audioCuesEnabled,
    cameraEnabled,
    cameraEyeEnabled,
    cameraHandEnabled,
    cameraPreview,
    conventionalSeekEnabled,
    displayScale,
    longWordSlowdown,
    showConventional,
    showReaderDisplay,
    viewMode,
  ]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!didHydrateRef.current) return;
    if (lastDocSaveTimerRef.current) {
      window.clearTimeout(lastDocSaveTimerRef.current);
    }
    if (lastDocIdleRef.current) {
      cancelIdle(lastDocIdleRef.current);
      lastDocIdleRef.current = null;
    }
    lastDocSaveTimerRef.current = window.setTimeout(() => {
      lastDocIdleRef.current = scheduleIdle(() => {
        const payload: LastDocument = {
          sourceText,
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
      }, 1500);
    }, 400);
    return () => {
      if (lastDocSaveTimerRef.current) {
        window.clearTimeout(lastDocSaveTimerRef.current);
      }
      if (lastDocIdleRef.current) {
        cancelIdle(lastDocIdleRef.current);
        lastDocIdleRef.current = null;
      }
    };
  }, [primaryFileMeta, sourceKind, sourceText, title]);

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
      const step = chunkSizeRef.current;
      const nextWordIndex = getSteppedWordIndex(
        wordIndexRef.current,
        direction === 'back' ? 'back' : 'next',
        step,
        segmentStartIndicesRef.current,
        segmentsRef.current,
      );
      handleSeekRef.current(nextWordIndex);
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
    const maxIndex = Math.max(0, tokens.length - 1);
    const prevRange = prevActiveRangeRef.current;
    if (prevRange) {
      const start = clamp(prevRange.start, 0, maxIndex);
      const end = clamp(prevRange.end, 0, maxIndex);
      for (let i = start; i <= end; i += 1) {
        const prevEl = container.querySelector(`[data-word-index="${i}"]`) as HTMLElement | null;
        prevEl?.classList.remove('is-active');
      }
    }
    if (activeSegmentRange) {
      const start = clamp(activeSegmentRange.start, 0, maxIndex);
      const end = clamp(activeSegmentRange.end, 0, maxIndex);
      for (let i = start; i <= end; i += 1) {
        const nextEl = container.querySelector(`[data-word-index="${i}"]`) as HTMLElement | null;
        nextEl?.classList.add('is-active');
      }
    }
    prevActiveRangeRef.current = activeSegmentRange;

    // Classify chunks: read / current / upcoming
    const chunkSize = getConventionalChunkSize(tokens.length);
    if (chunkSize > 0) {
      const currentChunkIdx = Math.floor(wordIndex / chunkSize);
      if (currentChunkIdx !== prevChunkClassRef.current) {
        const chunks = container.querySelectorAll<HTMLElement>('[data-chunk-index]');
        for (const chunk of chunks) {
          const idx = Number(chunk.dataset.chunkIndex);
          chunk.classList.toggle('chunk-read', idx < currentChunkIdx);
          chunk.classList.toggle('chunk-current', idx === currentChunkIdx);
        }
        prevChunkClassRef.current = currentChunkIdx;
      }
    }
  }, [activeSegmentRange, conventionalMode, conventionalWindow, showConventional, tokens.length, wordIndex]);

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
      if (
        parsed.granularity === 'word' ||
        parsed.granularity === 'bigram' ||
        parsed.granularity === 'trigram' ||
        parsed.granularity === 'sentence' ||
        parsed.granularity === 'tweet'
      ) {
        setGranularity(parsed.granularity);
      } else {
        setGranularity('word');
      }
      setContextRadius(
        typeof parsed.contextRadius === 'number' ? clamp(parsed.contextRadius, 0, 6) : 0,
      );
      setMinWordMs(
        typeof parsed.minWordMs === 'number' ? clamp(parsed.minWordMs, 80, 400) : 160,
      );
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
    if (saveIdleRef.current) {
      cancelIdle(saveIdleRef.current);
      saveIdleRef.current = null;
    }
    saveTimerRef.current = window.setTimeout(() => {
      saveIdleRef.current = scheduleIdle(() => {
      const payload: SavedSession = {
        index: wordIndex,
        wpm,
        chunkSize,
        granularity,
        contextRadius,
        minWordMs,
        sentencePauseMs,
        sessionElapsedMs,
        title,
          bookmarks,
          furthestRead,
          updatedAt: Date.now(),
        };
        localStorage.setItem(docKey, JSON.stringify(payload));
        // Sync session to server (uses docKey as the server document ID reference)
        if (syncService.isAuthenticated() && docKey) {
          syncService.pushSession(docKey, payload);
        }
      }, 1500);
    }, 300);
    return () => {
      if (saveTimerRef.current) {
        window.clearTimeout(saveTimerRef.current);
      }
      if (saveIdleRef.current) {
        cancelIdle(saveIdleRef.current);
        saveIdleRef.current = null;
      }
    };
  }, [
    docKey,
    wordIndex,
    wpm,
    chunkSize,
    granularity,
    contextRadius,
    minWordMs,
    sentencePauseMs,
    sessionElapsedMs,
    title,
    bookmarks,
    furthestRead,
  ]);

  useEffect(() => {
    if (!tokens.length) {
      setIsPlaying(false);
      setWordIndex(0);
      return;
    }
    setWordIndex((prev) => clamp(prev, 0, Math.max(0, tokens.length - 1)));
  }, [tokens.length]);

  useEffect(() => {
    if (granularity === 'word') return;
    const current = segments[segmentIndex];
    if (current && wordIndex !== current.startIndex) {
      setWordIndex(current.startIndex);
    }
  }, [granularity, segmentIndex, segments, wordIndex]);

  useEffect(() => {
    if (!isPlaying) {
      postPauseDelayRef.current = 0;
    }
  }, [isPlaying]);

  useEffect(() => {
    if (!isPlaying || segments.length === 0 || ttsEnabled) return;
    const wordsInChunk = Math.max(1, currentSegmentWordCount || 1);
    const baseIntervalMs = (60000 / Math.max(60, wpm)) * wordsInChunk;
    const minIntervalMs = Math.max(baseIntervalMs, minWordMs * wordsInChunk);
    const chunkText = currentSegments.map((segment) => segment.text).join(' ');
    const complexityMultiplier = getReadingComplexityMultiplier(chunkText);
    const slowdownIntensity = longWordSlowdown / 100;
    const intervalMs =
      Math.max(80, minIntervalMs) * (1 + (complexityMultiplier - 1) * slowdownIntensity);
    const resumeDelayMs = postPauseDelayRef.current;
    const lastSegmentText = currentSegments[currentSegments.length - 1]?.text ?? '';
    const pauseMs = getSentencePauseMs(lastSegmentText, sentencePauseMs);
    postPauseDelayRef.current = pauseMs > 0 ? Math.round(Math.min(160, pauseMs * 0.5)) : 0;
    const timerId = window.setTimeout(() => {
      setWordIndex((prev) => {
        const currentSegmentIndex = getSegmentIndexForWordIndex(prev, segmentStartIndices);
        const nextSegmentIndex = currentSegmentIndex + chunkSize;
        if (nextSegmentIndex >= segments.length) {
          setIsPlaying(false);
          return prev;
        }
        const nextSegment = segments[nextSegmentIndex];
        return nextSegment ? nextSegment.startIndex : prev;
      });
    }, intervalMs + pauseMs + resumeDelayMs);
    return () => window.clearTimeout(timerId);
  }, [
    chunkSize,
    currentSegmentWordCount,
    currentSegments,
    isPlaying,
    minWordMs,
    segmentStartIndices,
    segments,
    sentencePauseMs,
    longWordSlowdown,
    ttsEnabled,
    wpm,
  ]);

  const conventionalTokens = useMemo(() => tokens, [tokens]);
  const conventionalChunkSize = useMemo(
    () => getConventionalChunkSize(conventionalTokens.length),
    [conventionalTokens.length],
  );
  const conventionalBufferChunks = useMemo(
    () => getConventionalBufferChunks(conventionalChunkSize),
    [conventionalChunkSize],
  );
  const effectiveConventionalBuffer = useMemo(
    () => Math.max(1, conventionalBufferChunks + conventionalBufferBoost),
    [conventionalBufferBoost, conventionalBufferChunks],
  );
  const conventionalChunkCount = useMemo(
    () => Math.ceil(conventionalTokens.length / conventionalChunkSize),
    [conventionalChunkSize, conventionalTokens.length],
  );
  const getConventionalChunkHeight = (index: number) =>
    conventionalChunkHeightsRef.current[index] ?? conventionalAvgChunkHeightRef.current;
  const conventionalOffsets = useMemo(() => {
    if (conventionalChunkCount === 0) return [0];
    const offsets = new Array(conventionalChunkCount + 1);
    offsets[0] = conventionalLayoutVersion * 0;
    for (let i = 0; i < conventionalChunkCount; i += 1) {
      offsets[i + 1] = offsets[i] + getConventionalChunkHeight(i);
    }
    return offsets;
  }, [conventionalChunkCount, conventionalLayoutVersion]);
  const conventionalSpacerHeights = useMemo(() => {
    if (conventionalMode !== 'excerpt' || conventionalChunkCount === 0) {
      return { top: 0, bottom: 0 };
    }
    const startIndex = clamp(conventionalWindow.start, 0, Math.max(0, conventionalChunkCount - 1));
    const endIndex = clamp(
      conventionalWindow.end,
      startIndex,
      Math.max(startIndex, conventionalChunkCount - 1),
    );
    const total = conventionalOffsets[conventionalOffsets.length - 1] ?? 0;
    const startOffset = conventionalOffsets[startIndex] ?? 0;
    const endOffset = conventionalOffsets[Math.min(endIndex + 1, conventionalOffsets.length - 1)] ?? total;
    return {
      top: startOffset,
      bottom: Math.max(0, total - endOffset),
    };
  }, [conventionalChunkCount, conventionalMode, conventionalOffsets, conventionalWindow.end, conventionalWindow.start]);
  const registerConventionalChunk = useCallback((chunkIndex: number) => (node: HTMLDivElement | null) => {
    if (!node) return;
    const height = node.getBoundingClientRect().height;
    if (!height) return;
    const prev = conventionalChunkHeightsRef.current[chunkIndex];
    if (prev && Math.abs(prev - height) < 1) return;
    conventionalChunkHeightsRef.current[chunkIndex] = height;
    const measured = conventionalChunkHeightsRef.current.filter(
      (value): value is number => typeof value === 'number' && !Number.isNaN(value),
    );
    if (measured.length > 0) {
      conventionalAvgChunkHeightRef.current = measured.reduce((sum, value) => sum + value, 0) / measured.length;
    }
    setConventionalLayoutVersion((value) => value + 1);
  }, []);
  const conventionalNodes = useMemo(() => {
    if (conventionalMode !== 'excerpt') return null;
    if (conventionalChunkCount === 0) return null;
    const startChunk = clamp(conventionalWindow.start, 0, Math.max(0, conventionalChunkCount - 1));
    const endChunk = clamp(
      conventionalWindow.end,
      startChunk,
      Math.max(startChunk, conventionalChunkCount - 1),
    );
    const nodes = [];
    for (let chunkIndex = startChunk; chunkIndex <= endChunk; chunkIndex += 1) {
      const start = chunkIndex * conventionalChunkSize;
      const end = Math.min(start + conventionalChunkSize, conventionalTokens.length);
      const slice = conventionalTokens.slice(start, end);
      nodes.push(
        <div
          key={`chunk-${chunkIndex}`}
          className="excerpt-chunk"
          ref={registerConventionalChunk(chunkIndex)}
          data-chunk-index={chunkIndex}
        >
          {slice.map((token, offset) => {
            const wordIndex = start + offset;
            return (
              <span key={`${wordIndex}-${token}`} data-word-index={wordIndex} className="word">
                {token}{' '}
              </span>
            );
          })}
        </div>,
      );
    }
    return nodes;
  }, [
    conventionalChunkCount,
    conventionalChunkSize,
    conventionalMode,
    conventionalTokens,
    conventionalWindow,
    registerConventionalChunk,
  ]);
  const updateConventionalWindowForChunk = useCallback(
    (chunkIndex: number) => {
      if (conventionalMode !== 'excerpt') return;
      if (conventionalChunkCount === 0) return;
      setConventionalWindow((prev) => {
        const maxIndex = Math.max(0, conventionalChunkCount - 1);
        const nextStart = clamp(chunkIndex - effectiveConventionalBuffer, 0, maxIndex);
        const nextEnd = clamp(chunkIndex + effectiveConventionalBuffer, 0, maxIndex);
        if (prev.start === nextStart && prev.end === nextEnd) return prev;
        if (chunkIndex >= prev.start + 1 && chunkIndex <= prev.end - 1) return prev;
        return { start: nextStart, end: nextEnd };
      });
    },
    [conventionalChunkCount, conventionalMode, effectiveConventionalBuffer],
  );
  const ensureConventionalWindowForIndex = useCallback(
    (targetIndex: number) => {
      const chunkIndex = Math.floor(targetIndex / conventionalChunkSize);
      updateConventionalWindowForChunk(chunkIndex);
    },
    [conventionalChunkSize, updateConventionalWindowForChunk],
  );
  const findChunkIndexForScroll = useCallback(
    (scrollTop: number) => {
      if (conventionalOffsets.length <= 1) return 0;
      let low = 0;
      let high = conventionalOffsets.length - 1;
      while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (conventionalOffsets[mid] <= scrollTop) {
          low = mid + 1;
        } else {
          high = mid;
        }
      }
      return Math.max(0, low - 1);
    },
    [conventionalOffsets],
  );

  useEffect(() => {
    conventionalChunkHeightsRef.current = [];
    conventionalAvgChunkHeightRef.current = DEFAULT_CONVENTIONAL_CHUNK_HEIGHT;
    pendingConventionalScrollRef.current = null;
    setConventionalWindow(() => {
      if (conventionalChunkCount === 0) return { start: 0, end: 0 };
      const maxIndex = Math.max(0, conventionalChunkCount - 1);
      const anchorIndex = clamp(wordIndexRef.current ?? 0, 0, Math.max(0, conventionalTokens.length - 1));
      const anchorChunk = Math.floor(anchorIndex / conventionalChunkSize);
      const start = clamp(anchorChunk - conventionalBufferChunks, 0, maxIndex);
      const end = clamp(anchorChunk + conventionalBufferChunks, 0, maxIndex);
      return { start, end };
    });
    setConventionalLayoutVersion((value) => value + 1);
  }, [conventionalBufferChunks, conventionalChunkCount, conventionalChunkSize, conventionalMode, conventionalTokens.length]);

  useEffect(() => {
    if (!showConventional || conventionalMode !== 'excerpt') return;
    ensureConventionalWindowForIndex(wordIndex);
  }, [conventionalMode, ensureConventionalWindowForIndex, showConventional, wordIndex]);

  useEffect(() => {
    const pending = pendingConventionalScrollRef.current;
    if (!pending || !showConventional) return;
    const container = conventionalRef.current;
    if (!container) return;
    if (conventionalMode === 'rendered') {
      const ratio = tokens.length ? clamp(pending.index / Math.max(1, tokens.length - 1), 0, 1) : 0;
      const maxScroll = container.scrollHeight - container.clientHeight;
      scrollLockRef.current = true;
      container.scrollTo({ top: maxScroll * ratio, behavior: pending.behavior });
      window.setTimeout(() => {
        scrollLockRef.current = false;
      }, 120);
      pendingConventionalScrollRef.current = null;
      return;
    }
    if (conventionalMode !== 'excerpt') return;
    const activeWord = container.querySelector(
      `[data-word-index="${pending.index}"]`,
    ) as HTMLElement | null;
    if (!activeWord) return;
    pendingConventionalScrollRef.current = null;
    const rect = container.getBoundingClientRect();
    const wordRect = activeWord.getBoundingClientRect();
    const targetTop = container.scrollTop + (wordRect.top - rect.top) - rect.height / 2 + wordRect.height / 2;
    scrollLockRef.current = true;
    container.scrollTo({ top: targetTop, behavior: pending.behavior });
    window.setTimeout(() => {
      scrollLockRef.current = false;
    }, 120);
  }, [conventionalMode, conventionalWindow, showConventional, tokens.length]);
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
  const sessionWordCount = wordsRead;
  const totalWordCount = tokens.length;
  const sessionWordPairLabel = totalWordCount
    ? `${sessionWordCount.toLocaleString()} / ${totalWordCount.toLocaleString()}`
    : '0 / 0';
  const awards = useMemo(() => {
    const items: AwardStatus[] = AWARD_DEFINITIONS.map((award) => {
      const value = award.kind === 'time'
        ? sessionElapsedMs
        : award.kind === 'progress'
          ? sessionPercent
          : sessionWordCount;
      return { ...award, value, earned: value >= award.threshold };
    });
    const earnedCount = items.filter((item) => item.earned).length;
    const nextAward = items.find((item) => !item.earned) ?? null;
    return { items, earnedCount, total: items.length, nextAward };
  }, [sessionElapsedMs, sessionPercent, sessionWordCount]);

  const formatAwardRemaining = (award: AwardStatus) => {
    const remaining = Math.max(0, award.threshold - award.value);
    if (award.kind === 'time') {
      return `${formatDuration(remaining)} left`;
    }
    if (award.kind === 'progress') {
      return `${remaining}% to go`;
    }
    return `${remaining.toLocaleString()} words to go`;
  };
  const nextAwardRemaining = awards.nextAward ? formatAwardRemaining(awards.nextAward) : null;

  const dailyProgress = dailyGoalWords > 0
    ? Math.min(100, Math.round((streakStats.dailyWords / dailyGoalWords) * 100))
    : 0;
  const dailyRemainingWords = Math.max(0, dailyGoalWords - streakStats.dailyWords);
  const dailyProgressLabel = `${streakStats.dailyWords.toLocaleString()} / ${dailyGoalWords.toLocaleString()} (${dailyProgress}%)`;
  const streakLabel = `${streakStats.streak} ${streakStats.streak === 1 ? 'day' : 'days'}`;
  const bestStreakLabel = `${streakStats.bestStreak} ${streakStats.bestStreak === 1 ? 'day' : 'days'}`;
  const nextGoalLabel = dailyRemainingWords === 0
    ? 'Goal met today'
    : `${dailyRemainingWords.toLocaleString()} words to ${dailyGoalWords.toLocaleString()}`;

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

  useEffect(() => {
    const todayKey = getLocalDayKey();
    if (streakStats.dayKey === todayKey) return;
    setStreakStats((prev: StreakStats) => ({
      ...prev,
      dayKey: todayKey,
      dailyWords: 0,
    }));
    lastCountedRef.current = { dayKey: todayKey, docKey, words: sessionWordCount };
  }, [docKey, sessionWordCount, streakStats.dayKey]);

  useEffect(() => {
    if (!docKey) return;
    const todayKey = getLocalDayKey();
    if (streakStats.dayKey !== todayKey) return;
    if (lastCountedRef.current.dayKey !== todayKey || lastCountedRef.current.docKey !== docKey) {
      lastCountedRef.current = { dayKey: todayKey, docKey, words: sessionWordCount };
      return;
    }
    const delta = sessionWordCount - lastCountedRef.current.words;
    if (delta > 0) {
      setStreakStats((prev: StreakStats) => ({
        ...prev,
        dailyWords: prev.dailyWords + delta,
      }));
    }
    lastCountedRef.current.words = sessionWordCount;
  }, [docKey, sessionWordCount, streakStats.dayKey]);

  useEffect(() => {
    if (!dailyGoalWords || streakStats.dailyWords < dailyGoalWords) return;
    if (streakStats.lastCompletedDay === streakStats.dayKey) return;
    setStreakStats((prev: StreakStats) => {
      if (prev.lastCompletedDay === prev.dayKey) return prev;
      const continueStreak = prev.lastCompletedDay ? isPreviousDay(prev.lastCompletedDay, prev.dayKey) : false;
      const nextStreak = continueStreak ? prev.streak + 1 : 1;
      return {
        ...prev,
        streak: nextStreak,
        bestStreak: Math.max(prev.bestStreak, nextStreak),
        lastCompletedDay: prev.dayKey,
      };
    });
  }, [dailyGoalWords, streakStats.dailyWords, streakStats.dayKey, streakStats.lastCompletedDay]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const payload = { dailyGoalWords, stats: streakStats };
    window.localStorage.setItem(STREAK_STORAGE_KEY, JSON.stringify(payload));
    // Sync streak to server
    if (syncService.isAuthenticated()) {
      syncService.pushPreferences({ streak: payload });
    }
  }, [dailyGoalWords, streakStats]);

  useEffect(() => {
    if (docKey) {
      hasSessionActivityRef.current = false;
      setShowSessionRecap(false);
    }
  }, [docKey]);

  useEffect(() => {
    if (isPlaying) {
      hasSessionActivityRef.current = true;
      setShowSessionRecap(false);
      return;
    }
    if (!hasSessionActivityRef.current) return;
    if (sessionElapsedMs === 0 && sessionWordCount === 0) return;
    setShowSessionRecap(true);
  }, [isPlaying, sessionElapsedMs, sessionWordCount]);

  useEffect(() => {
    if (earnedAwardIdsRef.current.size === 0) {
      awards.items.filter((item) => item.earned).forEach((item) => {
        earnedAwardIdsRef.current.add(item.id);
      });
      return;
    }
    const newlyEarned = awards.items.filter((item) => item.earned && !earnedAwardIdsRef.current.has(item.id));
    if (!newlyEarned.length) return;
    newlyEarned.forEach((item) => {
      earnedAwardIdsRef.current.add(item.id);
    });
    const nextAward = newlyEarned[0];
    setAwardToast(nextAward);
    setHighlightAwardId(nextAward.id);
    if (awardToastTimerRef.current) {
      window.clearTimeout(awardToastTimerRef.current);
    }
    awardToastTimerRef.current = window.setTimeout(() => {
      setAwardToast(null);
    }, 2800);
    if (awardHighlightTimerRef.current) {
      window.clearTimeout(awardHighlightTimerRef.current);
    }
    awardHighlightTimerRef.current = window.setTimeout(() => {
      setHighlightAwardId(null);
    }, 1600);
  }, [awards.items]);

  useEffect(() => {
    earnedAwardIdsRef.current = new Set();
    setAwardToast(null);
    setHighlightAwardId(null);
  }, [docKey]);

  useEffect(() => {
    return () => {
      if (awardToastTimerRef.current) {
        window.clearTimeout(awardToastTimerRef.current);
      }
      if (awardHighlightTimerRef.current) {
        window.clearTimeout(awardHighlightTimerRef.current);
      }
    };
  }, []);

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
  const pageJumpMax = pdfState?.pageCount ?? 1;
  const pageJumpDisabled = sourceKind !== 'pdf' || !pdfState || isPdfBusy;
  const pageJumpPlaceholder = pdfState ? `1-${pdfState.pageCount}` : 'Load PDF';
  const showPageJump = sourceKind === 'pdf' && !!pdfState;
  const findCountLabel = isFinding
    ? 'Searching'
    : findQuery.trim()
      ? findMatches.length
        ? `${findCursor + 1}/${findMatches.length}`
        : '0'
      : '-';
  const formatStoredDate = useCallback((timestamp: number) => {
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }, []);
  const granularityLabel =
    granularity === 'word'
      ? 'Word'
      : granularity === 'bigram'
        ? 'Bi-gram'
        : granularity === 'trigram'
          ? 'Tri-gram'
          : granularity === 'sentence'
            ? 'Sentence'
            : 'Tweet';

  const applyPdfRange = useCallback(
    (start: number, end: number, options?: { focusPage?: number; pdfOverride?: PdfState }) => {
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
    },
    [pdfState],
  );

  const buildTtsChunk = useCallback((startIndex: number) => {
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
  }, []);

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

  const stopTtsPlayback = useCallback((options?: { clearNotice?: boolean }) => {
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
  }, []);

  const playCue = useCallback((tone: 'start' | 'pause' | 'back' | 'next' | 'bookmark') => {
    if (!audioCuesEnabledRef.current) return;
    if (typeof window === 'undefined') return;
    const AudioContextCtor =
      window.AudioContext ||
      (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) return;
    const now = performance.now();
    if (now - audioCueLastAtRef.current < 40) return;
    audioCueLastAtRef.current = now;

    const context = audioContextRef.current ?? new AudioContextCtor();
    audioContextRef.current = context;
    if (context.state === 'suspended') {
      void context.resume().catch(() => {});
    }

    const cueMap = {
      start: { freq: 520, duration: 0.09, gain: 0.08 },
      pause: { freq: 260, duration: 0.09, gain: 0.07 },
      back: { freq: 340, duration: 0.06, gain: 0.06 },
      next: { freq: 640, duration: 0.06, gain: 0.06 },
      bookmark: { freq: 760, duration: 0.12, gain: 0.07 },
    };
    const cue = cueMap[tone];
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    const startTime = context.currentTime;
    oscillator.type = 'triangle';
    oscillator.frequency.setValueAtTime(cue.freq, startTime);
    gain.gain.setValueAtTime(0.0001, startTime);
    gain.gain.exponentialRampToValueAtTime(cue.gain, startTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + cue.duration);
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start(startTime);
    oscillator.stop(startTime + cue.duration + 0.02);
    oscillator.onended = () => {
      oscillator.disconnect();
      gain.disconnect();
    };
  }, []);

  const startTtsFromIndex = useCallback(async (startIndex: number) => {
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
  }, [buildTtsChunk, stopTtsPlayback, ttsAvailable]);

  const scrollConventionalToIndex = useCallback(
    (targetIndex: number, behavior: ScrollBehavior = 'smooth') => {
      if (!showConventional) return;
      const container = conventionalRef.current;
      if (!container) return;
      if (tokens.length === 0) return;
      if (conventionalMode === 'rendered') {
        const ratio = tokens.length ? clamp(targetIndex / Math.max(1, tokens.length - 1), 0, 1) : 0;
        const maxScroll = container.scrollHeight - container.clientHeight;
        scrollLockRef.current = true;
        container.scrollTo({ top: maxScroll * ratio, behavior });
        window.setTimeout(() => {
          scrollLockRef.current = false;
        }, 120);
        return;
      }
      if (conventionalMode !== 'excerpt') return;
      pendingConventionalScrollRef.current = { index: targetIndex, behavior };
      ensureConventionalWindowForIndex(targetIndex);
    },
    [conventionalMode, ensureConventionalWindowForIndex, showConventional, tokens.length],
  );

  useEffect(() => {
    if (!autoFollowConventional || !showConventional || !isPlaying || conventionalMode !== 'excerpt') return;
    const container = conventionalRef.current;
    if (!container) return;
    const activeWord = container.querySelector(`[data-word-index="${wordIndex}"]`) as HTMLElement | null;
    if (!activeWord) {
      scrollConventionalToIndex(wordIndex, 'auto');
      return;
    }
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
  }, [autoFollowConventional, conventionalMode, scrollConventionalToIndex, showConventional, isPlaying, wordIndex]);

  const handleSeek = useCallback(
    (nextIndex: number, options?: { pause?: boolean; focusConventional?: boolean; scrollBehavior?: ScrollBehavior }) => {
      const total = Math.max(0, tokensRef.current.length - 1);
      const clampedIndex = clamp(nextIndex, 0, total);
      if (options?.pause) {
        setIsPlaying(false);
        stopTtsPlayback();
      }
      setWordIndex(clampedIndex);
      if (options?.focusConventional) {
        scrollConventionalToIndex(clampedIndex, options.scrollBehavior ?? 'smooth');
      }
      if (!options?.pause && ttsEnabledRef.current && ttsPlayingRef.current) {
        startTtsFromIndex(clampedIndex);
      }
    },
    [scrollConventionalToIndex, startTtsFromIndex, stopTtsPlayback],
  );

  const requestFullscreen = useCallback(async () => {
    if (typeof document === 'undefined') return;
    if (!document.fullscreenEnabled) return;
    if (document.fullscreenElement) return;
    try {
      await document.documentElement.requestFullscreen();
    } catch (error) {
      console.warn('Fullscreen request failed.', error);
    }
  }, []);

  const exitFullscreen = useCallback(async () => {
    if (typeof document === 'undefined') return;
    if (!document.fullscreenElement) return;
    try {
      await document.exitFullscreen();
    } catch (error) {
      console.warn('Fullscreen exit failed.', error);
    }
  }, []);

  const handleViewModeChange = useCallback(
    (mode: ViewMode, options?: { requestFullscreen?: boolean }) => {
      setViewMode(mode);
      setShowLibrary(false);
      if (mode === 'focus' || mode === 'focus-text') {
        if (options?.requestFullscreen) {
          void requestFullscreen();
        }
        return;
      }
      void exitFullscreen();
    },
    [exitFullscreen, requestFullscreen],
  );

  const adjustWpm = useCallback((delta: number) => {
    setWpm((prev) => clamp(prev + delta, 120, 900));
  }, []);

  const handleTogglePlay = useCallback(() => {
    if (!tokensRef.current.length) return;
    const nextState = !isPlayingRef.current;
    setIsPlaying(nextState);
    playCue(nextState ? 'start' : 'pause');
  }, [playCue]);

  const handleTransportStep = useCallback(
    (direction: 'back' | 'next') => {
      if (!tokensRef.current.length) return;
      const nextIndex = getSteppedWordIndex(
        wordIndexRef.current,
        direction,
        chunkSizeRef.current,
        segmentStartIndicesRef.current,
        segmentsRef.current,
      );
      handleSeekRef.current(nextIndex);
      playCue(direction);
    },
    [playCue],
  );

  const handleSentenceStep = useCallback(
    (direction: 'back' | 'next') => {
      if (!sentenceSegments.length) return;
      const currentIndex = getSegmentIndexForWordIndex(wordIndexRef.current, sentenceStartIndices);
      const nextIndex = clamp(
        currentIndex + (direction === 'back' ? -1 : 1),
        0,
        Math.max(0, sentenceSegments.length - 1),
      );
      const target = sentenceSegments[nextIndex]?.startIndex ?? wordIndexRef.current;
      handleSeekRef.current(target, { focusConventional: true });
      playCue(direction);
    },
    [playCue, sentenceSegments, sentenceStartIndices],
  );

  const jumpToFindMatch = useCallback(
    (matchIndex: number) => {
      const target = findMatches[matchIndex];
      if (typeof target !== 'number') return;
      setFindCursor(matchIndex);
      handleSeek(target, { pause: true, focusConventional: true });
    },
    [findMatches, handleSeek],
  );

  const handleFindStart = useCallback(() => {
    if (!findMatches.length) return;
    const currentIndex = wordIndexRef.current;
    const nextIndex = findMatches.findIndex((index) => index >= currentIndex);
    jumpToFindMatch(nextIndex === -1 ? 0 : nextIndex);
  }, [findMatches, jumpToFindMatch]);

  const handleFindStep = useCallback(
    (direction: 'prev' | 'next') => {
      if (!findMatches.length) return;
      const delta = direction === 'next' ? 1 : -1;
      const nextIndex = (findCursor + delta + findMatches.length) % findMatches.length;
      jumpToFindMatch(nextIndex);
    },
    [findCursor, findMatches.length, jumpToFindMatch],
  );

  useEffect(() => {
    handleSeekRef.current = handleSeek;
  }, [handleSeek]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isEditableTarget(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === '?' || (key === '/' && event.shiftKey)) {
        event.preventDefault();
        setShowHotkeys((prev) => !prev);
        return;
      }
      if (event.code === 'Space' || key === ' ') {
        event.preventDefault();
        handleTogglePlay();
        return;
      }
      if (event.code === 'ArrowUp') {
        event.preventDefault();
        adjustWpm(20);
        return;
      }
      if (event.code === 'ArrowDown') {
        event.preventDefault();
        adjustWpm(-20);
        return;
      }
      if (event.shiftKey && event.code === 'ArrowLeft') {
        event.preventDefault();
        handleSentenceStep('back');
        return;
      }
      if (event.shiftKey && event.code === 'ArrowRight') {
        event.preventDefault();
        handleSentenceStep('next');
        return;
      }
      if (key === 'v') {
        event.preventDefault();
        if (showReaderDisplay && showConventional) {
          // Both → text-only
          setShowReaderDisplay(false);
          setAutoFollowConventional(true);
          if (viewMode === 'focus') handleViewModeChange('focus-text');
        } else if (!showReaderDisplay && showConventional) {
          // Text-only → word-only
          setShowReaderDisplay(true);
          setShowConventional(false);
          if (viewMode === 'focus-text') handleViewModeChange('focus');
        } else {
          // Word-only (or neither) → both
          setShowReaderDisplay(true);
          setShowConventional(true);
          setAutoFollowConventional(true);
          if (viewMode === 'focus') handleViewModeChange('focus-text');
        }
        return;
      }
      if (key === 'f') {
        event.preventDefault();
        if (document.fullscreenElement || viewMode === 'focus' || viewMode === 'focus-text') {
          handleViewModeChange('deck');
        } else {
          handleViewModeChange('focus', { requestFullscreen: true });
        }
        return;
      }
      if (event.code === 'ArrowLeft') {
        event.preventDefault();
        sentenceMode ? handleSentenceStep('back') : handleTransportStep('back');
        return;
      }
      if (event.code === 'ArrowRight') {
        event.preventDefault();
        sentenceMode ? handleSentenceStep('next') : handleTransportStep('next');
        return;
      }
      if (key === 'b') {
        event.preventDefault();
        handleBookmarkRef.current();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [adjustWpm, handleSentenceStep, handleTogglePlay, handleTransportStep, handleViewModeChange, sentenceMode, showConventional, showReaderDisplay, viewMode]);

  useEffect(() => {
    const pending = pendingBookmarkRef.current;
    if (!pending) return;
    if (!tokens.length) return;
    if (pending.docHash && hashText(sourceText) !== pending.docHash) {
      if (!pendingBookmarkMismatchRef.current) {
        setBookmarkNotice({
          kind: 'info',
          message: 'Bookmark link targets a different document. Load the matching text to jump.',
        });
        pendingBookmarkMismatchRef.current = true;
      }
      return;
    }
    if (pendingBookmarkMismatchRef.current) {
      setBookmarkNotice(null);
    }
    pendingBookmarkMismatchRef.current = false;
    if (sourceKind === 'pdf' && pdfState && pending.pageNumber != null) {
      if (pending.pageNumber < pageRange.start || pending.pageNumber > pageRange.end) {
        applyPdfRange(
          Math.min(pageRange.start, pending.pageNumber),
          Math.max(pageRange.end, pending.pageNumber),
        );
        return;
      }
      if (pending.pageOffset != null) {
        const pageOffset = pageOffsets[pending.pageNumber - pageRange.start];
        if (typeof pageOffset === 'number') {
          handleSeek(pageOffset + pending.pageOffset, { pause: true, focusConventional: true });
          pendingBookmarkRef.current = null;
          return;
        }
      }
    }
    const clampedIndex = clamp(pending.index, 0, Math.max(0, tokens.length - 1));
    handleSeek(clampedIndex, { pause: true, focusConventional: true });
    pendingBookmarkRef.current = null;
  }, [
    applyPdfRange,
    handleSeek,
    pageOffsets,
    pageRange.end,
    pageRange.start,
    pdfState,
    sourceKind,
    sourceText,
    tokens.length,
  ]);

  const handleConventionalScroll = () => {
    if (conventionalMode !== 'excerpt') return;
    const container = conventionalRef.current;
    if (container) {
      const now = performance.now();
      const lastTop = lastConventionalScrollTopRef.current;
      const lastAt = lastConventionalScrollAtRef.current;
      const signedDelta = container.scrollTop - lastTop;
      const delta = Math.abs(signedDelta);
      const dt = now - lastAt;
      if (signedDelta !== 0) {
        conventionalScrollDirectionRef.current = signedDelta > 0 ? 1 : -1;
      }
      lastConventionalScrollTopRef.current = container.scrollTop;
      lastConventionalScrollAtRef.current = now;
      const velocity = dt > 0 ? delta / dt : 0;
      let nextBoost = 0;
      if (velocity > 1.6) {
        nextBoost = 2;
      } else if (velocity > 0.8) {
        nextBoost = 1;
      }
      if (nextBoost !== conventionalBufferBoostRef.current) {
        setConventionalBufferBoost(nextBoost);
      }
      if (nextBoost > 0) {
        if (conventionalBoostTimeoutRef.current) {
          window.clearTimeout(conventionalBoostTimeoutRef.current);
        }
        conventionalBoostTimeoutRef.current = window.setTimeout(() => {
          setConventionalBufferBoost((prev) => (prev === 0 ? prev : 0));
        }, 220);
      }
    }
    if (container && conventionalChunkCount > 0 && !scrollRafRef.current) {
      scrollRafRef.current = window.requestAnimationFrame(() => {
        scrollRafRef.current = null;
        const lead = Math.min(160, container.clientHeight * 0.4);
        const targetScroll = container.scrollTop + lead * conventionalScrollDirectionRef.current;
        const chunkIndex = findChunkIndexForScroll(targetScroll);
        updateConventionalWindowForChunk(chunkIndex);
      });
    }
    if (!conventionalSeekEnabledRef.current || scrollLockRef.current) return;
    if (scrollEndTimeoutRef.current) {
      window.clearTimeout(scrollEndTimeoutRef.current);
    }
    scrollEndTimeoutRef.current = window.setTimeout(() => {
      scrollEndTimeoutRef.current = null;
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
  }, [isPlaying, startTtsFromIndex, stopTtsPlayback, tokens.length, ttsEnabled, wordIndex]);

  useEffect(() => {
    if (ttsEnabled && isPlaying && tokens.length > 0) {
      startTtsFromIndex(wordIndex);
    }
  }, [isPlaying, startTtsFromIndex, tokens.length, ttsChunkWords, ttsEnabled, ttsLanguage, ttsSpeed, ttsVoice, wordIndex]);

  const ensurePdfWorker = useCallback(() => {
    if (pdfWorkerRef.current) return pdfWorkerRef.current;
    try {
      const worker = new Worker(new URL('./pdfWorker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (event) => {
        const message = event.data as {
          id?: number;
          type?: 'progress' | 'result';
          current?: number;
          total?: number;
          ok?: boolean;
          payload?: {
            pageTexts: string[];
            pageTokenCounts: number[];
            outline: PdfOutlineItem[];
            pageCount: number;
          };
          error?: string;
        };
        if (typeof message?.id !== 'number') return;
        const callbacks = pdfWorkerCallbacksRef.current.get(message.id);
        if (!callbacks) return;
        if (message.type === 'progress') {
          callbacks.onProgress?.(message.current ?? 0, message.total ?? 0);
          return;
        }
        if (message.type === 'result') {
          pdfWorkerCallbacksRef.current.delete(message.id);
          if (message.ok && message.payload) {
            callbacks.resolve(message.payload);
          } else {
            callbacks.reject(new Error(message.error || 'Failed to read PDF.'));
          }
        }
      };
      worker.onerror = (error) => {
        console.warn('PDF worker error, falling back to main thread.', error);
        pdfWorkerCallbacksRef.current.forEach((callbacks) => {
          callbacks.reject(new Error('PDF worker failed.'));
        });
        pdfWorkerCallbacksRef.current.clear();
        worker.terminate();
        pdfWorkerRef.current = null;
      };
      pdfWorkerRef.current = worker;
      return worker;
    } catch (error) {
      console.warn('PDF worker unavailable, using main thread.', error);
      return null;
    }
  }, []);

  useEffect(() => {
    const callbacks = pdfWorkerCallbacksRef.current;
    return () => {
      if (pdfWorkerRef.current) {
        pdfWorkerRef.current.terminate();
        pdfWorkerRef.current = null;
      }
      callbacks.forEach((entry) => {
        entry.reject(new Error('PDF worker terminated.'));
      });
      callbacks.clear();
    };
  }, []);

  const extractPdfData = useCallback(
    async (data: Uint8Array, onProgress?: (current: number, total: number) => void) => {
      if (typeof Worker === 'undefined') {
        return extractPdfDataMainThread(data, onProgress);
      }
      const worker = ensurePdfWorker();
      if (!worker) {
        return extractPdfDataMainThread(data, onProgress);
      }
      const id = pdfWorkerIdRef.current + 1;
      pdfWorkerIdRef.current = id;
      const task = new Promise<{
        pageTexts: string[];
        pageTokenCounts: number[];
        outline: PdfOutlineItem[];
        pageCount: number;
      }>((resolve, reject) => {
        pdfWorkerCallbacksRef.current.set(id, { resolve, reject, onProgress });
      });
      const bufferCopy = data.slice().buffer;
      worker.postMessage({ id, type: 'extract', buffer: bufferCopy }, [bufferCopy]);
      try {
        return await task;
      } catch (error) {
        console.warn('PDF worker failed, falling back to main thread.', error);
        return extractPdfDataMainThread(data, onProgress);
      }
    },
    [ensurePdfWorker],
  );

  const loadTextFile = async (
    file: File,
    setNotice: React.Dispatch<React.SetStateAction<Notice | null>>,
  ): Promise<{ text: string; wordCount: number }> => {
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
          return { text: '', wordCount: 0 };
        }
        const wordCount = pdfData.pageTokenCounts.reduce((sum, count) => sum + count, 0);
        setNotice({ kind: 'success', message: `PDF loaded (${wordCount.toLocaleString()} words)` });
        return { text, wordCount };
      } catch (error) {
        console.error('Failed to read PDF:', error);
        setNotice({ kind: 'error', message: 'Failed to read PDF.' });
        return { text: '', wordCount: 0 };
      }
    }

    const text = await file.text();
    const wordCount = tokenize(text).length;
    setNotice({ kind: 'success', message: `Text loaded (${wordCount.toLocaleString()} words)` });
    return { text, wordCount };
  };

  const loadEpubFile = async (
    file: File,
    setNotice: React.Dispatch<React.SetStateAction<Notice | null>>,
  ): Promise<{ text: string; wordCount: number }> => {
    setNotice({ kind: 'info', message: 'Reading EPUB...' });
    try {
      const zip = await JSZip.loadAsync(await file.arrayBuffer());
      const containerEntry = zip.file('META-INF/container.xml');
      if (!containerEntry) {
        setNotice({ kind: 'error', message: 'Missing EPUB container.' });
        return { text: '', wordCount: 0 };
      }
      const containerText = await containerEntry.async('string');
      const parser = new DOMParser();
      const containerDoc = parser.parseFromString(containerText, 'application/xml');
      const rootfile = containerDoc.getElementsByTagName('rootfile')[0];
      const opfPath = rootfile?.getAttribute('full-path');
      if (!opfPath) {
        setNotice({ kind: 'error', message: 'EPUB manifest not found.' });
        return { text: '', wordCount: 0 };
      }
      const opfEntry = zip.file(opfPath);
      if (!opfEntry) {
        setNotice({ kind: 'error', message: 'EPUB package document missing.' });
        return { text: '', wordCount: 0 };
      }
      const opfText = await opfEntry.async('string');
      const opfDoc = parser.parseFromString(opfText, 'application/xml');
      const manifestEl = opfDoc.getElementsByTagName('manifest')[0];
      const spineEl = opfDoc.getElementsByTagName('spine')[0];
      if (!manifestEl || !spineEl) {
        setNotice({ kind: 'error', message: 'Invalid EPUB manifest.' });
        return { text: '', wordCount: 0 };
      }
      const manifestItems = Array.from(manifestEl.getElementsByTagName('item'))
        .map((item) => ({
          id: item.getAttribute('id') ?? '',
          href: item.getAttribute('href') ?? '',
          mediaType: item.getAttribute('media-type') ?? '',
        }))
        .filter((item) => item.id && item.href);
      const manifestMap = new Map(manifestItems.map((item) => [item.id, item]));
      const basePath = opfPath.includes('/') ? opfPath.slice(0, opfPath.lastIndexOf('/') + 1) : '';
      const spineRefs = Array.from(spineEl.getElementsByTagName('itemref'))
        .map((item) => item.getAttribute('idref'))
        .filter((value): value is string => Boolean(value));
      const readableTypes = new Set(['application/xhtml+xml', 'text/html', 'application/x-dtbook+xml']);
      const chapters: string[] = [];
      for (const idref of spineRefs) {
        const item = manifestMap.get(idref);
        if (!item || (item.mediaType && !readableTypes.has(item.mediaType))) continue;
        const resolvedPath = resolveEpubPath(basePath, item.href);
        if (!resolvedPath) continue;
        const entry = zip.file(resolvedPath);
        if (!entry) continue;
        const html = await entry.async('string');
        const chapterText = extractEpubHtmlText(html);
        if (chapterText) chapters.push(chapterText);
      }
      if (chapters.length === 0) {
        const fallbackItems = manifestItems.filter((item) => readableTypes.has(item.mediaType));
        for (const item of fallbackItems) {
          const resolvedPath = resolveEpubPath(basePath, item.href);
          if (!resolvedPath) continue;
          const entry = zip.file(resolvedPath);
          if (!entry) continue;
          const html = await entry.async('string');
          const chapterText = extractEpubHtmlText(html);
          if (chapterText) chapters.push(chapterText);
        }
      }
      const combined = chapters.join('\n\n').trim();
      if (!combined) {
        setNotice({ kind: 'error', message: 'No readable text found. Is the EPUB DRM-free?' });
        return { text: '', wordCount: 0 };
      }
      const wordCount = tokenize(combined).length;
      setNotice({ kind: 'success', message: `EPUB loaded (${wordCount.toLocaleString()} words)` });
      return { text: combined, wordCount };
    } catch (error) {
      console.error('Failed to read EPUB:', error);
      setNotice({ kind: 'error', message: 'Failed to read EPUB.' });
      return { text: '', wordCount: 0 };
    }
  };

  const loadHtmlFile = async (
    file: File,
    setNotice: React.Dispatch<React.SetStateAction<Notice | null>>,
  ): Promise<{ text: string; wordCount: number }> => {
    setNotice({ kind: 'info', message: 'Reading HTML...' });
    try {
      const rawHtml = await file.text();
      const text = extractEpubHtmlText(rawHtml);
      if (!text.trim()) {
        setNotice({ kind: 'error', message: 'No readable text found in HTML.' });
        return { text: '', wordCount: 0 };
      }
      const wordCount = tokenize(text).length;
      setNotice({ kind: 'success', message: `HTML loaded (${wordCount.toLocaleString()} words)` });
      return { text, wordCount };
    } catch (error) {
      console.error('Failed to read HTML:', error);
      setNotice({ kind: 'error', message: 'Failed to read HTML file.' });
      return { text: '', wordCount: 0 };
    }
  };

  const loadDocxFile = async (
    file: File,
    setNotice: React.Dispatch<React.SetStateAction<Notice | null>>,
  ): Promise<{ text: string; wordCount: number }> => {
    setNotice({ kind: 'info', message: 'Reading Word document...' });
    try {
      const mammoth = await import('mammoth');
      const arrayBuffer = await file.arrayBuffer();
      const result = await mammoth.default.extractRawText({ arrayBuffer });
      const text = result.value.replace(/\n{3,}/g, '\n\n').trim();
      if (!text) {
        setNotice({ kind: 'error', message: 'No readable text found in document.' });
        return { text: '', wordCount: 0 };
      }
      const wordCount = tokenize(text).length;
      setNotice({ kind: 'success', message: `Word document loaded (${wordCount.toLocaleString()} words)` });
      return { text, wordCount };
    } catch (error) {
      console.error('Failed to read DOCX:', error);
      setNotice({ kind: 'error', message: 'Failed to read Word document.' });
      return { text: '', wordCount: 0 };
    }
  };

  const loadPrimaryPdf = async (
    file: File,
  ): Promise<{ pdfState: PdfState; wordCount: number; text: string; pageRange: PageRange } | null> => {
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
        const range = normalizeRange(1, nextPdfState.pageCount, nextPdfState.pageCount);
        setPrimaryNotice({ kind: 'error', message: 'No text found. Is this a scanned PDF?' });
        setSourceKind('pdf');
        setPdfState(nextPdfState);
        setTitle(file.name.replace(/\.[^/.]+$/, ''));
        applyPdfRange(range.start, range.end, { pdfOverride: nextPdfState });
        return {
          pdfState: nextPdfState,
          wordCount: 0,
          text: combinedText,
          pageRange: range,
        };
      }
      const nextPdfState: PdfState = {
        fileName: file.name,
        pageCount: pdfData.pageCount,
        pageTexts: pdfData.pageTexts,
        pageTokenCounts: pdfData.pageTokenCounts,
        outline: pdfData.outline,
      };
      const wordCount = pdfData.pageTokenCounts.reduce((sum, count) => sum + count, 0);
      const range = normalizeRange(1, nextPdfState.pageCount, nextPdfState.pageCount);
      setPdfState(nextPdfState);
      setSourceKind('pdf');
      setTitle(file.name.replace(/\.[^/.]+$/, ''));
      applyPdfRange(range.start, range.end, { pdfOverride: nextPdfState });
      setPrimaryNotice({ kind: 'success', message: `PDF loaded (${wordCount.toLocaleString()} words)` });
      return { pdfState: nextPdfState, wordCount, text: combinedText, pageRange: range };
    } catch (error) {
      console.error('Failed to read PDF:', error);
      setPrimaryNotice({ kind: 'error', message: 'Failed to read PDF.' });
      return null;
    } finally {
      setIsPdfBusy(false);
    }
  };

  const handlePageRangeApply = () => {
    if (!pdfState) return;
    const range = normalizeRange(pageRangeDraft.start, pageRangeDraft.end, pdfState.pageCount);
    applyPdfRange(range.start, range.end);
  };

  const jumpToPage = (pageNumber: number, options?: { focusConventional?: boolean }) => {
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
    handleSeek(pageOffsets[localIndex] ?? 0, { focusConventional: options?.focusConventional });
  };

  const handleJumpToPage = () => {
    if (!pdfState) return;
    if (!jumpPage.trim()) return;
    const raw = Number(jumpPage);
    if (!Number.isFinite(raw)) return;
    const safePage = clamp(Math.round(raw), 1, pdfState.pageCount);
    setJumpPage(String(safePage));
    jumpToPage(safePage, { focusConventional: true });
  };

  const handleJumpToPercent = () => {
    if (!tokens.length) return;
    if (!jumpPercent.trim()) return;
    const raw = Number(jumpPercent);
    if (!Number.isFinite(raw)) return;
    const safePercent = clamp(raw, 0, 100);
    const targetIndex = Math.round((safePercent / 100) * (tokens.length - 1));
    setJumpPercent(String(Math.round(safePercent)));
    handleSeek(targetIndex, { focusConventional: true });
  };

  const runOcrForRange = async () => {
    if (!pdfState || !pdfDataRef.current || isPdfBusy) return;
    const range = normalizeRange(pageRange.start, pageRange.end, pdfState.pageCount);
    setIsPdfBusy(true);
    setPrimaryNotice({ kind: 'info', message: `Running OCR for pages ${range.start}-${range.end}` });

    let worker: TesseractWorker | null = null;
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

  const persistStoredDocument = useCallback((doc: StoredDocument) => {
    void writeStoredDocumentToDb(doc);
    setSavedDocuments((prev) => {
      return [doc, ...prev.filter((item) => item.id !== doc.id)];
    });
    // Sync: push document to server (skip docs pulled from server, identified by lastModified === 0)
    if (syncService.isAuthenticated() && doc.primaryFileMeta?.lastModified !== 0) {
      const blob = new Blob([doc.sourceText], { type: 'text/plain' });
      const fileName = doc.primaryFileMeta?.name || `${doc.title || 'untitled'}.txt`;
      const file = new File([blob], fileName, { type: 'text/plain' });
      void syncService.pushDocument(file, doc.title || 'Untitled', doc.sourceKind, doc.wordCount);
    }
  }, []);

  const handleLoadStoredDocument = useCallback(
    (doc: StoredDocument) => {
      const titleValue =
        doc.title ||
        doc.primaryFileMeta?.name?.replace(/\.[^/.]+$/, '') ||
        'Untitled Session';
      if (doc.sourceKind === 'pdf' && doc.pdfState) {
        pdfDataRef.current = null;
        setPdfState(doc.pdfState);
        setSourceKind('pdf');
        setPrimaryFileMeta(doc.primaryFileMeta ?? null);
        const range = doc.pageRange ?? { start: 1, end: doc.pdfState.pageCount };
        applyPdfRange(range.start, range.end, { pdfOverride: doc.pdfState });
        setTitle(titleValue);
      } else {
        setSourceKind(doc.sourceKind);
        setPdfState(null);
        setPageOffsets([]);
        setPageRange({ start: 1, end: 1 });
        setPageRangeDraft({ start: 1, end: 1 });
        startTransition(() => {
          setSourceText(doc.sourceText);
          setTitle(titleValue);
        });
        setWordIndex(0);
        setIsPlaying(false);
        setPrimaryFileMeta(doc.primaryFileMeta ?? null);
      }
      setPrimaryNotice({ kind: 'info', message: 'Loaded saved file.' });
      persistStoredDocument({ ...doc, title: titleValue, updatedAt: Date.now() });
    },
    [applyPdfRange, persistStoredDocument],
  );

  const handleClearStoredDocuments = useCallback(() => {
    void clearStoredDocumentsFromDb();
    setSavedDocuments([]);
    // Sync: delete all documents on server
    if (syncService.isAuthenticated()) {
      void syncService.pushDeleteAll();
    }
  }, []);

  const handlePrimaryFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const nextMeta = {
      name: file.name,
      size: file.size,
      lastModified: file.lastModified,
    };
    setPrimaryFileMeta(nextMeta);
    if (isPdfFile(file)) {
      const result = await loadPrimaryPdf(file);
      if (result) {
        const nextDoc: StoredDocument = {
          id: buildDocumentKey('pdf', nextMeta, result.text),
          sourceText: result.text,
          title: file.name.replace(/\.[^/.]+$/, ''),
          sourceKind: 'pdf',
          primaryFileMeta: nextMeta,
          wordCount: result.wordCount,
          updatedAt: Date.now(),
          pageRange: result.pageRange,
          pdfState: result.pdfState,
        };
        persistStoredDocument(nextDoc);
      }
    } else {
      const isEpub = isEpubFile(file);
      const isHtml = isHtmlFile(file);
      const isDocx = isDocxFile(file);
      const { text, wordCount } = isEpub
        ? await loadEpubFile(file, setPrimaryNotice)
        : isHtml
          ? await loadHtmlFile(file, setPrimaryNotice)
          : isDocx
            ? await loadDocxFile(file, setPrimaryNotice)
            : await loadTextFile(file, setPrimaryNotice);
      if (text) {
        setSourceKind(isEpub ? 'epub' : 'text');
        setPdfState(null);
        setPageOffsets([]);
        setPageRange({ start: 1, end: 1 });
        setPageRangeDraft({ start: 1, end: 1 });
        startTransition(() => {
          setSourceText(text);
          setTitle(file.name.replace(/\.[^/.]+$/, ''));
        });
        setWordIndex(0);
        setIsPlaying(false);
        const nextDoc: StoredDocument = {
          id: buildDocumentKey(isEpub ? 'epub' : 'text', nextMeta, text),
          sourceText: text,
          title: file.name.replace(/\.[^/.]+$/, ''),
          sourceKind: isEpub ? 'epub' : 'text',
          primaryFileMeta: nextMeta,
          wordCount,
          updatedAt: Date.now(),
        };
        persistStoredDocument(nextDoc);
      }
    }
    event.target.value = '';
  };

  const handleBookmark = useCallback(() => {
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
    playCue('bookmark');
  }, [activePageIndex, activePageNumber, pageOffsets, playCue, sourceKind, tokens, wordIndex]);

  useEffect(() => {
    handleBookmarkRef.current = handleBookmark;
  }, [handleBookmark]);

  const bookmarkUrlFormat: 'encoded' | 'readable' = 'readable';

  const buildBookmarkPayload = useCallback(
    (bookmark: { index: number; pageNumber?: number; pageOffset?: number }) => {
      const payload: BookmarkPayload = {
        v: BOOKMARK_VERSION,
        index: bookmark.index,
        sourceKind,
        title: title || undefined,
        docHash: sourceText ? hashText(sourceText) : undefined,
      };
      if (bookmark.pageNumber != null && bookmark.pageOffset != null) {
        payload.pageNumber = bookmark.pageNumber;
        payload.pageOffset = bookmark.pageOffset;
      }
      return payload;
    },
    [sourceKind, sourceText, title],
  );

  const setBookmarkUrl = useCallback((payload: BookmarkPayload) => {
    if (typeof window === 'undefined') return '';
    const url = buildBookmarkUrl(window.location.href, payload, bookmarkUrlFormat);
    window.history.replaceState(null, '', url);
    return url;
  }, [bookmarkUrlFormat]);

  const handleShareBookmark = useCallback(
    async (bookmark: Bookmark) => {
      const payload = buildBookmarkPayload(bookmark);
      const url = setBookmarkUrl(payload);
      if (!url) return;
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(url);
          setBookmarkNotice({ kind: 'success', message: 'Bookmark link copied.' });
          return;
        } catch {
          // Ignore clipboard failures.
        }
      }
      setBookmarkNotice({ kind: 'info', message: 'Bookmark link set in the URL bar.' });
    },
    [buildBookmarkPayload, setBookmarkUrl],
  );

  const handleShareFurthest = useCallback(async () => {
    if (!furthestRead) return;
    const payload = buildBookmarkPayload(furthestRead);
    const url = setBookmarkUrl(payload);
    if (!url) return;
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(url);
        setBookmarkNotice({ kind: 'success', message: 'Furthest read link copied.' });
        return;
      } catch {
        // Ignore clipboard failures.
      }
    }
    setBookmarkNotice({ kind: 'info', message: 'Furthest read link set in the URL bar.' });
  }, [buildBookmarkPayload, furthestRead, setBookmarkUrl]);

  const handleFurthestJump = useCallback(
    (furthest: ResolvedFurthest) => {
      const resolvedIndex = furthest.resolvedIndex ?? null;
      const payload = buildBookmarkPayload(furthest);
      setBookmarkUrl(payload);
      if (resolvedIndex === null || furthest.outOfRange) return;
      handleSeek(resolvedIndex, { focusConventional: true });
    },
    [buildBookmarkPayload, handleSeek, setBookmarkUrl],
  );

  const handleBookmarkJump = useCallback(
    (bookmark: ResolvedBookmark) => {
      const resolvedIndex = bookmark.resolvedIndex ?? null;
      const payload = buildBookmarkPayload(bookmark);
      setBookmarkUrl(payload);
      if (resolvedIndex === null || bookmark.outOfRange) return;
      handleSeek(resolvedIndex, { focusConventional: true });
    },
    [buildBookmarkPayload, handleSeek, setBookmarkUrl],
  );

  const handleDeleteBookmark = useCallback((id: string) => {
    setBookmarks((prev) => prev.filter((bookmark) => bookmark.id !== id));
  }, []);
  const handleClearBookmarks = useCallback(() => {
    setBookmarks([]);
  }, []);

  const bookmarkRows = useMemo(() => {
    if (!resolvedFurthest && resolvedBookmarks.length === 0) return null;
    return (
      <div className="bookmark-list">
        {resolvedFurthest && (() => {
          const pageTag = resolvedFurthest.pageNumber ? `p. ${resolvedFurthest.pageNumber}` : null;
          const progress = resolvedFurthest.resolvedIndex !== null && tokens.length
            ? Math.round((resolvedFurthest.resolvedIndex / tokens.length) * 100)
            : null;
          const labelParts = ['Furthest read'];
          if (pageTag) labelParts.push(pageTag);
          labelParts.push(progress !== null ? `${progress}%` : 'out of range');
          return (
            <div key="furthest-read" className="bookmark-row auto">
              <button
                type="button"
                className="bookmark-jump"
                onClick={() => handleFurthestJump(resolvedFurthest)}
                disabled={resolvedFurthest.outOfRange || resolvedFurthest.resolvedIndex === null}
              >
                {labelParts.join(' · ')}
              </button>
              <button
                type="button"
                className="bookmark-share ghost"
                onClick={() => void handleShareFurthest()}
              >
                Share
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
          return (
            <div key={bookmark.id} className="bookmark-row">
              <button
                type="button"
                className="bookmark-jump"
                onClick={() => handleBookmarkJump(bookmark)}
                disabled={bookmark.outOfRange || bookmark.resolvedIndex === null}
              >
                {labelParts.join(' · ')}
              </button>
              <button
                type="button"
                className="bookmark-share ghost"
                onClick={() => void handleShareBookmark(bookmark)}
              >
                Share
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
    );
  }, [
    handleBookmarkJump,
    handleDeleteBookmark,
    handleFurthestJump,
    handleShareBookmark,
    handleShareFurthest,
    resolvedBookmarks,
    resolvedFurthest,
    tokens.length,
  ]);

  const handleCompanionSend = () => {
    if (!companionInput.trim()) return;
    setCompanionLog((prev) => [`You: ${companionInput.trim()}`, ...prev]);
    setCompanionInput('');
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
    <div className={`app-shell view-${viewMode}${isFullscreen ? ' is-fullscreen' : ''}`}>
      {awardToast && (
        <div className="award-toast" role="status" aria-live="polite">
          <span className="award-toast-label">Milestone unlocked</span>
          <strong className="award-toast-title">{awardToast.label}</strong>
          <span className="award-toast-detail">{awardToast.detail}</span>
        </div>
      )}
      <header className="hero">
        <div className="hero-brand">
          <BrandSigil className="brand-sigil" decorative />
          <h1>Pacer</h1>
        </div>
        <div className="hero-actions">
          <div className="view-toggle" role="group" aria-label="View options">
            <button
              type="button"
              className={viewMode === 'deck' ? 'active' : ''}
              onClick={() => handleViewModeChange('deck')}
            >
              Deck
            </button>
            <button
              type="button"
              className={viewMode === 'mobile' ? 'active' : ''}
              onClick={() => handleViewModeChange('mobile')}
            >
              Stack
            </button>
            <button
              type="button"
              className={viewMode === 'focus' ? 'active' : ''}
              onClick={() => handleViewModeChange('focus', { requestFullscreen: true })}
            >
              Focus
            </button>
            <button
              type="button"
              className={viewMode === 'focus-text' ? 'active' : ''}
              onClick={() => handleViewModeChange('focus-text', { requestFullscreen: true })}
            >
              Focus+
            </button>
          </div>
          <div className="session-actions">
            <SyncStatusPill />
            <button
              type="button"
              className={`library-btn${showLibrary ? ' active' : ''}`}
              onClick={() => {
                const isFocusMode = viewMode === 'focus' || viewMode === 'focus-text';
                if (isFocusMode) {
                  setShowLibrary((prev) => !prev);
                } else {
                  sourcePanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
              }}
            >
              Library
            </button>
            <button
              type="button"
              className="reset-btn"
              onClick={() => {
                handleSeek(0, { pause: true });
              }}
            >
              Reset
            </button>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className={`panel source-panel${showLibrary ? ' show-library' : ''}`} ref={sourcePanelRef}>
          <div className="panel-header">
            <h2>Source Rack</h2>
            <p>Load a text, then cue up the deck.</p>
          </div>

          <label className="field">
            <span>Session title</span>
            <input
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Untitled Session"
            />
          </label>

          <div className="load-panel">
            <label className="field file">
              <span>Import .txt/.md/.pdf/.epub/.html/.docx</span>
              <input
                type="file"
                accept=".txt,.md,.text,.pdf,application/pdf,.epub,application/epub+zip,.html,.htm,text/html,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                onChange={handlePrimaryFile}
              />
            </label>
            <p className="hint">Drop a text, EPUB, PDF, HTML, or Word file to load the deck.</p>
          </div>
          {primaryNotice && (
            <div className="notice-row">
              <p className={`notice ${primaryNotice.kind}`}>Source: {primaryNotice.message}</p>
            </div>
          )}
          {workerNotice && (
            <p className={`notice ${workerNotice.kind}`}>Indexer: {workerNotice.message}</p>
          )}
          {isIndexingPrimary && <p className="hint">Indexing text...</p>}

          {savedDocuments.length > 0 && (
            <div className="recent-docs">
              <div className="recent-docs-header">
                <h3>Recent uploads</h3>
                <button type="button" className="ghost small" onClick={handleClearStoredDocuments}>
                  Clear
                </button>
              </div>
              <div className="recent-docs-list">
                {savedDocuments.map((doc) => (
                  <button
                    key={doc.id}
                    type="button"
                    className="recent-doc"
                    onClick={() => handleLoadStoredDocument(doc)}
                  >
                    <div className="recent-doc-main">
                      <span className="recent-doc-title">
                        {doc.title || doc.primaryFileMeta?.name || 'Untitled Session'}
                      </span>
                      <span className="recent-doc-meta">
                        {doc.sourceKind.toUpperCase()} ·{' '}
                        {(Number.isFinite(doc.wordCount) ? doc.wordCount : tokenize(doc.sourceText).length).toLocaleString()}{' '}
                        words
                      </span>
                    </div>
                    <span className="recent-doc-time">{formatStoredDate(doc.updatedAt)}</span>
                  </button>
                ))}
              </div>
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

          <div className="meta-grid">
            <div>
              <span className="meta-label">Session / Total</span>
              <strong>{sessionWordPairLabel}</strong>
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
            Kindle/EPUB imports require DRM-free exports. PDF text extraction is best on text-based PDFs.
          </p>

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
              Swipe left/right to move back/next. Long blink toggles pause. Video stays on-device.
            </p>
            {cameraNotice && <p className={`notice ${cameraNotice.kind}`}>{cameraNotice.message}</p>}
          </div>
        </section>

        <section className="panel reader-panel reader-sticky">
            {showReaderDisplay && (
            <div className="reader-display" style={readerDisplayStyle} ref={readerDisplayRef}>
              <div className="focus-rail" aria-hidden="true" />
              <ReaderDisplay
                currentSegments={currentSegments}
                currentSegmentText={currentSegmentText}
                singleWordMode={singleWordMode}
                contextRadius={contextRadius}
                tokens={tokens}
                wordIndex={wordIndex}
              />
            </div>
            )}
            {!showReaderDisplay && showConventional && conventionalMode === 'excerpt' && (
              <ConventionalExcerpt
                ref={conventionalRef}
                nodes={conventionalNodes}
                spacerHeights={conventionalSpacerHeights}
                onScroll={handleConventionalScroll}
                onClick={handleConventionalClick}
              />
            )}
            {!showReaderDisplay && showConventional && conventionalMode === 'rendered' && (
              <ConventionalRendered ref={conventionalRef} content={renderedMarkdown} />
            )}

            <div className="deck-panel">
              <div className="deck-header">
                <span className="deck-label">Main Deck</span>
                <span className={`deck-led ${isPlaying ? 'live' : ''}`} aria-hidden="true" />
              </div>
              <div className="controls deck-controls">
                <button
                  type="button"
                  className={`deck-btn primary ${isPlaying ? 'live' : ''}`}
                  aria-pressed={isPlaying}
                  onClick={handleTogglePlay}
                  disabled={!tokens.length}
                >
                  {isPlaying ? 'Pause' : 'Start'}
                </button>
                <button type="button" className="deck-btn ghost" onClick={handleBookmark} disabled={!tokens.length}>
                  Bookmark
                </button>
                <button
                  type="button"
                  className={`deck-btn mode ${sentenceMode ? 'active' : ''}`}
                  onClick={() => setSentenceMode((m) => !m)}
                  aria-pressed={sentenceMode}
                >
                  {sentenceMode ? 'Sentence' : 'Word'}
                </button>
                <button
                  type="button"
                  className="deck-btn mode"
                  onClick={() => {
                    if (showReaderDisplay && showConventional) {
                      setShowReaderDisplay(false);
                      setAutoFollowConventional(true);
                      if (viewMode === 'focus') handleViewModeChange('focus-text');
                    } else if (!showReaderDisplay && showConventional) {
                      setShowReaderDisplay(true);
                      setShowConventional(false);
                      if (viewMode === 'focus-text') handleViewModeChange('focus');
                    } else {
                      setShowReaderDisplay(true);
                      setShowConventional(true);
                      setAutoFollowConventional(true);
                      if (viewMode === 'focus') handleViewModeChange('focus-text');
                    }
                  }}
                >
                  {showReaderDisplay && showConventional ? 'Both' : showReaderDisplay ? 'Word' : 'Text'}
                </button>
                <button
                  type="button"
                  className="deck-btn back"
                  onClick={() => sentenceMode ? handleSentenceStep('back') : handleTransportStep('back')}
                  disabled={sentenceMode ? !sentenceSegments.length : !tokens.length}
                >
                  Back
                </button>
                <button
                  type="button"
                  className="deck-btn next"
                  onClick={() => sentenceMode ? handleSentenceStep('next') : handleTransportStep('next')}
                  disabled={sentenceMode ? !sentenceSegments.length : !tokens.length}
                >
                  Next
                </button>
              </div>
              <p className="deck-hint">
                Hotkeys: Space play/pause, Left/Right back/next, Shift + Left/Right sentence, Up/Down speed, V view,
                F fullscreen, B bookmark, ? help.
              </p>
            </div>
        </section>

        <section className="panel reader-panel reader-controls-panel">
          <div className="mixer-panel">
            <div className="mixer-header">
              <span className="mixer-label">Mixer</span>
              <span className="mixer-led" aria-hidden="true" />
            </div>
            <div className="slider-group">
              <label className="slider">
                <span>Granularity</span>
                <select value={granularity} onChange={(event) => setGranularity(event.target.value as Granularity)}>
                  <option value="word">Word</option>
                  <option value="bigram">Bi-gram</option>
                  <option value="trigram">Tri-gram</option>
                  <option value="sentence">Sentence</option>
                  <option value="tweet">Tweet</option>
                </select>
                <strong>{granularityLabel}</strong>
              </label>
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
                <span>Text size</span>
                <input
                  type="range"
                  min={80}
                  max={135}
                  value={Math.round(displayScale * 100)}
                  onChange={(event) => {
                    const next = Number(event.target.value) / 100;
                    setDisplayScale(clamp(next, 0.8, 1.35));
                  }}
                />
                <strong>{Math.round(displayScale * 100)}%</strong>
              </label>
              <label className="slider">
                <span>Chunk size</span>
                <input
                  type="range"
                  min={1}
                  max={5}
                  value={chunkSize}
                  onChange={(event) => setChunkSize(Number(event.target.value))}
                  disabled={granularity === 'sentence' || granularity === 'tweet'}
                />
                <strong>
                  {chunkSize} {granularity === 'word' ? 'word' : 'segment'}
                  {chunkSize > 1 ? 's' : ''}
                </strong>
              </label>
              <label className="slider">
                <span>Context words</span>
                <input
                  type="range"
                  min={0}
                  max={6}
                  value={contextRadius}
                  onChange={(event) => setContextRadius(Number(event.target.value))}
                  disabled={chunkSize !== 1 || granularity !== 'word'}
                />
                <strong>{contextRadius ? `${contextRadius} each side` : 'Off'}</strong>
              </label>
              <label className="slider">
                <span>Long word slowdown</span>
                <input
                  type="range"
                  min={0}
                  max={80}
                  value={longWordSlowdown}
                  onChange={(event) => setLongWordSlowdown(clamp(Number(event.target.value), 0, 80))}
                />
                <strong>{longWordSlowdown}%</strong>
              </label>
              <label className="slider">
                <span>Minimum word time</span>
                <input
                  type="range"
                  min={80}
                  max={300}
                  step={10}
                  value={minWordMs}
                  onChange={(event) => setMinWordMs(Number(event.target.value))}
                />
                <strong>{minWordMs} ms</strong>
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
            <div className="mixer-aux">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={audioCuesEnabled}
                  onChange={(event) => setAudioCuesEnabled(event.target.checked)}
                />
                <span>Cue sounds</span>
              </label>
            </div>
          </div>

          <div className="session-panel">
            <div className="session-header">
              <h3>Readout</h3>
            </div>
            <div className="session-metrics">
              <div>
                <span className="meta-label">Time</span>
                <strong>{sessionDurationLabel}</strong>
              </div>
              <div>
                <span className="meta-label">Session words</span>
                <strong>{sessionWordCount.toLocaleString()}</strong>
              </div>
              <div>
                <span className="meta-label">Total words</span>
                <strong>{totalWordCount.toLocaleString()}</strong>
              </div>
              <div>
                <span className="meta-label">Completion</span>
                <strong>{sessionPercent}%</strong>
              </div>
            </div>
            <div className="session-awards">
              <div className="session-awards-header">
                <h4>Milestones</h4>
                <span className="award-count">
                  {awards.earnedCount}/{awards.total} unlocked
                </span>
              </div>
              <div className="award-grid">
                {awards.items.map((award) => (
                  <div
                    key={award.id}
                    className={`award-chip ${award.earned ? 'earned' : 'locked'}${
                      highlightAwardId === award.id ? ' is-new' : ''
                    }`}
                    title={award.detail}
                  >
                    <span className="award-label">{award.label}</span>
                    <span className="award-detail">{award.detail}</span>
                  </div>
                ))}
              </div>
              {awards.nextAward && nextAwardRemaining && (
                <div className="award-next">
                  <span className="meta-label">Next up</span>
                  <strong>{awards.nextAward.label}</strong>
                  <span className="award-progress">{nextAwardRemaining}</span>
                </div>
              )}
            </div>
            <div className="session-goals">
              <div className="goal-row">
                <span className="meta-label">Daily goal</span>
                <div className="goal-input">
                  <input
                    type="number"
                    min={200}
                    max={20000}
                    value={dailyGoalWords}
                    onChange={(event) => {
                      const next = Number(event.target.value);
                      if (!Number.isFinite(next)) return;
                      setDailyGoalWords(clamp(next, 200, 20000));
                    }}
                  />
                  <span>words</span>
                </div>
                <span className="goal-progress">{dailyProgressLabel}</span>
              </div>
              <div className="goal-row">
                <span className="meta-label">Streak</span>
                <strong>{streakLabel}</strong>
                <span className="goal-progress">Best {bestStreakLabel}</span>
              </div>
            </div>
            {showSessionRecap && (
              <div className="session-recap">
                <div className="session-recap-header">
                  <h4>Session recap</h4>
                  <span className="recap-pill">Paused</span>
                </div>
                <div className="session-recap-grid">
                  <div>
                    <span className="meta-label">Time</span>
                    <strong>{sessionDurationLabel}</strong>
                  </div>
                  <div>
                    <span className="meta-label">Words</span>
                    <strong>{sessionWordCount.toLocaleString()}</strong>
                  </div>
                  <div>
                    <span className="meta-label">Best streak</span>
                    <strong>{bestStreakLabel}</strong>
                  </div>
                  <div>
                    <span className="meta-label">Next goal</span>
                    <strong>{nextGoalLabel}</strong>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="deck-panel nav-deck">
            <div className="deck-header">
              <span className="deck-label">Navigator Deck</span>
              <span className="deck-led" aria-hidden="true" />
            </div>
            <label className="progress deck-progress">
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

            <div className="jump-nav">
              {showPageJump && (
                <div className="jump-field">
                  <span>Go to page</span>
                  <div className="jump-input">
                    <input
                      type="number"
                      min={1}
                      max={pageJumpMax}
                      inputMode="numeric"
                      value={jumpPage}
                      onChange={(event) => setJumpPage(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.preventDefault();
                          handleJumpToPage();
                        }
                      }}
                      placeholder={pageJumpPlaceholder}
                      disabled={pageJumpDisabled}
                    />
                    <button type="button" className="ghost small" onClick={handleJumpToPage} disabled={pageJumpDisabled}>
                      Go
                    </button>
                  </div>
                </div>
              )}
              <div className="jump-field">
                <span>Go to %</span>
                <div className="jump-input">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    inputMode="numeric"
                    value={jumpPercent}
                    onChange={(event) => setJumpPercent(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        event.preventDefault();
                        handleJumpToPercent();
                      }
                    }}
                    placeholder="0-100"
                    disabled={!tokens.length}
                  />
                  <button
                    type="button"
                    className="ghost small"
                    onClick={handleJumpToPercent}
                    disabled={!tokens.length}
                  >
                    Go
                  </button>
                </div>
              </div>
            </div>

            <BookmarksPanel onClear={handleClearBookmarks} bookmarkRows={bookmarkRows} notice={bookmarkNotice} />
          </div>

          <div className="find-panel">
            <div className="find-header">
              <h3>Find</h3>
              <span className="find-count">{findCountLabel}</span>
            </div>
            <div className="find-row">
              <input
                type="text"
                value={findQuery}
                onChange={(event) => setFindQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    handleFindStart();
                  }
                }}
                placeholder="Search the text"
              />
              <button
                type="button"
                className="find-btn"
                onClick={handleFindStart}
                disabled={!findQuery.trim()}
              >
                Find
              </button>
            </div>
            <div className="find-actions">
              <button
                type="button"
                className="find-btn secondary"
                onClick={() => handleFindStep('prev')}
                disabled={!findMatches.length}
              >
                Prev
              </button>
              <button
                type="button"
                className="find-btn secondary"
                onClick={() => handleFindStep('next')}
                disabled={!findMatches.length}
              >
                Next
              </button>
            </div>
            {findQuery.trim() && !findMatches.length && !isFinding && (
              <p className="hint">No matches found.</p>
            )}
          </div>
        </section>

        <section className="panel companion-panel">
          {showReaderDisplay && showConventional && conventionalMode === 'excerpt' && (
            <ConventionalExcerpt
              ref={conventionalRef}
              nodes={conventionalNodes}
              spacerHeights={conventionalSpacerHeights}
              onScroll={handleConventionalScroll}
              onClick={handleConventionalClick}
            />
          )}
          {showReaderDisplay && showConventional && conventionalMode === 'rendered' && (
            <ConventionalRendered ref={conventionalRef} content={renderedMarkdown} />
          )}

          <div className="toggle-row conventional-toggles">
            <label className="toggle">
              <input
                type="checkbox"
                checked={showReaderDisplay}
                onChange={(event) => setShowReaderDisplay(event.target.checked)}
              />
              <span>Show word</span>
            </label>
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
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={conventionalSeekEnabled}
                  onChange={(event) => setConventionalSeekEnabled(event.target.checked)}
                  disabled={!showConventional}
                />
                <span>Scroll to seek</span>
              </label>
            )}
            <label className="toggle">
              <input
                type="checkbox"
                checked={autoFollowConventional}
                onChange={(event) => setAutoFollowConventional(event.target.checked)}
                disabled={!showConventional || conventionalMode !== 'excerpt'}
              />
              <span>Auto-follow</span>
            </label>
          </div>
          {showConventional && conventionalMode === 'excerpt' && (
            <p className="hint">Scrolling the excerpt moves the reader to the word near the top edge.</p>
          )}
          {showConventional && conventionalMode === 'rendered' && (
            <p className="hint">Rendered view is read-only. Switch to Excerpt to scroll/seek.</p>
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

      {showHotkeys && (
        <div className="hotkey-sheet" role="dialog" aria-label="Hotkey guide">
          <div className="hotkey-header">
            <span>Hotkeys</span>
            <button type="button" className="ghost small" onClick={() => setShowHotkeys(false)}>
              Close
            </button>
          </div>
          <div className="hotkey-list">
            <div className="hotkey-row">
              <span className="hotkey-key">Space</span>
              <span className="hotkey-label">Play / Pause</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">Left</span>
              <span className="hotkey-label">Back</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">Right</span>
              <span className="hotkey-label">Next</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">Shift + ← / →</span>
              <span className="hotkey-label">Sentence step</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">Up</span>
              <span className="hotkey-label">Faster</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">Down</span>
              <span className="hotkey-label">Slower</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">B</span>
              <span className="hotkey-label">Bookmark</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">V</span>
              <span className="hotkey-label">Cycle display mode</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">F</span>
              <span className="hotkey-label">Toggle fullscreen</span>
            </div>
            <div className="hotkey-row">
              <span className="hotkey-key">?</span>
              <span className="hotkey-label">Toggle help</span>
            </div>
          </div>
          <p className="hotkey-hint">Press ? anytime to show or hide this panel.</p>
        </div>
      )}
    </div>
  );
}

export default App;

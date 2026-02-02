/**
 * SyncService — Push/pull sync for the Focus Reader.
 *
 * When authenticated, syncs documents, sessions, and preferences
 * to the server. Falls back to local-only when anonymous.
 */

import { buildApiUrl } from '../env';

type SyncQueueItem = {
  id: string;
  action: 'push-doc' | 'push-session' | 'push-preferences' | 'delete-doc';
  payload: unknown;
  createdAt: number;
};

type ServerDocument = {
  id: string;
  title: string;
  sourceKind: string;
  fileName: string;
  fileSize: number;
  wordCount: number;
  contentHash: string;
  createdAt: string;
  updatedAt: string;
};

type ServerSession = {
  wordIndex: number;
  wpm: number;
  chunkSize: number;
  granularity: string;
  contextRadius: number;
  minWordMs: number;
  sentencePauseMs: number;
  sessionElapsedMs: number;
  bookmarks: unknown[];
  furthestRead: unknown;
  updatedAt: string;
};

type ServerPreferences = {
  preferences: Record<string, unknown>;
  streak: Record<string, unknown>;
  conventionalMode: string;
  updatedAt: string;
};

export type SyncStatus = 'idle' | 'syncing' | 'error' | 'offline';

type SyncListener = (status: SyncStatus) => void;

const QUEUE_DB = 'reader-sync-queue';
const QUEUE_STORE = 'queue';

// IndexedDB helpers for offline queue
const openQueueDb = (): Promise<IDBDatabase> =>
  new Promise((resolve, reject) => {
    const request = indexedDB.open(QUEUE_DB, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(QUEUE_STORE)) {
        db.createObjectStore(QUEUE_STORE, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

const enqueueItem = async (item: SyncQueueItem) => {
  try {
    const db = await openQueueDb();
    const tx = db.transaction(QUEUE_STORE, 'readwrite');
    tx.objectStore(QUEUE_STORE).put(item);
    await new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
    db.close();
  } catch {
    // IndexedDB unavailable — drop silently
  }
};

const dequeueAll = async (): Promise<SyncQueueItem[]> => {
  try {
    const db = await openQueueDb();
    const tx = db.transaction(QUEUE_STORE, 'readonly');
    const store = tx.objectStore(QUEUE_STORE);
    const items: SyncQueueItem[] = await new Promise((resolve, reject) => {
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return items;
  } catch {
    return [];
  }
};

const clearQueue = async () => {
  try {
    const db = await openQueueDb();
    const tx = db.transaction(QUEUE_STORE, 'readwrite');
    tx.objectStore(QUEUE_STORE).clear();
    await new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
    db.close();
  } catch {
    // ignore
  }
};

let queueCounter = 0;

export class SyncService {
  private token: string | null = null;
  private listeners: Set<SyncListener> = new Set();
  private _status: SyncStatus = 'idle';
  private debounceTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  get status(): SyncStatus {
    return this._status;
  }

  private setStatus(s: SyncStatus) {
    this._status = s;
    this.listeners.forEach((fn) => fn(s));
  }

  onStatusChange(fn: SyncListener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  setToken(t: string | null) {
    this.token = t;
    if (t) {
      this.flushQueue();
      this.startPolling();
    } else {
      this.stopPolling();
    }
  }

  isAuthenticated(): boolean {
    return !!this.token;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  }

  private async apiCall<T>(method: string, path: string, body?: unknown): Promise<T | null> {
    if (!this.token) return null;
    try {
      const opts: RequestInit = { method, headers: this.headers() };
      if (body !== undefined) opts.body = JSON.stringify(body);
      const res = await fetch(buildApiUrl(path), opts);
      if (!res.ok) {
        if (res.status === 401) {
          this.token = null;
          this.setStatus('idle');
          return null;
        }
        throw new Error(`${res.status}`);
      }
      const text = await res.text();
      return text ? JSON.parse(text) : null;
    } catch {
      this.setStatus('error');
      return null;
    }
  }

  private debounce(key: string, fn: () => void, ms: number) {
    const existing = this.debounceTimers.get(key);
    if (existing) clearTimeout(existing);
    this.debounceTimers.set(key, setTimeout(fn, ms));
  }

  // ========================================================================
  // Push operations
  // ========================================================================

  /** Upload a document file to the server */
  async pushDocument(file: File, title: string, sourceKind: string, wordCount: number): Promise<ServerDocument | null> {
    if (!this.token) {
      await enqueueItem({
        id: `q_${++queueCounter}_${Date.now()}`,
        action: 'push-doc',
        payload: { title, sourceKind, fileName: file.name, wordCount },
        createdAt: Date.now(),
      });
      return null;
    }
    this.setStatus('syncing');
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', title);
      formData.append('sourceKind', sourceKind);
      formData.append('fileName', file.name);
      formData.append('wordCount', String(wordCount));

      const res = await fetch(buildApiUrl('/api/reader-docs'), {
        method: 'POST',
        headers: { Authorization: `Bearer ${this.token}` },
        body: formData,
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const doc = await res.json();
      this.setStatus('idle');
      return doc;
    } catch {
      this.setStatus('error');
      return null;
    }
  }

  /** Push reading session state (debounced) */
  pushSession(docId: string, session: Record<string, unknown>) {
    if (!this.token) return;
    this.debounce(`session:${docId}`, () => {
      this.setStatus('syncing');
      this.apiCall('PUT', `/api/reader-docs/${docId}/session`, session).then(() => {
        if (this._status === 'syncing') this.setStatus('idle');
      });
    }, 600);
  }

  /** Push preferences + streak (debounced) */
  pushPreferences(data: { preferences?: Record<string, unknown>; streak?: Record<string, unknown>; conventionalMode?: string }) {
    if (!this.token) return;
    this.debounce('preferences', () => {
      this.setStatus('syncing');
      this.apiCall('PUT', '/api/reader-docs/preferences', data).then(() => {
        if (this._status === 'syncing') this.setStatus('idle');
      });
    }, 1000);
  }

  /** Push document deletion */
  async pushDelete(docId: string) {
    if (!this.token) return;
    this.setStatus('syncing');
    await this.apiCall('DELETE', `/api/reader-docs/${docId}`);
    if (this._status === 'syncing') this.setStatus('idle');
  }

  /** Push deletion of all user documents (iterate server-side list) */
  async pushDeleteAll() {
    if (!this.token) return;
    this.setStatus('syncing');
    const docs = await this.pullDocumentList();
    if (docs) {
      for (const doc of docs) {
        await this.apiCall('DELETE', `/api/reader-docs/${doc.id}`);
      }
    }
    if (this._status === 'syncing') this.setStatus('idle');
  }

  // ========================================================================
  // Pull operations
  // ========================================================================

  /** Pull document list from server */
  async pullDocumentList(): Promise<ServerDocument[] | null> {
    return this.apiCall('GET', '/api/reader-docs');
  }

  /** Pull session for a document */
  async pullSession(docId: string): Promise<ServerSession | null> {
    return this.apiCall('GET', `/api/reader-docs/${docId}/session`);
  }

  /** Pull preferences */
  async pullPreferences(): Promise<ServerPreferences | null> {
    return this.apiCall('GET', '/api/reader-docs/preferences');
  }

  /** Download a document file */
  async pullFile(docId: string): Promise<Blob | null> {
    if (!this.token) return null;
    try {
      const res = await fetch(buildApiUrl(`/api/reader-docs/${docId}/file`), {
        headers: { Authorization: `Bearer ${this.token}` },
      });
      if (!res.ok) return null;
      return await res.blob();
    } catch {
      return null;
    }
  }

  // ========================================================================
  // Queue flush & polling
  // ========================================================================

  private async flushQueue() {
    const items = await dequeueAll();
    if (!items.length) return;
    // For now we just clear the queue — file re-upload from IndexedDB
    // is complex and the user can re-open the document to trigger a fresh push
    await clearQueue();
  }

  private startPolling() {
    this.stopPolling();
    // Poll every 60s for new documents from other devices
    this.pollTimer = setInterval(() => {
      if (this.token && this._status !== 'syncing') {
        this.listeners.forEach((fn) => fn(this._status));
      }
    }, 60_000);
  }

  private stopPolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  destroy() {
    this.stopPolling();
    this.debounceTimers.forEach((t) => clearTimeout(t));
    this.debounceTimers.clear();
    this.listeners.clear();
  }
}

// Singleton
export const syncService = new SyncService();

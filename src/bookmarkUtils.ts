export type BookmarkPayload = {
  v: number;
  index: number;
  pageNumber?: number;
  pageOffset?: number;
  docHash?: string;
  sourceKind?: 'text' | 'pdf';
  title?: string;
};

export const BOOKMARK_VERSION = 1;
const BOOKMARK_PARAM = 'bm';

export const hashText = (text: string) => {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash << 5) - hash + text.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash).toString(36);
};

const encodeBase64 = (value: string) => {
  if (typeof btoa === 'function') {
    return btoa(value);
  }
  return Buffer.from(value, 'binary').toString('base64');
};

const decodeBase64 = (value: string) => {
  if (typeof atob === 'function') {
    return atob(value);
  }
  return Buffer.from(value, 'base64').toString('binary');
};

const base64UrlEncode = (value: string) => {
  const bytes = new TextEncoder().encode(value);
  let binary = '';
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return encodeBase64(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};

const base64UrlDecode = (value: string) => {
  let base64 = value.replace(/-/g, '+').replace(/_/g, '/');
  while (base64.length % 4) {
    base64 += '=';
  }
  const binary = decodeBase64(base64);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
};

export const encodeBookmarkPayload = (payload: BookmarkPayload) =>
  base64UrlEncode(JSON.stringify(payload));

export const decodeBookmarkPayload = (value: string) => {
  try {
    const decoded = base64UrlDecode(value);
    const parsed = JSON.parse(decoded) as BookmarkPayload;
    if (!parsed || typeof parsed !== 'object') return null;
    if (typeof parsed.index !== 'number' || !Number.isFinite(parsed.index)) return null;
    if (typeof parsed.v === 'number' && parsed.v !== BOOKMARK_VERSION) return null;
    return parsed;
  } catch {
    return null;
  }
};

const parseReadablePairs = (value: string) => {
  const parts = value.split(':').filter(Boolean);
  if (parts.length < 2 || parts.length % 2 !== 0) return null;
  const pairs = new Map<string, string>();
  for (let i = 0; i < parts.length; i += 2) {
    pairs.set(parts[i].toLowerCase(), parts[i + 1]);
  }
  return pairs;
};

export const encodeReadableBookmark = (payload: BookmarkPayload) => {
  const segments: string[] = [];
  if (payload.pageNumber != null && payload.pageOffset != null) {
    segments.push('page', String(payload.pageNumber), 'offset', String(payload.pageOffset));
    if (Number.isFinite(payload.index)) {
      segments.push('idx', String(Math.max(0, Math.round(payload.index))));
    }
  } else {
    segments.push('idx', String(Math.max(0, Math.round(payload.index))));
  }
  if (payload.docHash) {
    segments.push('doc', payload.docHash);
  }
  return segments.join(':');
};

export const decodeReadableBookmark = (value: string) => {
  const pairs = parseReadablePairs(value);
  if (!pairs) return null;
  const idxValue = pairs.get('idx') ?? pairs.get('index');
  const pageValue = pairs.get('page') ?? pairs.get('p');
  const offsetValue = pairs.get('offset') ?? pairs.get('o');
  const docHash = pairs.get('doc');
  let index = 0;
  if (idxValue) {
    const raw = Number(idxValue);
    if (!Number.isFinite(raw)) return null;
    index = Math.max(0, Math.round(raw));
  }
  let pageNumber: number | undefined;
  let pageOffset: number | undefined;
  if (pageValue && offsetValue) {
    const pageRaw = Number(pageValue);
    const offsetRaw = Number(offsetValue);
    if (!Number.isFinite(pageRaw) || !Number.isFinite(offsetRaw)) return null;
    pageNumber = Math.max(1, Math.round(pageRaw));
    pageOffset = Math.max(0, Math.round(offsetRaw));
  }
  if (!idxValue && pageNumber == null) return null;
  return {
    v: BOOKMARK_VERSION,
    index,
    pageNumber,
    pageOffset,
    docHash: docHash || undefined,
  } satisfies BookmarkPayload;
};

export const parseBookmarkValue = (value: string) => {
  if (!value) return null;
  if (value.includes(':')) {
    const readable = decodeReadableBookmark(value);
    if (readable) return readable;
  }
  return decodeBookmarkPayload(value);
};

const getHashParams = (hash: string) => {
  if (!hash) return { base: '', params: new URLSearchParams() };
  const [base, query = ''] = hash.split('?');
  return { base, params: new URLSearchParams(query) };
};

export const buildBookmarkUrl = (currentUrl: string, payload: BookmarkPayload, format: 'encoded' | 'readable') => {
  const url = new URL(currentUrl);
  const value = format === 'readable' ? encodeReadableBookmark(payload) : encodeBookmarkPayload(payload);
  if (url.hash) {
    const { base, params } = getHashParams(url.hash);
    params.set(BOOKMARK_PARAM, value);
    const query = params.toString();
    url.hash = query ? `${base}?${query}` : base;
    return url.toString();
  }
  url.searchParams.set(BOOKMARK_PARAM, value);
  return url.toString();
};

export const readBookmarkParam = (currentUrl: string) => {
  const url = new URL(currentUrl);
  const searchValue = url.searchParams.get(BOOKMARK_PARAM);
  if (searchValue) return searchValue;
  if (url.hash) {
    const { params } = getHashParams(url.hash);
    return params.get(BOOKMARK_PARAM);
  }
  return null;
};

import { segmentTextBySentence, segmentTokens, tokenize, type Granularity, type Segment } from './textUtils';

type Kind = 'primary' | 'secondary';

type AnalyzeRequest = {
  id: number;
  type: 'analyze';
  docId: number;
  kind: Kind;
  text: string;
  granularity: Granularity;
};

type SegmentRequest = {
  id: number;
  type: 'segment';
  docId: number;
  kind: Kind;
  granularity: Granularity;
};

type FindRequest = {
  id: number;
  type: 'find';
  docId: number;
  query: string;
};

type WorkerRequest = AnalyzeRequest | SegmentRequest | FindRequest;

type AnalyzeResponse = {
  id: number;
  type: 'analyze-result';
  docId: number;
  kind: Kind;
  tokens: string[];
  segments: Segment[];
};

type SegmentResponse = {
  id: number;
  type: 'segment-result';
  docId: number;
  kind: Kind;
  segments: Segment[];
};

type FindResponse = {
  id: number;
  type: 'find-result';
  docId: number;
  query: string;
  matches: number[];
};

type WorkerResponse = AnalyzeResponse | SegmentResponse | FindResponse;

type CacheEntry = {
  text: string;
  tokens: string[];
  normalizedTokens: string[];
};

const normalizeTokenForSearch = (value: string) =>
  value.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');

const cache = new Map<number, CacheEntry>();

const getSegments = (entry: CacheEntry, granularity: Granularity) => {
  if (granularity === 'sentence') {
    return segmentTextBySentence(entry.text);
  }
  return segmentTokens(entry.tokens, granularity);
};

const computeTokens = (text: string) => {
  const tokens = tokenize(text);
  const normalizedTokens = tokens.map((token) => normalizeTokenForSearch(token));
  return { tokens, normalizedTokens };
};

const getEntry = (docId: number, text?: string) => {
  const existing = cache.get(docId);
  if (existing && (!text || existing.text === text)) {
    return existing;
  }
  if (typeof text !== 'string') return null;
  const computed = computeTokens(text);
  const entry: CacheEntry = {
    text,
    tokens: computed.tokens,
    normalizedTokens: computed.normalizedTokens,
  };
  cache.set(docId, entry);
  return entry;
};

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  if (!message || typeof message !== 'object') return;
  if (message.type === 'analyze') {
    const entry = getEntry(message.docId, message.text);
    const segments = entry ? getSegments(entry, message.granularity) : [];
    const response: AnalyzeResponse = {
      id: message.id,
      type: 'analyze-result',
      docId: message.docId,
      kind: message.kind,
      tokens: entry ? entry.tokens : [],
      segments,
    };
    self.postMessage(response satisfies WorkerResponse);
    return;
  }
  if (message.type === 'segment') {
    const entry = getEntry(message.docId);
    const response: SegmentResponse = {
      id: message.id,
      type: 'segment-result',
      docId: message.docId,
      kind: message.kind,
      segments: entry ? getSegments(entry, message.granularity) : [],
    };
    self.postMessage(response satisfies WorkerResponse);
    return;
  }
  if (message.type === 'find') {
    const entry = getEntry(message.docId);
    const queryTokens = tokenize(message.query)
      .map((token) => normalizeTokenForSearch(token))
      .filter(Boolean);
    const matches: number[] = [];
    if (entry && entry.normalizedTokens.length && queryTokens.length) {
      const maxStart = entry.normalizedTokens.length - queryTokens.length;
      for (let i = 0; i <= maxStart; i += 1) {
        let isMatch = true;
        for (let j = 0; j < queryTokens.length; j += 1) {
          const token = entry.normalizedTokens[i + j];
          if (!token || !token.includes(queryTokens[j])) {
            isMatch = false;
            break;
          }
        }
        if (isMatch) {
          matches.push(i);
        }
      }
    }
    const response: FindResponse = {
      id: message.id,
      type: 'find-result',
      docId: message.docId,
      query: message.query,
      matches,
    };
    self.postMessage(response satisfies WorkerResponse);
  }
});

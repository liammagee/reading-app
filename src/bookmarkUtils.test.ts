import { describe, expect, it } from 'vitest';
import {
  BOOKMARK_VERSION,
  buildBookmarkUrl,
  decodeReadableBookmark,
  encodeBookmarkPayload,
  encodeReadableBookmark,
  parseBookmarkValue,
  readBookmarkParam,
} from './bookmarkUtils';

describe('bookmarkUtils', () => {
  it('round-trips encoded payloads', () => {
    const payload = {
      v: BOOKMARK_VERSION,
      index: 128,
      pageNumber: 12,
      pageOffset: 34,
      docHash: 'abc123',
      sourceKind: 'pdf' as const,
    };
    const encoded = encodeBookmarkPayload(payload);
    const decoded = parseBookmarkValue(encoded);
    expect(decoded).toEqual(payload);
  });

  it('round-trips readable payloads', () => {
    const payload = {
      v: BOOKMARK_VERSION,
      index: 42,
      pageNumber: 5,
      pageOffset: 9,
      docHash: 'doc-hash',
    };
    const readable = encodeReadableBookmark(payload);
    const decoded = decodeReadableBookmark(readable);
    expect(decoded).toEqual({
      v: BOOKMARK_VERSION,
      index: 42,
      pageNumber: 5,
      pageOffset: 9,
      docHash: 'doc-hash',
    });
  });

  it('builds and reads bookmark params for hash routes', () => {
    const payload = {
      v: BOOKMARK_VERSION,
      index: 7,
      docHash: 'hashy',
    };
    const url = buildBookmarkUrl('https://example.com/#/reading', payload, 'readable');
    const value = readBookmarkParam(url);
    expect(value).toBe(encodeReadableBookmark(payload));
    const parsed = parseBookmarkValue(value ?? '');
    expect(parsed?.index).toBe(7);
    expect(parsed?.docHash).toBe('hashy');
  });
});

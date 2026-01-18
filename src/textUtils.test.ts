import { describe, expect, it } from 'vitest';
import { segmentTextBySentence, segmentTokens, tokenize } from './textUtils';

describe('segmentTokens', () => {
  it('segments into bigrams', () => {
    const tokens = tokenize('one two three four five');
    const segments = segmentTokens(tokens, 'bigram');
    expect(segments.map((segment) => segment.text)).toEqual([
      'one two',
      'three four',
      'five',
    ]);
    expect(segments.map((segment) => segment.startIndex)).toEqual([0, 2, 4]);
  });
});

describe('segmentTextBySentence', () => {
  it('segments into sentences', () => {
    const segments = segmentTextBySentence('Hello world. Another line here! Final sentence');
    expect(segments.length).toBe(3);
    expect(segments[0].text).toBe('Hello world.');
    expect(segments[1].text).toBe('Another line here!');
    expect(segments[2].text).toBe('Final sentence');
    expect(segments[0].startIndex).toBe(0);
    expect(segments[1].startIndex).toBe(2);
  });

  it('preserves sentence punctuation boundaries from raw text', () => {
    const segments = segmentTextBySentence('He said "Hello world." Next sentence? Final line');
    expect(segments.map((segment) => segment.text)).toEqual([
      'He said "Hello world."',
      'Next sentence?',
      'Final line',
    ]);
    expect(segments[1].startIndex).toBe(4);
  });
});

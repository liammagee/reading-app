import { describe, expect, it } from 'vitest';
import { segmentTextBySentence, segmentTextByTweet, segmentTokens, tokenize } from './textUtils';

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

describe('segmentTextByTweet', () => {
  it('groups sentences into tweet-sized chunks', () => {
    const text = 'First sentence here. Second sentence there. Third sentence now.';
    const segments = segmentTextByTweet(text, 35);
    expect(segments.map((segment) => segment.text)).toEqual([
      'First sentence here.',
      'Second sentence there.',
      'Third sentence now.',
    ]);
    expect(segments[0].startIndex).toBe(0);
    expect(segments[1].startIndex).toBe(3);
  });

  it('splits long sentences into smaller chunks', () => {
    const text = 'This sentence should be split into smaller chunks because it is long.';
    const segments = segmentTextByTweet(text, 22);
    expect(segments.length).toBeGreaterThan(1);
    expect(segments[0].wordCount).toBeGreaterThan(0);
    expect(segments[0].startIndex).toBe(0);
  });
});

const BRACKET_PAIRS: Array<[string, string]> = [
  ['[', ']'],
  ['(', ')'],
  ['{', '}'],
  ['<', '>'],
];

export const mergeBracketedTokens = (tokens: string[]) => {
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

export const tokenize = (text: string) => {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) return [];
  return mergeBracketedTokens(normalized.split(' '));
};

export type Granularity = 'word' | 'bigram' | 'trigram' | 'sentence' | 'tweet';

export type Segment = {
  text: string;
  startIndex: number;
  wordCount: number;
};

const SENTENCE_END_RE = /[.!?]["')\]]*$/;

export const segmentTokens = (tokens: string[], granularity: Granularity): Segment[] => {
  if (!tokens.length) return [];
  if (granularity === 'word') {
    return tokens.map((token, index) => ({
      text: token,
      startIndex: index,
      wordCount: 1,
    }));
  }
  if (granularity === 'bigram' || granularity === 'trigram') {
    const size = granularity === 'bigram' ? 2 : 3;
    const segments: Segment[] = [];
    for (let i = 0; i < tokens.length; i += size) {
      const slice = tokens.slice(i, i + size);
      if (!slice.length) continue;
      segments.push({
        text: slice.join(' '),
        startIndex: i,
        wordCount: slice.length,
      });
    }
    return segments;
  }
  if (granularity === 'tweet') {
    return segmentTextByTweet(tokens.join(' '));
  }
  const segments: Segment[] = [];
  let buffer: string[] = [];
  let startIndex = 0;
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (!buffer.length) {
      startIndex = i;
    }
    buffer.push(token);
    if (SENTENCE_END_RE.test(token)) {
      segments.push({
        text: buffer.join(' '),
        startIndex,
        wordCount: buffer.length,
      });
      buffer = [];
    }
  }
  if (buffer.length) {
    segments.push({
      text: buffer.join(' '),
      startIndex,
      wordCount: buffer.length,
    });
  }
  return segments;
};

const splitTextIntoSentences = (text: string) => {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) return [];
  const sentenceEndRe = /[.!?](?:["')\]\u201d\u2019])*(?=\s|$)/g;
  const sentences: string[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null = null;
  while ((match = sentenceEndRe.exec(normalized))) {
    const endIndex = match.index + match[0].length;
    const slice = normalized.slice(lastIndex, endIndex).trim();
    if (slice) {
      sentences.push(slice);
    }
    lastIndex = endIndex;
  }
  const tail = normalized.slice(lastIndex).trim();
  if (tail) {
    sentences.push(tail);
  }
  return sentences;
};

export const segmentTextBySentence = (text: string): Segment[] => {
  const sentences = splitTextIntoSentences(text);
  if (!sentences.length) return [];
  const segments: Segment[] = [];
  let startIndex = 0;
  for (const sentence of sentences) {
    const sentenceTokens = tokenize(sentence);
    if (!sentenceTokens.length) continue;
    segments.push({
      text: sentenceTokens.join(' '),
      startIndex,
      wordCount: sentenceTokens.length,
    });
    startIndex += sentenceTokens.length;
  }
  return segments;
};

export const segmentTextByTweet = (text: string, maxChars = 280): Segment[] => {
  const sentences = segmentTextBySentence(text);
  if (!sentences.length) return [];
  const segments: Segment[] = [];
  let bufferText = '';
  let bufferStartIndex = 0;
  let bufferWordCount = 0;

  const flushBuffer = () => {
    if (!bufferText) return;
    segments.push({
      text: bufferText,
      startIndex: bufferStartIndex,
      wordCount: bufferWordCount,
    });
    bufferText = '';
    bufferWordCount = 0;
  };

  for (const sentence of sentences) {
    const sentenceText = sentence.text.trim();
    if (!sentenceText) continue;
    if (sentenceText.length > maxChars) {
      flushBuffer();
      const tokens = tokenize(sentenceText);
      if (!tokens.length) continue;
      let tokenBuffer: string[] = [];
      let tokenStartIndex = sentence.startIndex;
      let tokenCharCount = 0;

      for (let i = 0; i < tokens.length; i += 1) {
        const token = tokens[i];
        const tokenLength = token.length;
        if (!tokenBuffer.length && tokenLength > maxChars) {
          segments.push({
            text: token,
            startIndex: tokenStartIndex,
            wordCount: 1,
          });
          tokenStartIndex += 1;
          tokenCharCount = 0;
          tokenBuffer = [];
          continue;
        }
        const nextLength = tokenBuffer.length ? tokenCharCount + 1 + tokenLength : tokenLength;
        if (tokenBuffer.length && nextLength > maxChars) {
          segments.push({
            text: tokenBuffer.join(' '),
            startIndex: tokenStartIndex,
            wordCount: tokenBuffer.length,
          });
          tokenStartIndex += tokenBuffer.length;
          tokenBuffer = [token];
          tokenCharCount = tokenLength;
          continue;
        }
        tokenBuffer.push(token);
        tokenCharCount = nextLength;
      }

      if (tokenBuffer.length) {
        segments.push({
          text: tokenBuffer.join(' '),
          startIndex: tokenStartIndex,
          wordCount: tokenBuffer.length,
        });
      }
      continue;
    }

    if (!bufferText) {
      bufferText = sentenceText;
      bufferStartIndex = sentence.startIndex;
      bufferWordCount = sentence.wordCount;
      continue;
    }
    const nextLength = bufferText.length + 1 + sentenceText.length;
    if (nextLength <= maxChars) {
      bufferText = `${bufferText} ${sentenceText}`;
      bufferWordCount += sentence.wordCount;
    } else {
      flushBuffer();
      bufferText = sentenceText;
      bufferStartIndex = sentence.startIndex;
      bufferWordCount = sentence.wordCount;
    }
  }

  flushBuffer();
  return segments;
};

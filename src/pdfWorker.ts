import { getDocument } from 'pdfjs-dist';
import { tokenize } from './textUtils';

type PdfOutlineItem = {
  title: string;
  pageNumber: number | null;
  url?: string;
  items: PdfOutlineItem[];
};

type ExtractRequest = {
  id: number;
  type: 'extract';
  buffer: ArrayBuffer;
};

type ProgressMessage = {
  id: number;
  type: 'progress';
  current: number;
  total: number;
};

type ResultMessage =
  | {
      id: number;
      type: 'result';
      ok: true;
      payload: {
        pageTexts: string[];
        pageTokenCounts: number[];
        outline: PdfOutlineItem[];
        pageCount: number;
      };
    }
  | {
      id: number;
      type: 'result';
      ok: false;
      error: string;
    };

const extractPdfData = async (
  data: ArrayBuffer,
  onProgress?: (current: number, total: number) => void,
) => {
  const loadingTask = getDocument({ data: new Uint8Array(data), disableWorker: true });
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

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.addEventListener('message', (event) => {
  const data = event.data as ExtractRequest;
  if (!data || data.type !== 'extract') return;
  const { id, buffer } = data;
  void (async () => {
    try {
      const payload = await extractPdfData(buffer, (current, total) => {
        const progress: ProgressMessage = { id, type: 'progress', current, total };
        ctx.postMessage(progress);
      });
      const result: ResultMessage = { id, type: 'result', ok: true, payload };
      ctx.postMessage(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to extract PDF.';
      const result: ResultMessage = { id, type: 'result', ok: false, error: message };
      ctx.postMessage(result);
    }
  })();
});

export {};

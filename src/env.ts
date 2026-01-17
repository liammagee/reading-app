const normalizeBaseUrl = (value: string) => value.trim().replace(/\/+$/, '');

const apiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL || '');

export const buildApiUrl = (path: string) => {
  if (!apiBaseUrl) return path;
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${apiBaseUrl}${normalizedPath}`;
};

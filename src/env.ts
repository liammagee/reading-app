const normalizeBaseUrl = (value: string) => value.trim().replace(/\/+$/, '');

const configuredApiBaseUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL || '');

const shouldUseRelativeApiOnLocalhost = (() => {
  if (!configuredApiBaseUrl || typeof window === 'undefined') return false;
  const localHosts = new Set(['localhost', '127.0.0.1']);
  if (!localHosts.has(window.location.hostname)) return false;
  try {
    const configured = new URL(configuredApiBaseUrl, window.location.origin);
    return localHosts.has(configured.hostname) && configured.origin !== window.location.origin;
  } catch {
    return false;
  }
})();

const apiBaseUrl = shouldUseRelativeApiOnLocalhost ? '' : configuredApiBaseUrl;

export const buildApiUrl = (path: string) => {
  if (!apiBaseUrl) return path;
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${apiBaseUrl}${normalizedPath}`;
};

import { createContext, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { buildApiUrl } from '../env';
import type { AuthState } from './useAuth';

export const AuthContext = createContext<AuthState | null>(null);

const TOKEN_KEY = 'reader:auth_token';
const USER_KEY = 'reader:auth_user';

const isEmbedded = () => {
  try {
    return window.self !== window.top;
  } catch {
    return true;
  }
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthState['user']>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const didInit = useRef(false);

  const persistAuth = useCallback((t: string | null, u: AuthState['user']) => {
    setToken(t);
    setUser(u);
    if (t && u) {
      try {
        localStorage.setItem(TOKEN_KEY, t);
        localStorage.setItem(USER_KEY, JSON.stringify(u));
      } catch { /* quota */ }
    } else {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  }, []);

  const clearAuth = useCallback(() => {
    persistAuth(null, null);
  }, [persistAuth]);

  // Validate a token against the server
  const validateToken = useCallback(async (t: string): Promise<AuthState['user']> => {
    try {
      const res = await fetch(buildApiUrl('/api/auth/me'), {
        headers: { Authorization: `Bearer ${t}` },
      });
      if (!res.ok) return null;
      const data = await res.json();
      const u = data?.user ?? data;
      if (!u?.id) return null;
      return { id: u.id, email: u.email, name: u.name, role: u.role };
    } catch {
      return null;
    }
  }, []);

  // Bootstrap: listen for parent postMessage (embedded) or restore from localStorage
  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;

    const handleMessage = async (event: MessageEvent) => {
      if (event.data?.type === 'auth-token' && event.data.token) {
        const t = event.data.token as string;
        const u = event.data.user as AuthState['user'];
        if (u?.id) {
          persistAuth(t, u);
        } else {
          const validated = await validateToken(t);
          if (validated) {
            persistAuth(t, validated);
          }
        }
        setIsLoading(false);
      }
      if (event.data?.type === 'auth-token-revoked') {
        clearAuth();
      }
    };

    window.addEventListener('message', handleMessage);

    if (isEmbedded()) {
      // Request token from parent
      window.parent.postMessage({ type: 'auth-token-request' }, '*');
      // Give parent 2s to respond, then fall back to localStorage
      const timer = setTimeout(async () => {
        if (!token) {
          const saved = localStorage.getItem(TOKEN_KEY);
          if (saved) {
            const validated = await validateToken(saved);
            if (validated) {
              persistAuth(saved, validated);
            } else {
              clearAuth();
            }
          }
          setIsLoading(false);
        }
      }, 2000);
      return () => {
        window.removeEventListener('message', handleMessage);
        clearTimeout(timer);
      };
    }

    // Standalone mode: restore from localStorage
    (async () => {
      const saved = localStorage.getItem(TOKEN_KEY);
      if (saved) {
        const validated = await validateToken(saved);
        if (validated) {
          persistAuth(saved, validated);
        } else {
          clearAuth();
        }
      }
      setIsLoading(false);
    })();

    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const requestMagicLink = useCallback(async (email: string): Promise<{ token?: string }> => {
    setError(null);
    try {
      const res = await fetch(buildApiUrl('/api/auth/magic-link'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to send login link');
      // In dev mode, the server returns the token directly for auto-verify
      return { token: data.token };
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to send login link';
      setError(msg);
      throw err;
    }
  }, []);

  const verifyMagicLink = useCallback(async (magicToken: string) => {
    setError(null);
    try {
      const res = await fetch(buildApiUrl('/api/auth/magic-link/verify'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: magicToken }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Verification failed');
      const t = data.token || data.accessToken;
      const u = data.user;
      persistAuth(t, { id: u.id, email: u.email, name: u.name, role: u.role });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Verification failed';
      setError(msg);
      throw err;
    }
  }, [persistAuth]);

  const logout = useCallback(() => {
    if (token) {
      fetch(buildApiUrl('/api/auth/logout'), {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {});
    }
    clearAuth();
  }, [token, clearAuth]);

  const value: AuthState = {
    user,
    token,
    isAuthenticated: !!user && !!token,
    isLoading,
    requestMagicLink,
    verifyMagicLink,
    logout,
    error,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

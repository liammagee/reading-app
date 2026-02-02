import { useContext } from 'react';
import { AuthContext } from './AuthProvider';

export type AuthState = {
  user: { id: string; email: string; name?: string; role: string } | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  requestMagicLink: (email: string) => Promise<{ token?: string }>;
  verifyMagicLink: (token: string) => Promise<void>;
  logout: () => void;
  error: string | null;
};

export const useAuth = (): AuthState => {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    return {
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      requestMagicLink: async () => ({}),
      verifyMagicLink: async () => {},
      logout: () => {},
      error: null,
    };
  }
  return ctx;
};

import { useState, type FormEvent } from 'react';
import { useAuth } from './useAuth';

type Mode = 'login' | 'register';

export function LoginPanel({ onClose }: { onClose?: () => void }) {
  const { login, register, error, isLoading } = useAuth();
  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      if (mode === 'register') {
        await register(email, password, name || undefined);
      } else {
        await login(email, password);
      }
      onClose?.();
    } catch {
      // error is set in auth state
    }
  };

  return (
    <div className="login-panel">
      <div className="login-panel-header">
        <h3>{mode === 'login' ? 'Sign In' : 'Create Account'}</h3>
        {onClose && (
          <button type="button" className="ghost" onClick={onClose} aria-label="Close">
            &times;
          </button>
        )}
      </div>
      <form onSubmit={handleSubmit} className="login-form">
        {mode === 'register' && (
          <label>
            <span>Name</span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
              autoComplete="name"
            />
          </label>
        )}
        <label>
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            required
            autoComplete="email"
          />
        </label>
        <label>
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Min 8 characters"
            required
            minLength={8}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          />
        </label>
        {error && <p className="login-error">{error}</p>}
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Loading...' : mode === 'login' ? 'Sign In' : 'Register'}
        </button>
      </form>
      <p className="login-toggle">
        {mode === 'login' ? (
          <>
            No account?{' '}
            <button type="button" className="ghost" onClick={() => setMode('register')}>
              Register
            </button>
          </>
        ) : (
          <>
            Have an account?{' '}
            <button type="button" className="ghost" onClick={() => setMode('login')}>
              Sign In
            </button>
          </>
        )}
      </p>
    </div>
  );
}

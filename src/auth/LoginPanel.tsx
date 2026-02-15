import { useState, type FormEvent } from 'react';
import { useAuth } from './useAuth';

type Step = 'email' | 'sent' | 'verify';
type LoginMode = 'magic' | 'password';

export function LoginPanel({ onClose }: { onClose?: () => void }) {
  const { requestMagicLink, verifyMagicLink, loginWithPassword, registerWithPassword, error, isLoading } = useAuth();
  const [step, setStep] = useState<Step>('email');
  const [mode, setMode] = useState<LoginMode>('password');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [isCreateAccount, setIsCreateAccount] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleRequestLink = async (e: FormEvent) => {
    e.preventDefault();
    setHint(null);
    setIsSubmitting(true);
    try {
      const result = await requestMagicLink(email);
      if (result.token) {
        // Dev mode: auto-verify with returned token
        await verifyMagicLink(result.token);
        onClose?.();
      } else {
        setStep('sent');
      }
    } catch {
      // error is set in auth state
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleVerify = async (e: FormEvent) => {
    e.preventDefault();
    setHint(null);
    setIsSubmitting(true);
    try {
      await verifyMagicLink(code);
      onClose?.();
    } catch {
      // error is set in auth state
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePasswordLogin = async (e: FormEvent) => {
    e.preventDefault();
    setHint(null);
    setIsSubmitting(true);
    try {
      if (isCreateAccount) {
        await registerWithPassword(email, password, name);
      } else {
        await loginWithPassword(email, password);
      }
      onClose?.();
    } catch {
      // error is set in auth state
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="login-panel">
      <div className="login-panel-header">
        <h3>
          {mode === 'password'
            ? (isCreateAccount ? 'Create account' : 'Sign in')
            : step === 'sent'
              ? 'Check your email'
              : 'Sign in with magic link'}
        </h3>
        {onClose && (
          <button type="button" className="ghost" onClick={onClose} aria-label="Close">
            &times;
          </button>
        )}
      </div>

      {mode === 'magic' && step === 'email' && (
        <form onSubmit={handleRequestLink} className="login-form">
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
          {hint && <p className="login-hint">{hint}</p>}
          {error && <p className="login-error">{error}</p>}
          <button type="submit" disabled={isLoading || isSubmitting}>
            {isSubmitting ? 'Sending...' : 'Send login link'}
          </button>
          <p className="login-toggle">
            <button type="button" className="ghost" onClick={() => setMode('password')}>
              Use password instead
            </button>
          </p>
        </form>
      )}

      {mode === 'magic' && step === 'sent' && (
        <div className="login-form">
          <p className="login-sent-msg">
            We sent a login link to <strong>{email}</strong>. Check your inbox and click the link, or paste the code below.
          </p>
          <form onSubmit={handleVerify}>
            <label>
              <span>Verification code</span>
              <input
                type="text"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Paste code from email"
                required
                autoComplete="one-time-code"
              />
            </label>
            {hint && <p className="login-hint">{hint}</p>}
            {error && <p className="login-error">{error}</p>}
            <button type="submit" disabled={isLoading || isSubmitting}>
              {isSubmitting ? 'Verifying...' : 'Verify'}
            </button>
          </form>
          <p className="login-toggle">
            <button type="button" className="ghost" onClick={() => setStep('email')}>
              Try a different email
            </button>
          </p>
          <p className="login-toggle">
            <button
              type="button"
              className="ghost"
              onClick={() => {
                setMode('password');
                setStep('email');
                setIsCreateAccount(false);
              }}
            >
              Use password instead
            </button>
          </p>
        </div>
      )}

      {mode === 'password' && (
        <form onSubmit={handlePasswordLogin} className="login-form">
          {isCreateAccount && (
            <label>
              <span>Name (optional)</span>
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
              placeholder={isCreateAccount ? 'At least 8 characters' : 'Your password'}
              required
              autoComplete={isCreateAccount ? 'new-password' : 'current-password'}
            />
          </label>
          {hint && <p className="login-hint">{hint}</p>}
          {error && <p className="login-error">{error}</p>}
          <button type="submit" disabled={isLoading || isSubmitting}>
            {isSubmitting ? (isCreateAccount ? 'Creating...' : 'Signing in...') : (isCreateAccount ? 'Create account' : 'Sign in')}
          </button>
          <p className="login-toggle">
            <button
              type="button"
              className="ghost"
              onClick={() => setIsCreateAccount((v) => !v)}
            >
              {isCreateAccount ? 'Already have an account? Sign in' : 'New here? Create account'}
            </button>
          </p>
          <p className="login-toggle">
            <button
              type="button"
              className="ghost"
              onClick={() => {
                setMode('magic');
                setStep('email');
                setIsCreateAccount(false);
                setPassword('');
                setName('');
              }}
            >
              Use magic link instead
            </button>
          </p>
        </form>
      )}
    </div>
  );
}

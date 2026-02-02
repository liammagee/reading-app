import { useState, type FormEvent } from 'react';
import { useAuth } from './useAuth';

type Step = 'email' | 'sent' | 'verify';

export function LoginPanel({ onClose }: { onClose?: () => void }) {
  const { requestMagicLink, verifyMagicLink, error, isLoading } = useAuth();
  const [step, setStep] = useState<Step>('email');
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');

  const handleRequestLink = async (e: FormEvent) => {
    e.preventDefault();
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
    }
  };

  const handleVerify = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await verifyMagicLink(code);
      onClose?.();
    } catch {
      // error is set in auth state
    }
  };

  return (
    <div className="login-panel">
      <div className="login-panel-header">
        <h3>{step === 'sent' ? 'Check your email' : 'Sign In'}</h3>
        {onClose && (
          <button type="button" className="ghost" onClick={onClose} aria-label="Close">
            &times;
          </button>
        )}
      </div>

      {step === 'email' && (
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
          {error && <p className="login-error">{error}</p>}
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Sending...' : 'Send login link'}
          </button>
        </form>
      )}

      {step === 'sent' && (
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
            {error && <p className="login-error">{error}</p>}
            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Verifying...' : 'Verify'}
            </button>
          </form>
          <p className="login-toggle">
            <button type="button" className="ghost" onClick={() => setStep('email')}>
              Try a different email
            </button>
          </p>
        </div>
      )}
    </div>
  );
}

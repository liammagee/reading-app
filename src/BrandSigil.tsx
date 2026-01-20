import React from 'react';

type BrandSigilProps = {
  className?: string;
  label?: string;
  decorative?: boolean;
};

export const BrandSigil: React.FC<BrandSigilProps> = ({
  className = 'w-8 h-8',
  label = 'Pacer',
  decorative = false,
}) => {
  const accessibilityProps = decorative
    ? { 'aria-hidden': true }
    : { role: 'img', 'aria-label': label };

  return (
    <svg
      viewBox="0 0 512 512"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      {...accessibilityProps}
    >
      <defs>
        <radialGradient id="pacer-sigil-bg" cx="30%" cy="20%" r="75%">
          <stop offset="0%" stopColor="#271712" />
          <stop offset="55%" stopColor="#140F0D" />
          <stop offset="100%" stopColor="#0A0A0A" />
        </radialGradient>
        <linearGradient id="pacer-sigil-ring" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#F7C977" />
          <stop offset="50%" stopColor="#FF8A4B" />
          <stop offset="100%" stopColor="#E63946" />
        </linearGradient>
        <linearGradient id="pacer-sigil-core" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#FFE9C7" />
          <stop offset="100%" stopColor="#FF8F5A" />
        </linearGradient>
      </defs>
      <rect width="512" height="512" rx="120" fill="url(#pacer-sigil-bg)" />
      <circle cx="256" cy="256" r="176" fill="none" stroke="url(#pacer-sigil-ring)" strokeWidth="30" />
      <circle
        cx="256"
        cy="256"
        r="114"
        fill="none"
        stroke="#F9E9D6"
        strokeOpacity="0.35"
        strokeWidth="6"
      />
      <rect
        x="196"
        y="196"
        width="120"
        height="120"
        rx="14"
        transform="rotate(45 256 256)"
        fill="url(#pacer-sigil-core)"
      />
      <circle cx="256" cy="256" r="28" fill="#FFF8EA" />
      <circle cx="256" cy="256" r="12" fill="#E63946" />
    </svg>
  );
};

export default BrandSigil;

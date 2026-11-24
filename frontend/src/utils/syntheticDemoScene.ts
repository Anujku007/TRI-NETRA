const escapeXml = (value: string) => value.replace(/[&<>"']/g, (character) => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&apos;',
}[character] || character));

/** A camera-style illustration for deterministic demo scenarios; it contains no real face image. */
export function createSyntheticDemoScene(
  title: string,
  zone: string,
  accent: string,
  subjectCount = 1,
  masked = false,
): string {
  const centers = subjectCount > 1 ? [440, 1110] : [800];
  const silhouettes = centers.map((center, index) => `
    <g transform="translate(${center} 0)" opacity="${index === 0 ? '1' : '.84'}">
      <path d="M-198 850c14-161 79-238 198-238s184 77 198 238v28h-396z" fill="url(#uniform)" stroke="${accent}" stroke-opacity=".58" stroke-width="4"/>
      <path d="M-115 651c24-36 62-58 115-58s91 22 115 58l-50 65h-130z" fill="#17243a"/>
      <ellipse cx="0" cy="390" rx="108" ry="137" fill="url(#face)" stroke="#a8bdcf" stroke-opacity=".34" stroke-width="3"/>
      <path d="M-111 366c9-98 53-156 111-156s102 58 111 156c-22-22-40-55-49-91-37 30-95 45-173 45z" fill="#172338"/>
      ${masked && index === 0 ? `<path d="M-78 414q78-34 156 0v62q-78 38-156 0z" fill="${accent}" fill-opacity=".92"/>` : '<path d="M-43 423h22m42 0h22" stroke="#b7cad7" stroke-width="7" stroke-linecap="round" opacity=".65"/>'}
      <path d="M0 431v36m-24 37q24 14 48 0" fill="none" stroke="#8297ac" stroke-width="5" stroke-linecap="round" opacity=".42"/>
      <circle cx="-133" cy="275" r="5" fill="${accent}"/>
      <circle cx="133" cy="275" r="5" fill="${accent}"/>
    </g>`).join('');

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
    <defs>
      <linearGradient id="wall" x2="0" y2="1"><stop stop-color="#17243a"/><stop offset="1" stop-color="#080e1a"/></linearGradient>
      <linearGradient id="floor" x2="0" y2="1"><stop stop-color="#16243a"/><stop offset="1" stop-color="#070b12"/></linearGradient>
      <linearGradient id="uniform" x2="0" y2="1"><stop stop-color="#384c67"/><stop offset="1" stop-color="#101a2a"/></linearGradient>
      <radialGradient id="face"><stop stop-color="#9aafc2"/><stop offset="1" stop-color="#50657c"/></radialGradient>
      <pattern id="grid" width="64" height="64" patternUnits="userSpaceOnUse"><path d="M64 0H0v64" fill="none" stroke="#9eb2c8" stroke-opacity=".055" stroke-width="1"/></pattern>
      <filter id="glow"><feGaussianBlur stdDeviation="18"/></filter>
    </defs>
    <rect width="1600" height="900" fill="url(#wall)"/>
    <ellipse cx="800" cy="185" rx="540" ry="100" fill="${accent}" opacity=".12" filter="url(#glow)"/>
    <path d="M0 0h1600v94H0z" fill="#080d16" opacity=".62"/>
    <path d="M0 610 800 480l800 130v290H0z" fill="url(#floor)"/>
    <path d="M0 611 800 480l800 131M800 480v420M340 555l-115 345m1035-345 115 345M0 728l800-105 800 105M0 820l800-42 800 42" fill="none" stroke="#9fb6ce" stroke-opacity=".12" stroke-width="3"/>
    <g fill="#40546a" opacity=".42"><path d="M85 115h210v465H85zM1305 115h210v465h-210z"/><path d="M375 160h90v350h-90zm760 0h90v350h-90z"/></g>
    <g fill="#d6e6f3" opacity=".12"><path d="M145 142h90v6h-90zm0 28h90v3h-90zm0 24h90v3h-90zm1220-52h90v6h-90zm0 28h90v3h-90zm0 24h90v3h-90z"/></g>
    <rect width="1600" height="900" fill="url(#grid)"/>
    ${silhouettes}
    <g fill="none" stroke="${accent}" stroke-width="3" opacity=".88">
      <path d="M48 152v-48h48M1552 152v-48h-48M48 748v48h48M1552 748v48h-48"/>
      <path d="M70 220h36m-18-18v36m1424-18h36m-18-18v36" opacity=".55"/>
    </g>
    <rect x="48" y="46" width="540" height="80" rx="12" fill="#070c14" fill-opacity=".74" stroke="${accent}" stroke-opacity=".4"/>
    <text x="76" y="79" fill="${accent}" font-family="monospace" font-size="17" font-weight="700" letter-spacing="3">SYNTHETIC DEMO SCENE</text>
    <text x="76" y="108" fill="#e7eef7" font-family="sans-serif" font-size="23" font-weight="600">${escapeXml(title)}</text>
    <text x="1518" y="72" text-anchor="end" fill="#c1cedc" font-family="monospace" font-size="16">${escapeXml(zone.toUpperCase())} · SIMULATION</text>
    <text x="1518" y="838" text-anchor="end" fill="#91a4b8" font-family="monospace" font-size="14">ILLUSTRATION ONLY · NO BIOMETRIC IMAGE</text>
    <rect x="48" y="832" width="246" height="34" rx="6" fill="#080d16" fill-opacity=".78"/>
    <circle cx="68" cy="849" r="5" fill="${accent}"/><text x="84" y="855" fill="#c7d5e2" font-family="monospace" font-size="13">DETERMINISTIC EVENT PLAYBACK</text>
  </svg>`;

  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

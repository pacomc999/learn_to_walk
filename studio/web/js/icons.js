// Icons and small illustrations, drawn as inline SVG so the studio works offline.

const stroke = (paths, extra = "") =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" ${extra}>${paths}</svg>`;

export const icon = {
  home: stroke('<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>'),
  plus: stroke('<path d="M12 5v14M5 12h14"/>'),
  sparkle: stroke('<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>'),
  walkers: stroke('<circle cx="9" cy="5" r="2"/><path d="M7 21l2-6 3 2v4"/><path d="M6 11l3-3 3 2 3 1"/><path d="M9 8l-1 7"/><path d="M17 21l1-4-2-3"/>'),
  library: stroke('<path d="M4 4h4v16H4zM10 4h4v16h-4z"/><path d="M16 5l3.8-1 3.2 15.5-3.8 1z"/>'),
  book: stroke('<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5z"/><path d="M4 21.5V4.5"/><path d="M8 7h8M8 11h6"/>'),
  creature: stroke('<circle cx="12" cy="9" r="5"/><path d="M8 20l1-6M16 20l-1-6"/><circle cx="10.5" cy="8" r=".6" fill="currentColor"/>'),
  world: stroke('<path d="M2 18c3-4 5-4 8 0s5 4 8 0 3-2 4-1"/><path d="M2 22h20"/><circle cx="17" cy="6" r="3"/>'),
  brain: stroke('<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h1V4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h-1V4z"/>'),
  plan: stroke('<rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 9h18M8 2v4M16 2v4M7 13h4M7 17h8"/>'),
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4.5" height="16" rx="1"/><rect x="13.5" y="4" width="4.5" height="16" rx="1"/></svg>',
  stop: stroke('<rect x="6" y="6" width="12" height="12" rx="2"/>'),
  restart: stroke('<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>'),
  camera: stroke('<path d="M3 8a2 2 0 0 1 2-2h2l2-2h6l2 2h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="12" cy="13" r="4"/>'),
  follow: stroke('<circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>'),
  expand: stroke('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>'),
  check: stroke('<path d="M5 12.5l4.5 4.5L19 7.5"/>', 'stroke-width="2.6"'),
  chevron: stroke('<path d="M6 9l6 6 6-6"/>'),
  chevronRight: stroke('<path d="M9 6l6 6-6 6"/>'),
  arrowRight: stroke('<path d="M5 12h14M13 6l6 6-6 6"/>'),
  arrowLeft: stroke('<path d="M19 12H5M11 6l-6 6 6 6"/>'),
  info: stroke('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/>'),
  reset: stroke('<path d="M4 12a8 8 0 1 0 2.4-5.7L4 8.5"/><path d="M4 4v4.5h4.5"/>'),
  save: stroke('<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M7 3v6h9V3M7 21v-7h10v7"/>'),
  trash: stroke('<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>'),
  edit: stroke('<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M14 6l4 4"/>'),
  copy: stroke('<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>'),
  sun: stroke('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
  moon: stroke('<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>'),
  clock: stroke('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
  trophy: stroke('<path d="M8 4h8v5a4 4 0 0 1-8 0z"/><path d="M8 6H4a3 3 0 0 0 4 4M16 6h4a3 3 0 0 1-4 4M12 13v4M8 21h8M9 17h6"/>'),
  target: stroke('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>'),
  fall: stroke('<path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 21h16"/>'),
  speed: stroke('<path d="M12 14l4-4"/><path d="M4 18a9 9 0 1 1 16 0"/>'),
  ruler: stroke('<path d="M3 17L17 3l4 4L7 21z"/><path d="M7 13l2 2M10 10l2 2M13 7l2 2"/>'),
  weight: stroke('<path d="M6 8h12l2 12H4z"/><circle cx="12" cy="5" r="2"/>'),
  muscle: stroke('<path d="M4 16c2-5 5-8 9-8 3 0 5 2 5 4s-2 4-5 4H7"/><path d="M13 8l1-4h3"/>'),
  palette: stroke('<path d="M12 3a9 9 0 0 0 0 18c1.5 0 2-1 2-2s-1-1.5-1-2.5 1-1.5 2-1.5h2a4 4 0 0 0 4-4c0-4.5-4-8-9-8z"/><circle cx="7.5" cy="11" r="1"/><circle cx="10" cy="7" r="1"/><circle cx="15" cy="7" r="1"/>'),
  leg: stroke('<path d="M10 3v8l-2 6h7"/><path d="M14 3v7"/>'),
  physics: stroke('<circle cx="12" cy="12" r="2"/><ellipse cx="12" cy="12" rx="10" ry="4"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)"/>'),
  flag: stroke('<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>'),
  star: stroke('<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>'),
  eye: stroke('<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'),
  wind: stroke('<path d="M3 8h10a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h7"/>'),
  layers: stroke('<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>'),
  cpu: stroke('<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>'),
  grad: stroke('<path d="M2 9l10-5 10 5-10 5z"/><path d="M6 11v5c3 2 9 2 12 0v-5"/>'),
  more: stroke('<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>'),
  close: stroke('<path d="M6 6l12 12M18 6L6 18"/>'),
  folder: stroke('<path d="M3 6a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
  lab: stroke('<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6A2 2 0 0 0 19 18l-5-9V3"/><path d="M7 15h10"/>'),
  stairs: stroke('<path d="M3 20h5v-5h5v-5h5V5h3"/>'),
};

// Group icons for setting cards
export const groupIcon = {
  size: icon.ruler, legs: icon.leg, weight: icon.weight, muscles: icon.muscle, looks: icon.palette,
  ground: icon.world, physics: icon.physics, task: icon.flag, reward: icon.star, extras: icon.wind,
  cells: icon.brain, senses: icon.eye, habits: icon.sparkle, length: icon.clock, study: icon.grad,
};

// ---------- Animals ----------

function duckArt(c = {}) {
  const body = c.body_colour || "#f5c430", wing = c.wing_colour || "#e0b021", leg = c.leg_colour || "#ff8c1a",
    beak = c.beak_colour || "#ff8c1a";
  return `<svg class="animal" viewBox="0 0 120 100" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="60" cy="94" rx="34" ry="4" fill="rgba(0,0,0,0.12)"/>
    <path d="M52 66 L49 82 L44 88 L56 88 M58 66 L60 82 L56 88 L68 88" stroke="${leg}" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <ellipse cx="25" cy="50" rx="10" ry="6" fill="${body}" transform="rotate(-25 25 50)"/>
    <ellipse cx="55" cy="54" rx="34" ry="19" fill="${body}" transform="rotate(-6 55 54)"/>
    <ellipse cx="52" cy="50" rx="21" ry="9" fill="${wing}" transform="rotate(-8 52 50)"/>
    <path d="M78 46 C82 34 82 28 86 22" stroke="${body}" stroke-width="12" fill="none" stroke-linecap="round"/>
    <circle cx="89" cy="20" r="12" fill="${body}"/>
    <path d="M99 19 L112 22 L99 26 Z" fill="${beak}"/>
    <circle cx="93" cy="16" r="3.2" fill="#fff"/><circle cx="94.2" cy="16" r="1.7" fill="#222"/>
  </svg>`;
}

function puppyArt(c = {}) {
  const fur = c.fur_colour || "#c98a4b", ear = c.ear_colour || "#7a4a26", paw = c.paw_colour || "#f2e3cf",
    nose = c.nose_colour || "#2a1d17";
  return `<svg class="animal" viewBox="0 0 120 100" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="60" cy="94" rx="38" ry="4" fill="rgba(0,0,0,0.12)"/>
    <path d="M22 46 C14 36 12 30 16 24" stroke="${fur}" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M36 58 L33 74 L36 88 M46 60 L46 88 M76 60 L74 74 L78 88 M84 58 L86 88" stroke="${fur}" stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <g fill="${paw}"><ellipse cx="37" cy="89" rx="5" ry="3"/><ellipse cx="47" cy="89" rx="5" ry="3"/><ellipse cx="79" cy="89" rx="5" ry="3"/><ellipse cx="88" cy="89" rx="5" ry="3"/></g>
    <ellipse cx="60" cy="52" rx="34" ry="15" fill="${fur}"/>
    <path d="M86 46 C90 38 92 34 94 30" stroke="${fur}" stroke-width="12" fill="none" stroke-linecap="round"/>
    <circle cx="96" cy="27" r="12" fill="${fur}"/>
    <rect x="100" y="25" width="15" height="11" rx="5" fill="${fur}"/>
    <circle cx="114" cy="27" r="3" fill="${nose}"/>
    <ellipse cx="91" cy="27" rx="5" ry="10" fill="${ear}" transform="rotate(18 91 27)"/>
    <circle cx="101" cy="22" r="2.8" fill="#fff"/><circle cx="102" cy="22" r="1.5" fill="#222"/>
  </svg>`;
}

function customArt(values = {}) {
  const body = (values && values.body) || {};
  const torso = (body.torso && body.torso.colour) || "#f59e0b";
  const parts = body.parts || [];
  const leg = (parts[0] && parts[0].colour) || "#4f46e5";
  const tip = (parts[1] && parts[1].colour) || leg;
  return `<svg class="animal" viewBox="0 0 120 100" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="60" cy="94" rx="36" ry="4" fill="rgba(0,0,0,0.12)"/>
    <path d="M34 52 L28 70 M86 52 L92 70 M46 54 L44 70 M74 54 L76 70" stroke="${leg}" stroke-width="6" stroke-linecap="round"/>
    <path d="M28 70 L30 88 M92 70 L90 88 M44 70 L42 88 M76 70 L78 88" stroke="${tip}" stroke-width="5" stroke-linecap="round"/>
    <g fill="#fff" stroke="#1f2937" stroke-width="1.5"><circle cx="28" cy="70" r="3.5"/><circle cx="92" cy="70" r="3.5"/><circle cx="44" cy="70" r="3.5"/><circle cx="76" cy="70" r="3.5"/></g>
    <rect x="24" y="36" width="72" height="20" rx="5" fill="${torso}"/>
    <circle cx="86" cy="43" r="3" fill="#1f2937"/>
    <path d="M60 36 L60 26" stroke="#1f2937" stroke-width="2"/><circle cx="60" cy="24" r="3" fill="#ef4444"/>
  </svg>`;
}

export function animalArt(key, colours) {
  if (key === "custom") return customArt(colours);
  return key === "puppy" ? puppyArt(colours) : duckArt(colours);
}

// ---------- Worlds ----------

export const STYLE_COLOURS = {
  meadow: { sky: ["#bfe3ff", "#eaf6ff"], ground: "#7dbb63", dark: "#5f9a48" },
  street: { sky: ["#cfd8e3", "#eef2f6"], ground: "#9a9a9a", dark: "#7b7b7b" },
  desert: { sky: ["#ffd9a8", "#fff1dc"], ground: "#e0b77a", dark: "#c4975a" },
  snow: { sky: ["#d6e9f7", "#f4f9fd"], ground: "#e8f1f8", dark: "#bcd2e3" },
  moon: { sky: ["#1b1f33", "#3a3f5c"], ground: "#a7a7ad", dark: "#7f7f86" },
  night: { sky: ["#141b38", "#2c3566"], ground: "#4a6b4a", dark: "#365236" },
  studio: { sky: ["#e7ecf3", "#f7f9fc"], ground: "#d8dee8", dark: "#c3cad6" },
};

function terrainPath(terrain, height) {
  const h = Math.min(1, (height || 0) / 10 + 0.25);
  const base = 66;
  if (terrain === "hills") return `M0 ${base} C 25 ${base - 22 * h}, 45 ${base - 22 * h}, 60 ${base - 6} S 100 ${base - 30 * h}, 120 ${base - 12} L120 100 L0 100 Z`;
  if (terrain === "stairs") {
    let d = `M0 ${base}`, x = 0, y = base;
    for (let i = 0; i < 4; i++) { x += 12; d += ` L${x} ${y}`; y -= 6 * h + 3; d += ` L${x} ${y}`; }
    x += 16; d += ` L${x} ${y}`;
    for (let i = 0; i < 4; i++) { y += 6 * h + 3; d += ` L${x} ${y}`; x += 12; d += ` L${x} ${y}`; }
    return d + ` L120 ${y} L120 100 L0 100 Z`;
  }
  if (terrain === "cobblestones" || terrain === "blocks") {
    const w = terrain === "blocks" ? 14 : 7;
    let d = `M0 ${base}`;
    for (let x = 0; x <= 120; x += w) {
      const y = base - ((Math.sin(x * 1.7) + 1) * 3 + 1) * h * (terrain === "blocks" ? 2 : 1);
      d += ` L${x + 0.8} ${y} L${x + w - 0.8} ${y} L${x + w - 0.8} ${base} `;
    }
    return d + " L120 100 L0 100 Z";
  }
  if (terrain === "craters") return `M0 ${base - 4} L20 ${base - 4} Q 25 ${base - 9} 30 ${base - 4} Q 42 ${base + 8} 54 ${base - 4} Q 58 ${base - 9} 62 ${base - 4} L80 ${base - 4} Q 84 ${base - 8} 88 ${base - 4} Q 96 ${base + 5} 104 ${base - 4} L120 ${base - 4} L120 100 L0 100 Z`;
  return `M0 ${base} L120 ${base} L120 100 L0 100 Z`;
}

export function worldArt(values = {}) {
  const style = STYLE_COLOURS[values.style] || STYLE_COLOURS.meadow;
  const id = "g" + Math.random().toString(36).slice(2, 8);
  const terrain = values.terrain_height > 0 ? values.terrain : "flat";
  const extras = [];
  if (values.style === "moon" || values.style === "night") {
    extras.push(`<circle cx="96" cy="18" r="7" fill="#f4f1de" opacity="0.9"/>`);
    for (const [x, y] of [[14, 12], [30, 24], [52, 9], [70, 20], [110, 34], [84, 6]]) extras.push(`<circle cx="${x}" cy="${y}" r="0.9" fill="#fff"/>`);
  } else {
    extras.push(`<circle cx="98" cy="18" r="8" fill="#fff6c9" opacity="0.95"/>`);
  }
  if (values.wind) extras.push(`<path d="M10 30 h26 a4 4 0 1 0 -4 -4 M14 40 h34 a4 4 0 1 1 -4 4" stroke="#fff" stroke-width="2" fill="none" opacity="0.9" stroke-linecap="round"/>`);
  if (values.rocks) extras.push(`<circle cx="70" cy="36" r="3.5" fill="#6b7280"/><path d="M58 40 l8 -3" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="2 2"/>`);
  const ice = values.ground_grip < 40 ? `<path d="M0 66 L120 66" stroke="#fff" stroke-width="2" opacity="0.8"/>` : "";
  return `<svg class="world-art" viewBox="0 0 120 80" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${style.sky[0]}"/><stop offset="1" stop-color="${style.sky[1]}"/></linearGradient></defs>
    <rect width="120" height="80" fill="url(#${id})"/>
    ${extras.join("")}
    <path d="${terrainPath(terrain, values.terrain_height)}" fill="${style.ground}" stroke="${style.dark}" stroke-width="1"/>
    ${ice}
  </svg>`;
}

export function logoMark() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 18c2-6 5-9 8-9"/><circle cx="15.5" cy="6.5" r="2.5" fill="currentColor"/><path d="M9 14l-2 6M13 13l1 7"/></svg>`;
}

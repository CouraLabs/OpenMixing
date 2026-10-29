import { st } from './store.js';
import { store } from './dom.js';

const THEMES = {
  neon: { low: [255, 106, 43], mid: [240, 79, 208], high: [60, 162, 255] },
  club: { low: [47, 107, 255], mid: [255, 162, 31], high: [242, 246, 255] },
  aurora: { low: [255, 61, 90], mid: [43, 227, 138], high: [60, 200, 255] },
  ember: { low: [150, 22, 40], mid: [214, 92, 28], high: [255, 196, 120] },
  abyss: { low: [24, 52, 150], mid: [20, 138, 150], high: [150, 190, 235] },
};
export const BAND = { low: [], mid: [], high: [] };

export function setTheme(n) {
  if (!THEMES[n]) n = 'neon';
  const r = document.documentElement.style;
  for (const k in BAND) { BAND[k] = THEMES[n][k]; r.setProperty('--' + k, `rgb(${BAND[k]})`) }
  document.querySelectorAll('.th').forEach(b => b.setAttribute('aria-pressed', b.dataset.th === n));
  store('om.theme', n);
  if (st.S) st.S.clips.forEach(c => c._ov = null);
}

export function mixRGB(l, m, h) {
  const a = Math.pow(l, 2.4), b = Math.pow(m, 2.4), c = Math.pow(h, 2.4), s = a + b + c;
  if (s < 1e-4) return null;
  const ch = i => (BAND.low[i] * a + BAND.mid[i] * b + BAND.high[i] * c) / s, rgb = [ch(0), ch(1), ch(2)], av = (rgb[0] + rgb[1] + rgb[2]) / 3;
  return rgb.map(v => Math.max(0, Math.min(255, av + (v - av) * 1.3)));
}

export function column(ctx, px, mid, amp, rgb, alpha) {
  const [r, g, b] = rgb, h = Math.max(amp * 2, 1), ca = amp * .5, lift = v => (v + (255 - v) * .3) | 0;
  ctx.fillStyle = `rgb(${r | 0} ${g | 0} ${b | 0}/${alpha})`; ctx.fillRect(px, mid - amp, 1, h);
  if (ca > 1) { ctx.fillStyle = `rgb(${lift(r)} ${lift(g)} ${lift(b)}/${alpha})`; ctx.fillRect(px, mid - ca, 1, ca * 2) }
}

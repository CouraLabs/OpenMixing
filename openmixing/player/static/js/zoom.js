import { $, au, dpr, MONO, SANS } from './dom.js';
import { bpmAt, eqGain, lane } from './automation.js';
import { barBeat, loopWraps, outBeat, passTxt, posOf, sizeTxt, srcBeat } from './position.js';
import { live, name, st } from './store.js';
import { column, mixRGB } from './theme.js';

export const zv = $('zv');
const zg = zv.getContext('2d'), RED = '#ff3b3b', LOOP = '#34d399', RULER = 26, SPANS = [4, 8, 16, 32];

export function pairAt(t) {
  const s = st.sorted, A = s.find(c => live(c, t)) || s.find(c => c.start > t) || s[s.length - 1] || null;
  return [A, A ? s[s.indexOf(A) + 1] || null : null];
}

function wave(c, y, h, t, w) {
  const wf = c.wf, B = wf.b, n = wf.a.length, mid = y + h / 2, at = (k, f, j) => wf[k][j] * (1 - f) + wf[k][Math.min(j + 1, n - 1)] * f;
  for (let px = 0; px < w; px++) {
    const tt = t + (px - w / 2) / w * st.zoomSec, x = (tt - c.start) / B - .5, j = Math.floor(x);
    if (j < -1 || j >= n) continue;
    const f = x - j, j0 = Math.max(j, 0), l = at('l', f, j0) / 100, m = at('m', f, j0) / 100, hi = at('h', f, j0) / 100, raw = l + m + hi || 1;
    const v = Math.min(Math.max(lane(c, 'volume', tt), 0), 1.2), gl = eqGain(lane(c, 'eq_low', tt)), gm = eqGain(lane(c, 'eq_mid', tt)), gh = eqGain(lane(c, 'eq_high', tt));
    const rgb = mixRGB(l * gl, m * gm, hi * gh); if (!rgb) continue;
    const amp = Math.min(at('a', f, j0) / 100, 1) * (h / 2 * .8) * v * Math.min((l * gl + m * gm + hi * gh) / raw, 1);
    column(zg, px, mid, amp, rgb, (px < w / 2 ? .55 : 1) * Math.min(.3 + .7 * v, 1));
  }
}

function tick(px, y, h, color, flag) {
  const X = Math.round(px);
  zg.fillStyle = '#00000026'; zg.fillRect(X - 1, y, 3, h); zg.fillStyle = color; zg.fillRect(X, y, 1, h);
  if (flag) { zg.fillStyle = flag; zg.beginPath(); zg.moveTo(px - 6, y); zg.lineTo(px + 7, y); zg.lineTo(px + .5, y + 8); zg.closePath(); zg.fill() }
}

function grid(c, y, h, t, w, X) {
  const Z = st.zoomSec, k0 = Math.max(Math.floor(outBeat(c, t - Z / 2)), 0), k1 = Math.min(Math.ceil(outBeat(c, t + Z / 2)), c.beats.length - 1);
  zg.font = '9px ' + MONO; zg.textBaseline = 'alphabetic';
  for (let k = k0; k <= k1; k++) {
    const px = X(c.beats[k]), s = srcBeat(c, k), p = barBeat(s), bar = p.beat === 1 && Math.abs(s - Math.round(s)) < 1e-3;
    if (px < -20 || px > w + 20) continue;
    tick(px, y, h, bar ? '#ff3b3b99' : '#ffffff40', bar && RED);
    const room = ((c.beats[k + 1] ?? c.beats[k]) - c.beats[k]) / Z * w >= 26;
    if (bar || room) { zg.fillStyle = bar ? RED : '#ffffffb3'; zg.fillText(`${p.bar}.${p.beat}`, px + (bar ? 10 : 3), y + 9) }
  }
}

function loops(c, y, h, t, w, X) {
  zg.font = '600 10px ' + MONO; zg.textBaseline = 'alphabetic';
  c.loops.forEach(l => {
    const x0 = X(l.start), x1 = X(l.end);
    if (x1 < 0 || x0 > w) return;
    zg.fillStyle = LOOP + '1a'; zg.fillRect(x0, y, x1 - x0, h);
    zg.fillStyle = LOOP; zg.fillRect(x0, y + h - 3, x1 - x0, 3); zg.fillRect(Math.round(x0), y, 2, h); zg.fillRect(Math.round(x1) - 1, y, 2, h);
    zg.fillText(`${l.roll ? 'ROLL' : 'LOOP'} ${sizeTxt(l.size_beats)}`, x0 + 4, y + h - 8);
    loopWraps(c, l).forEach((tw, j) => {
      const px = X(tw); zg.fillStyle = LOOP; zg.fillRect(Math.round(px) - 1, y + 12, 2, h - 12);
      zg.fillText(`↺ ${j + 2}/${l.passes}`, px + 4, y + 22);
    });
  });
}

function label(c, y, h, t) {
  const tt = Math.min(Math.max(t, c.start), c.end), p = posOf(c, tt), title = `${c.n}  ${name(c)}`;
  const info = `${bpmAt(c, tt).toFixed(1)} bpm · ${c.camelot} · ${p.bar}.${p.beat}` + (p.loop ? '  ' + passTxt(p.loop) : '');
  zg.fillStyle = c.color; zg.fillRect(0, y, 4, h);
  zg.font = '600 12px ' + SANS; const tw = Math.max(zg.measureText(title).width, zg.measureText(info).width * 1.05);
  zg.fillStyle = '#00000099'; zg.fillRect(8, y + h - 34, tw + 14, 30);
  zg.fillStyle = '#fff'; zg.fillText(title, 14, y + h - 20);
  zg.font = '11px ' + MONO; zg.fillStyle = p.loop ? LOOP : '#ffffffb8'; zg.fillText(info, 14, y + h - 7);
}

function drawLane(c, y, h, t, w) {
  zg.fillStyle = '#07080c'; zg.fillRect(0, y, w, h);
  if (!c || !c.wf) { zg.fillStyle = '#ffffff26'; zg.fillRect(0, y + h / 2, w, 1); return }
  const X = tt => (tt - t) / st.zoomSec * w + w / 2;
  loops(c, y, h, t, w, X); wave(c, y, h, t, w); grid(c, y, h, t, w, X); label(c, y, h, t);
}

function ruler(c, y, t, w) {
  zg.fillStyle = '#07080c'; zg.fillRect(0, y, w, RULER);
  if (!c) return;
  const x0 = w / 2, bw = 60 / bpmAt(c, Math.min(Math.max(t, c.start), c.end)) / st.zoomSec * w, rh = RULER / SPANS.length;
  zg.font = '8px ' + MONO; zg.textBaseline = 'middle';
  SPANS.forEach((n, i) => {
    const x1 = x0 + n * bw, ry = y + i * rh + 1;
    zg.fillStyle = '#ffffff1f'; zg.fillRect(x0, ry, Math.min(x1, w) - x0, rh - 2);
    zg.fillStyle = '#ffffff8c'; zg.fillRect(Math.round(x1) - 1, ry, 2, rh - 2);
    zg.textAlign = 'right'; zg.fillStyle = '#ffffffb3'; zg.fillText(`${n} beats`, Math.min(x1, w) - 6, ry + rh / 2 - .5); zg.textAlign = 'left';
  });
}

export function drawZoom() {
  if (!st.S) return;
  const w = zv.clientWidth, h = zv.clientHeight, lh = (h - 6 - RULER) / 2;
  if (zv.width !== Math.round(w * dpr) || zv.height !== h * dpr) { zv.width = w * dpr; zv.height = h * dpr }
  zg.setTransform(dpr, 0, 0, dpr, 0, 0); zg.clearRect(0, 0, w, h);
  const t = au.currentTime, [A, B] = pairAt(t);
  drawLane(A, 0, lh, t, w); drawLane(B, lh + 6, lh, t, w); ruler(A, 2 * lh + 6, t, w);
  zg.fillStyle = '#8f0f1a'; zg.fillRect(w / 2 - 1, 0, 2, h);
}

export function initZoom() {
  zv.onclick = e => { if (st.S) au.currentTime += (e.offsetX / zv.clientWidth - .5) * st.zoomSec };
  zv.addEventListener('wheel', e => { e.preventDefault(); st.zoomSec = Math.min(Math.max(st.zoomSec * (e.deltaY > 0 ? 1.15 : 1 / 1.15), 6), 60) }, { passive: false });
}

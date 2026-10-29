import { $, au, dpr, fmt, MONO, SANS } from './dom.js';
import { val } from './automation.js';
import { loopWraps } from './position.js';
import { name, st } from './store.js';
import { BAND, column, mixRGB } from './theme.js';

export const wrap = $('tlwrap'), cv = $('tl');
const g = cv.getContext('2d'), ROW = 76, TOP = 80, LOOP = '#34d399';

export function layoutRows() {
  const ends = [];
  [...st.S.clips].sort((a, b) => a.start - b.start).forEach(c => {
    let r = ends.findIndex(e => e <= c.start + 0.01);
    if (r < 0) { r = ends.length; ends.push(0) }
    ends[r] = c.end; c.row = r; c.color = '#c9cfdb'; c.n = st.S.clips.indexOf(c) + 1;
  });
  st.rows = ends.length;
}

function size() {
  const vw = wrap.clientWidth, h = Math.max(170, TOP + st.rows * ROW + 26), w = Math.max(vw, Math.min(Math.round(st.S.duration_sec * st.pxPerSec), 12000));
  if (cv.style.width !== w + 'px') cv.style.width = w + 'px';
  if (cv.style.height !== h + 'px') cv.style.height = h + 'px';
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (cv.width !== W) cv.width = W;
  if (cv.height !== H) cv.height = H;
  return [w, h, vw];
}

function grid(x, w, h, D) {
  const step = [5, 10, 15, 30, 60, 120, 300, 600].find(s => s * w / D >= 80) || 600;
  g.font = '11px ' + MONO;
  for (let t = 0; t <= D; t += step) { g.fillStyle = '#ffffff0d'; g.fillRect(Math.round(x(t)), 0, 1, h - 18); g.fillStyle = '#a9b0bd'; g.fillText(fmt(t), x(t) + (t ? 4 : 6), h - 4) }
  st.trans.forEach(tr => { g.fillStyle = '#ffffff12'; g.fillRect(x(tr.start), 0, Math.max(x(tr.end) - x(tr.start), 2), h - 18) });
}

function overview(w, vw, ph) {
  const pk = st.S.peaks || [], e = st.S.bands, sl = wrap.scrollLeft, mid = 34;
  for (let px = Math.max(0, Math.floor(sl) - 2); px < Math.min(w, sl + vw + 2); px++) {
    const i = Math.floor(px / w * pk.length), v = pk[i] || 0, rgb = e ? mixRGB(e.low[i] || 0, e.mid[i] || 0, e.high[i] || 0) : null;
    if (rgb) column(g, px, mid, Math.max(v * 30, .5), rgb, px < ph ? 1 : .5);
    else { g.fillStyle = px < ph ? '#8fa3c4' : '#3a4659'; g.fillRect(px, mid - v * 30, 1, Math.max(v * 60, 1)) }
  }
}

function lowKills(c, X, y, Wd) {
  const lk = c.lanes.eq_low; if (!lk) return;
  g.fillStyle = `rgb(${BAND.low}/.6)`;
  const n = Math.max(Math.floor(Wd), 1); let on = false, s0 = 0;
  for (let p = 0; p <= n; p++) {
    const k = val(lk, c.start + (c.end - c.start) * p / n, 0) < -30;
    if (k && !on) { on = true; s0 = p }
    if ((!k || p == n) && on) { on = false; g.fillRect(X + s0, y + 57, p - s0, 4) }
  }
}

function fader(c, X, y, Wd) {
  g.strokeStyle = '#ffffffb0'; g.lineWidth = 1.25; g.lineJoin = 'round'; g.beginPath();
  const n = Math.max(Math.floor(Wd / 2), 2);
  for (let p = 0; p <= n; p++) {
    const v = Math.min(val(c.lanes.volume, c.start + (c.end - c.start) * p / n, 1), 1.2), px = X + Wd * p / n, py = y + 54 - v * 16;
    p ? g.lineTo(px, py) : g.moveTo(px, py);
  }
  g.stroke();
}

function loops(c, x, y) {
  c.loops.forEach(l => {
    const x0 = x(l.start), x1 = x(l.end);
    g.fillStyle = LOOP + '1f'; g.fillRect(x0, y, Math.max(x1 - x0, 2), 68);
    g.fillStyle = LOOP; g.fillRect(x0, y, Math.max(x1 - x0, 2), 4);
    loopWraps(c, l).forEach(t => g.fillRect(Math.round(x(t)), y, 1, 12));
  });
}

function clipBox(c, x, sl) {
  const y = TOP + c.row * ROW, X = x(c.start), Wd = Math.max(x(c.end) - X, 2), on = au.currentTime >= c.start && au.currentTime < c.end;
  g.beginPath(); g.roundRect(X, y, Wd, 68, 8); g.fillStyle = c.color + (on ? '33' : '1f'); g.fill(); g.strokeStyle = c.color + (on ? 'aa' : '55'); g.lineWidth = 1; g.stroke();
  g.save(); g.beginPath(); g.roundRect(X, y, Wd, 68, 8); g.clip();
  lowKills(c, X, y, Wd); fader(c, X, y, Wd); loops(c, x, y);
  c.fx.forEach(f => { g.fillStyle = '#a78bfa'; g.fillRect(x(f.start), y + 62, Math.max(x(f.end) - x(f.start), 2), 4) });
  g.fillStyle = '#fff'; g.font = '600 12.5px ' + SANS; g.fillText(c.n + '  ' + name(c), Math.max(X, sl) + 10, y + 20);
  g.fillStyle = '#ffffffb8'; g.font = '11px ' + MONO; g.fillText(`${c.native_bpm} bpm · ${c.camelot}`, Math.max(X, sl) + 10, y + 35);
  g.restore();
}

function flagMarks(x, h) {
  st.flags.forEach((f, i) => {
    const X = Math.round(x(f.t));
    g.fillStyle = '#ffe14d'; g.fillRect(X, 0, 1, h - 18); g.beginPath(); g.moveTo(X, 0); g.lineTo(X + 12, 5); g.lineTo(X, 10); g.fill();
    g.fillStyle = '#000'; g.font = '600 8px ' + MONO; g.fillText(i + 1, X + 2, 8);
  });
}

export function draw() {
  if (!st.S || wrap.hidden) return;
  const [w, h, vw] = size(), D = st.S.duration_sec, x = t => (t - (st.S.start_sec || 0)) / D * w, ph = x(au.currentTime);
  g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.scale(dpr, dpr);
  grid(x, w, h, D); overview(w, vw, ph);
  st.S.clips.forEach(c => clipBox(c, x, wrap.scrollLeft));
  flagMarks(x, h);
  g.fillStyle = '#fff'; g.fillRect(ph - .75, 0, 1.5, h - 18); g.beginPath(); g.arc(ph, 4, 4, 0, 7); g.fill();
}

export function follow(t) {
  if (wrap.hidden) return;
  const px = t / st.S.duration_sec * cv.clientWidth, sl = wrap.scrollLeft, vw = wrap.clientWidth;
  if (px < sl || px > sl + vw || (!au.paused && Date.now() > st.userScrollUntil && px > sl + vw * .8)) wrap.scrollLeft = Math.max(px - vw * .2, 0);
}

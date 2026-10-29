import { $, cam, dpr, esc, fmt, seek } from './dom.js';
import { bpmAt, filtTxt, lane } from './automation.js';
import { passTxt, posOf, sizeTxt } from './position.js';
import { live, name, st } from './store.js';
import { column, mixRGB } from './theme.js';
import { pairAt } from './zoom.js';

const PHRASE_H = 22;
const beatsTxt = b => `${sizeTxt(b)} ${b > 1 ? 'beats' : 'beat'}`;

function pads(c) {
  const p = [{ k: '', n: 'Start', t: c.start, s: c.start, e: c.start + 2 }];
  c.loops.forEach(l => p.push({ k: 'loop', n: `${l.roll ? 'Roll' : 'Loop'} ${sizeTxt(l.size_beats)} ×${l.passes}`, t: l.start, s: l.start, e: l.end }));
  c.fx.forEach(f => p.push({ k: 'fx', n: f.fx, t: f.start, s: f.start, e: f.end }));
  p.push({ k: '', n: 'Last 8 s', t: Math.max(c.end - 8, c.start), s: c.end - 1e9, e: -1 });
  return p.slice(0, 8);
}

const deckHTML = c => `<div class="deck off enter" style="--c:${c.color}">
 <div class="dh"><span class="idx">${c.n}</span><div class="who"><h2>${esc(name(c))}</h2><div>${esc(c.artist || '')}</div></div><span class="key" style="color:${cam(c.camelot)}">${esc(c.camelot)}</span></div>
 <canvas class="wv" aria-label="Track waveform and position"></canvas>
 <div class="read"><span class="bpmv"><span class="bpm"></span><small>BPM</small></span><span class="tt"><b class="el"></b><br><span class="rem"></span> · bar <span class="pos"></span></span></div>
 <div class="pads">${pads(c).map(p => `<button class="pad ${p.k}" data-s="${p.s}" data-e="${p.e}" data-t="${p.t}" title="Jump to ${esc(p.n)}">${esc(p.n)}</button>`).join('')}</div></div>`;

const knob = b => `<div class="eq"><div class="tr eqt"><div class="fill" data-b="${b}" style="background:var(--${b})"></div></div><span class="v" data-v="${b}"></span>${b}</div>`;

const stripHTML = c => `<div class="sname">${c.n} · ${esc(name(c))}</div><div class="eqs">${['low', 'mid', 'high'].map(knob).join('')}
 <div class="eq"><div class="tr"><div class="fill" data-b="volume" style="background:#fff"></div></div><span class="v" data-v="volume"></span>fader</div></div>
 <div class="flt"><span>filter</span><span class="v" data-v="filter" title="LP = low-pass (muffles the highs), HP = high-pass (thins out the lows)"></span></div><div class="chips"></div>`;

function overview(c, w, h) {
  if (c._ov && c._ov.w === w) return c._ov.cv;
  const o = document.createElement('canvas'); o.width = w * dpr; o.height = h * dpr;
  const x = o.getContext('2d'); x.scale(dpr, dpr);
  const wf = c.wf;
  if (wf) {
    const n = wf.a.length;
    for (let px = 0; px < w; px++) {
      const i0 = Math.floor(px / w * n), i1 = Math.max(Math.floor((px + 1) / w * n), i0 + 1); let a = 0, l = 0, m = 0, hi = 0;
      for (let i = i0; i < i1 && i < n; i++) { a = Math.max(a, wf.a[i]); l += wf.l[i]; m += wf.m[i]; hi += wf.h[i] }
      const k = 100 * (i1 - i0), rgb = mixRGB(l / k, m / k, hi / k);
      if (rgb) column(x, px, h, Math.min(a / 100, 1) * (h - 3), rgb, 1);
    }
  }
  c._ov = { w, cv: o };
  return o;
}

function drawDeckWave(el, c, t) {
  const cvs = el.querySelector('.wv'), w = cvs.clientWidth, h = 60, H = h + 4 + PHRASE_H; if (!w) return;
  if (cvs.width !== Math.round(w * dpr) || cvs.height !== H * dpr) { cvs.width = Math.round(w * dpr); cvs.height = H * dpr }
  const x = cvs.getContext('2d'), ov = overview(c, w, h), p = Math.min(Math.max((t - c.start) / (c.end - c.start), 0), 1) * w;
  x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, w, H);
  x.globalAlpha = .4; x.drawImage(ov, 0, 0, w * dpr, h * dpr, 0, 0, w, h); x.globalAlpha = 1;
  if (p > 0) x.drawImage(ov, 0, 0, p * dpr, h * dpr, 0, 0, p, h);
  x.fillStyle = '#fff'; x.fillRect(Math.min(p, w - 1.5), 0, 1.5, h);
  drawPhrases(x, c, w, h, p);
}

const PHRASE = { intro: ['#d63a32', 'INTRO'], up: ['#8a47f0', 'UP'], chorus: ['#3fae3a', 'CHORUS'], down: ['#a8803f', 'DOWN'], outro: ['#2f5fd0', 'OUTRO'] };

function drawPhrases(x, c, w, h, p) {
  const top = h + 4, hh = PHRASE_H, span = c.end - c.start;
  x.font = '600 11px system-ui, sans-serif'; x.textBaseline = 'middle';
  (c.phrases || []).forEach(ph => {
    const a = (ph.start - c.start) / span * w, b = (ph.end - c.start) / span * w, [col, txt] = PHRASE[ph.kind];
    x.globalAlpha = b <= p ? 1 : .55; x.fillStyle = col; x.fillRect(a + .5, top, Math.max(b - a - 1, 1), hh);
    x.globalAlpha = 1; x.fillStyle = '#fff';
    const label = x.measureText(txt).width + 8 <= b - a ? txt : x.measureText(txt[0]).width + 8 <= b - a ? txt[0] : '';
    if (label) x.fillText(label, a + 5, top + hh / 2 + .5);
  });
}

function updateDeck(el, c, t) {
  const on = live(c, t), tt = Math.min(Math.max(t, c.start), c.end), p = posOf(c, t);
  el.classList.toggle('off', !on); el.classList.toggle('ending', on && c.end - t <= 20);
  el.querySelector('.bpm').textContent = bpmAt(c, tt).toFixed(1);
  el.querySelector('.el').textContent = fmt(tt - c.start); el.querySelector('.rem').textContent = '-' + fmt(c.end - tt);
  el.querySelectorAll('.pad[data-s]').forEach(b => b.classList.toggle('on', on && t >= +b.dataset.s && t < +b.dataset.e));
  const pos = `${p.bar}.${p.beat}` + (p.loop ? `<span class="lp">${passTxt(p.loop)}</span>` : '');
  const pe = el.querySelector('.pos'); if (pe.innerHTML !== pos) pe.innerHTML = pos;
  drawDeckWave(el, c, t);
}

function updateStrip(el, c, t) {
  const on = live(c, t), tt = Math.min(Math.max(t, c.start), c.end), q = k => lane(c, k, tt);
  el.classList.toggle('off', !on);
  ['low', 'mid', 'high'].forEach(b => {
    const v = q('eq_' + b), ve = el.querySelector(`[data-v=${b}]`);
    el.querySelector(`[data-b=${b}]`).style.height = Math.max((v + 40) / 46, 0.02) * 100 + '%';
    ve.classList.toggle('kill', v < -30); ve.title = v < -30 ? 'Band fully cut' : ''; ve.textContent = v < -30 ? 'KILL' : (v >= 0 ? '+' : '') + v.toFixed(0);
  });
  const vol = q('volume');
  el.querySelector('[data-b=volume]').style.height = Math.min(vol / 1.2, 1) * 100 + '%';
  el.querySelector('[data-v=volume]').textContent = Math.round(vol * 100) + '%';
  el.querySelector('[data-v=filter]').textContent = filtTxt(q('filter'));
  const p = posOf(c, t), chips = on ? c.fx.filter(x => t >= x.start && t < x.end).map(x => `<span class="chip fx">${x.fx} ${beatsTxt(x.beats)}</span>`) : [];
  if (on && p.loop) chips.push(`<span class="chip loop">${p.loop.roll ? 'roll' : 'loop'} ${beatsTxt(p.loop.size_beats)} ${passTxt(p.loop)}</span>`);
  const ch = el.querySelector('.chips'), html = chips.join(''); if (ch.innerHTML !== html) ch.innerHTML = html;
}

function mount(slot, strip, c) {
  const cid = c ? c.id : '';
  if (slot.dataset.c !== cid) {
    slot.dataset.c = cid; slot.innerHTML = c ? deckHTML(c) : '';
    slot.querySelectorAll('.pad[data-t]').forEach(b => b.onclick = () => seek(+b.dataset.t));
    const wv = slot.querySelector('.wv'); if (wv) wv.onclick = e => seek(c.start + e.offsetX / wv.clientWidth * (c.end - c.start));
  }
  if (strip.dataset.c !== cid) { strip.dataset.c = cid; strip.innerHTML = c ? stripHTML(c) : ''; strip.style.setProperty('--c', c ? c.color : '#888') }
}

export function updateDecks(t) {
  pairAt(t).forEach((c, k) => {
    const slot = $(k ? 'sR' : 'sL'), strip = $(k ? 'mR' : 'mL');
    mount(slot, strip, c);
    if (c) { updateDeck(slot.firstElementChild, c, t); updateStrip(strip, c, t) }
  });
}

export function buildList() {
  ['sL', 'sR', 'mL', 'mR'].forEach(id => $(id).dataset.c = '\0');
  $('rows').innerHTML = st.sorted.map(c => `<tr id="r_${c.id}" style="--c:${c.color}" data-t="${c.start}"><td class="n">${c.n}</td><td>${esc(name(c))}</td><td class="dim">${esc(c.artist || '')}</td><td class="n">${c.native_bpm}</td><td class="ky" style="color:${cam(c.camelot)}">${esc(c.camelot)}</td><td class="n">${c.energy}</td><td class="n">${fmt(c.start)}</td><td class="n">${fmt(c.end - c.start)}</td></tr>`).join('');
  $('rows').querySelectorAll('tr').forEach(r => { r.tabIndex = 0; r.onclick = () => seek(+r.dataset.t); r.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); r.click() } } });
}

export const $ = id => document.getElementById(id);
export const au = $('au'), dpr = devicePixelRatio || 1;
export const MONO = 'ui-monospace,Menlo,monospace', SANS = 'ui-sans-serif,-apple-system,system-ui,sans-serif';
export const esc = x => String(x ?? '').replace(/[&<>"]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[m]));
export const cam = k => `oklch(.8 .14 ${((parseInt(k) || 1) - 1) * 30 + 15})`;
export const fmt = t => { t = Math.max(t, 0); const m = Math.floor(t / 60); return m + ':' + String(Math.floor(t % 60)).padStart(2, '0') };
export const seek = t => { au.currentTime = t };
export const stored = (k, fallback) => { try { return localStorage.getItem(k) ?? fallback } catch (e) { return fallback } };
export const store = (k, v) => { try { localStorage.setItem(k, v) } catch (e) { } };

let timer;
export function status(msg, kind, ms) {
  clearTimeout(timer); const el = $('status');
  if (!msg) { el.hidden = true; el.dataset.k = ''; return }
  el.hidden = false; el.dataset.k = kind || ''; $('stxt').textContent = msg; $('retry').hidden = kind !== 'net';
  if (ms) timer = setTimeout(() => status(''), ms);
}

export const play = () => au.play().catch(e => { if (e.name !== 'AbortError') status('Playback was blocked or failed. Press Play to try again.', 'warn', 6000) });

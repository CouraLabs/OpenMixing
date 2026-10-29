import { $, au, fmt, status, store, stored } from './dom.js';
import { bpmAt, filtTxt, lane } from './automation.js';
import { posOf, sizeTxt } from './position.js';
import { live, name, st } from './store.js';

const plural = (n, w) => `${n} ${w}${n > 1 ? 's' : ''}`;
const key = () => 'om.flags.' + st.S.set + ':' + st.S.version;

function deckNote(c, t) {
  const p = posOf(c, t), q = k => lane(c, k, t);
  const eq = ['low', 'mid', 'high'].map(b => b + ' ' + (q('eq_' + b) < -30 ? 'KILL' : Math.round(q('eq_' + b)) + ' dB')).join(', ');
  const fx = c.fx.filter(x => t >= x.start && t < x.end).map(x => 'fx ' + x.fx);
  const loop = p.loop ? [`${p.loop.roll ? 'roll' : 'loop'} ${sizeTxt(p.loop.size_beats)} beats pass ${p.loop.pass} of ${p.loop.passes}`] : [];
  const live = [...fx, ...loop];
  return `#${c.n} "${name(c)}" clip bar ${p.clipBar}, track ${p.bar}.${p.beat} (${bpmAt(c, t).toFixed(1)} bpm; EQ ${eq}; fader ${Math.round(q('volume') * 100)}%; filter ${filtTxt(q('filter'))}${live.length ? '; ' + live.join(', ') : ''})`;
}

export const describe = t => `${st.S.set} @ ${fmt(t)}: ` + (st.sorted.filter(c => live(c, t)).map(c => deckNote(c, t)).join(' × ') || 'between tracks');

function update() {
  const n = st.flags.length;
  $('copyflags').hidden = $('clrflags').hidden = !n;
  $('copyflags').textContent = 'Copy ' + plural(n, 'flag');
}

function save() { store(key(), JSON.stringify(st.flags)); update() }

export function readFlags() {
  try { st.flags = JSON.parse(stored(key(), '[]')) } catch (e) { st.flags = [] }
  update();
}

const copy = (txt, ok) => navigator.clipboard.writeText(txt).then(() => status(ok, 'ok', 5000), () => status('Couldn\'t copy to the clipboard. Use Copy flags to try again.', 'warn', 6000));

export function initFlags() {
  $('flagbtn').onclick = () => {
    if (!st.S) return;
    const f = { t: au.currentTime, text: describe(au.currentTime) };
    st.flags.push(f); save(); copy(f.text, `Flag ${st.flags.length} copied: ${f.text}`);
  };
  $('copyflags').onclick = () => copy(st.flags.map((f, i) => `${i + 1}. ${f.text}`).join('\n'), plural(st.flags.length, 'flag') + ' copied.');
  $('clrflags').onclick = () => { const n = st.flags.length; st.flags = []; save(); status(plural(n, 'flag') + ' cleared.', 'ok', 3000) };
}

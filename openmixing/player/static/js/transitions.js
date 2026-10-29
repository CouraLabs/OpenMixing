import { $, au, seek } from './dom.js';
import { st } from './store.js';

const preRoll = tr => 8 * 4 * 60 / (tr.a.native_bpm || 120);
const tStart = tr => Math.max(tr.start - preRoll(tr), tr.a.start, st.S.start_sec || 0);

export function buildTrans() {
  const s = st.sorted;
  st.trans = s.slice(0, -1).map((a, i) => ({ start: s[i + 1].start, end: a.end, a, b: s[i + 1] })).filter(x => x.start < x.end);
  $('prevT').disabled = $('nextT').disabled = !st.trans.length;
  $('trn').textContent = st.trans.length ? '' : 'No transitions';
}

export function updTrans(t) {
  const tr = st.trans; if (!tr.length) return;
  const i = tr.findIndex(x => t < x.end);
  $('trn').textContent = i < 0 ? tr.length + ' transitions' : (t >= tr[i].start ? 'In ' : 'Next: ') + (i + 1) + ' of ' + tr.length;
}

export function jumpTrans(d) {
  const t = au.currentTime, r = d > 0 ? st.trans.find(x => tStart(x) > t + .5) : [...st.trans].reverse().find(x => tStart(x) < t - 1.5);
  if (r) seek(tStart(r));
}

export function initTrans() {
  $('nextT').onclick = () => jumpTrans(1);
  $('prevT').onclick = () => jumpTrans(-1);
}

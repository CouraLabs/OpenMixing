import { $, au, esc, fmt, play, seek, status, store, stored } from './dom.js';
import { buildList, updateDecks } from './decks.js';
import { initFlags, readFlags } from './flags.js';
import { live, name, st } from './store.js';
import { setTheme } from './theme.js';
import { cv, draw, follow, layoutRows, wrap } from './timeline.js';
import { buildTrans, initTrans, jumpTrans, updTrans } from './transitions.js';
import { drawZoom, initZoom } from './zoom.js';

let stamp = '';
async function load(set) {
  let r;
  const q = set ? '?set=' + encodeURIComponent(set) : '';
  try { r = await fetch('/api/version' + q) }
  catch (e) { status('Can\'t reach the player server. Check that it is still running; retrying every 3 s.', 'net'); return }
  if (!r.ok) { status('No rendered set yet. Ask your agent to render one and it will appear here.', 'info'); return }
  if (['net', 'info'].includes($('status').dataset.k)) status('');
  const s = (await r.json()).stamp;
  if (s === stamp) return;
  r = await fetch('/api/state' + q);
  if (!r.ok) return;
  stamp = s;
  const j = await r.json(), key = j.set + ':' + j.version + ':' + j.duration_sec;
  if (key === st.version) return;
  const had = !!st.S, keep = st.S && j.set === st.S.set ? au.currentTime : 0, was = !au.paused;
  st.version = key; st.S = j;
  if (had) status('New render loaded.', 'ok', 3000);
  layoutRows(); $('title').textContent = j.set; st.sorted = [...j.clips].sort((a, b) => a.start - b.start);
  $('stats').innerHTML = [`${fmt(j.duration_sec)} long`, j.integrated_lufs != null && `${j.integrated_lufs} LUFS`, j.true_peak_dbtp != null && `${j.true_peak_dbtp} dBTP`, `${j.clips.length} tracks`]
    .filter(Boolean).map(x => `<span>${x}</span>`).join('');
  au.src = '/audio/' + encodeURIComponent(j.audio) + '?v=' + encodeURIComponent(key); au.currentTime = keep;
  if (was) play();
  buildList(); buildTrans(); readFlags(); draw();
}

async function sets() {
  let l; try { l = await (await fetch('/api/sets')).json() } catch (e) { return }
  const cur = $('sets').value;
  $('sets').innerHTML = l.map(n => `<option ${n === cur ? 'selected' : ''}>${esc(n)}</option>`).join('');
}

let lastSec = -1;
function aria(t) {
  const s = Math.floor(t); if (lastSec === s) return; lastSec = s;
  cv.setAttribute('aria-valuenow', s); cv.setAttribute('aria-valuemax', Math.floor(st.S.duration_sec)); cv.setAttribute('aria-valuetext', fmt(t) + ' of ' + fmt(st.S.duration_sec));
}

function tick() {
  if (st.S) {
    const t = au.currentTime;
    $('time').innerHTML = fmt(t) + ' <span>/ ' + fmt(st.S.duration_sec) + '</span>';
    updateDecks(t);
    st.sorted.forEach(c => $('r_' + c.id)?.classList.toggle('live', live(c, t)));
    $('now').textContent = st.sorted.filter(c => live(c, t)).map(name).join('  ×  ');
    updTrans(t); aria(t);
    const ps = au.paused ? 'play' : 'pause';
    if ($('pp').dataset.s !== ps) { $('pp').dataset.s = ps; $('pp').setAttribute('aria-label', au.paused ? 'Play' : 'Pause') }
    follow(t); draw(); drawZoom();
  }
  requestAnimationFrame(tick);
}

function mixPanel(on) { wrap.hidden = !on; $('mixbtn').setAttribute('aria-pressed', on); $('mixst').textContent = on ? 'on' : 'off'; store('om.mix', on ? '1' : '0') }

function keys(e) {
  if (e.target.tagName === 'SELECT') return;
  if (e.code === 'Space' && !e.target.matches('button')) { e.preventDefault(); $('pp').click() }
  if (!e.ctrlKey && !e.metaKey && !e.altKey) {
    if (e.key === '[') jumpTrans(-1);
    else if (e.key === ']') jumpTrans(1);
    else if ((e.key === 'f' || e.key === 'F') && !e.target.matches('button')) $('flagbtn').click();
  }
  if (e.code === 'ArrowRight') au.currentTime += e.shiftKey ? 30 : 5;
  if (e.code === 'ArrowLeft') au.currentTime -= e.shiftKey ? 30 : 5;
}

au.addEventListener('error', () => status('Couldn\'t load the audio for this render. Ask your agent to render again, or reload.', 'warn'));
$('retry').onclick = () => load($('sets').value || undefined);
$('pp').onclick = () => au.paused ? play() : au.pause();
['wheel', 'pointerdown'].forEach(ev => wrap.addEventListener(ev, () => { st.userScrollUntil = Date.now() + 4000 }, { passive: true }));
document.querySelectorAll('.th').forEach(b => b.onclick = () => setTheme(b.dataset.th));
$('mixbtn').onclick = () => mixPanel(wrap.hidden);
cv.onclick = e => { if (st.S) seek(e.offsetX / cv.clientWidth * st.S.duration_sec + (st.S.start_sec || 0)) };
$('sets').onchange = () => { st.version = null; load($('sets').value) };
addEventListener('keydown', keys);
addEventListener('resize', draw);
setTheme(stored('om.theme', 'neon'));
mixPanel(stored('om.mix', '1') !== '0');
initTrans(); initFlags(); initZoom();
sets(); load();
setInterval(() => { if (!document.hidden) { sets(); load($('sets').value || undefined) } }, 3000);
requestAnimationFrame(tick);

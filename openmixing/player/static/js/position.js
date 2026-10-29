const bisect = (a, t) => { let lo = 0, hi = a.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; a[m] <= t ? lo = m : hi = m } return lo };

export function outBeat(c, t) {
  const b = c.beats, n = b.length - 1;
  if (t <= b[0]) return 0;
  if (t >= b[n]) return c.length_beats;
  const i = bisect(b, t);
  return Math.min(i + (t - b[i]) / (b[i + 1] - b[i]), c.length_beats);
}

export function beatTime(c, o) {
  const b = c.beats, i = Math.max(0, Math.min(Math.floor(o), b.length - 2));
  return b[i] + (b[i + 1] - b[i]) * (o - i);
}

export const segAt = (c, o) => c.segs.find(g => o >= g[0] && o < g[1]) || c.segs[c.segs.length - 1];
export const srcBeat = (c, o) => { const g = segAt(c, o); return g[2] + (g[3] ? (o - g[0]) % g[3] : o - g[0]) };
export const barBeat = src => { const s = Math.max(src, 0) + 1e-6; return { bar: Math.floor(s / 4) + 1, beat: Math.floor(s % 4) + 1 } };
export const sizeTxt = s => s < 1 ? '1/' + Math.round(1 / s) : String(s);
export const passTxt = l => `↺ ${l.pass}/${l.passes}`;

export function loopAt(c, o) {
  const l = c.loops.find(l => o >= l.beat && o < l.beat + l.length_beats);
  return l && { ...l, pass: Math.floor((o - l.beat) / l.size_beats) + 1 };
}

export function posOf(c, t) {
  const o = outBeat(c, Math.min(Math.max(t, c.start), c.end));
  return { ...barBeat(srcBeat(c, o)), clipBar: Math.floor(o / 4) + 1, loop: loopAt(c, o) };
}

export function loopWraps(c, l) {
  const out = [];
  for (let j = 1; j < l.passes; j++) out.push(beatTime(c, l.beat + j * l.size_beats));
  return out;
}

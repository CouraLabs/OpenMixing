export const NEUTRAL = { eq_low: 0, eq_mid: 0, eq_high: 0, filter: 0, volume: 1 };

export function val(pts, t, neutral) {
  if (!pts || !pts.length) return neutral;
  if (t <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i][0]) { const [a, x] = pts[i - 1], [b, y] = pts[i]; return b == a ? y : x + (y - x) * (t - a) / (b - a) }
  }
  return pts[pts.length - 1][1];
}

export const lane = (c, k, t) => val(c.lanes[k], t, NEUTRAL[k]);
export const bpmAt = (c, t) => val(c.tempo, t, c.native_bpm);
export const eqGain = db => db <= -30 ? 0 : Math.min(Math.pow(10, db / 20), 1.5);
export const filtTxt = f => Math.abs(f) < .02 ? 'off' : (f < 0 ? 'LP ' : 'HP ') + Math.round(Math.abs(f) * 100) + '%';

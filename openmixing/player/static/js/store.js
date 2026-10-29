export const st = { S: null, version: null, sorted: [], trans: [], flags: [], rows: 0, zoomSec: 20, pxPerSec: 3, userScrollUntil: 0 };
export const live = (c, t) => t >= c.start && t < c.end;
export const name = c => c.title || c.label;

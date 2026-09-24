// Lyrics text -> timed lines. Synced lyrics come as LRC ("[01:02.34]words"); plain lyrics have no
// times, so their lines are spread over the stretches where the analysis hears vocals.

import type { LyricLine, LyricTrack } from './types';

const STAMP = /\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]/g;

/** Seconds of the last line when nothing follows it. */
const LAST_LINE = 6;
/** A line never lasts longer than this (a long instrumental gap after it is not part of it). */
const MAX_LINE = 12;

/** LRC text -> lines sorted by time (blank stamped lines end the line before them). */
export function parseLrc(text: string, duration?: number): LyricTrack {
  let offset = 0;
  const marks: { t: number; text: string }[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    const off = /^\[offset:\s*([+-]?\d+)\s*\]$/i.exec(line);
    if (off) {
      // Positive offset: lyrics appear sooner.
      offset = parseInt(off[1], 10) / 1000;
      continue;
    }
    STAMP.lastIndex = 0;
    const times: number[] = [];
    let m: RegExpExecArray | null;
    let rest = 0;
    while ((m = STAMP.exec(line)) && m.index === rest) {
      const frac = m[3] ? parseInt(m[3], 10) / 10 ** m[3].length : 0;
      times.push(parseInt(m[1], 10) * 60 + parseInt(m[2], 10) + frac);
      rest = STAMP.lastIndex;
    }
    if (!times.length) continue;
    // Enhanced LRC word stamps (<00:12.34>) are dropped.
    const words = line.slice(rest).replace(/<\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?>/g, '').replace(/\s+/g, ' ').trim();
    for (const t of times) marks.push({ t, text: words });
  }
  marks.sort((a, b) => a.t - b.t);
  const lines: LyricLine[] = [];
  for (let i = 0; i < marks.length; i++) {
    const mk = marks[i];
    if (!mk.text) continue;
    const t = Math.max(0, mk.t - offset);
    const next = i + 1 < marks.length ? Math.max(0, marks[i + 1].t - offset) : Math.min(t + LAST_LINE, duration ?? Infinity);
    lines.push({ t, end: Math.max(t + 0.2, Math.min(next, t + MAX_LINE)), text: mk.text });
  }
  return { lines, synced: true };
}

/** Stretches of vocals [start, end) in seconds from a per-frame presence envelope. */
export function vocalRegions(presence: ArrayLike<number>, frameRate: number, thr = 0.12): [number, number][] {
  const out: [number, number][] = [];
  const n = presence.length;
  if (!n || !(frameRate > 0)) return out;
  // Smooth over ~0.5 s so single dips do not split a phrase.
  const w = Math.max(1, Math.round(frameRate * 0.25));
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + (presence[i] ?? 0);
  const sm = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - w);
    const hi = Math.min(n, i + w + 1);
    sm[i] = (pre[hi] - pre[lo]) / (hi - lo);
  }
  let start = -1;
  for (let i = 0; i <= n; i++) {
    const on = i < n && sm[i] > thr;
    if (on && start < 0) start = i;
    else if (!on && start >= 0) {
      out.push([start / frameRate, i / frameRate]);
      start = -1;
    }
  }
  // Merge short gaps, drop short blips.
  const merged: [number, number][] = [];
  for (const r of out) {
    const last = merged[merged.length - 1];
    if (last && r[0] - last[1] < 1.5) last[1] = r[1];
    else merged.push([r[0], r[1]]);
  }
  return merged.filter((r) => r[1] - r[0] >= 1);
}

/**
 * Plain lyrics spread over the vocal stretches: each line gets time in proportion to its length,
 * laid along the vocals only (gaps between stretches are skipped). Without any vocals the lines
 * cover 5%..95% of the song.
 */
export function spreadPlain(text: string, duration: number, regions: [number, number][]): LyricTrack {
  const texts = text.split(/\r?\n/).map((s) => s.replace(/\s+/g, ' ').trim()).filter(Boolean);
  if (!texts.length || !(duration > 0)) return { lines: [], synced: false };
  let regs = regions.filter((r) => r[1] > r[0] && r[0] < duration).map((r) => [Math.max(0, r[0]), Math.min(duration, r[1])] as [number, number]);
  if (!regs.length) regs = [[duration * 0.05, duration * 0.95]];
  const total = regs.reduce((s, r) => s + r[1] - r[0], 0);
  const weights = texts.map((s) => 8 + s.length);
  const wsum = weights.reduce((a, b) => a + b, 0);
  // Lay the lines along the vocals; a line that would mostly fall past the end of a stretch moves to
  // the next one. When that pushes the lines past the last stretch, shrink them all and lay again.
  const layout = (scale: number): { lines: LyricLine[]; used: number } => {
    const lines: LyricLine[] = [];
    let ri = 0;
    let t = regs[0][0];
    let used = 0;
    texts.forEach((s, i) => {
      const d = (weights[i] / wsum) * total * scale;
      while (ri < regs.length - 1 && regs[ri][1] - t < d * 0.5) {
        used += regs[ri][1] - t;
        ri++;
        t = regs[ri][0];
      }
      const end = Math.min(regs[ri][1], t + d);
      lines.push({ t, end: Math.max(t + 0.2, end), text: s });
      used += d;
      t = end;
    });
    return { lines, used };
  };
  let scale = 1;
  let res = layout(scale);
  for (let k = 0; k < 4 && res.used > total * 1.001; k++) {
    scale *= total / res.used;
    res = layout(scale);
  }
  const lines = res.lines;
  return { lines, synced: false };
}

/** Index of the line playing at `time` (-1 between lines or outside them); binary search. */
export function lineAt(track: LyricTrack, time: number): number {
  const L = track.lines;
  let lo = 0;
  let hi = L.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (L[mid].t <= time) {
      best = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return best >= 0 && time < L[best].end ? best : -1;
}

// Synced lyrics -> the audio's clock. LRC times are made for one release of a song; a music video
// rip or a remaster often starts a few seconds later, or runs a little slower, so the lines would
// show early or drift. The analysis hears where sung phrases start (the vocal stem's onsets); this
// finds the shift (and, if clearly better, a small speed change) that puts the line starts on them.
//
// Score of a candidate (offset, scale): the mean vocal-onset strength (z-scored over the song) in
// the first second of every line, with each line at t * scale + offset. Lines pushed outside the song
// count as zero, so a shift cannot win by dropping lines. Confidence comes from how far the best
// candidate stands above leaving the times alone (gain), above the best candidate more than 1.5 s
// away (prominence), and above the best fit to the song played backwards (significance: the chance
// level of the whole search), in standard deviations of the score over offsets.

import type { LyricLine, LyricTrack } from './types';

export interface AlignFeatures {
  /** Vocal onset strength per analysis frame (AnalysisResult.stemOnsets.vocals). */
  onsets: ArrayLike<number>;
  frameRate: number;
  duration: number;
}

export interface Alignment {
  /** Seconds added to every (scaled) line time; positive: the lyrics come later. */
  offset: number;
  /** Speed factor on the line times (1: none); a line at t moves to t * scale + offset. */
  scale: number;
  /** 0..1 how clearly the audio supports this alignment. */
  confidence: number;
  /** Whether the alignment was confident enough to use (else the times stay as they were). */
  applied: boolean;
  /** Best score above the unshifted times, in standard deviations of the offset curve. */
  gain: number;
  /** Best score above the best candidate more than 1.5 s away, in standard deviations. */
  prominence: number;
  /** Best score above the best fit to the song played backwards, in standard deviations. */
  significance: number;
}

export interface AlignOptions {
  /** Largest shift tried either way (seconds). */
  range?: number;
  /** Largest speed change tried either way (0.04: 4%). */
  stretch?: number;
  /** Confidence needed to apply. */
  minConfidence?: number;
}

const FPS = 50;
/** The window after a line start that should hold its first sung onset (seconds). */
const WINDOW = 1;
const STEP = 0.05;
const SCALE_STEP = 0.0025;
/** A speed change must beat the best plain shift by this many standard deviations. */
const STRETCH_MARGIN = 0.5;
/** Shifts below this are left alone (the LRC is already as close as the analysis can tell). */
const MIN_SHIFT = 0.1;
const NONE: Alignment = { offset: 0, scale: 1, confidence: 0, applied: false, gain: 0, prominence: 0, significance: 0 };

/** The z-scored onset curve at FPS frames per second (null: no vocal onsets at all). */
function onsetCurve(f: AlignFeatures): { pre: Float64Array; n: number } | null {
  const n = Math.floor(f.duration * FPS);
  if (n < FPS * 5 || !(f.frameRate > 0) || !f.onsets.length) return null;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor((i / FPS) * f.frameRate);
    const b = Math.max(a + 1, Math.floor(((i + 1) / FPS) * f.frameRate));
    let s = 0;
    let c = 0;
    for (let k = a; k < b && k < f.onsets.length; k++) {
      s += f.onsets[k] ?? 0;
      c++;
    }
    x[i] = c ? s / c : 0;
  }
  let m = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
  const sd = Math.sqrt(v / n);
  if (!(sd > 1e-9)) return null;
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + (x[i] - m) / sd;
  return { pre, n };
}

/** Finds the offset (and speed) that best puts the line starts on the song's vocal onsets. */
export function alignLines(lines: LyricLine[], f: AlignFeatures, opts: AlignOptions = {}): Alignment {
  const range = opts.range ?? 30;
  const stretch = opts.stretch ?? 0.04;
  const minConf = opts.minConfidence ?? 0.5;
  const L = lines.filter((l) => l.text.trim());
  if (L.length < 4) return NONE;
  const c = onsetCurve(f);
  if (!c) return NONE;
  const { n } = c;
  const rev = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) rev[i + 1] = rev[i] + (c.pre[n - i] - c.pre[n - i - 1]);
  const steps = Math.round(range / STEP);
  const ks = Math.round(stretch / SCALE_STEP);

  /** Best (offset, scale) for a set of line starts; the offset curve at that scale. */
  const search = (starts: number[], widths: number[], pre: Float64Array) => {
    const norm = starts.length * WINDOW * FPS;
    const score = (off: number, k: number): number => {
      let s = 0;
      for (let i = 0; i < starts.length; i++) {
        const a = starts[i] * k + off;
        const i0 = Math.max(0, Math.min(n, Math.round(a * FPS)));
        const i1 = Math.max(0, Math.min(n, Math.round((a + widths[i] * k) * FPS)));
        s += pre[i1] - pre[i0];
      }
      return s / norm;
    };
    const curveAt = (k: number): { cv: Float64Array; bi: number } => {
      const cv = new Float64Array(2 * steps + 1);
      let bi = 0;
      for (let j = -steps; j <= steps; j++) {
        cv[j + steps] = score(j * STEP, k);
        if (cv[j + steps] > cv[bi]) bi = j + steps;
      }
      return { cv, bi };
    };
    // Plain shift first; then speeds, kept only when clearly better.
    const base = curveAt(1);
    const baseSd = sdOf(base.cv);
    let k = 1;
    let cur = base;
    for (let q = -ks; q <= ks; q++) {
      const kk = 1 + q * SCALE_STEP;
      if (Math.abs(kk - 1) < 0.005) continue;
      const c2 = curveAt(kk);
      if (c2.cv[c2.bi] > cur.cv[cur.bi] && c2.cv[c2.bi] - base.cv[base.bi] > STRETCH_MARGIN * baseSd) {
        k = kk;
        cur = c2;
      }
    }
    // Refine the offset to 0.01 s.
    let off = (cur.bi - steps) * STEP;
    let best = cur.cv[cur.bi];
    for (let d = -STEP; d <= STEP + 1e-9; d += 0.01) {
      const s = score((cur.bi - steps) * STEP + d, k);
      if (s > best) {
        best = s;
        off = (cur.bi - steps) * STEP + d;
      }
    }
    return { off, k, best, cv: cur.cv, bi: cur.bi, baseSd, score };
  };

  const starts = L.map((l) => l.t);
  const widths = L.map((l) => Math.min(WINDOW, Math.max(0.2, l.end - l.t)));
  const r = search(starts, widths, c.pre);
  if (!(r.baseSd > 1e-9)) return NONE;
  // Chance level: the same search against the song played backwards. Its onsets have the same
  // density, loudness and beat grid, so whatever fits lines on any song's grid fits them as well;
  // only the real direction has the sung phrases where the lyrics say.
  const nullBest = search(starts, widths, rev).best;
  const sd = sdOf(r.cv);
  let second = -Infinity;
  for (let i = 0; i < r.cv.length; i++) if (Math.abs(i - r.bi) * STEP > 1.5) second = Math.max(second, r.cv[i]);
  const gain = (r.best - r.score(0, 1)) / sd;
  const prominence = (r.best - second) / sd;
  const significance = (r.best - nullBest) / r.baseSd;
  // All must hold: clearly better than doing nothing, not one of several equally good shifts (a line
  // period apart, say), and clearly better than the backwards song fits.
  const confidence = clamp01(gain / 3) * clamp01(prominence / 0.4) * clamp01(significance / 0.6);
  const shift = Math.abs(r.off) >= MIN_SHIFT || r.k !== 1;
  const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;
  return {
    offset: round(r.off, 2),
    scale: round(r.k, 4),
    confidence: round(confidence, 3),
    applied: shift && confidence >= minConf,
    gain: round(gain, 2),
    prominence: round(prominence, 2),
    significance: round(significance, 2),
  };
}

function sdOf(a: Float64Array): number {
  let m = 0;
  for (const v of a) m += v;
  m /= a.length;
  let v2 = 0;
  for (const v of a) v2 += (v - m) ** 2;
  return Math.sqrt(v2 / a.length);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** The track with every line moved to t * scale + offset (clipped to the song). */
export function shiftTrack(track: LyricTrack, offset: number, scale = 1, duration = Infinity): LyricTrack {
  if (offset === 0 && scale === 1) return track;
  const lines = track.lines
    .map((l) => ({ ...l, t: l.t * scale + offset, end: l.end * scale + offset }))
    .filter((l) => l.end > 0 && l.t < duration)
    .map((l) => ({ ...l, t: Math.max(0, l.t), end: Math.min(duration, l.end) }));
  return { ...track, lines };
}

/**
 * How well line starts sit on sung onsets, as a diagnostic. `strength`: the mean z-scored vocal
 * onset strength in the first second of the lines (what the alignment maximises). `residual`: the
 * median, over lines, of how far each line's start would have to move (within +-2 s) to sit on the
 * strongest half second of onsets near it: about 0.5 s or less for well placed lines, 1 s and up
 * for misplaced ones.
 */
export function lineOnsetFit(lines: LyricLine[], f: AlignFeatures): { strength: number; residual: number } {
  const c = onsetCurve(f);
  const L = lines.filter((l) => l.text.trim());
  if (!c || !L.length) return { strength: 0, residual: NaN };
  const { pre, n } = c;
  const mean = (a: number, w: number) => {
    const i0 = Math.max(0, Math.min(n, Math.round(a * FPS)));
    const i1 = Math.max(0, Math.min(n, Math.round((a + w) * FPS)));
    return i1 > i0 ? (pre[i1] - pre[i0]) / (i1 - i0) : 0;
  };
  let str = 0;
  const res: number[] = [];
  for (const l of L) {
    str += mean(l.t, WINDOW);
    let bd = 0;
    let bs = -Infinity;
    for (let d = -2; d <= 2 + 1e-9; d += STEP) {
      const s = mean(l.t + d, 0.5) - 0.02 * Math.abs(d);
      if (s > bs) {
        bs = s;
        bd = d;
      }
    }
    res.push(Math.abs(bd));
  }
  res.sort((a, b) => a - b);
  return { strength: str / L.length, residual: res[res.length >> 1] };
}

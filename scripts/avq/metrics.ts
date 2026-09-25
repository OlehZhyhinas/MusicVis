// Reactivity metrics for one rendered clip (pure TS, no DOM / Node APIs).
//
// Input is a ClipLike: per-frame columns (music + visual, see format.ts), the frame clock,
// the song's beat grid / sections / moments / hooks, and 32x18 thumbnails. Output is a
// ReportCard with headline scores in 0..1 plus the raw numbers behind them.
//
// Definitions (fps = 30, frame i shows the MusicState at time t[i]):
//  * response(i): composite visual change, sum over change signals (frame difference, flow
//    magnitude, |d luminance|, |d coverage|, |d colourfulness|, |d spread|, centroid speed,
//    thumbnail change), each scaled by its own 95th percentile. A response PEAK is a local
//    max of response above (local median over +-1 s) + 1.5 x (local MAD) and above 0.15.
//  * sync (onset hit rate): musical events = local maxima of an onset strength (all stems,
//    or one stem) above 0.3, at least 100 ms apart. A hit is a response peak inside the
//    audiovisual integration window, from 45 ms before to 125 ms after the event (visual
//    leads are noticed sooner than lags). chance = hit rate of the same events shifted by
//    random offsets; lift = (hit - chance) / (1 - chance). meanLag over hits, in ms.
//  * coupling: each stem envelope vs each visual feature, both high-passed (minus a 2 s
//    moving average), max Pearson r over visual lags 0..200 ms. uniqueR2: drop-one-stem
//    increase in R^2 of a linear model of the feature on all four stems (the variance only
//    that stem explains). specificity: distinct stems that are the top unique driver of at
//    least one feature (uniqueR2 > 0.03), over the stems present.
//  * hook rhyme: for each hook occurrence inside the clip, the trajectory of z-scored change
//    features resampled to 32 slots. S_hook = mean correlation between occurrences; S_base
//    = mean correlation between an occurrence and another occurrence shifted by k beats
//    (k = 1 .. beats-per-hook - 1: same beat phase, different place in the motif).
//    rhyme = (S_hook - S_base) / (1 - S_base). A preset that only flashes on the beat
//    scores ~0; one whose response follows the riff scores high.
//  * melody following: melody pitch (salient frames only) vs visual height (cy), hue angle
//    (unwrapped), rotation (curl) and spread, both detrended over 4 s, max |r| over 0..200 ms.
//  * correspondences (perceptual, signed): pitch -> height (+), loudness -> size (coverage,
//    spread) and brightness (+), timbre brightness -> luminance / sharpness (edge) (+),
//    rhythmic density (onset rate) -> motion speed (+). Each is r of 2 s-smoothed curves.
//  * structure: at each section boundary / drop with >= 2 s either side inside the clip,
//    look change = thumbnail distance between the 1 s means before and after, over the
//    median of the same measure at non-boundary times (>= 3 s from any boundary).
//    score = 1 - exp(-max(0, ratio - 1)).
//  * flow: jerkiness = mean |d2 motion| / mean |d motion| over flow (x, y, div, curl);
//    stillness = share of frames with almost no change; flicker = RMS of luminance
//    band-passed to 3..15 Hz (Nyquist at 30 fps); strobe = WCAG-style flash count: a flash
//    is a pair of opposing luminance changes >= 0.1 in a thumbnail cell, flagged when > 3
//    per second over >= 25 % of the frame; balance = 1 - 2 x mean centroid distance from
//    the centre.
//  * interest: activity (mean thumbnail change) through a bump that is 0 when static and
//    falls off when frantic, times predictability (R^2 of an AR model of the response on
//    lags 1 frame, 1 beat and 1 bar): static and chaotic both score low.

import type { Hook, Moment, SectionLite } from './music';

export interface ClipLike {
  fps: number;
  n: number;
  col(name: string): Float32Array;
  has(name: string): boolean;
  thumb(i: number): Uint8Array;
  thumbW: number;
  thumbH: number;
  beats: number[];
  downbeats: number[];
  beatsPerBar: number;
  bpm: number;
  sections: SectionLite[];
  moments: Moment[];
  hooks: Hook[];
}

// ------------------------------------------------------------------ helpers

export function mean(x: ArrayLike<number>): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i])) (s += x[i]), n++;
  return n ? s / n : 0;
}

export function quantile(x: ArrayLike<number>, q: number): number {
  const a = Array.from(x).filter(Number.isFinite).sort((p, r) => p - r);
  if (!a.length) return 0;
  const k = Math.max(0, Math.min(a.length - 1, q * (a.length - 1)));
  const lo = Math.floor(k);
  const hi = Math.ceil(k);
  return a[lo] + (a[hi] - a[lo]) * (k - lo);
}

export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let sa = 0, sb = 0, c = 0;
  for (let i = 0; i < n; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) (sa += a[i]), (sb += b[i]), c++;
  if (c < 3) return 0;
  const ma = sa / c, mb = sb / c;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) continue;
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  return da > 1e-12 && db > 1e-12 ? num / Math.sqrt(da * db) : 0;
}

/** Correlation of a[i] with b[i + lag] for lag in 0..maxLag; returns the r with the largest |r| and its lag. */
export function laggedCorr(a: ArrayLike<number>, b: ArrayLike<number>, maxLag: number): { r: number; lag: number } {
  let best = { r: 0, lag: 0 };
  const n = Math.min(a.length, b.length);
  for (let lag = 0; lag <= maxLag; lag++) {
    const x = new Float32Array(n - lag);
    const y = new Float32Array(n - lag);
    for (let i = 0; i + lag < n; i++) (x[i] = a[i]), (y[i] = b[i + lag]);
    const r = pearson(x, y);
    if (Math.abs(r) > Math.abs(best.r)) best = { r, lag };
  }
  return best;
}

/** Circular shift (for null distributions: same signal, broken timing). */
export function roll(x: ArrayLike<number>, k: number): Float32Array {
  const n = x.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x[(((i - k) % n) + n) % n];
  return out;
}

/** Shifts for null baselines: three offsets spread over the clip, each at least minF frames. */
export function nullShifts(n: number, minF: number): number[] {
  return [0.29, 0.5, 0.71].map((q) => Math.round(q * n)).filter((k) => k >= minF && n - k >= minF);
}

export function movingAverage(x: ArrayLike<number>, half: number): Float32Array {
  const n = x.length;
  const out = new Float32Array(n);
  const pre = new Float64Array(n + 1);
  const cnt = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const ok = Number.isFinite(x[i]);
    pre[i + 1] = pre[i] + (ok ? x[i] : 0);
    cnt[i + 1] = cnt[i] + (ok ? 1 : 0);
  }
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half);
    const b = Math.min(n, i + half + 1);
    const c = cnt[b] - cnt[a];
    out[i] = c ? (pre[b] - pre[a]) / c : NaN;
  }
  return out;
}

export function highpass(x: ArrayLike<number>, half: number): Float32Array {
  const m = movingAverage(x, half);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] - m[i];
  return out;
}

function absDiff(x: ArrayLike<number>): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 1; i < x.length; i++) out[i] = Math.abs(x[i] - x[i - 1]);
  out[0] = out[1] ?? 0;
  return out;
}

function scaleBy(x: Float32Array, q = 0.95): Float32Array {
  const s = quantile(x, q) || quantile(x, 1) || 1;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / s;
  return out;
}

function zscore(x: ArrayLike<number>): Float32Array {
  const m = mean(x);
  let v = 0;
  for (let i = 0; i < x.length; i++) v += (x[i] - m) ** 2;
  const sd = Math.sqrt(v / Math.max(1, x.length)) || 1;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) / sd;
  return out;
}

/** Deterministic PRNG for baselines. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp01 = (x: number) => (Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : 0);

/** Mean absolute thumbnail change per frame, 0..1. */
export function thumbChange(c: ClipLike): Float32Array {
  const out = new Float32Array(c.n);
  for (let i = 1; i < c.n; i++) {
    const a = c.thumb(i);
    const b = c.thumb(i - 1);
    let s = 0;
    for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
    out[i] = s / a.length / 255;
  }
  out[0] = out[1] ?? 0;
  return out;
}

// ------------------------------------------------------------------ response

export const CHANGE_FIELDS = ['diff', 'flowMag'] as const;

export interface Response {
  resp: Float32Array;
  peaks: Uint8Array;
  peakIdx: number[];
}

/** The composite visual change signal and its peaks. */
export function visualResponse(c: ClipLike, tch = thumbChange(c)): Response {
  const parts: Float32Array[] = [
    scaleBy(c.col('diff')),
    scaleBy(c.col('flowMag')),
    scaleBy(absDiff(c.col('lum'))),
    scaleBy(absDiff(c.col('coverage'))),
    scaleBy(absDiff(c.col('colorful'))),
    scaleBy(absDiff(c.col('spread'))),
    scaleBy(Float32Array.from(absDiff(c.col('cx')), (v, i) => Math.hypot(v, absDiff(c.col('cy'))[i]))),
    scaleBy(tch),
  ];
  const n = c.n;
  const resp = new Float32Array(n);
  for (const p of parts) for (let i = 0; i < n; i++) resp[i] += p[i] / parts.length;
  // Peaks against a local baseline (median, MAD over +-1 s).
  const half = Math.round(c.fps);
  const peaks = new Uint8Array(n);
  const peakIdx: number[] = [];
  for (let i = 1; i + 1 < n; i++) {
    if (!(resp[i] >= resp[i - 1] && resp[i] > resp[i + 1])) continue;
    const a = Math.max(0, i - half);
    const b = Math.min(n, i + half + 1);
    const w = resp.subarray(a, b);
    const med = quantile(w, 0.5);
    const mad = quantile(Array.from(w, (v) => Math.abs(v - med)), 0.5);
    if (resp[i] > med + 1.5 * mad + 1e-3 && resp[i] > 0.15) {
      peaks[i] = 1;
      peakIdx.push(i);
    }
  }
  return { resp, peaks, peakIdx };
}

// ------------------------------------------------------------------ sync

export const SYNC_EARLY_MS = 45;
export const SYNC_LATE_MS = 125;

export interface SyncStats {
  events: number;
  hitRate: number;
  chance: number;
  lift: number;
  meanLagMs: number;
}

/** Local maxima of an onset strength above thr, at least minGap seconds apart. Returns times (s from clip start, frame units / fps). */
export function onsetEvents(x: ArrayLike<number>, fps: number, thr = 0.3, minGap = 0.1): number[] {
  const out: number[] = [];
  let last = -Infinity;
  for (let i = 1; i + 1 < x.length; i++) {
    if (x[i] >= thr && x[i] >= x[i - 1] && x[i] > x[i + 1] && i / fps - last >= minGap) {
      out.push(i);
      last = i / fps;
    }
  }
  return out;
}

export function syncStats(eventsIdx: number[], peaks: Uint8Array, fps: number, seed = 7): SyncStats {
  const n = peaks.length;
  const lo = -Math.floor((SYNC_EARLY_MS / 1000) * fps + 1e-9);
  const hi = Math.floor((SYNC_LATE_MS / 1000) * fps + 1e-9);
  const hitAt = (e: number): number | null => {
    let best: number | null = null;
    for (let d = lo; d <= hi; d++) {
      const j = e + d;
      if (j >= 0 && j < n && peaks[j]) if (best === null || Math.abs(d) < Math.abs(best)) best = d;
    }
    return best;
  };
  let hits = 0;
  let lag = 0;
  const ev = eventsIdx.filter((e) => e + lo >= 0 && e + hi < n);
  for (const e of ev) {
    const d = hitAt(e);
    if (d !== null) (hits++, (lag += d));
  }
  // Chance: the same number of events at random frames (several draws).
  const r = rng(seed);
  let ch = 0;
  let draws = 0;
  for (let k = 0; k < 20; k++) {
    for (let m = 0; m < Math.max(ev.length, 10); m++) {
      const e = Math.floor(-lo + r() * Math.max(1, n - hi + lo - 1));
      if (hitAt(e) !== null) ch++;
      draws++;
    }
  }
  const hitRate = ev.length ? hits / ev.length : 0;
  const chance = draws ? ch / draws : 0;
  return {
    events: ev.length,
    hitRate,
    chance,
    lift: chance < 1 ? (hitRate - chance) / (1 - chance) : 0,
    meanLagMs: hits ? ((lag / hits) * 1000) / fps : NaN,
  };
}

// ------------------------------------------------------------------ coupling

export const STEM_KEYS = ['drums', 'bass', 'vocals', 'other'] as const;
const STEM_ON = { drums: 'onDrums', bass: 'onBass', vocals: 'onVocals', other: 'onOther' } as const;
const STEM_PR = { drums: 'prDrums', bass: 'prBass', vocals: 'prVocals', other: 'prOther' } as const;

export const COUPLING_FEATURES = ['lum', 'lumStd', 'colorful', 'coverage', 'sat', 'cy', 'spread', 'diff', 'edge', 'flowMag', 'div', 'curl', 'hueSpeed', 'thumbChange'] as const;

function hueAngle(c: ClipLike): Float32Array {
  const hx = c.col('hueX');
  const hy = c.col('hueY');
  const out = new Float32Array(c.n);
  let prev = 0;
  let off = 0;
  for (let i = 0; i < c.n; i++) {
    let a = Math.atan2(hy[i], hx[i]);
    if (i > 0) {
      let d = a + off - prev;
      while (d > Math.PI) (off -= 2 * Math.PI), (d -= 2 * Math.PI);
      while (d < -Math.PI) (off += 2 * Math.PI), (d += 2 * Math.PI);
    }
    a += off;
    out[i] = a;
    prev = a;
  }
  return out;
}

export function featureSeries(c: ClipLike, tch: Float32Array): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  for (const f of COUPLING_FEATURES) {
    if (f === 'hueSpeed') out[f] = absDiff(hueAngle(c));
    else if (f === 'thumbChange') out[f] = tch;
    else out[f] = c.col(f);
  }
  return out;
}

/** Least squares R^2 of y on the columns of X (plus intercept), via normal equations. */
export function r2(y: ArrayLike<number>, X: ArrayLike<number>[]): number {
  const n = y.length;
  const k = X.length + 1;
  const A = new Float64Array(k * k);
  const b = new Float64Array(k);
  const row = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    row[0] = 1;
    for (let j = 0; j < X.length; j++) row[j + 1] = X[j][i];
    for (let p = 0; p < k; p++) {
      b[p] += row[p] * y[i];
      for (let q = 0; q < k; q++) A[p * k + q] += row[p] * row[q];
    }
  }
  for (let p = 0; p < k; p++) A[p * k + p] += 1e-6;
  // Gaussian elimination.
  const M = Array.from({ length: k }, (_, p) => [...Array.from(A.subarray(p * k, p * k + k)), b[p]]);
  for (let p = 0; p < k; p++) {
    let piv = p;
    for (let q = p + 1; q < k; q++) if (Math.abs(M[q][p]) > Math.abs(M[piv][p])) piv = q;
    [M[p], M[piv]] = [M[piv], M[p]];
    const d = M[p][p] || 1e-12;
    for (let q = p; q <= k; q++) M[p][q] /= d;
    for (let r = 0; r < k; r++) {
      if (r === p) continue;
      const f = M[r][p];
      for (let q = p; q <= k; q++) M[r][q] -= f * M[p][q];
    }
  }
  const beta = M.map((r) => r[k]);
  const my = mean(y);
  let ssr = 0, sst = 0;
  for (let i = 0; i < n; i++) {
    let pred = beta[0];
    for (let j = 0; j < X.length; j++) pred += beta[j + 1] * X[j][i];
    ssr += (y[i] - pred) ** 2;
    sst += (y[i] - my) ** 2;
  }
  return sst > 1e-12 ? Math.max(0, 1 - ssr / sst) : 0;
}

export interface CouplingStats {
  /** r[stem][feature] (max |r| over lags, sign kept). */
  r: Record<string, Record<string, number>>;
  /** Best feature per stem and its |r|. */
  top: Record<string, { feature: string; r: number }>;
  uniqueR2: Record<string, Record<string, number>>;
  /** Stems that uniquely drive at least one feature. */
  drivers: string[];
  present: string[];
  specificity: number;
  /** Per stem: the best |r| with the stem's timing broken (circular shift). */
  nullTop: Record<string, number>;
  /** Mean over present stems of (top |r| - null |r|), floored at 0. */
  score: number;
}

export function couplingStats(c: ClipLike, feats: Record<string, Float32Array>): CouplingStats {
  const half = Math.round(c.fps);
  const maxLag = Math.round(0.2 * c.fps);
  const present = STEM_KEYS.filter((s) => mean(c.col(STEM_PR[s])) > 0.15);
  const stemHP: Record<string, Float32Array> = {};
  for (const s of STEM_KEYS) {
    const env = c.col(s);
    const on = c.col(STEM_ON[s]);
    stemHP[s] = highpass(Float32Array.from(env, (v, i) => v + 0.5 * on[i]), half);
  }
  const featHP: Record<string, Float32Array> = {};
  for (const [f, x] of Object.entries(feats)) featHP[f] = highpass(x, half);
  const r: CouplingStats['r'] = {};
  const top: CouplingStats['top'] = {};
  for (const s of STEM_KEYS) {
    r[s] = {};
    top[s] = { feature: '', r: 0 };
    for (const f of Object.keys(feats)) {
      const v = laggedCorr(stemHP[s], featHP[f], maxLag).r;
      r[s][f] = v;
      if (Math.abs(v) > Math.abs(top[s].r)) top[s] = { feature: f, r: v };
    }
  }
  const uniqueR2: CouplingStats['uniqueR2'] = {};
  for (const s of STEM_KEYS) uniqueR2[s] = {};
  const drivers = new Set<string>();
  for (const f of Object.keys(feats)) {
    const y = featHP[f];
    const X = STEM_KEYS.map((s) => stemHP[s]);
    const full = r2(y, X);
    let bestS = '';
    let bestU = 0;
    STEM_KEYS.forEach((s, k) => {
      const u = Math.max(0, full - r2(y, X.filter((_, j) => j !== k)));
      uniqueR2[s][f] = u;
      if (u > bestU) (bestU = u), (bestS = s);
    });
    if (bestU > 0.03 && present.includes(bestS as (typeof STEM_KEYS)[number])) drivers.add(bestS);
  }
  const specificity = present.length ? drivers.size / present.length : 0;
  // Null: the best |r| each stem reaches against the features when its timing is broken
  // (circularly shifted by > 2 s). Coupling counts only what beats that.
  const shifts = nullShifts(c.n, Math.round(2 * c.fps));
  const nullTop: Record<string, number> = {};
  for (const s of STEM_KEYS) {
    let acc = 0;
    for (const k of shifts) {
      const sh = roll(stemHP[s], k);
      let best = 0;
      for (const f of Object.keys(feats)) best = Math.max(best, Math.abs(laggedCorr(sh, featHP[f], maxLag).r));
      acc += best;
    }
    nullTop[s] = shifts.length ? acc / shifts.length : 0;
  }
  const lift = (s: string) => Math.max(0, Math.abs(top[s].r) - nullTop[s]);
  const score = present.length ? mean(present.map(lift)) : 0;
  return { r, top, nullTop, uniqueR2, drivers: [...drivers], present, specificity, score };
}

// ------------------------------------------------------------------ hook rhyme

export const RHYME_FIELDS = ['diff', 'flowMag', 'dLum', 'dColor', 'div', 'curl', 'dCx', 'dCy', 'dSpread', 'dCover', 'thumbChange'] as const;

export interface RhymeStats {
  hook: number;
  occurrences: number;
  sHook: number;
  sBase: number;
  rhyme: number;
  /** 0..1 headline: max(0, rhyme). */
  score: number;
}

function trajectory(Z: Float32Array[], startF: number, lenF: number, slots = 32): Float32Array | null {
  if (startF < 0 || startF + lenF >= Z[0].length) return null;
  const out = new Float32Array(Z.length * slots);
  for (let k = 0; k < Z.length; k++) {
    for (let s = 0; s < slots; s++) {
      const a = startF + Math.floor((s * lenF) / slots);
      const b = Math.max(a + 1, startF + Math.floor(((s + 1) * lenF) / slots));
      let v = 0;
      for (let i = a; i < b; i++) v += Z[k][i];
      out[k * slots + s] = v / (b - a);
    }
  }
  return out;
}

export function rhymeInputs(c: ClipLike, tch: Float32Array): Float32Array[] {
  const src: Record<string, Float32Array> = {
    diff: c.col('diff'), flowMag: c.col('flowMag'), dLum: absDiff(c.col('lum')), dColor: absDiff(c.col('colorful')),
    div: c.col('div'), curl: c.col('curl'), dCx: absDiff(c.col('cx')), dCy: absDiff(c.col('cy')), dSpread: absDiff(c.col('spread')),
    dCover: absDiff(c.col('coverage')), thumbChange: tch,
  };
  return RHYME_FIELDS.map((f) => zscore(src[f]));
}

/**
 * Hook rhyme for one hook. clipStart is the song time of frame 0 minus one frame, i.e. frame
 * i is at song time clipStart + (i + 1) / fps; trajectories are cut on song time.
 */
export function hookRhyme(c: ClipLike, Z: Float32Array[], hook: Hook, clipT0: number): RhymeStats {
  const fps = c.fps;
  const beat = 60 / (c.bpm || 120);
  const lenF = Math.round(hook.len * fps);
  const beatsPer = Math.max(2, Math.round(hook.len / beat));
  const toF = (t: number) => Math.round((t - clipT0) * fps) - 1;
  const occ = hook.occurrences.map((o) => toF(o.start)).filter((f) => f >= 0 && f + lenF < c.n);
  const trajs = occ.map((f) => trajectory(Z, f, lenF)!).filter(Boolean);
  const empty = { hook: hook.id, occurrences: trajs.length, sHook: 0, sBase: 0, rhyme: 0, score: 0 };
  if (trajs.length < 2) return empty;
  let sh = 0, nh = 0, sb = 0, nb = 0;
  for (let a = 0; a < trajs.length; a++) {
    for (let b = a + 1; b < trajs.length; b++) {
      sh += pearson(trajs[a], trajs[b]);
      nh++;
    }
  }
  for (let a = 0; a < occ.length; a++) {
    for (let b = 0; b < occ.length; b++) {
      if (a === b) continue;
      for (let k = 1; k < beatsPer; k++) {
        const t = trajectory(Z, occ[b] + Math.round(k * beat * fps), lenF);
        if (!t) continue;
        sb += pearson(trajs[a], t);
        nb++;
      }
    }
  }
  if (!nb) return empty;
  const sHook = sh / nh;
  const sBase = sb / nb;
  const rhyme = sBase < 1 ? (sHook - sBase) / (1 - sBase) : 0;
  return { hook: hook.id, occurrences: trajs.length, sHook, sBase, rhyme, score: clamp01(rhyme) };
}

// ------------------------------------------------------------------ melody + correspondences

export interface MelodyStats {
  frames: number;
  r: Record<string, number>;
  best: { target: string; r: number };
  /** Best |r| with the melody's timing broken (circular shift). */
  nullR: number;
  score: number;
}

export function melodyStats(c: ClipLike): MelodyStats {
  const fps = c.fps;
  const midi = c.col('melMidi');
  const sal = c.col('melSal');
  const pitch = Float32Array.from(midi, (v, i) => (sal[i] > 0.3 && Number.isFinite(v) ? v : NaN));
  // Smooth over ~100 ms and fill tiny gaps.
  const sm = movingAverage(pitch, Math.round(0.05 * fps));
  const det = Float32Array.from(sm, (v, i) => (Number.isFinite(pitch[i]) ? v : NaN));
  const trend = movingAverage(det, Math.round(2 * fps));
  const contour = Float32Array.from(det, (v, i) => v - trend[i]);
  const targets: Record<string, Float32Array> = { height: c.col('cy'), hue: hueAngle(c), rotation: c.col('curl'), spread: c.col('spread') };
  const r: Record<string, number> = {};
  let best = { target: '', r: 0 };
  const frames = Array.from(contour).filter(Number.isFinite).length;
  const hps: Float32Array[] = [];
  for (const [k, x] of Object.entries(targets)) {
    const hp = highpass(movingAverage(x, Math.round(0.05 * fps)), Math.round(2 * fps));
    hps.push(hp);
    const v = frames > fps * 2 ? laggedCorr(contour, hp, Math.round(0.2 * fps)).r : 0;
    r[k] = v;
    if (Math.abs(v) > Math.abs(best.r)) best = { target: k, r: v };
  }
  let nullR = 0;
  const shifts = nullShifts(c.n, Math.round(2 * fps));
  if (frames > fps * 2) {
    for (const k of shifts) {
      const sh = roll(contour, k);
      nullR += Math.max(...hps.map((hp) => Math.abs(laggedCorr(sh, hp, Math.round(0.2 * fps)).r)));
    }
    nullR /= Math.max(1, shifts.length);
  }
  return { frames, r, best, nullR, score: clamp01((Math.abs(best.r) - nullR) / 0.35) };
}

export interface CorrespondenceStats {
  /** Signed r for each rule; positive means the expected direction. */
  rules: Record<string, number>;
  score: number;
}

export function correspondences(c: ClipLike): CorrespondenceStats {
  const fps = c.fps;
  const s2 = (x: ArrayLike<number>) => movingAverage(x, Math.round(fps));
  const midi = c.col('melMidi');
  const sal = c.col('melSal');
  const pitch = Float32Array.from(midi, (v, i) => (sal[i] > 0.3 && Number.isFinite(v) ? v : NaN));
  const onRate = Float32Array.from(c.col('onDrums'), (v, i) => v + c.col('onBass')[i] + c.col('onVocals')[i] + c.col('onOther')[i]);
  const loud = s2(c.col('loud'));
  const size = s2(Float32Array.from(c.col('coverage'), (v, i) => v + c.col('spread')[i]));
  const rules: Record<string, number> = {
    'pitch>height': pearson(s2(pitch), s2(c.col('cy'))),
    'loud>size': pearson(loud, size),
    'loud>bright': pearson(loud, s2(c.col('lum'))),
    'timbre>bright': pearson(s2(c.col('tBright')), s2(c.col('lum'))),
    'timbre>sharp': pearson(s2(c.col('tBright')), s2(c.col('edge'))),
    'density>speed': pearson(s2(onRate), s2(Float32Array.from(c.col('flowMag'), (v, i) => v + 5 * c.col('diff')[i]))),
  };
  const score = mean(Object.values(rules).map((v) => clamp01(v / 0.5)));
  return { rules, score };
}

// ------------------------------------------------------------------ structure

export interface StructureStats {
  boundaries: { t: number; label: string; kind: string; change: number; ratio: number }[];
  baseline: number;
  score: number;
}

function meanThumb(c: ClipLike, a: number, b: number): Float32Array {
  const len = c.thumbW * c.thumbH * 3;
  const out = new Float32Array(len);
  let n = 0;
  for (let i = Math.max(0, a); i < Math.min(c.n, b); i++, n++) {
    const t = c.thumb(i);
    for (let k = 0; k < len; k++) out[k] += t[k];
  }
  if (n) for (let k = 0; k < len; k++) out[k] /= n * 255;
  return out;
}

function lookChange(c: ClipLike, f: number, w: number): number {
  const a = meanThumb(c, f - w, f);
  const b = meanThumb(c, f + 1, f + 1 + w);
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
  return s / a.length;
}

export function structureStats(c: ClipLike, clipT0: number): StructureStats {
  const fps = c.fps;
  const toF = (t: number) => Math.round((t - clipT0) * fps) - 1;
  const w = Math.round(fps);
  const bounds = c.moments.map((m) => ({ m, f: toF(m.t) })).filter((x) => x.f - 2 * fps >= 0 && x.f + 2 * fps < c.n);
  const allB = c.moments.map((m) => toF(m.t));
  const base: number[] = [];
  for (let f = w + 1; f + w + 1 < c.n; f += Math.round(fps / 2)) {
    if (allB.every((b) => Math.abs(b - f) >= 3 * fps)) base.push(lookChange(c, f, w));
  }
  const baseline = quantile(base, 0.5);
  const boundaries = bounds.map(({ m, f }) => {
    const ch = lookChange(c, f, w);
    return { t: m.t, label: m.label, kind: m.kind, change: ch, ratio: baseline > 1e-6 ? ch / baseline : ch > 1e-4 ? 10 : 1 };
  });
  const score = boundaries.length ? mean(boundaries.map((b) => 1 - Math.exp(-Math.max(0, b.ratio - 1)))) : NaN;
  return { boundaries, baseline, score };
}

// ------------------------------------------------------------------ flow quality

export interface FlowStats {
  jerk: number;
  stillness: number;
  flicker: number;
  flashesPerSec: number;
  flashArea: number;
  strobe: boolean;
  balance: number;
  score: number;
}

export function flowStats(c: ClipLike, tch: Float32Array): FlowStats {
  const fps = c.fps;
  const n = c.n;
  const comps = ['flowX', 'flowY', 'div', 'curl'].map((f) => movingAverage(c.col(f), 1));
  let d1 = 0, d2 = 0;
  for (let i = 2; i < n; i++) {
    for (const x of comps) {
      d1 += Math.abs(x[i] - x[i - 1]);
      d2 += Math.abs(x[i] - 2 * x[i - 1] + x[i - 2]);
    }
  }
  const jerk = d1 > 1e-9 ? d2 / d1 : 0;
  const diff = c.col('diff');
  const fm = c.col('flowMag');
  let still = 0;
  for (let i = 1; i < n; i++) if (diff[i] < 0.0015 && fm[i] < 0.02 && tch[i] < 0.002) still++;
  const stillness = still / Math.max(1, n - 1);
  // Flicker: luminance band-passed to 3..15 Hz (difference of a 1-frame and a 5-frame smoothing ~ >3 Hz).
  const lum = c.col('lum');
  const lp = movingAverage(lum, Math.round(fps / 6 / 2));
  let fl = 0;
  for (let i = 0; i < n; i++) fl += (lum[i] - lp[i]) ** 2;
  const flicker = Math.sqrt(fl / Math.max(1, n));
  // WCAG-style flashes per thumbnail cell.
  const cells = c.thumbW * c.thumbH;
  const cellLum = (i: number, k: number) => {
    const t = c.thumb(i);
    return (0.2126 * t[k * 3] + 0.7152 * t[k * 3 + 1] + 0.0722 * t[k * 3 + 2]) / 255;
  };
  let worstRate = 0;
  let worstArea = 0;
  const win = Math.round(fps);
  // Count flash transitions (direction reversals of >= 0.1) per cell, then the max over 1 s windows.
  const trans: Uint8Array[] = [];
  for (let k = 0; k < cells; k++) {
    const tr = new Uint8Array(n);
    let ext = cellLum(0, k);
    let dir = 0;
    for (let i = 1; i < n; i++) {
      const v = cellLum(i, k);
      const d = v - ext;
      if (dir >= 0 && d <= -0.1) (dir = -1), (ext = v), (tr[i] = 1);
      else if (dir <= 0 && d >= 0.1) (dir = 1), (ext = v), (tr[i] = 1);
      else if ((dir > 0 && v > ext) || (dir < 0 && v < ext) || dir === 0) ext = dir === 0 ? ext : v;
    }
    trans.push(tr);
  }
  for (let s = 0; s + win <= n; s += Math.max(1, Math.round(fps / 4))) {
    let flashing = 0;
    let rateSum = 0;
    for (let k = 0; k < cells; k++) {
      let cnt = 0;
      for (let i = s; i < s + win; i++) cnt += trans[k][i];
      const flashes = cnt / 2; // a flash = a pair of opposing transitions
      if (flashes > 3) flashing++;
      rateSum += flashes;
    }
    const area = flashing / cells;
    if (area > worstArea) worstArea = area;
    worstRate = Math.max(worstRate, rateSum / cells);
  }
  const strobe = worstArea >= 0.25;
  const cx = c.col('cx');
  const cy = c.col('cy');
  let off = 0;
  for (let i = 0; i < n; i++) off += Math.hypot(cx[i] - 0.5, cy[i] - 0.5);
  const balance = clamp01(1 - (2 * off) / Math.max(1, n));
  const smooth = clamp01(1 - (jerk - 0.5) / 1.5);
  const score = strobe ? 0 : clamp01(0.4 * smooth + 0.4 * (1 - stillness) + 0.2 * balance - Math.max(0, flicker - 0.03) * 5);
  return { jerk, stillness, flicker, flashesPerSec: worstRate, flashArea: worstArea, strobe, balance, score };
}

// ------------------------------------------------------------------ interest

export interface InterestStats {
  activity: number;
  predictability: number;
  score: number;
}

export function interestStats(c: ClipLike, resp: Float32Array, tch: Float32Array): InterestStats {
  const fps = c.fps;
  const activity = mean(tch);
  const beatF = Math.max(2, Math.round((60 / (c.bpm || 120)) * fps));
  const barF = beatF * Math.max(2, c.beatsPerBar || 4);
  const y: number[] = [];
  const X: number[][] = [[], [], []];
  for (let i = barF; i < c.n; i++) {
    y.push(resp[i]);
    X[0].push(resp[i - 1]);
    X[1].push(resp[i - beatF]);
    X[2].push(resp[i - barF]);
  }
  const predictability = y.length > 10 ? r2(y, X) : 0;
  // Activity bump: 0 when static (< 0.001), 1 around 0.005..0.03, falling to ~0.2 at 0.12.
  const a = activity;
  const rise = clamp01((a - 0.001) / 0.004);
  const fall = a <= 0.03 ? 1 : Math.exp(-(a - 0.03) / 0.06);
  const score = rise * fall * (0.3 + 0.7 * predictability);
  return { activity, predictability, score };
}

// ------------------------------------------------------------------ report card

export interface ReportCard {
  preset: string;
  song: string;
  clip: string;
  frames: number;
  headline: {
    sync: number;
    coupling: number;
    hookRhyme: number;
    melody: number;
    structure: number;
    flow: number;
    interest: number;
    correspond: number;
    overall: number;
  };
  sync: { all: SyncStats; hits: SyncStats; perStem: Record<string, SyncStats>; beats: SyncStats };
  coupling: CouplingStats;
  rhyme: RhymeStats[];
  melody: MelodyStats;
  correspond: CorrespondenceStats;
  structure: StructureStats;
  flow: FlowStats;
  interest: InterestStats;
  /** Present when counterfactual renders exist for the same clip. */
  counterfactual?: CfSummary;
  notes: string[];
}

export const HEADLINE_WEIGHTS = { sync: 0.22, coupling: 0.14, hookRhyme: 0.16, melody: 0.08, structure: 0.12, flow: 0.12, interest: 0.1, correspond: 0.06 };

/** clipT0 = song time of the frame before row 0 (row i is at clipT0 + (i + 1) / fps). */
export function reportCard(c: ClipLike, meta: { preset: string; song: string; clip: string; clipT0: number }): ReportCard {
  const fps = c.fps;
  const tch = thumbChange(c);
  const R = visualResponse(c, tch);
  const all = onsetEvents(Float32Array.from(c.col('onDrums'), (v, i) => Math.max(v, c.col('onBass')[i], c.col('onVocals')[i], c.col('onOther')[i])), fps);
  const hitsEv = onsetEvents(c.col('onDrums'), fps, 0.5, 0.15);
  const perStem: Record<string, SyncStats> = {};
  for (const s of STEM_KEYS) perStem[s] = syncStats(onsetEvents(c.col(STEM_ON[s]), fps), R.peaks, fps);
  const beatEv: number[] = [];
  const ob = c.col('onBeat');
  for (let i = 0; i < c.n; i++) if (ob[i] > 0.5) beatEv.push(i);
  const sync = { all: syncStats(all, R.peaks, fps), hits: syncStats(hitsEv, R.peaks, fps), perStem, beats: syncStats(beatEv, R.peaks, fps) };
  const feats = featureSeries(c, tch);
  const coupling = couplingStats(c, feats);
  const Z = rhymeInputs(c, tch);
  const rhyme = c.hooks.map((h) => hookRhyme(c, Z, h, meta.clipT0)).filter((r) => r.occurrences >= 2);
  const melody = melodyStats(c);
  const correspond = correspondences(c);
  const structure = structureStats(c, meta.clipT0);
  const flow = flowStats(c, tch);
  const interest = interestStats(c, R.resp, tch);
  const syncScore = clamp01(0.5 * Math.max(0, sync.all.lift) / 0.5 + 0.5 * Math.max(0, sync.hits.lift) / 0.6);
  const headline = {
    sync: syncScore,
    coupling: clamp01((coupling.score / 0.35) * (0.6 + 0.4 * coupling.specificity)),
    hookRhyme: rhyme.length ? Math.max(...rhyme.map((r) => r.score)) : NaN,
    melody: melody.score,
    structure: structure.score,
    flow: flow.score,
    interest: interest.score,
    correspond: correspond.score,
    overall: 0,
  };
  let ws = 0;
  let acc = 0;
  for (const [k, w] of Object.entries(HEADLINE_WEIGHTS)) {
    const v = headline[k as keyof typeof headline];
    if (Number.isFinite(v)) (acc += w * v), (ws += w);
  }
  headline.overall = ws ? acc / ws : 0;
  const notes: string[] = [];
  const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : 'n/a');
  if (flow.strobe) notes.push(`STROBE: ${(flow.flashArea * 100).toFixed(0)}% of the frame flashes > 3/s`);
  if (flow.stillness > 0.5) notes.push(`still ${(flow.stillness * 100).toFixed(0)}% of frames`);
  if (sync.hits.events >= 8 && sync.hits.lift < 0.15) notes.push(`drum hits barely land (lift ${f2(sync.hits.lift)}, hit ${f2(sync.hits.hitRate)} vs chance ${f2(sync.hits.chance)})`);
  if (rhyme.length && headline.hookRhyme < 0.15) notes.push(`hook rhyme ${f2(headline.hookRhyme)}: the ${c.hooks[rhyme[0].hook]?.bars ?? '?'}-bar motif gets no consistent visual response (S_hook ${f2(rhyme[0].sHook)} vs beat-shifted ${f2(rhyme[0].sBase)})`);
  if (coupling.present.length >= 2 && coupling.specificity <= 0.25) notes.push(`stems not distinguishable: only ${coupling.drivers.join(',') || 'none'} uniquely drive a feature`);
  if (interest.activity > 0.08 && interest.predictability < 0.15) notes.push('chaotic: high change with little predictability');
  if (Number.isFinite(structure.score) && structure.score < 0.2 && structure.boundaries.length) notes.push(`section changes barely visible (look-change ratio ${f2(mean(structure.boundaries.map((b) => b.ratio)))})`);
  return { preset: meta.preset, song: meta.song, clip: meta.clip, frames: c.n, headline, sync, coupling, rhyme, melody, correspond, structure, flow, interest, notes };
}

// ------------------------------------------------------------------ counterfactuals

/** Minimal shape of a counterfactual result (scripts/avq/page.ts counterfactual()). */
export interface CfLike {
  motion: number;
  variants: { id: string; kind: string; rel: number; stem?: string; reaction?: number; beats?: number }[];
  reactions: { src: string; target: string; gain: number }[];
}

export interface CfSummary {
  /** Divergence under a half-bar desync, relative to the preset's own half-bar motion. */
  desync: number;
  /** Divergence under an inaudible 2 % level change (amplification of meaningless perturbations: chaos). */
  chaos: number;
  /** desync - chaos, floored at 0: what the timing itself explains. */
  syncSensitivity: number;
  /** Divergence with the music from elsewhere in the song. */
  content: number;
  /** Each stem's visual footprint (divergence when it is removed, relative), minus the chaos floor. */
  stems: Record<string, number>;
  /** dead: no visible effect beyond the chaos floor; masked: the preset is too chaotic (floor >= 0.3) to tell. */
  reactions: { index: number; src: string; target: string; rel: number; dead: boolean; masked: boolean }[];
  dead: number;
  /** 0..1 headline: timing sensitivity. */
  score: number;
}

export const DEAD_REACTION = 0.03;
/** Chaos floor above which single-change counterfactuals cannot be told from amplified noise. */
export const CHAOTIC = 0.3;

export function cfSummary(cf: CfLike): CfSummary {
  const find = (pred: (v: CfLike['variants'][number]) => boolean) => cf.variants.find(pred)?.rel ?? NaN;
  const shifts = cf.variants.filter((v) => v.kind === 'shift').sort((a, b) => (a.beats ?? 0) - (b.beats ?? 0));
  const desync = shifts.length ? shifts[shifts.length - 1].rel : NaN;
  const chaos = find((v) => v.kind === 'gain');
  const floor = Number.isFinite(chaos) ? chaos : 0;
  const stems: Record<string, number> = {};
  for (const v of cf.variants) if (v.kind === 'mute' && v.stem) stems[v.stem] = Math.max(0, v.rel - floor);
  const reactions = cf.variants
    .filter((v) => v.kind === 'ablate' && v.reaction !== undefined)
    .map((v) => {
      const masked = floor >= CHAOTIC && v.rel < floor * 1.5;
      return { index: v.reaction!, src: cf.reactions[v.reaction!]?.src ?? '?', target: cf.reactions[v.reaction!]?.target ?? '?', rel: v.rel, masked, dead: !masked && v.rel - floor < DEAD_REACTION };
    });
  const syncSensitivity = Math.max(0, (Number.isFinite(desync) ? desync : shifts.at(-1)?.rel ?? 0) - floor);
  return {
    desync, chaos, syncSensitivity, content: find((v) => v.kind === 'offset'), stems, reactions,
    dead: reactions.filter((r) => r.dead).length, score: clamp01(syncSensitivity / 0.6),
  };
}

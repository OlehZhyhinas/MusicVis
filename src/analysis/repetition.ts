// Section repetition: which sections of a song come back (the second chorus, a returning riff).
//
// Each section becomes a beat-synchronous feature sequence (chroma plus an energy / instrument
// profile). Two sections are compared by the best diagonal alignment of their cross-similarity
// matrix (a small lag search absorbs boundaries that are off by a bar or two), with the chroma of the
// later section rotated to the key that fits best, so a chorus that returns a step higher still
// counts. The raw similarity is rescaled against the song's own baseline (two unrelated sections of
// one song already share a key and a mix), then sections are grouped greedily in time order: a
// section joins the group of the earlier section it resembles most, when that resemblance is strong.
//
// Pure and deterministic; uses only AnalysisResult fields, so it also runs on results analysed before
// this module existed (the TimelineSampler calls it when `repeats` is missing).

import type { AnalysisResult, SectionRepeat, StemName } from '../types';
import { STEM_NAMES } from '../types';

/** Normalised similarity a section needs to count as a return of an earlier one. */
export const REPEAT_THRESHOLD = 0.6;
/** Sections shorter than this many beats never repeat or get repeated. */
const MIN_BEATS = 6;
/** Largest alignment lag, as a fraction of the shorter section. */
const MAX_LAG = 0.25;
/** Fraction of the shorter section an alignment must overlap. */
const MIN_OVERLAP = 0.6;
/** Raw similarity factor of a return in another key. */
const TRANSPOSE_PENALTY = 0.93;

type Input = Pick<AnalysisResult, 'duration' | 'frameRate' | 'numFrames' | 'chroma' | 'loudness' | 'beats' | 'sections'> &
  Partial<Pick<AnalysisResult, 'stems' | 'stemPresence' | 'complexity'>>;

const ENERGY_DIMS = 2 + STEM_NAMES.length * 2; // loudness, complexity, stems, presence

/** One beat-synchronous frame grid: its window start times and per-window features. */
interface Grid {
  n: number;
  t: Float64Array; // window start times (n + 1 entries, the last = end)
  chroma: Float64Array; // n * 12, unit norm
  energy: Float64Array; // n * ENERGY_DIMS, standardised over the song
}

function buildGrid(r: Input): Grid {
  const dur = Math.max(r.duration, 1e-3);
  let times = Array.from(r.beats ?? []).filter((b) => b >= 0 && b < dur);
  // Too few beats (ambient, silence): a half-second grid instead.
  if (times.length < 8) {
    times = [];
    for (let t = 0; t < dur; t += 0.5) times.push(t);
  }
  if (times[0] > 1e-3) times.unshift(0);
  const n = times.length;
  const t = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) t[i] = times[i];
  t[n] = dur;
  const T = r.numFrames;
  const fr = r.frameRate;
  const chroma = new Float64Array(n * 12);
  const energy = new Float64Array(n * ENERGY_DIMS);
  const at = (x: ArrayLike<number> | undefined, i: number) => (x && i < x.length && Number.isFinite(x[i]) ? x[i] : 0);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, Math.min(T - 1, Math.floor(t[i] * fr)));
    const b = Math.max(a + 1, Math.min(T, Math.ceil(t[i + 1] * fr)));
    const cnt = Math.max(1, b - a);
    for (let f = a; f < b && f < T; f++) {
      for (let k = 0; k < 12; k++) chroma[i * 12 + k] += at(r.chroma, f * 12 + k);
      const e = i * ENERGY_DIMS;
      energy[e] += at(r.loudness, f);
      energy[e + 1] += at(r.complexity, f);
      STEM_NAMES.forEach((s: StemName, j) => {
        energy[e + 2 + j] += at(r.stems?.[s], f);
        energy[e + 2 + STEM_NAMES.length + j] += at(r.stemPresence?.[s], f);
      });
    }
    let s = 0;
    for (let k = 0; k < 12; k++) s += chroma[i * 12 + k] ** 2;
    s = Math.sqrt(s);
    for (let k = 0; k < 12; k++) chroma[i * 12 + k] = s > 1e-9 ? chroma[i * 12 + k] / s : 0;
    for (let d = 0; d < ENERGY_DIMS; d++) energy[i * ENERGY_DIMS + d] /= cnt;
  }
  // Standardise each energy dimension over the song (a floor keeps flat dimensions from exploding).
  for (let d = 0; d < ENERGY_DIMS; d++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += energy[i * ENERGY_DIMS + d];
    m /= Math.max(1, n);
    let v = 0;
    for (let i = 0; i < n; i++) v += (energy[i * ENERGY_DIMS + d] - m) ** 2;
    const sd = Math.max(0.05, Math.sqrt(v / Math.max(1, n)));
    for (let i = 0; i < n; i++) energy[i * ENERGY_DIMS + d] = Math.max(-4, Math.min(4, (energy[i * ENERGY_DIMS + d] - m) / sd));
  }
  return { n, t, chroma, energy };
}

/** Grid window range [a, b) a section covers. */
function span(g: Grid, start: number, end: number): [number, number] {
  let a = 0;
  while (a < g.n && g.t[a + 1] <= start + 1e-6) a++;
  let b = a;
  while (b < g.n && g.t[b] < end - 1e-6) b++;
  return [a, b];
}

/** Rotation (semitones) of B's chroma that best matches A's mean chroma. */
function bestRotation(g: Grid, a0: number, a1: number, b0: number, b1: number): number {
  const ma = new Float64Array(12);
  const mb = new Float64Array(12);
  for (let i = a0; i < a1; i++) for (let k = 0; k < 12; k++) ma[k] += g.chroma[i * 12 + k];
  for (let i = b0; i < b1; i++) for (let k = 0; k < 12; k++) mb[k] += g.chroma[i * 12 + k];
  const dot = (r: number) => {
    let v = 0;
    for (let k = 0; k < 12; k++) v += ma[k] * mb[(k + r) % 12];
    return v;
  };
  // No transposition unless another key fits clearly better (the common case is a plain repeat).
  let best = 0;
  let bestV = dot(0) * 1.05 + 1e-9;
  for (let r = 1; r < 12; r++) {
    const v = dot(r);
    if (v > bestV) {
      bestV = v;
      best = r;
    }
  }
  return best;
}

/** Similarity of two grid windows 0..1 (chroma cosine and energy-profile closeness). */
function beatSim(g: Grid, i: number, j: number, rot: number, sig2: number): number {
  let c = 0;
  for (let k = 0; k < 12; k++) c += g.chroma[i * 12 + k] * g.chroma[j * 12 + ((k + rot) % 12)];
  let d = 0;
  for (let k = 0; k < ENERGY_DIMS; k++) d += (g.energy[i * ENERGY_DIMS + k] - g.energy[j * ENERGY_DIMS + k]) ** 2;
  return 0.45 * Math.max(0, c) + 0.55 * Math.exp(-d / sig2);
}

/** Raw similarity of two sections: the best mean along an aligned diagonal. */
function sectionSim(g: Grid, a0: number, a1: number, b0: number, b1: number, sig2: number): number {
  const la = a1 - a0;
  const lb = b1 - b0;
  const short = Math.min(la, lb);
  if (short < MIN_BEATS) return 0;
  const rot = bestRotation(g, a0, a1, b0, b1);
  // A transposed return is still a return, but a plain one is the stronger match.
  const keyPen = rot === 0 ? 1 : TRANSPOSE_PENALTY;
  const maxLag = Math.max(1, Math.floor(short * MAX_LAG));
  let best = 0;
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    // Align B's start with A's start + lag.
    let sum = 0;
    let cnt = 0;
    for (let i = 0; i < la; i++) {
      const j = i - lag;
      if (j < 0 || j >= lb) continue;
      sum += beatSim(g, a0 + i, b0 + j, rot, sig2);
      cnt++;
    }
    if (cnt < short * MIN_OVERLAP) continue;
    const v = (sum / cnt) * keyPen;
    if (v > best) best = v;
  }
  return best;
}

/**
 * Repetition structure of a song: one entry per section (same order as `sections`).
 * Deterministic; sections too short to compare are never grouped.
 */
export function detectRepeats(r: Input): SectionRepeat[] {
  const secs = r.sections ?? [];
  const out: SectionRepeat[] = secs.map((_, i) => ({ group: i, of: -1, n: 0, sim: 0, returnSim: 0 }));
  if (secs.length < 2 || !(r.numFrames > 0) || !r.chroma || r.chroma.length < r.numFrames * 12) return renumber(out);
  const g = buildGrid(r);
  // Energy distance scale: median pairwise distance over a sample of window pairs.
  const samples: number[] = [];
  for (let i = 0; i < g.n; i += 2) {
    for (let j = i + 1; j < g.n; j += 3) {
      let d = 0;
      for (let k = 0; k < ENERGY_DIMS; k++) d += (g.energy[i * ENERGY_DIMS + k] - g.energy[j * ENERGY_DIMS + k]) ** 2;
      samples.push(d);
    }
  }
  samples.sort((a, b) => a - b);
  const sig2 = Math.max(0.5, samples[samples.length >> 1] || 1);

  const spans = secs.map((s) => span(g, s.start, s.end));
  const m = secs.length;
  const raw = new Float64Array(m * m);
  const pairs: number[] = [];
  for (let j = 1; j < m; j++) {
    for (let i = 0; i < j; i++) {
      const v = sectionSim(g, spans[i][0], spans[i][1], spans[j][0], spans[j][1], sig2);
      raw[i * m + j] = v;
      if (v > 0) pairs.push(v);
    }
  }
  if (!pairs.length) return renumber(out);
  // Baseline: what two sections of this song share anyway (the lower half of the pair similarities).
  pairs.sort((a, b) => a - b);
  const base = pairs[Math.floor((pairs.length - 1) * 0.35)];
  const top = Math.max(base + 0.08, 0.97);
  const norm = (v: number) => Math.max(0, Math.min(1, (v - base) / (top - base)));

  for (let j = 1; j < m; j++) {
    let bi = -1;
    let bv = 0;
    for (let i = 0; i < j; i++) {
      const v = raw[i * m + j];
      if (v > bv + 1e-9) {
        bv = v;
        bi = i;
      }
    }
    if (bi >= 0 && norm(bv) >= REPEAT_THRESHOLD) {
      out[j].group = out[bi].group;
      out[j].sim = norm(bv);
    }
  }
  // First appearance, occurrence index and how strongly each group comes back.
  const first = new Map<number, number>();
  const count = new Map<number, number>();
  for (let j = 0; j < m; j++) {
    const gid = out[j].group;
    if (!first.has(gid)) first.set(gid, j);
    const c = count.get(gid) ?? 0;
    out[j].n = c;
    out[j].of = c === 0 ? -1 : first.get(gid)!;
    count.set(gid, c + 1);
  }
  for (let j = 0; j < m; j++) {
    let best = 0;
    for (let k = j + 1; k < m; k++) if (out[k].group === out[j].group) best = Math.max(best, out[k].sim);
    out[j].returnSim = best;
  }
  return renumber(out);
}

/** Group ids 0, 1, 2... in order of first appearance. */
function renumber(out: SectionRepeat[]): SectionRepeat[] {
  const ids = new Map<number, number>();
  for (const x of out) {
    if (!ids.has(x.group)) ids.set(x.group, ids.size);
    x.group = ids.get(x.group)!;
  }
  return out;
}

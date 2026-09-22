// Chroma features and key estimation (Krumhansl-Schmuckler profiles on
// bar-synchronous chroma, smoothed with a 24-state Viterbi).

import type { KeySegment } from '../types';
import { stftPower } from './stft';
import { slidingMedian } from './dsp';

export interface ChromaResult {
  /** Tc * 12, log-compressed, each frame normalized to max 1 (0 for silent frames). */
  chroma: Float32Array;
  /** Tc, per-frame chroma energy weight 0..1 (relative loudness). */
  weight: Float32Array;
  numFrames: number;
  frameRate: number;
}

/**
 * Chroma from a long-window STFT (good pitch resolution in the bass), with a
 * temporal median over bins to suppress transients (harmonic enhancement).
 */
export function computeChroma(x: Float32Array, sr: number, n: number, hop: number): ChromaResult {
  const Tc = Math.max(1, Math.floor((x.length - 1) / hop) + 1);
  const binHz = sr / n;
  const kLo = Math.max(1, Math.ceil(55 / binHz));
  const kHi = Math.min(n / 2, Math.floor(5000 / binHz));
  const K = Math.max(1, kHi - kLo + 1);
  const M = new Float32Array(Tc * K);
  stftPower(x, n, hop, Tc, (f, pow) => {
    const o = f * K;
    for (let k = 0; k < K; k++) M[o + k] = Math.sqrt(pow[kLo + k]);
  });
  // Harmonic enhancement: median over ~7 frames per bin.
  const H = new Float32Array(Tc * K);
  const w = 7;
  const win = new Float64Array(w + 2);
  const pad = new Float64Array(Tc + w + 2);
  for (let k = 0; k < K; k++) slidingMedian(M, k, K, Tc, w, H, k, K, win, pad);

  // Bin -> pitch-class weights (two nearest classes, cosine falloff).
  const pcA = new Int32Array(K);
  const pcB = new Int32Array(K);
  const wA = new Float32Array(K);
  const wB = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    const f = (kLo + k) * binHz;
    const midi = 69 + 12 * Math.log2(f / 440);
    const lo = Math.floor(midi);
    const frac = midi - lo;
    pcA[k] = ((lo % 12) + 12) % 12;
    pcB[k] = (pcA[k] + 1) % 12;
    // Weight concentrated near semitone centers.
    const ca = Math.cos((Math.PI / 2) * Math.min(1, frac * 2));
    const cb = Math.cos((Math.PI / 2) * Math.min(1, (1 - frac) * 2));
    wA[k] = ca * ca;
    wB[k] = cb * cb;
  }
  const raw = new Float32Array(Tc * 12);
  const energy = new Float32Array(Tc);
  let maxE = 0;
  for (let t = 0; t < Tc; t++) {
    const o = t * K;
    const c = t * 12;
    let e = 0;
    for (let k = 0; k < K; k++) {
      const v = H[o + k];
      const p = v * v;
      raw[c + pcA[k]] += p * wA[k];
      raw[c + pcB[k]] += p * wB[k];
      e += p;
    }
    energy[t] = e;
    if (e > maxE) maxE = e;
  }
  const chroma = new Float32Array(Tc * 12);
  const weight = new Float32Array(Tc);
  if (maxE <= 0) return { chroma, weight, numFrames: Tc, frameRate: sr / hop };
  let gMax = 0;
  for (let i = 0; i < raw.length; i++) if (raw[i] > gMax) gMax = raw[i];
  for (let t = 0; t < Tc; t++) {
    const rel = energy[t] / maxE;
    const db = 10 * Math.log10(rel + 1e-12);
    const wgt = Math.max(0, Math.min(1, (db + 60) / 60));
    weight[t] = wgt;
    if (wgt <= 0) continue;
    let m = 0;
    for (let k = 0; k < 12; k++) {
      const v = Math.log1p((1000 * raw[t * 12 + k]) / gMax);
      chroma[t * 12 + k] = v;
      if (v > m) m = v;
    }
    if (m > 0) for (let k = 0; k < 12; k++) chroma[t * 12 + k] /= m;
  }
  return { chroma, weight, numFrames: Tc, frameRate: sr / hop };
}

/** Resample chroma (Tc frames at rate rc) to T frames at rate r by linear interpolation. */
export function resampleChroma(c: ChromaResult, T: number, frameRate: number): Float32Array {
  const out = new Float32Array(T * 12);
  const Tc = c.numFrames;
  for (let t = 0; t < T; t++) {
    const pos = (t / frameRate) * c.frameRate;
    let i = Math.floor(pos);
    let f = pos - i;
    if (i >= Tc - 1) {
      i = Tc - 1;
      f = 0;
    }
    if (i < 0) {
      i = 0;
      f = 0;
    }
    const j = Math.min(Tc - 1, i + 1);
    for (let k = 0; k < 12; k++) out[t * 12 + k] = c.chroma[i * 12 + k] * (1 - f) + c.chroma[j * 12 + k] * f;
  }
  return out;
}

const MIN_KEY_SECONDS = 10;

const KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function pearson(a: ArrayLike<number>, b: ArrayLike<number>, rot: number): number {
  let ma = 0,
    mb = 0;
  for (let i = 0; i < 12; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= 12;
  mb /= 12;
  let num = 0,
    da = 0,
    db = 0;
  for (let i = 0; i < 12; i++) {
    const x = a[(i + rot) % 12] - ma;
    const y = b[i] - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
}

/** Correlation with all 24 keys: index = tonic + 12 * (minor ? 1 : 0). */
export function keyCorrelations(ch: ArrayLike<number>, out: Float64Array): void {
  for (let t = 0; t < 12; t++) {
    out[t] = pearson(ch, KS_MAJOR, t);
    out[t + 12] = pearson(ch, KS_MINOR, t);
  }
}

/**
 * Key segments from chroma accumulated between consecutive boundaries (bar
 * starts, in seconds). Emission = correlation of each unit (neighbours lightly weighted);
 * Viterbi with a switching penalty yields few, confident segments.
 */
export function detectKeys(c: ChromaResult, boundaries: number[], duration: number): KeySegment[] {
  const edges = boundaries.filter((b) => b > 0 && b < duration);
  edges.unshift(0);
  edges.push(duration);
  const U = edges.length - 1;
  const fallback: KeySegment[] = [{ start: 0, end: duration, tonic: 0, mode: 'major', confidence: 0 }];
  if (U <= 0 || duration <= 0) return fallback;
  const unit = new Float64Array(U * 12);
  const unitW = new Float64Array(U);
  for (let u = 0; u < U; u++) {
    const a = Math.max(0, Math.floor(edges[u] * c.frameRate));
    const b = Math.min(c.numFrames, Math.max(a + 1, Math.floor(edges[u + 1] * c.frameRate)));
    for (let t = a; t < b; t++) {
      const w = c.weight[t];
      unitW[u] += w;
      for (let k = 0; k < 12; k++) unit[u * 12 + k] += w * c.chroma[t * 12 + k];
    }
  }
  let totalW = 0;
  for (let u = 0; u < U; u++) totalW += unitW[u];
  if (totalW <= 0) return fallback;

  const S = 24;
  const em = new Float64Array(U * S);
  const tmp = new Float64Array(12);
  const corr = new Float64Array(S);
  for (let u = 0; u < U; u++) {
    tmp.fill(0);
    let w = 0;
    // Own unit at full weight, neighbours lightly (stabilizes very short bars).
    for (let v = Math.max(0, u - 1); v <= Math.min(U - 1, u + 1); v++) {
      const g = v === u ? 1 : 0.25;
      for (let k = 0; k < 12; k++) tmp[k] += g * unit[v * 12 + k];
      w += g * unitW[v];
    }
    if (w <= 1e-6) continue; // silent: neutral emission
    keyCorrelations(tmp, corr);
    // Evidence scaled by how much signal the unit has (silent bars carry little).
    const conf = Math.min(1, unitW[u] / Math.max(1e-9, (totalW / U) * 0.5));
    // Evidence per second rather than per unit, so the switching penalty means
    // the same amount of music at any tempo.
    const secs = Math.min(4, edges[u + 1] - edges[u]) / 2;
    for (let s = 0; s < S; s++) em[u * S + s] = 10 * conf * secs * corr[s];
  }
  const penalty = 12;
  const score = new Float64Array(S);
  const next = new Float64Array(S);
  const back = new Int32Array(U * S);
  for (let s = 0; s < S; s++) score[s] = em[s];
  for (let u = 1; u < U; u++) {
    let bestPrev = 0;
    for (let s = 1; s < S; s++) if (score[s] > score[bestPrev]) bestPrev = s;
    for (let s = 0; s < S; s++) {
      const stay = score[s];
      const sw = score[bestPrev] - penalty;
      if (stay >= sw) {
        next[s] = stay + em[u * S + s];
        back[u * S + s] = s;
      } else {
        next[s] = sw + em[u * S + s];
        back[u * S + s] = bestPrev;
      }
    }
    score.set(next);
  }
  let s = 0;
  for (let k = 1; k < S; k++) if (score[k] > score[s]) s = k;
  const path = new Int32Array(U);
  for (let u = U - 1; u >= 0; u--) {
    path[u] = s;
    s = back[u * S + s];
  }
  // Absorb short excursions (< MIN_KEY_SECONDS) into the better-fitting neighbour.
  const runs = (): [number, number][] => {
    const r: [number, number][] = [];
    let a = 0;
    for (let u = 1; u <= U; u++) if (u === U || path[u] !== path[a]) {
      r.push([a, u]);
      a = u;
    }
    return r;
  };
  for (let guard = 0; guard < U; guard++) {
    const r = runs();
    if (r.length <= 1) break;
    let shortest = -1;
    let shortDur = Infinity;
    for (let i = 0; i < r.length; i++) {
      const d = edges[r[i][1]] - edges[r[i][0]];
      if (d < shortDur) {
        shortDur = d;
        shortest = i;
      }
    }
    if (shortDur >= MIN_KEY_SECONDS) break;
    const [a, b] = r[shortest];
    tmp.fill(0);
    for (let v = a; v < b; v++) for (let k = 0; k < 12; k++) tmp[k] += unit[v * 12 + k];
    keyCorrelations(tmp, corr);
    const left = shortest > 0 ? path[r[shortest - 1][0]] : -1;
    const right = shortest + 1 < r.length ? path[r[shortest + 1][0]] : -1;
    const target = left < 0 ? right : right < 0 ? left : corr[left] >= corr[right] ? left : right;
    for (let v = a; v < b; v++) path[v] = target;
  }

  const segs: KeySegment[] = [];
  let startU = 0;
  for (let u = 1; u <= U; u++) {
    if (u === U || path[u] !== path[startU]) {
      const st = path[startU];
      // Confidence: correlation of the segment's summed chroma with the chosen profile.
      tmp.fill(0);
      for (let v = startU; v < u; v++) for (let k = 0; k < 12; k++) tmp[k] += unit[v * 12 + k];
      keyCorrelations(tmp, corr);
      segs.push({
        start: edges[startU],
        end: edges[u],
        tonic: st % 12,
        mode: st >= 12 ? 'minor' : 'major',
        confidence: Math.max(0, Math.min(1, corr[st])),
      });
      startU = u;
    }
  }
  segs[0].start = 0;
  segs[segs.length - 1].end = duration;
  return segs;
}

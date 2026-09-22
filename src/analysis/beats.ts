// Onset envelope, tempo estimation, dynamic-programming beat tracking
// (Ellis 2007) and downbeat phase selection.

import { stftPower, logBands } from './stft';
import { movingAverage, percentiles } from './dsp';

/**
 * Log-magnitude spectral flux on ~40 log-spaced bands with a frequency max-filter
 * on the reference frame (suppresses vibrato), half-wave rectified, local-mean
 * removed and gain-normalized so quiet passages still carry beat evidence.
 */
export function onsetEnvelope(x: Float32Array, sr: number, n: number, hop: number, T: number): Float32Array {
  const bands = logBands(n, sr, 30, Math.min(11000, sr * 0.48), 40);
  const B = bands.count;
  const S = new Float32Array(T * B);
  stftPower(x, n, hop, T, (f, pow) => {
    const o = f * B;
    for (let b = 0; b < B; b++) {
      let s = 0;
      for (let k = bands.lo[b]; k < bands.hi[b]; k++) s += pow[k];
      S[o + b] = 10 * Math.log10(s + 1e-20);
    }
  });
  let top = -Infinity;
  for (let i = 0; i < S.length; i++) if (S[i] > top) top = S[i];
  const floor = top - 80;
  for (let i = 0; i < S.length; i++) if (S[i] < floor) S[i] = floor;
  const raw = new Float32Array(T);
  for (let t = 1; t < T; t++) {
    const o = t * B;
    const p = (t - 1) * B;
    let acc = 0;
    for (let b = 0; b < B; b++) {
      let ref = S[p + b];
      if (b > 0 && S[p + b - 1] > ref) ref = S[p + b - 1];
      if (b + 1 < B && S[p + b + 1] > ref) ref = S[p + b + 1];
      const d = S[o + b] - ref;
      if (d > 0) acc += d;
    }
    raw[t] = acc / B;
  }
  return finishOnset(raw, sr / hop);
}

function finishOnset(raw: Float32Array, fr: number): Float32Array {
  const T = raw.length;
  const w = Math.max(3, Math.round(0.5 * fr) | 1);
  const local = movingAverage(raw, w);
  const env = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const d = raw[t] - local[t];
    env[t] = d > 0 ? d : 0;
  }
  // Adaptive gain: divide by local RMS over ~4 s mixed with the global RMS.
  const sq = new Float32Array(T);
  let g = 0;
  for (let t = 0; t < T; t++) {
    sq[t] = env[t] * env[t];
    g += sq[t];
  }
  const grms = Math.sqrt(g / Math.max(1, T));
  if (!(grms > 1e-9)) return env;
  const lrms = movingAverage(sq, Math.max(3, Math.round(4 * fr) | 1));
  for (let t = 0; t < T; t++) {
    env[t] = env[t] / (Math.sqrt(lrms[t]) + 0.35 * grms);
  }
  return env;
}

export interface TempoEstimate {
  bpm: number;
  period: number; // frames per beat
  confidence: number;
}

/** Autocorrelation tempo with a log-Gaussian prior around 120 BPM, folded into 70..180. */
export function estimateTempo(env: Float32Array, fr: number): TempoEstimate {
  const T = env.length;
  const maxLag = Math.min(T - 1, Math.ceil((fr * 60) / 30));
  if (maxLag < 4) return { bpm: 120, period: (60 * fr) / 120, confidence: 0 };
  let mean = 0;
  for (let t = 0; t < T; t++) mean += env[t];
  mean /= T;
  const e = new Float64Array(T);
  for (let t = 0; t < T; t++) e[t] = env[t] - mean;
  const acf = new Float64Array(maxLag + 2);
  for (let l = 0; l <= maxLag; l++) {
    let s = 0;
    for (let t = 0; t + l < T; t++) s += e[t] * e[t + l];
    acf[l] = s / (T - l);
  }
  const a0 = acf[0] > 0 ? acf[0] : 1;
  const at = (lag: number): number => {
    if (lag >= maxLag) return 0;
    const i = Math.floor(lag);
    const f = lag - i;
    return (acf[i] * (1 - f) + acf[i + 1] * f) / a0;
  };
  let bestBpm = 120;
  let bestScore = -Infinity;
  const scores: number[] = [];
  for (let bpm = 40; bpm <= 240; bpm += 0.05) {
    const L = (60 * fr) / bpm;
    const s = at(L) + 0.5 * at(2 * L) + 0.25 * at(4 * L);
    const oct = Math.log2(bpm / 120);
    const prior = Math.exp(-0.5 * (oct / 0.9) * (oct / 0.9));
    const v = s * prior;
    scores.push(v);
    if (v > bestScore) {
      bestScore = v;
      bestBpm = bpm;
    }
  }
  if (!(bestScore > 0)) return { bpm: 120, period: (60 * fr) / 120, confidence: 0 };
  let bpm = bestBpm;
  while (bpm < 70) bpm *= 2;
  while (bpm > 180) bpm /= 2;
  const sorted = scores.slice().sort((a, b) => a - b);
  const med = sorted[sorted.length >> 1];
  const confidence = Math.max(0, Math.min(1, (bestScore - med) / (Math.abs(bestScore) + 1e-9)));
  return { bpm, period: (60 * fr) / bpm, confidence };
}

/** Ellis 2007 DP beat tracker. Returns fractional beat frame positions. */
export function trackBeats(env: Float32Array, period: number, tightness = 100): Float64Array {
  const T = env.length;
  if (T < 2 || !(period > 1)) return new Float64Array(0);
  // Normalize onset by its std.
  let m = 0;
  for (let t = 0; t < T; t++) m += env[t];
  m /= T;
  let v = 0;
  for (let t = 0; t < T; t++) v += (env[t] - m) * (env[t] - m);
  const sd = Math.sqrt(v / T) || 1;
  const o = new Float64Array(T);
  for (let t = 0; t < T; t++) o[t] = env[t] / sd;

  const score = new Float64Array(T);
  const back = new Int32Array(T).fill(-1);
  const lo = Math.max(1, Math.round(period / 2));
  const hi = Math.max(lo + 1, Math.round(2 * period));
  const pen = new Float64Array(hi + 1);
  for (let d = lo; d <= hi; d++) {
    const r = Math.log(d / period);
    pen[d] = -tightness * r * r;
  }
  for (let t = 0; t < T; t++) {
    let best = -Infinity;
    let arg = -1;
    const dMax = Math.min(hi, t);
    for (let d = lo; d <= dMax; d++) {
      const c = score[t - d] + pen[d];
      if (c > best) {
        best = c;
        arg = t - d;
      }
    }
    if (arg >= 0 && best > 0) {
      score[t] = o[t] + best;
      back[t] = arg;
    } else {
      score[t] = o[t];
      back[t] = -1;
    }
  }
  // End: best score within the last period, preferring an onset peak.
  let end = T - 1;
  let bestEnd = -Infinity;
  for (let t = Math.max(0, T - Math.ceil(period)); t < T; t++) {
    if (score[t] > bestEnd) {
      bestEnd = score[t];
      end = t;
    }
  }
  const out: number[] = [];
  let t = end;
  while (t >= 0) {
    out.push(t);
    t = back[t];
  }
  out.reverse();
  return Float64Array.from(out);
}

/**
 * Sub-frame refinement and grid regularization. Beats in weak-onset stretches
 * are re-spaced uniformly between strong anchors (the DP drifts toward integer
 * periods where there is no evidence), and leading / trailing weak beats are
 * extrapolated from the local tempo. Missing beats before/after the DP chain are
 * filled out to cover [0, T).
 */
export function refineBeats(beats: Float64Array, env: Float32Array, period: number): Float64Array {
  const n = beats.length;
  const T = env.length;
  if (n === 0) return beats;
  const pos = Float64Array.from(beats);
  const strength = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const b = Math.round(pos[i]);
    let best = Math.max(1, Math.min(T - 2, b));
    for (let k = Math.max(1, b - 2); k <= Math.min(T - 2, b + 2); k++) if (env[k] > env[best]) best = k;
    let s = env[Math.max(0, Math.min(T - 1, best))];
    if (best > 0 && best < T - 1) {
      const y0 = env[best - 1];
      const y1 = env[best];
      const y2 = env[best + 1];
      const den = y0 - 2 * y1 + y2;
      let delta = den < 0 ? (0.5 * (y0 - y2)) / den : 0;
      if (delta > 0.5) delta = 0.5;
      if (delta < -0.5) delta = -0.5;
      if (Math.abs(best - b) <= 1) pos[i] = best + delta;
      s = y1;
    }
    strength[i] = s;
  }
  const [p75] = percentiles(strength, [0.75]);
  const thr = 0.3 * p75;
  const strong: boolean[] = [];
  for (let i = 0; i < n; i++) strong.push(strength[i] >= thr && thr > 0);
  // Interpolate interior weak runs.
  let prevStrong = -1;
  for (let i = 0; i < n; i++) {
    if (!strong[i]) continue;
    if (prevStrong >= 0 && i - prevStrong > 1) {
      const a = pos[prevStrong];
      const b = pos[i];
      const k = i - prevStrong;
      for (let j = 1; j < k; j++) pos[prevStrong + j] = a + ((b - a) * j) / k;
    }
    prevStrong = i;
  }
  const firstStrong = strong.indexOf(true);
  const lastStrong = strong.lastIndexOf(true);
  const localPeriod = (from: number, dir: number): number => {
    // Regression over up to 16 beats starting at `from` going in `dir`.
    const idx: number[] = [];
    for (let i = from; i >= 0 && i < n && idx.length < 17; i += dir) idx.push(i);
    if (idx.length < 3) return period;
    idx.sort((a, b) => a - b);
    const m = idx.length;
    let sx = 0,
      sy = 0;
    for (const i of idx) {
      sx += i;
      sy += pos[i];
    }
    sx /= m;
    sy /= m;
    let num = 0,
      den = 0;
    for (const i of idx) {
      num += (i - sx) * (pos[i] - sy);
      den += (i - sx) * (i - sx);
    }
    const p = den > 0 ? num / den : period;
    return p > period * 0.8 && p < period * 1.25 ? p : period;
  };
  if (firstStrong > 0) {
    const p = localPeriod(firstStrong, 1);
    for (let i = firstStrong - 1; i >= 0; i--) pos[i] = pos[firstStrong] - (firstStrong - i) * p;
  }
  if (lastStrong >= 0 && lastStrong < n - 1) {
    const p = localPeriod(lastStrong, -1);
    for (let i = lastStrong + 1; i < n; i++) pos[i] = pos[lastStrong] + (i - lastStrong) * p;
  }
  // Extend the grid to the edges.
  const out: number[] = [];
  const pStart = localPeriod(0, 1);
  const lead: number[] = [];
  for (let x = pos[0] - pStart; x >= 0; x -= pStart) lead.push(x);
  lead.reverse();
  out.push(...lead);
  for (let i = 0; i < n; i++) if (pos[i] >= 0 && pos[i] < T) out.push(pos[i]);
  const pEnd = localPeriod(n - 1, -1);
  for (let x = pos[n - 1] + pEnd; x < T; x += pEnd) out.push(x);
  // Ensure strictly increasing.
  const res: number[] = [];
  for (const x of out) if (res.length === 0 || x > res[res.length - 1] + period * 0.4) res.push(x);
  return Float64Array.from(res);
}

/** Max of env within +-r frames of position p. */
function peakNear(env: Float32Array, p: number, r: number): number {
  const c = Math.round(p);
  let m = 0;
  for (let k = Math.max(0, c - r); k <= Math.min(env.length - 1, c + r); k++) if (env[k] > m) m = env[k];
  return m;
}

function zscore(x: Float64Array): Float64Array {
  const n = x.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= Math.max(1, n);
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) * (x[i] - m);
  const sd = Math.sqrt(v / Math.max(1, n)) || 1;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - m) / sd;
  return out;
}

/**
 * Choose the downbeat phase (0..beatsPerBar-1) from low-frequency onsets, bass
 * onsets and chroma change, penalizing snare hits (backbeats).
 * `chroma` is T*12 at the same frame rate as beat positions.
 */
export function downbeatPhase(
  beats: Float64Array,
  kick: Float32Array,
  bassOnset: Float32Array,
  snare: Float32Array,
  chroma: Float32Array,
  beatsPerBar: number,
): number {
  const n = beats.length;
  if (n < beatsPerBar * 2) return 0;
  const low = new Float64Array(n);
  const bas = new Float64Array(n);
  const sn = new Float64Array(n);
  const chg = new Float64Array(n);
  const T = kick.length;
  // Beat-synchronous chroma.
  const bc = new Float64Array(n * 12);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, Math.round(beats[i]));
    const b = Math.min(T, Math.max(a + 1, Math.round(i + 1 < n ? beats[i + 1] : beats[i] + (beats[i] - beats[Math.max(0, i - 1)]))));
    for (let t = a; t < b; t++) for (let k = 0; k < 12; k++) bc[i * 12 + k] += chroma[t * 12 + k];
  }
  const cos = (i0: number, i1: number, j0: number, j1: number): number => {
    let dot = 0,
      na = 0,
      nb = 0;
    for (let k = 0; k < 12; k++) {
      let a = 0,
        b = 0;
      for (let i = i0; i < i1; i++) a += bc[i * 12 + k];
      for (let j = j0; j < j1; j++) b += bc[j * 12 + k];
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 1;
  };
  for (let i = 0; i < n; i++) {
    low[i] = peakNear(kick, beats[i], 2);
    bas[i] = peakNear(bassOnset, beats[i], 2);
    sn[i] = peakNear(snare, beats[i], 2);
    const i0 = Math.max(0, i - 2);
    const i1 = Math.min(n, i + 2);
    chg[i] = i0 < i && i < i1 ? 1 - cos(i0, i, i, i1) : 0;
  }
  const zl = zscore(low);
  const zb = zscore(bas);
  const zs = zscore(sn);
  const zc = zscore(chg);
  let best = 0;
  let bestScore = -Infinity;
  for (let ph = 0; ph < beatsPerBar; ph++) {
    let s = 0;
    let c = 0;
    for (let i = ph; i < n; i += beatsPerBar) {
      s += zl[i] + zb[i] + 1.5 * zc[i] - 0.5 * zs[i];
      c++;
    }
    s /= Math.max(1, c);
    if (s > bestScore) {
      bestScore = s;
      best = ph;
    }
  }
  return best;
}

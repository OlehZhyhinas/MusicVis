// Song structure: beat-synchronous self-similarity, Foote novelty, boundary
// selection on downbeats with a length-aware dynamic program, then
// energy-based labeling (intro / verse / build / chorus / drop / breakdown / outro).

import type { Section, SectionLabel } from '../types';
import { clamp01, mean, movingAverage } from './dsp';

export interface StructureInput {
  duration: number;
  frameRate: number;
  numFrames: number;
  beats: Float64Array; // seconds
  downbeats: Float64Array; // seconds (subset of beats)
  chroma: Float32Array; // T * 12
  timbre: Float32Array; // T * timbreBands (dB)
  timbreBands: number;
  loudness: Float32Array; // 0..1
  drums: Float32Array; // 0..1
  bass: Float32Array; // 0..1
  drumOnsets: Float32Array; // 0..1
  /** Absolute measures (complexity.ts). When given, they gate 'drop' labels and segmentation density. */
  complexity?: Float32Array; // 0..1
  drumsPresence?: Float32Array; // 0..1
  bassPresence?: Float32Array; // 0..1
  songComplexity?: number;
}

// A 'drop' must be dense and drum + bass heavy in absolute terms, and arrive
// with a clear lift (after a build, or a large jump in complexity).
const DROP_MIN_COMPLEXITY = 0.65;
const DROP_MIN_PRESENCE = 0.4;
const DROP_MIN_CX_JUMP = 0.15; // when the preceding section is not a build
const DROP_MIN_CX_JUMP_AFTER_BUILD = -0.1; // a build already carries the lift (its energy jump is checked separately)

interface SegStats {
  start: number;
  end: number;
  a: number; // frame range
  b: number;
  loud: number;
  drums: number;
  bass: number;
  act: number; // activity: loudness + onset density
  e: number; // combined energy
  ramp: number; // rise of activity over the span
  cx: number; // mean absolute complexity (0.5 when unknown)
  pd: number; // mean drums presence
  pb: number; // mean bass presence
  label: SectionLabel;
}

function frameOf(t: number, fr: number, T: number): number {
  return Math.max(0, Math.min(T, Math.round(t * fr)));
}

export function detectSections(inp: StructureInput): Section[] {
  const { duration, frameRate: fr, numFrames: T } = inp;
  const whole: Section[] = [
    {
      start: 0,
      end: Math.max(duration, 1e-3),
      label: 'verse',
      energy: clamp01(mean(inp.loudness)),
    },
  ];
  const beats = Array.from(inp.beats).filter((b) => b >= 0 && b < duration);
  const nB = beats.length;
  if (nB < 16 || T < 4) return whole;

  // --- Beat-synchronous features ---
  const D = inp.timbreBands;
  const bc = new Float64Array(nB * 12);
  const bt = new Float64Array(nB * D);
  for (let i = 0; i < nB; i++) {
    const a = frameOf(beats[i], fr, T);
    const b = Math.max(a + 1, frameOf(i + 1 < nB ? beats[i + 1] : duration, fr, T));
    const bb = Math.min(T, b);
    const n = Math.max(1, bb - a);
    for (let t = a; t < bb; t++) {
      for (let k = 0; k < 12; k++) bc[i * 12 + k] += inp.chroma[t * 12 + k];
      for (let k = 0; k < D; k++) bt[i * D + k] += inp.timbre[t * D + k];
    }
    for (let k = 0; k < 12; k++) bc[i * 12 + k] /= n;
    for (let k = 0; k < D; k++) bt[i * D + k] /= n;
  }
  // Standardize timbre dims over the song.
  for (let k = 0; k < D; k++) {
    // Robust scale (MAD): quiet intros / outros must not dominate the variance,
    // or level changes between the louder sections become invisible.
    const dev = new Float64Array(nB);
    const col = new Float64Array(nB);
    for (let i = 0; i < nB; i++) col[i] = bt[i * D + k];
    col.sort();
    const med = col[nB >> 1];
    for (let i = 0; i < nB; i++) dev[i] = Math.abs(bt[i * D + k] - med);
    dev.sort();
    const sd = Math.max(1, 1.4826 * dev[nB >> 1]);
    for (let i = 0; i < nB; i++) bt[i * D + k] = Math.max(-6, Math.min(6, (bt[i * D + k] - med) / sd));
  }
  // Chroma: unit-normalize.
  for (let i = 0; i < nB; i++) {
    let s = 0;
    for (let k = 0; k < 12; k++) s += bc[i * 12 + k] ** 2;
    s = Math.sqrt(s);
    if (s > 0) for (let k = 0; k < 12; k++) bc[i * 12 + k] /= s;
  }

  // --- Self-similarity ---
  const Sm = new Float32Array(nB * nB);
  const d2 = new Float32Array(nB * nB);
  const samples: number[] = [];
  for (let i = 0; i < nB; i++) {
    for (let j = i; j < nB; j++) {
      let d = 0;
      for (let k = 0; k < D; k++) {
        const x = bt[i * D + k] - bt[j * D + k];
        d += x * x;
      }
      d2[i * nB + j] = d;
      d2[j * nB + i] = d;
      if ((i * 7 + j) % 5 === 0) samples.push(d);
    }
  }
  samples.sort((a, b) => a - b);
  const sig2 = Math.max(1e-6, samples[samples.length >> 1] || 1);
  for (let i = 0; i < nB; i++) {
    for (let j = i; j < nB; j++) {
      let c = 0;
      for (let k = 0; k < 12; k++) c += bc[i * 12 + k] * bc[j * 12 + k];
      const s = 0.4 * c + 0.6 * Math.exp(-d2[i * nB + j] / sig2);
      Sm[i * nB + j] = s;
      Sm[j * nB + i] = s;
    }
  }

  // --- Foote novelty with a Gaussian-tapered checkerboard kernel ---
  const K = Math.min(16, Math.max(2, Math.floor(nB / 4)));
  const g = new Float64Array(2 * K);
  for (let a = 0; a < 2 * K; a++) {
    const x = (a - K + 0.5) / (K * 0.5);
    g[a] = Math.exp(-0.5 * x * x);
  }
  let gTotal = 0;
  for (let a = 0; a < 2 * K; a++) for (let b = 0; b < 2 * K; b++) gTotal += g[a] * g[b];
  const nov = new Float64Array(nB);
  for (let i = 0; i < nB; i++) {
    let s = 0;
    let wsum = 0;
    for (let a = 0; a < 2 * K; a++) {
      const ia = i - K + a;
      if (ia < 0 || ia >= nB) continue;
      const sa = a < K ? -1 : 1;
      for (let b = 0; b < 2 * K; b++) {
        const ib = i - K + b;
        if (ib < 0 || ib >= nB) continue;
        const sb = b < K ? -1 : 1;
        const w = g[a] * g[b];
        s += sa * sb * w * Sm[ia * nB + ib];
        wsum += w;
      }
    }
    // Taper near the song edges, where half the kernel falls outside the song.
    nov[i] = wsum > 0 ? Math.max(0, s / wsum) * (wsum / gTotal) : 0;
  }

  // --- Candidate boundaries at downbeats ---
  const beatPeriod = (beats[nB - 1] - beats[0]) / Math.max(1, nB - 1);
  const cands: number[] = [0];
  const candNov: number[] = [0];
  for (const d of inp.downbeats) {
    // A pickup shorter than half a bar belongs to the first bar.
    if (d < beatPeriod * 2 || d > duration - beatPeriod * 2) continue;
    // nearest beat index
    let lo = 0,
      hi = nB - 1;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (beats[m] < d) lo = m + 1;
      else hi = m;
    }
    let bi = lo;
    if (bi > 0 && Math.abs(beats[bi - 1] - d) < Math.abs(beats[bi] - d)) bi--;
    let v = 0;
    for (let k = Math.max(0, bi - 1); k <= Math.min(nB - 1, bi + 1); k++) v = Math.max(v, nov[k] * (k === bi ? 1 : 0.8));
    cands.push(d);
    candNov.push(v);
  }
  cands.push(duration);
  candNov.push(0);
  const M = cands.length;
  if (M < 3) return whole;
  const sortedNov = candNov.slice(1, M - 1).sort((a, b) => a - b);
  const ref = sortedNov[Math.floor(sortedNov.length * 0.95)] || 1;
  const nn = candNov.map((v) => Math.min(1.5, v / (ref || 1)));

  const totalBars = M - 1;
  // Sparse songs (solo piano, ambient) change texture continuously; without
  // the beat-driven structure of a band, the novelty curve over-segments them.
  const songCx = inp.songComplexity ?? 0.5;
  const sparse = clamp01((0.3 - songCx) / 0.15); // 0 at >= 0.3, 1 at <= 0.15
  const minBars = totalBars >= 24 && sparse > 0 ? (sparse > 0.5 ? 8 : 6) : totalBars >= 12 ? 4 : totalBars >= 4 ? 2 : 1;
  const maxSegs = Math.max(3, Math.min(10, Math.round(duration / (sparse > 0 ? 25 : 20))));
  const lenBonus = (L: number): number => (L % 8 === 0 ? 0.3 : L % 4 === 0 ? 0.18 : 0) - 0.03 * Math.max(0, L - 32);
  const best = new Float64Array(M);
  const prev = new Int32Array(M);
  const runDp = (lambda: number): number => {
    best.fill(-Infinity);
    prev.fill(-1);
    best[0] = 0;
    for (let j = 1; j < M; j++) {
      const gain = j < M - 1 ? nn[j] : 0;
      for (let i = 0; i < j; i++) {
        if (best[i] === -Infinity) continue;
        const L = j - i;
        const edge = i === 0 || j === M - 1;
        if (L < minBars && !(edge && L >= 1 && totalBars < minBars * 2)) continue;
        const v = best[i] + gain + lenBonus(L) - lambda;
        if (v > best[j]) {
          best[j] = v;
          prev[j] = i;
        }
      }
    }
    if (best[M - 1] === -Infinity) return -1;
    let cnt = 0;
    for (let q = M - 1; q > 0; q = prev[q]) cnt++;
    return cnt;
  };
  // Raise the per-boundary cost until the section count is reasonable.
  let lambda = 0.6 + 0.4 * sparse;
  let segCount = runDp(lambda);
  for (let it = 0; it < 12 && segCount > maxSegs; it++) {
    lambda += 0.15;
    segCount = runDp(lambda);
  }
  const bIdx: number[] = [];
  let j = M - 1;
  if (segCount < 0 || best[j] === -Infinity) return whole;
  while (j > 0) {
    bIdx.push(j);
    j = prev[j];
  }
  bIdx.reverse();
  const bounds = [0, ...bIdx.map((k) => cands[k])];

  // --- Stats ---
  const actRaw = movingAverage(inp.drumOnsets, Math.max(3, Math.round(fr * 1.5) | 1));
  let actMax = 0;
  for (let t = 0; t < T; t++) actMax = Math.max(actMax, actRaw[t]);
  const act = new Float32Array(T);
  for (let t = 0; t < T; t++) act[t] = 0.5 * inp.loudness[t] + 0.5 * (actMax > 0 ? actRaw[t] / actMax : 0);
  const mkStats = (start: number, end: number): SegStats => {
    const a = frameOf(start, fr, T);
    const b = Math.max(a + 1, frameOf(end, fr, T));
    const loud = mean(inp.loudness, a, b);
    const drums = mean(inp.drums, a, b);
    const bass = mean(inp.bass, a, b);
    const ac = mean(act, a, b);
    const q = Math.max(1, Math.floor((b - a) / 4));
    const ramp = mean(act, b - q, b) - mean(act, a, a + q);
    const cx = inp.complexity ? mean(inp.complexity, a, b) : 0.5;
    const pd = inp.drumsPresence ? mean(inp.drumsPresence, a, b) : drums;
    const pb = inp.bassPresence ? mean(inp.bassPresence, a, b) : bass;
    return {
      start,
      end,
      a,
      b,
      loud,
      drums,
      bass,
      act: ac,
      e: 0.6 * loud + 0.25 * drums + 0.15 * bass,
      ramp,
      cx,
      pd,
      pb,
      label: 'verse',
    };
  };
  let segs: SegStats[] = [];
  for (let k = 0; k + 1 < bounds.length; k++) segs.push(mkStats(bounds[k], bounds[k + 1]));

  const barLen = beatPeriod * 4;
  // Split a trailing ramp (4-8 bars) off a longer section when it leads into a louder one.
  const split: SegStats[] = [];
  for (let k = 0; k < segs.length; k++) {
    const s = segs[k];
    const nxt = segs[k + 1];
    let done = false;
    if (nxt && nxt.e > s.e + 0.1) {
      for (const bars of [8, 4]) {
        const cut = s.end - bars * barLen;
        if (cut - s.start < 4 * barLen - 1e-3) continue;
        // Snap to nearest downbeat.
        let snapped = cut;
        let bestD = Infinity;
        for (const d of inp.downbeats) {
          const dd = Math.abs(d - cut);
          if (dd < bestD) {
            bestD = dd;
            snapped = d;
          }
        }
        if (bestD > barLen * 0.5) continue;
        const head = mkStats(s.start, snapped);
        const tail = mkStats(snapped, s.end);
        if (tail.ramp > 0.12 && tail.act > head.act + 0.05 && tail.ramp > 2 * Math.max(0, head.ramp)) {
          split.push(head, tail);
          done = true;
          break;
        }
      }
    }
    if (!done) split.push(s);
  }
  segs = split;

  // --- Builds: rising sections that lead into a louder one ---
  const isBuild = segs.map((s, k) => {
    const nx = segs[k + 1];
    if (!nx || nx.e < s.e + 0.05) return false;
    return s.ramp >= (k === 0 ? 0.15 : 0.1);
  });
  // Extend a build backwards over a short rising section (e.g. a build whose
  // second half doubles the snare roll was cut in two by the novelty curve).
  const merged: SegStats[] = [];
  const mergedBuild: boolean[] = [];
  for (let k = 0; k < segs.length; k++) {
    const s = segs[k];
    const pv = merged[merged.length - 1];
    const pvBuild = mergedBuild[mergedBuild.length - 1];
    if (
      isBuild[k] &&
      pv &&
      !pvBuild &&
      k - 1 > 0 &&
      pv.end - pv.start <= 8 * barLen + 1e-3 &&
      pv.act < s.act &&
      pv.e < s.e &&
      pv.ramp >= 0.03 &&
      s.end - pv.start <= 16 * barLen + 1e-3
    ) {
      merged[merged.length - 1] = mkStats(pv.start, s.end);
      mergedBuild[mergedBuild.length - 1] = true;
      continue;
    }
    merged.push(s);
    mergedBuild.push(isBuild[k]);
  }
  segs = merged;

  // --- Labels ---
  const n = segs.length;
  const hasAbs = !!inp.complexity;
  const absDrop = (k: number, afterBuild: boolean): boolean => {
    if (!hasAbs) return true;
    const s = segs[k];
    const p = k > 0 ? segs[k - 1] : null;
    if (!p) return false;
    if (s.cx < DROP_MIN_COMPLEXITY || s.pd < DROP_MIN_PRESENCE || s.pb < DROP_MIN_PRESENCE) return false;
    const dcx = s.cx - p.cx;
    return afterBuild ? dcx >= DROP_MIN_CX_JUMP_AFTER_BUILD : dcx >= DROP_MIN_CX_JUMP;
  };
  let eMin = Infinity,
    eMax = -Infinity;
  for (const s of segs) {
    eMin = Math.min(eMin, s.e);
    eMax = Math.max(eMax, s.e);
  }
  const range = eMax - eMin;
  const rel = segs.map((s) => (range > 0.08 ? (s.e - eMin) / range : 0.5));
  const high = segs.map((s, k) => !mergedBuild[k] && range > 0.08 && (eMax - s.e <= 0.12 || rel[k] >= 0.85));
  const low = rel.map((r) => r <= 0.35);
  for (let k = 0; k < n; k++) {
    if (mergedBuild[k]) segs[k].label = 'build';
    if (!high[k]) continue;
    const p = k > 0 ? segs[k - 1] : null;
    const jump = p ? segs[k].e - p.e : 0;
    const lowJump = p ? segs[k].bass + segs[k].drums - (p.bass + p.drums) : 0;
    const strongLow = segs[k].drums >= 0.45 && segs[k].bass >= 0.45;
    // EDM-style drop: strong kick + bass, arriving after a lift (build), a low
    // passage, or with a very large jump. Otherwise it is a chorus.
    const afterLift = !!p && (mergedBuild[k - 1] || low[k - 1]);
    const relDrop = p && strongLow && ((afterLift && (jump >= 0.2 || lowJump >= 0.5)) || jump >= 0.35);
    segs[k].label = relDrop && absDrop(k, mergedBuild[k - 1]) ? 'drop' : 'chorus';
  }
  for (let k = 1; k < n - 1; k++) {
    if (segs[k].label !== 'verse' || !low[k]) continue;
    let before = false,
      after = false;
    for (let i = 0; i < k; i++) if (high[i]) before = true;
    for (let i = k + 1; i < n; i++) if (high[i]) after = true;
    if (before && after) segs[k].label = 'breakdown';
  }
  if (n >= 2 && !high[0] && (low[0] || rel[0] < 0.5) && segs[0].label !== 'build') segs[0].label = 'intro';
  if (n >= 3 && !high[n - 1] && segs[n - 1].label !== 'build' && (low[n - 1] || rel[n - 1] < 0.5)) segs[n - 1].label = 'outro';
  if (segs[0].label === 'drop') segs[0].label = 'chorus';
  // Quiet verses glued to the intro / outro are part of them.
  for (let k = 1; k < n; k++) if (segs[k].label === 'verse' && low[k] && segs[k - 1].label === 'intro') segs[k].label = 'intro';
  for (let k = n - 2; k >= 0; k--) if (segs[k].label === 'verse' && low[k] && segs[k + 1].label === 'outro') segs[k].label = 'outro';

  // Repeats share labels: a verse that is as similar to a chorus as the chorus
  // is to itself, at comparable energy, is another chorus.
  const beatIdx = (t: number): number => {
    let lo = 0,
      hi = nB - 1,
      ans = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (beats[m] <= t + 1e-6) {
        ans = m;
        lo = m + 1;
      } else hi = m - 1;
    }
    return ans;
  };
  const blockSim = (a: SegStats, b: SegStats): number => {
    const a0 = beatIdx(a.start),
      a1 = Math.max(a0 + 1, beatIdx(a.end));
    const b0 = beatIdx(b.start),
      b1 = Math.max(b0 + 1, beatIdx(b.end));
    // Diagonal (time-lag) similarity: compare beat i of a with beat i of b.
    const L = Math.min(a1 - a0, b1 - b0);
    let sum = 0;
    for (let i = 0; i < L; i++) sum += Sm[(a0 + i) * nB + (b0 + i)];
    return L > 0 ? sum / L : 0;
  };
  for (let k = 0; k < n; k++) {
    if (segs[k].label !== 'verse') continue;
    for (let c = 0; c < n; c++) {
      if (segs[c].label !== 'chorus' && segs[c].label !== 'drop') continue;
      const self = blockSim(segs[c], segs[c]);
      if (blockSim(segs[k], segs[c]) >= 0.92 * self && segs[c].e - segs[k].e <= 0.15) {
        segs[k].label = segs[c].label === 'drop' && k > 0 && segs[k - 1].label === 'build' && absDrop(k, true) ? 'drop' : 'chorus';
        break;
      }
    }
  }

  // Merge neighbours that ended up with the same intro / outro / build label.
  const out: Section[] = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    if (last && last.label === s.label && (s.label === 'intro' || s.label === 'outro' || s.label === 'build')) {
      const w0 = last.end - last.start;
      const w1 = s.end - s.start;
      last.energy = clamp01((last.energy * w0 + s.loud * w1) / (w0 + w1));
      last.end = s.end;
      continue;
    }
    out.push({
      start: s.start,
      end: s.end,
      label: s.label,
      energy: clamp01(s.loud),
    });
  }
  out[0].start = 0;
  out[out.length - 1].end = duration;
  return out;
}

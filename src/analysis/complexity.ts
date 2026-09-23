// Absolute (cross-song) measures: per-stem presence and musical complexity.
// Unlike the per-stem envelopes in stems.ts, nothing here is normalized by the
// song's own statistics: every mapping uses fixed constants, so a solo piano
// piece and a dense EDM drop land at very different values.

import type { StemName } from '../types';
import { STEM_NAMES } from '../types';
import { clamp01, movingAverage } from './dsp';
import type { RawStemFeatures } from './stems';

const TINY = 1e-20;

export interface ComplexityInput {
  raw: RawStemFeatures;
  T: number;
  frameRate: number;
  /** STFT size used for the band powers (for the dBFS reference). */
  fftSize: number;
  /** T * 12 chroma, each frame normalized to max 1. */
  chroma: Float32Array;
}

export interface ComplexityFeatures {
  stemPresence: Record<StemName, Float32Array>;
  complexity: Float32Array;
  songComplexity: number;
  /** Diagnostics (not part of AnalysisResult). */
  debug: {
    dbfs: Float32Array;
    onsetRate: Float32Array;
    flatness: Float32Array;
    polyphony: Float32Array;
    parts: Record<string, Float32Array>;
  };
}

// --- Fixed calibration constants -------------------------------------------
export const SILENCE_DB = -50; // mix below this is silent
export const GATE_DB = 10; // gate ramps from SILENCE_DB to SILENCE_DB + GATE_DB
export const SHARE_LO = 0.05; // stem share of the mix -> presence 0
export const SHARE_HI = 0.35; // -> presence 1
export const DRUM_HF_LO = 0.03; // HF fraction of percussive energy -> not drums
export const DRUM_HF_HI = 0.08; // -> clearly drums
export const DRUM_FLAT_LO = 0.12; // flatness of that HF percussive energy: tonal attack
export const DRUM_FLAT_HI = 0.32; // -> noise-like hits
const EXPAND_KNEE = 0.1;
const EXPAND_KNEE_OUT = 0.08;
const EXPAND_TOP = 0.65;
const EXPAND_TOP_OUT = 0.92;
const SHOULDER = 0.7;

/**
 * Final fixed curve from the weighted feature sum to the published scale:
 * the components rarely all saturate, so the useful range of the sum is
 * about 0.1 (solo instrument) .. 0.65 (dense, loud, drum-heavy).
 */
export function expand(v: number): number {
  if (v <= EXPAND_KNEE) return (v * EXPAND_KNEE_OUT) / EXPAND_KNEE;
  const y = EXPAND_KNEE_OUT + ((v - EXPAND_KNEE) * (EXPAND_TOP_OUT - EXPAND_KNEE_OUT)) / (EXPAND_TOP - EXPAND_KNEE);
  // Soft shoulder so only the very densest passages approach 1.
  if (y <= SHOULDER) return y;
  return SHOULDER + (1 - SHOULDER) * Math.tanh((y - SHOULDER) / (1 - SHOULDER));
}

function oddWidth(x: number, min = 3): number {
  let w = Math.max(min, Math.round(x));
  if (w % 2 === 0) w++;
  return w;
}

/** Moving average of a non-negative signal (clamps running-sum round-off below 0). */
function smoothPos(x: Float32Array, width: number): Float32Array {
  const y = movingAverage(x, width);
  for (let i = 0; i < y.length; i++) if (!(y[i] > 0)) y[i] = 0;
  return y;
}

/** Asymmetric one-pole smoothing (forward pass): fast rise, slow fall. */
function smoothAsym(x: Float32Array, fr: number, tauUp: number, tauDown: number): Float32Array {
  const out = new Float32Array(x.length);
  const ku = 1 - Math.exp(-1 / (tauUp * fr));
  const kd = 1 - Math.exp(-1 / (tauDown * fr));
  let y = x.length > 0 ? x[0] : 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    y += (v - y) * (v > y ? ku : kd);
    out[i] = y;
  }
  return out;
}

/**
 * Onset picking on an absolute flux (mean positive dB rise per band): a frame
 * is an onset when it is a local maximum and stands a fixed margin above the
 * local median-ish baseline (moving average). Returns a 0/1 train.
 */
function pickOnsets(flux: Float32Array, fr: number, delta: number, gate: Float32Array): Uint8Array {
  const T = flux.length;
  const out = new Uint8Array(T);
  const base = movingAverage(flux, oddWidth(0.5 * fr));
  const h = Math.max(1, Math.round(0.03 * fr));
  const minGap = Math.max(1, Math.round(0.06 * fr));
  let last = -minGap;
  for (let t = 1; t < T - 1; t++) {
    const v = flux[t];
    if (gate[t] < 0.5) continue;
    if (v - base[t] < delta) continue;
    let isMax = true;
    for (let k = Math.max(0, t - h); k <= Math.min(T - 1, t + h); k++) {
      if (flux[k] > v || (flux[k] === v && k < t)) {
        isMax = false;
        break;
      }
    }
    if (!isMax || t - last < minGap) continue;
    out[t] = 1;
    last = t;
  }
  return out;
}

/** Onsets per second over a centered window of `win` seconds. */
function rateOf(train: Uint8Array, fr: number, win: number): Float32Array {
  const f = new Float32Array(train.length);
  for (let i = 0; i < train.length; i++) f[i] = train[i];
  const m = movingAverage(f, oddWidth(win * fr));
  for (let i = 0; i < m.length; i++) m[i] *= fr;
  return m;
}

export function computeComplexity(inp: ComplexityInput): ComplexityFeatures {
  const { raw, T, frameRate: fr, fftSize: n, chroma } = inp;
  // Mean-square reference: one-sided Hann power spectrum sum of a signal with
  // mean square m is about m * 3 n^2 / 16, so 0 dBFS = full-scale square wave.
  const ref = 10 * Math.log10((3 * n * n) / 16);
  const wS = oddWidth(0.1 * fr);

  // Smoothed powers (~100 ms).
  const sp: Record<StemName, Float32Array> = {} as Record<StemName, Float32Array>;
  for (const s of STEM_NAMES) sp[s] = smoothPos(raw.weighted[s], wS);
  const tot = smoothPos(raw.total, wS);
  const pHigh = smoothPos(raw.percHigh, wS);
  // HF fraction of the percussive energy over ~1 s: hats, snares and cymbals
  // put a lot of percussive energy above 2 kHz, piano / guitar attacks do not.
  const wL = oddWidth(1.0 * fr);
  const pHighL = smoothPos(raw.percHigh, wL);
  const pDrL = smoothPos(raw.weighted.drums, wL);
  // Drum share over ~0.4 s: presence describes the groove, not single hits
  // (a 100 ms share dips to zero between kicks).
  const wD = oddWidth(0.4 * fr);
  const drumsGroove = smoothPos(raw.weighted.drums, wD);
  const restGroove = smoothPos(raw.weighted.bass, wD);
  {
    const v = smoothPos(raw.weighted.vocals, wD);
    const o = smoothPos(raw.weighted.other, wD);
    for (let t = 0; t < T; t++) restGroove[t] += v[t] + o[t];
  }
  // Power-weighted flatness of that HF percussive energy over ~1 s.
  const hfFlat = new Float32Array(T);
  {
    const num = new Float32Array(T);
    for (let t = 0; t < T; t++) num[t] = raw.percHighFlatness[t] * raw.percHigh[t];
    const numL = smoothPos(num, wL);
    for (let t = 0; t < T; t++) hfFlat[t] = pHighL[t] > TINY ? numL[t] / pHighL[t] : 0;
  }

  const dbfs = new Float32Array(T);
  const gate = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const d = 10 * Math.log10(tot[t] + TINY) - ref;
    dbfs[t] = d;
    gate[t] = clamp01((d - SILENCE_DB) / GATE_DB);
  }

  // Onsets (absolute flux, fixed thresholds).
  const onsetTrain = pickOnsets(raw.flux, fr, 1.2, gate);
  const percTrain = pickOnsets(raw.percFlux, fr, 1.5, gate);
  const onsetRate = rateOf(onsetTrain, fr, 2);
  const percRate = rateOf(percTrain, fr, 2);

  // Presence.
  const presence: Record<StemName, Float32Array> = {} as Record<StemName, Float32Array>;
  for (const s of STEM_NAMES) presence[s] = new Float32Array(T);
  const drumShare = new Float32Array(T);
  const hiShare = new Float32Array(T);
  const hiFrac = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let sum = 0;
    for (const s of STEM_NAMES) sum += sp[s][t];
    if (!(sum > TINY) || gate[t] <= 0) continue;
    for (const s of STEM_NAMES) {
      const share = s === 'drums' ? drumsGroove[t] / (drumsGroove[t] + restGroove[t] + TINY) : sp[s][t] / sum;
      // Per-stem absolute gate: the stem itself must be above the silence level.
      const sDb = dbfs[t] + 10 * Math.log10(share + TINY);
      const g = gate[t] * clamp01((sDb - SILENCE_DB) / GATE_DB);
      let p = clamp01((share - SHARE_LO) / (SHARE_HI - SHARE_LO)) * g;
      if (s === 'drums') {
        drumShare[t] = share;
        hiShare[t] = pHigh[t] / sum;
        hiFrac[t] = pHighL[t] / (pDrL[t] + TINY);
        // Real drums: percussive energy that is broadband (reaches the top
        // octaves) AND a steady supply of percussive onsets.
        // The HF percussive energy must also be noise-like (hats, snares,
        // cymbals), not the bright tonal partials of a piano or guitar attack.
        p *=
          clamp01((hiFrac[t] - DRUM_HF_LO) / (DRUM_HF_HI - DRUM_HF_LO)) *
          clamp01((hfFlat[t] - DRUM_FLAT_LO) / (DRUM_FLAT_HI - DRUM_FLAT_LO)) *
          clamp01((percRate[t] - 1) / 2);
      }
      presence[s][t] = p;
    }
  }
  for (const s of STEM_NAMES) presence[s] = smoothPos(presence[s], wS).map((v) => (v > 1 ? 1 : v));

  // Polyphony: strong peaks of the chroma, smoothed over ~0.5 s.
  const poly = new Float32Array(T);
  {
    const c = new Float32Array(12);
    const w = Math.max(1, Math.round(0.25 * fr));
    const acc = new Float64Array(12);
    // Running sum over a centered window.
    const at = (t: number, k: number) => chroma[t * 12 + k];
    for (let t = 0; t < Math.min(T, w); t++) for (let k = 0; k < 12; k++) acc[k] += at(t, k);
    for (let t = 0; t < T; t++) {
      const add = t + w;
      if (add < T) for (let k = 0; k < 12; k++) acc[k] += at(add, k);
      const rem = t - w - 1;
      if (rem >= 0) for (let k = 0; k < 12; k++) acc[k] -= at(rem, k);
      let m = 0;
      for (let k = 0; k < 12; k++) {
        c[k] = acc[k];
        if (c[k] > m) m = c[k];
      }
      if (!(m > 0)) continue;
      let cnt = 0;
      for (let k = 0; k < 12; k++) {
        const v = c[k] / m;
        const l = c[(k + 11) % 12] / m;
        const r = c[(k + 1) % 12] / m;
        if (v >= 0.8 && v >= l && v >= r) cnt++;
      }
      poly[t] = cnt;
    }
  }

  const flat = movingAverage(raw.flatness, oddWidth(0.5 * fr));

  // Components (each 0..1).
  const cStems = new Float32Array(T);
  const cOnset = new Float32Array(T);
  const cFlat = new Float32Array(T);
  const cPoly = new Float32Array(T);
  const cLoud = new Float32Array(T);
  const cDrum = new Float32Array(T);
  const rawC = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    // Layers: drums and bass count fully; the vocal / other split is unreliable
    // for a single instrument (it lands in both), so those two count half.
    const layers = presence.drums[t] + presence.bass[t] + 0.5 * (presence.vocals[t] + presence.other[t]);
    cStems[t] = clamp01((layers - 0.8) / 1.4);
    cDrum[t] = presence.drums[t];
    cOnset[t] = clamp01((onsetRate[t] - 1) / 7);
    cFlat[t] = clamp01((flat[t] - 0.05) / 0.3);
    cPoly[t] = clamp01((poly[t] - 1) / 4);
    cLoud[t] = clamp01((dbfs[t] + 35) / 25);
    const v = 0.2 * cStems[t] + 0.25 * cDrum[t] + 0.15 * cOnset[t] + 0.15 * cFlat[t] + 0.1 * cPoly[t] + 0.15 * cLoud[t];
    rawC[t] = expand(v) * gate[t];
  }
  const complexity = smoothAsym(rawC, fr, 0.6, 1.5);
  // Compensate the forward lag of the smoother by averaging with a reverse pass.
  const rev = smoothAsym(Float32Array.from(rawC).reverse(), fr, 1.5, 0.6).reverse();
  for (let t = 0; t < T; t++) complexity[t] = clamp01(0.5 * (complexity[t] + rev[t]));

  let sum = 0;
  let cnt = 0;
  for (let t = 0; t < T; t++) {
    if (dbfs[t] > SILENCE_DB) {
      sum += complexity[t];
      cnt++;
    }
  }
  const songComplexity = cnt > 0 ? sum / cnt : 0;
  return {
    stemPresence: presence,
    complexity,
    songComplexity,
    debug: {
      dbfs,
      onsetRate,
      flatness: flat,
      polyphony: poly,
      parts: { hfFlat, hiFrac, cStems, cDrum, cOnset, cFlat, cPoly, cLoud, percRate, drumShare, hiShare },
    },
  };
}

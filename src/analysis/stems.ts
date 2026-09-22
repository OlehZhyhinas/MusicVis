// Stem envelopes approximated with harmonic/percussive separation (median
// filtering, Fitzgerald 2010) plus mid/side analysis. Only envelopes are
// produced; no audio is reconstructed.

import type { StemName } from '../types';
import { movingAverage, percentiles, slidingMedian, clamp01 } from './dsp';
import type { BandLayout } from './stft';

export const TIMBRE_BANDS = 16;

export interface StemFeatures {
  stems: Record<StemName, Float32Array>;
  stemOnsets: Record<StemName, Float32Array>;
  loudness: Float32Array;
  /** Percussive low-frequency (kick) onset strength, 0..1. */
  kickOnset: Float32Array;
  /** Percussive mid-frequency (snare) onset strength, 0..1. */
  snareOnset: Float32Array;
  /** T * TIMBRE_BANDS log band energies (dB) for structure analysis. */
  timbre: Float32Array;
}

const TINY = 1e-20;
const MAX_RANGE_DB = 36;

function toDb(p: number): number {
  return 10 * Math.log10(p + TINY);
}

function aWeightPower(f: number): number {
  const f2 = f * f;
  const ra =
    (12194 * 12194 * f2 * f2) /
    ((f2 + 20.6 * 20.6) * Math.sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9)) * (f2 + 12194 * 12194));
  return ra * ra * 1.585; // +2 dB
}

function oddWidth(x: number, min = 3): number {
  let w = Math.max(min, Math.round(x));
  if (w % 2 === 0) w++;
  return w;
}

/** dB envelope -> 0..1 with robust percentiles and absolute floors relative to the song top level. */
function envelopeFromPower(p: Float32Array, top: number, smoothWidth: number, floorRel: number, ceilRel: number): Float32Array {
  const T = p.length;
  const db = new Float32Array(T);
  for (let i = 0; i < T; i++) db[i] = toDb(p[i]);
  const sm = movingAverage(db, smoothWidth);
  const [p5, p98] = percentiles(sm, [0.05, 0.98]);
  let hi = Math.max(p98, top + ceilRel);
  // Visual dynamic range: at most MAX_RANGE_DB below the stem's own peak level.
  const lo = Math.max(p5, top + floorRel, hi - MAX_RANGE_DB);
  if (hi < lo + 10) hi = lo + 10;
  const out = new Float32Array(T);
  const r = hi - lo;
  for (let i = 0; i < T; i++) out[i] = clamp01((sm[i] - lo) / r);
  return out;
}

function normalizeOnset(x: Float32Array, minRange: number): Float32Array {
  const [p50, p99] = percentiles(x, [0.5, 0.99]);
  const lo = p50;
  const hi = Math.max(p99, lo + minRange);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = clamp01((x[i] - lo) / (hi - lo));
  return out;
}

/**
 * mid, side: T * B band powers of the mid (L+R)/2 and side (L-R)/2 signals.
 * Both arrays are overwritten (used as scratch to keep memory down).
 */
export function computeStems(
  mid: Float32Array,
  side: Float32Array,
  bands: BandLayout,
  T: number,
  frameRate: number,
  progress?: (p: number) => void,
): StemFeatures {
  const B = bands.count;
  const freq = bands.freq;

  // Reference level: robust top of total frame power.
  const total = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let s = 0;
    const o = t * B;
    for (let b = 0; b < B; b++) s += mid[o + b];
    total[t] = s;
  }

  // Magnitudes, in place (the power is recovered as X^2).
  const X = mid;
  for (let i = 0; i < T * B; i++) X[i] = Math.sqrt(X[i]);

  // Horizontal (time) median -> harmonic-enhanced spectrogram.
  const wT = oddWidth(0.2 * frameRate, 5);
  const wF = 17;
  const H = new Float32Array(T * B);
  const win = new Float64Array(Math.max(wT, wF) + 2);
  const pad = new Float64Array(Math.max(T, B) + Math.max(wT, wF) + 2);
  for (let b = 0; b < B; b++) {
    slidingMedian(X, b, B, T, wT, H, b, B, win, pad);
  }
  // Side/mid power ratio per band, smoothed over ~100 ms so that interference
  // between neighbouring partials (which fluctuates) averages out.
  const ratio = side; // computed in place: the raw side power is not needed afterwards
  {
    const w = oddWidth(0.1 * frameRate, 3);
    const colM = new Float32Array(T);
    const colS = new Float32Array(T);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const xm = X[t * B + b];
        colM[t] = xm * xm;
        colS[t] = side[t * B + b];
      }
      const sm = movingAverage(colM, w);
      const ss = movingAverage(colS, w);
      for (let t = 0; t < T; t++) ratio[t * B + b] = sm[t] > 0 ? ss[t] / sm[t] : 1;
    }
  }
  progress?.(0.4);

  const totalDb = new Float32Array(T);
  for (let t = 0; t < T; t++) totalDb[t] = toDb(total[t]);
  const top = percentiles(totalDb, [0.98])[0];
  // Per-band floor for flux computations (about -75 dB below the loudest frames' per-band average).
  const floorDb = top - 10 * Math.log10(B) - 75;

  // Band roles.
  const wA = new Float32Array(B);
  const timbreIdx = new Int32Array(B);
  for (let b = 0; b < B; b++) {
    const f = freq[b];
    wA[b] = aWeightPower(Math.max(f, 1));
    const lf = Math.log(Math.max(f, 30) / 30) / Math.log(11000 / 30);
    timbreIdx[b] = Math.min(TIMBRE_BANDS - 1, Math.max(0, Math.floor(lf * TIMBRE_BANDS)));
  }

  const drumsP = new Float32Array(T);
  const bassP = new Float32Array(T);
  const vocP = new Float32Array(T);
  const otherP = new Float32Array(T);
  const loudP = new Float32Array(T);
  const drumsF = new Float32Array(T);
  const bassF = new Float32Array(T);
  const vocF = new Float32Array(T);
  const otherF = new Float32Array(T);
  const kickF = new Float32Array(T);
  const snareF = new Float32Array(T);
  const timbre = new Float32Array(T * TIMBRE_BANDS);
  const timbreAcc = new Float64Array(TIMBRE_BANDS);

  const P = new Float32Array(B);
  const prevPerc = new Float32Array(B).fill(floorDb);
  const prevHarm = new Float32Array(B).fill(floorDb);
  const prevCenter = new Float32Array(B).fill(floorDb);
  const prevOther = new Float32Array(B).fill(floorDb);
  const center = new Float32Array(B);

  let nDrum = 0,
    nBass = 0,
    nVoc = 0,
    nOther = 0,
    nKick = 0,
    nSnare = 0;
  for (let b = 0; b < B; b++) {
    const f = freq[b];
    if (f >= 40 && f <= 12000) nDrum++;
    if (f < 250) nBass++;
    if (f >= 250 && f <= 4000) nVoc++;
    if (f >= 250) nOther++;
    if (f < 150) nKick++;
    if (f >= 150 && f <= 5000) nSnare++;
  }
  const inv = (n: number) => (n > 0 ? 1 / n : 0);
  const iDrum = inv(nDrum),
    iBass = inv(nBass),
    iVoc = inv(nVoc),
    iOther = inv(nOther),
    iKick = inv(nKick),
    iSnare = inv(nSnare);

  const reportEvery = Math.max(1, Math.floor(T / 10));
  for (let t = 0; t < T; t++) {
    const o = t * B;
    // Vertical (frequency) median -> percussive-enhanced.
    slidingMedian(X, o, 1, B, wF, P, 0, 1, win, pad);

    let dr = 0,
      ba = 0,
      vo = 0,
      ot = 0,
      lo = 0;
    let fDr = 0,
      fBa = 0,
      fVo = 0,
      fOt = 0,
      fK = 0,
      fS = 0;
    timbreAcc.fill(0);
    // Flatness of the centered harmonic spectrum in the vocal range.
    let logSum = 0;
    let linSum = 0;
    let harmSum = 0;
    for (let b = 0; b < B; b++) {
      const xm = X[o + b];
      const x2 = xm * xm;
      const h = H[o + b];
      const p = P[b];
      const h2 = h * h;
      const p2 = p * p;
      const mh = h2 + p2 > 0 ? h2 / (h2 + p2) : 0.5;
      const harm = mh * x2;
      const perc = x2 - harm;
      const f = freq[b];
      lo += x2 * wA[b];
      timbreAcc[timbreIdx[b]] += x2;

      const percDb = Math.max(floorDb, toDb(perc));
      const dPerc = percDb - prevPerc[b];
      prevPerc[b] = percDb;
      if (f >= 40 && f <= 12000) {
        dr += perc;
        if (dPerc > 0) fDr += dPerc;
      }
      if (f < 150) {
        if (dPerc > 0) fK += dPerc;
      } else if (f <= 5000) {
        if (dPerc > 0) fS += dPerc;
      }

      if (f < 250) {
        ba += harm;
        const hDb = Math.max(floorDb, toDb(harm));
        const d = hDb - prevHarm[b];
        prevHarm[b] = hDb;
        if (d > 0) fBa += d;
        center[b] = 0;
      } else {
        // Centered harmonic energy: mid minus side (sources panned center have no side).
        // Peakiness: a tonal partial stands well above the local spectral median.
        const pk = x2 / (x2 + 12 * p2 + TINY);
        const cw = 1 - 1.3 * ratio[o + b];
        let c = cw > 0 ? harm * cw * pk : 0;
        if (f > 4000) c = 0;
        center[b] = c;
        if (f <= 4000) {
          linSum += c;
          harmSum += harm;
        }
        const oth = harm - c;
        ot += oth;
        const oDb = Math.max(floorDb, toDb(oth));
        const d = oDb - prevOther[b];
        prevOther[b] = oDb;
        if (d > 0) fOt += d;
      }
    }
    // Vocal-ness: centered harmonic energy weighted by tonality (low flatness).
    let tonal = 0;
    if (nVoc > 0 && linSum > 0 && harmSum > 0) {
      // Flatness of the harmonic (not side-subtracted) spectrum: noise-like
      // content (risers, snare rolls) is flat, voices and instruments are peaky.
      const fl = 1e-4 * harmSum * iVoc + TINY;
      for (let b = 0; b < B; b++) {
        const f = freq[b];
        if (f >= 250 && f <= 4000) {
          const xm = X[o + b];
          const x2 = xm * xm;
          const h = H[o + b];
          const p = P[b];
          const mh = h * h + p * p > 0 ? (h * h) / (h * h + p * p) : 0.5;
          logSum += Math.log(mh * x2 + fl);
        }
      }
      const flat = Math.exp(logSum * iVoc) / (harmSum * iVoc);
      tonal = clamp01((0.55 - flat) / 0.45);
    }
    for (let b = 0; b < B; b++) {
      const f = freq[b];
      if (f < 250 || f > 4000) continue;
      const c = center[b] * tonal;
      vo += c;
      const cDb = Math.max(floorDb, toDb(c));
      const d = cDb - prevCenter[b];
      prevCenter[b] = cDb;
      if (d > 0) fVo += d;
    }
    // Centered energy not credited to vocals (noisy / flat) stays in "other".
    ot += (1 - tonal) * linSum;

    drumsP[t] = dr;
    bassP[t] = ba;
    vocP[t] = vo;
    otherP[t] = ot;
    loudP[t] = lo;
    drumsF[t] = fDr * iDrum;
    bassF[t] = fBa * iBass;
    vocF[t] = fVo * iVoc;
    otherF[t] = fOt * iOther;
    kickF[t] = fK * iKick;
    snareF[t] = fS * iSnare;
    for (let k = 0; k < TIMBRE_BANDS; k++) timbre[t * TIMBRE_BANDS + k] = Math.max(floorDb, toDb(timbreAcc[k]));
    if (progress && t % reportEvery === 0) progress(0.4 + (0.6 * t) / T);
  }

  const smoothW = oddWidth(0.03 * frameRate, 1);
  const silent = !(top > -150);
  const zero = () => new Float32Array(T);
  const stems: Record<StemName, Float32Array> = silent
    ? { drums: zero(), bass: zero(), vocals: zero(), other: zero() }
    : {
        drums: envelopeFromPower(drumsP, top, smoothW, -60, -30),
        bass: envelopeFromPower(bassP, top, smoothW, -60, -30),
        vocals: envelopeFromPower(vocP, top, smoothW, -60, -30),
        other: envelopeFromPower(otherP, top, smoothW, -60, -30),
      };
  const loudTop = silent ? 0 : percentiles(Float32Array.from(loudP, (v) => toDb(v)), [0.99])[0];
  const loudness = silent ? zero() : envelopeFromPower(loudP, loudTop, smoothW, -50, -12);
  const stemOnsets: Record<StemName, Float32Array> = silent
    ? { drums: zero(), bass: zero(), vocals: zero(), other: zero() }
    : {
        drums: normalizeOnset(drumsF, 4),
        bass: normalizeOnset(bassF, 4),
        vocals: normalizeOnset(vocF, 4),
        other: normalizeOnset(otherF, 4),
      };
  return {
    stems,
    stemOnsets,
    loudness,
    kickOnset: silent ? zero() : normalizeOnset(kickF, 4),
    snareOnset: silent ? zero() : normalizeOnset(snareF, 4),
    timbre,
  };
}

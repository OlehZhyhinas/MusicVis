// Causal streaming features for the ML prototypes: one frame per 512-sample capture block at
// 44.1 kHz (86.13 fps, the same block LiveInput posts), so a frame is ready the moment its block
// arrives and describes the audio up to the block's last sample (no look-ahead).
//
// Frame = [80 log-spaced band log-magnitudes, 80 positive band differences (flux)], from a
// 2048-sample Hann window that ENDS at the block's last sample. Mono mid signal.
// This is the part an app integration would reuse as is (it is plain TS, ~0.05 ms per frame).

import { RealFFT } from '../../src/analysis/fft';

export const ML_SR = 44100;
export const ML_HOP = 512;
export const ML_WIN = 2048;
export const ML_BANDS = 80;
export const ML_FEAT = 2 * ML_BANDS;
export const ML_FPS = ML_SR / ML_HOP;

/** Triangular log-spaced filterbank (30 Hz .. 16 kHz), rows of [bin, weight]. */
function filterbank(n: number, sr: number, count: number, fMin = 30, fMax = 16000): { lo: number; w: Float32Array }[] {
  const nb = n / 2 + 1;
  const binHz = sr / n;
  const edges: number[] = [];
  for (let i = 0; i < count + 2; i++) edges.push(fMin * Math.pow(fMax / fMin, i / (count + 1)));
  const out: { lo: number; w: Float32Array }[] = [];
  for (let b = 0; b < count; b++) {
    const f0 = edges[b], f1 = edges[b + 1], f2 = edges[b + 2];
    // Guarantee at least one bin (low bands are narrower than a bin): widen to +-1 bin around f1.
    let lo = Math.floor(f0 / binHz), hi = Math.ceil(f2 / binHz);
    const c = f1 / binHz;
    lo = Math.max(0, Math.min(lo, Math.floor(c) - 1));
    hi = Math.min(nb - 1, Math.max(hi, Math.ceil(c) + 1));
    const w = new Float32Array(hi - lo + 1);
    let s = 0;
    for (let k = lo; k <= hi; k++) {
      const f = k * binHz;
      let v = f < f1 ? (f - f0) / (f1 - f0) : (f2 - f) / (f2 - f1);
      if (!(v > 0)) v = 0;
      // Narrow bands: a triangle over the bin distance instead.
      if (f2 - f0 < 2 * binHz) v = Math.max(0, 1 - Math.abs(k - c));
      w[k - lo] = v;
      s += v;
    }
    if (s > 0) for (let i = 0; i < w.length; i++) w[i] /= s;
    out.push({ lo, w });
  }
  return out;
}

export class StreamFeatures {
  private readonly fft = new RealFFT(ML_WIN);
  private readonly win = new Float64Array(ML_WIN);
  private readonly ring = new Float32Array(ML_WIN);
  private readonly buf = new Float64Array(ML_WIN);
  private readonly pow = new Float64Array(ML_WIN / 2 + 1);
  private readonly fb = filterbank(ML_WIN, ML_SR, ML_BANDS);
  private readonly prev = new Float32Array(ML_BANDS);
  readonly out = new Float32Array(ML_FEAT);
  private filled = 0;
  frames = 0;
  constructor() {
    for (let i = 0; i < ML_WIN; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / ML_WIN);
  }
  /** Push one 512-sample stereo block; returns the new feature frame (reused buffer). */
  push(l: Float32Array, r: Float32Array): Float32Array {
    const H = ML_HOP;
    this.ring.copyWithin(0, H);
    for (let i = 0; i < H; i++) this.ring[ML_WIN - H + i] = 0.5 * (l[i] + r[i]);
    this.filled = Math.min(ML_WIN, this.filled + H);
    for (let i = 0; i < ML_WIN; i++) this.buf[i] = this.ring[i] * this.win[i];
    this.fft.power(this.buf, this.pow);
    const o = this.out;
    for (let b = 0; b < ML_BANDS; b++) {
      const { lo, w } = this.fb[b];
      let s = 0;
      for (let k = 0; k < w.length; k++) s += w[k] * this.pow[lo + k];
      // log(1 + 1000 * magnitude): magnitude = sqrt(power), window-normalised.
      const v = Math.log1p(1000 * Math.sqrt(s) / (ML_WIN / 4));
      o[b] = v;
      o[ML_BANDS + b] = this.frames > 0 ? Math.max(0, v - this.prev[b]) : 0;
      this.prev[b] = v;
    }
    this.frames++;
    return o;
  }
}

/** All frames of a PCM buffer (frame k ends at sample (k + 1) * 512). */
export function featuresOf(left: Float32Array, right: Float32Array): Float32Array {
  const T = Math.floor(left.length / ML_HOP);
  const f = new StreamFeatures();
  const out = new Float32Array(T * ML_FEAT);
  for (let k = 0; k < T; k++) out.set(f.push(left.subarray(k * ML_HOP, (k + 1) * ML_HOP), right.subarray(k * ML_HOP, (k + 1) * ML_HOP)), k * ML_FEAT);
  return out;
}

/** Stream time of frame k (end of its block). */
export const frameTime = (k: number) => ((k + 1) * ML_HOP) / ML_SR;

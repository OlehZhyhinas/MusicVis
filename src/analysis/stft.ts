import { RealFFT } from './fft';
import { hann } from './dsp';

/**
 * Streaming short-time power spectrum. Frame i is centered on sample i * hop
 * (zero padding outside the signal), so frame i describes time i * hop / sr.
 * The callback receives a reused buffer of n/2 + 1 power values.
 */
export function stftPower(
  x: Float32Array,
  n: number,
  hop: number,
  numFrames: number,
  onFrame: (frame: number, power: Float64Array) => void,
): void {
  const fft = new RealFFT(n);
  const win = hann(n);
  const buf = new Float64Array(n);
  const pow = new Float64Array(n / 2 + 1);
  const len = x.length;
  const half = n >> 1;
  for (let f = 0; f < numFrames; f++) {
    const start = f * hop - half;
    if (start >= 0 && start + n <= len) {
      for (let i = 0; i < n; i++) buf[i] = x[start + i] * win[i];
    } else {
      for (let i = 0; i < n; i++) {
        const k = start + i;
        buf[i] = k >= 0 && k < len ? x[k] * win[i] : 0;
      }
    }
    fft.power(buf, pow);
    onFrame(f, pow);
  }
}

export interface BandLayout {
  count: number;
  lo: Int32Array; // first FFT bin (inclusive)
  hi: Int32Array; // last FFT bin (exclusive)
  freq: Float32Array; // center frequency in Hz
}

/**
 * Near-linear-then-coarser band layout used for the stem spectrogram:
 * single bins up to n/16, pairs up to n/4, quads up to Nyquist.
 */
export function stemBands(n: number, sr: number): BandLayout {
  const lo: number[] = [];
  const hi: number[] = [];
  const nb = n / 2 + 1;
  const a = n / 16;
  const b = n / 4;
  for (let k = 1; k < a; k++) {
    lo.push(k);
    hi.push(k + 1);
  }
  for (let k = a; k < b; k += 2) {
    lo.push(k);
    hi.push(k + 2);
  }
  for (let k = b; k < nb - 1; k += 4) {
    lo.push(k);
    hi.push(Math.min(nb, k + 4 >= nb - 1 ? nb : k + 4));
  }
  const count = lo.length;
  const freq = new Float32Array(count);
  for (let i = 0; i < count; i++) freq[i] = (((lo[i] + hi[i] - 1) / 2) * sr) / n;
  return { count, lo: Int32Array.from(lo), hi: Int32Array.from(hi), freq };
}

/** Log-spaced bands between fMin and fMax (each band has at least one bin). */
export function logBands(n: number, sr: number, fMin: number, fMax: number, count: number): BandLayout {
  const binHz = sr / n;
  const nb = n / 2 + 1;
  const lo: number[] = [];
  const hi: number[] = [];
  let prev = Math.max(1, Math.round(fMin / binHz));
  for (let i = 1; i <= count; i++) {
    const f = fMin * Math.pow(fMax / fMin, i / count);
    let edge = Math.min(nb, Math.round(f / binHz));
    if (edge <= prev) edge = prev + 1;
    if (edge > nb) break;
    lo.push(prev);
    hi.push(edge);
    prev = edge;
    if (prev >= nb) break;
  }
  const c = lo.length;
  const freq = new Float32Array(c);
  for (let i = 0; i < c; i++) freq[i] = (((lo[i] + hi[i] - 1) / 2) * sr) / n;
  return { count: c, lo: Int32Array.from(lo), hi: Int32Array.from(hi), freq };
}

export function bandPower(pow: Float64Array, bands: BandLayout, out: Float32Array, outOffset: number): void {
  for (let b = 0; b < bands.count; b++) {
    let s = 0;
    const h = bands.hi[b];
    for (let k = bands.lo[b]; k < h; k++) s += pow[k];
    out[outOffset + b] = s;
  }
}

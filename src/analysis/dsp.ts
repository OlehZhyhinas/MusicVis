// Small numeric helpers shared by the analysis stages.

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Periodic Hann window. */
export function hann(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/**
 * Integer-factor decimation with a windowed-sinc low-pass (Blackman window).
 * Returns the input unchanged for factor 1.
 */
export function decimate(x: Float32Array, factor: number): Float32Array {
  if (factor <= 1) return x;
  const taps = 16 * factor + 1;
  const half = (taps - 1) >> 1;
  const cutoff = 0.46 / factor; // cycles per input sample
  const h = new Float32Array(taps);
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const t = i - half;
    const sinc = t === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * t) / (Math.PI * t);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1));
    h[i] = sinc * w;
    sum += h[i];
  }
  for (let i = 0; i < taps; i++) h[i] /= sum;
  const outLen = Math.floor((x.length - 1) / factor) + 1;
  const y = new Float32Array(outLen);
  const n = x.length;
  for (let j = 0; j < outLen; j++) {
    const c = j * factor - half;
    let acc = 0;
    if (c >= 0 && c + taps <= n) {
      for (let i = 0; i < taps; i++) acc += h[i] * x[c + i];
    } else {
      for (let i = 0; i < taps; i++) {
        const k = c + i;
        if (k >= 0 && k < n) acc += h[i] * x[k];
      }
    }
    y[j] = acc;
  }
  return y;
}

/** p in [0, 1]. Sorts a copy. */
export function percentile(x: ArrayLike<number>, p: number): number {
  const n = x.length;
  if (n === 0) return 0;
  const a = Float64Array.from(x as ArrayLike<number>);
  a.sort();
  const pos = clamp01(p) * (n - 1);
  const i = Math.floor(pos);
  const f = pos - i;
  return i + 1 < n ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}

/** Several percentiles with a single sort. */
export function percentiles(x: ArrayLike<number>, ps: number[]): number[] {
  const n = x.length;
  if (n === 0) return ps.map(() => 0);
  const a = Float64Array.from(x as ArrayLike<number>);
  a.sort();
  return ps.map((p) => {
    const pos = clamp01(p) * (n - 1);
    const i = Math.floor(pos);
    const f = pos - i;
    return i + 1 < n ? a[i] * (1 - f) + a[i + 1] * f : a[i];
  });
}

/** Centered moving average of odd width, in place-safe (returns new array). */
export function movingAverage(x: Float32Array, width: number): Float32Array {
  const n = x.length;
  const out = new Float32Array(n);
  const h = Math.max(0, Math.floor(width / 2));
  if (h === 0) {
    out.set(x);
    return out;
  }
  let acc = 0;
  let count = 0;
  // Prime window [0, h)
  for (let i = 0; i < Math.min(h, n); i++) {
    acc += x[i];
    count++;
  }
  for (let i = 0; i < n; i++) {
    const add = i + h;
    if (add < n) {
      acc += x[add];
      count++;
    }
    const rem = i - h - 1;
    if (rem >= 0) {
      acc -= x[rem];
      count--;
    }
    out[i] = acc / count;
  }
  return out;
}

/**
 * Sliding median of odd width over a strided sequence. Reads src[offset + i*stride]
 * for i in [0, n) and writes dst[dOffset + i*dStride]. Edges use mirrored padding.
 * `win` is scratch of length >= width, `pad` scratch of length >= n + width.
 */
export function slidingMedian(
  src: Float32Array,
  offset: number,
  stride: number,
  n: number,
  width: number,
  dst: Float32Array,
  dOffset: number,
  dStride: number,
  win: Float64Array,
  pad: Float64Array,
): void {
  if (n <= 0) return;
  const h = width >> 1;
  // Gather with mirrored padding: pad[i + h] = x[i].
  for (let i = -h; i < n + h; i++) {
    let j = i;
    if (n === 1) j = 0;
    else {
      const period = 2 * (n - 1);
      j = ((j % period) + period) % period;
      if (j >= n) j = period - j;
    }
    pad[i + h] = src[offset + j * stride];
  }
  const w = 2 * h + 1;
  let len = 0;
  for (let i = 0; i < w; i++) {
    const v = pad[i];
    let k = len;
    while (k > 0 && win[k - 1] > v) {
      win[k] = win[k - 1];
      k--;
    }
    win[k] = v;
    len++;
  }
  dst[dOffset] = win[h];
  for (let i = 1; i < n; i++) {
    const out = pad[i - 1];
    const inn = pad[i + w - 1];
    if (out !== inn) {
      let lo = 0;
      let hi = len - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (win[mid] < out) lo = mid + 1;
        else hi = mid;
      }
      let k = lo;
      if (inn > out) {
        while (k + 1 < len && win[k + 1] < inn) {
          win[k] = win[k + 1];
          k++;
        }
      } else {
        while (k > 0 && win[k - 1] > inn) {
          win[k] = win[k - 1];
          k--;
        }
      }
      win[k] = inn;
    }
    dst[dOffset + i * dStride] = win[h];
  }
}

/** Linear-in-dB normalization: lo -> 0, hi -> 1, clamped. */
export function normalizeRange(x: Float32Array, lo: number, hi: number): Float32Array {
  const out = new Float32Array(x.length);
  const r = hi - lo;
  if (!(r > 1e-9)) return out;
  for (let i = 0; i < x.length; i++) out[i] = clamp01((x[i] - lo) / r);
  return out;
}

/** Sample a frame-rate envelope at fractional position with linear interpolation. */
export function sampleLinear(x: ArrayLike<number>, pos: number): number {
  const n = x.length;
  if (n === 0) return 0;
  if (pos <= 0) return x[0];
  if (pos >= n - 1) return x[n - 1];
  const i = Math.floor(pos);
  const f = pos - i;
  return x[i] + (x[i + 1] - x[i]) * f;
}

export function mean(x: ArrayLike<number>, a = 0, b = x.length): number {
  a = Math.max(0, a);
  b = Math.min(x.length, b);
  if (b <= a) return 0;
  let s = 0;
  for (let i = a; i < b; i++) s += x[i];
  return s / (b - a);
}

/** Least-squares slope of y over its index (per sample). */
export function slope(x: ArrayLike<number>, a: number, b: number): number {
  a = Math.max(0, a);
  b = Math.min(x.length, b);
  const n = b - a;
  if (n < 2) return 0;
  const mx = (n - 1) / 2;
  let my = 0;
  for (let i = a; i < b; i++) my += x[i];
  my /= n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const dx = i - mx;
    num += dx * (x[a + i] - my);
    den += dx * dx;
  }
  return den > 0 ? num / den : 0;
}

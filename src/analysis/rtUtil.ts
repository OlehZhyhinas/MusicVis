// Small streaming helpers for the real-time (live input) analysis.
// Everything here is causal and allocation-free after construction.

/**
 * Causal moving average over the last `len` pushed values of a non-negative
 * signal (running-sum round-off below 0 is clamped, like complexity.ts smoothPos).
 */
export class BoxAvg {
  private readonly ring: Float64Array;
  private idx = 0;
  private count = 0;
  private sum = 0;
  private pushes = 0;
  value = 0;

  constructor(len: number) {
    this.ring = new Float64Array(Math.max(1, Math.round(len)));
  }

  push(v: number): number {
    if (!Number.isFinite(v)) v = 0;
    const n = this.ring.length;
    if (this.count === n) this.sum -= this.ring[this.idx];
    else this.count++;
    this.ring[this.idx] = v;
    this.sum += v;
    this.idx = (this.idx + 1) % n;
    // Re-sum now and then so floating-point drift never accumulates.
    if (++this.pushes >= 4096) {
      this.pushes = 0;
      let s = 0;
      for (let i = 0; i < this.count; i++) s += this.ring[(this.idx - 1 - i + n * 2) % n];
      this.sum = s;
    }
    const m = this.sum / this.count;
    this.value = m > 0 ? m : 0;
    return this.value;
  }

  reset(): void {
    this.ring.fill(0);
    this.idx = 0;
    this.count = 0;
    this.sum = 0;
    this.value = 0;
  }
}

/** Ring buffer of the last `len` values with random access by age (0 = newest). */
export class History {
  readonly data: Float32Array;
  private head = 0; // index of the next write
  count = 0;

  constructor(len: number) {
    this.data = new Float32Array(Math.max(1, Math.round(len)));
  }

  get length(): number {
    return this.data.length;
  }

  push(v: number): void {
    this.data[this.head] = Number.isFinite(v) ? v : 0;
    this.head = (this.head + 1) % this.data.length;
    if (this.count < this.data.length) this.count++;
  }

  /** Value `age` frames ago (0 = newest); 0 outside the stored range. */
  at(age: number): number {
    if (age < 0 || age >= this.count) return 0;
    const n = this.data.length;
    return this.data[(this.head - 1 - age + n) % n];
  }

  /** Linear interpolation at a fractional age. */
  atFrac(age: number): number {
    const i = Math.floor(age);
    const f = age - i;
    return this.at(i) * (1 - f) + this.at(i + 1) * f;
  }

  /** Copy the stored values oldest-first into `out` (returns the count copied). */
  copyOrdered(out: Float32Array | Float64Array): number {
    const n = this.data.length;
    const c = Math.min(this.count, out.length);
    for (let i = 0; i < c; i++) out[i] = this.data[(this.head - c + i + n) % n];
    return c;
  }

  reset(): void {
    this.data.fill(0);
    this.head = 0;
    this.count = 0;
  }
}

/** Mean of `h` between ages [a, b) (0 = newest). */
export function histMean(h: History, a: number, b: number): number {
  a = Math.max(0, Math.floor(a));
  b = Math.min(h.count, Math.floor(b));
  if (b <= a) return 0;
  let s = 0;
  for (let i = a; i < b; i++) s += h.at(i);
  return s / (b - a);
}

/** Least-squares slope per frame of `h` over ages [0, n) (positive = rising toward now). */
export function histSlope(h: History, n: number): number {
  n = Math.min(h.count, Math.floor(n));
  if (n < 2) return 0;
  const mx = (n - 1) / 2;
  let my = 0;
  for (let i = 0; i < n; i++) my += h.at(n - 1 - i);
  my /= n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const dx = i - mx;
    num += dx * (h.at(n - 1 - i) - my);
    den += dx * dx;
  }
  return den > 0 ? num / den : 0;
}

/**
 * Streaming integer-factor decimator with the same windowed-sinc low-pass as
 * dsp.decimate (Blackman window, cutoff 0.46 / factor).
 */
export class StreamDecimator {
  readonly factor: number;
  private readonly h: Float32Array;
  private readonly buf: Float32Array; // circular input history, doubled for contiguous reads
  private pos = 0;
  private phase = 0;

  constructor(factor: number) {
    this.factor = Math.max(1, Math.round(factor));
    const f = this.factor;
    const taps = f <= 1 ? 1 : 16 * f + 1;
    const half = (taps - 1) >> 1;
    const cutoff = 0.46 / f;
    this.h = new Float32Array(taps);
    if (taps === 1) this.h[0] = 1;
    else {
      let sum = 0;
      for (let i = 0; i < taps; i++) {
        const t = i - half;
        const sinc = t === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * t) / (Math.PI * t);
        const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1));
        this.h[i] = sinc * w;
        sum += this.h[i];
      }
      for (let i = 0; i < taps; i++) this.h[i] /= sum;
    }
    this.buf = new Float32Array(taps * 2);
  }

  /** Feed input samples; calls `emit` for every output sample. */
  process(x: Float32Array, emit: (v: number) => void): void {
    const f = this.factor;
    if (f <= 1) {
      for (let i = 0; i < x.length; i++) emit(x[i]);
      return;
    }
    const taps = this.h.length;
    const h = this.h;
    const buf = this.buf;
    for (let i = 0; i < x.length; i++) {
      const v = Number.isFinite(x[i]) ? x[i] : 0;
      buf[this.pos] = v;
      buf[this.pos + taps] = v;
      this.pos = (this.pos + 1) % taps;
      if (++this.phase >= f) {
        this.phase = 0;
        // buf[pos .. pos + taps) holds the last `taps` samples, oldest first.
        let acc = 0;
        const o = this.pos;
        for (let k = 0; k < taps; k++) acc += h[k] * buf[o + k];
        emit(acc);
      }
    }
  }

  reset(): void {
    this.buf.fill(0);
    this.pos = 0;
    this.phase = 0;
  }
}

/** In-place quickselect median of the first n values of `a` (reorders `a`). */
export function medianInPlace(a: Float64Array, n: number): number {
  if (n <= 0) return 0;
  const k = n >> 1;
  let lo = 0;
  let hi = n - 1;
  while (hi > lo) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return a[k];
}

export function wrapHalf(x: number): number {
  return x - Math.round(x);
}

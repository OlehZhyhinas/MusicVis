// Radix-2 FFT for real input. A real signal of length n is packed into a
// complex signal of length n/2 (even samples real, odd samples imaginary),
// transformed with an iterative complex FFT, then split into the n/2 + 1
// non-redundant bins of the real spectrum.

export class RealFFT {
  readonly n: number;
  private readonly m: number;
  private readonly rev: Uint32Array;
  private readonly cosM: Float64Array; // twiddles for the size-m complex FFT
  private readonly sinM: Float64Array;
  private readonly cosN: Float64Array; // twiddles for the final split
  private readonly sinN: Float64Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;

  constructor(n: number) {
    if (n < 4 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two >= 4, got ${n}`);
    this.n = n;
    const m = n >> 1;
    this.m = m;
    this.re = new Float64Array(m);
    this.im = new Float64Array(m);
    this.rev = new Uint32Array(m);
    let bits = 0;
    while (1 << bits < m) bits++;
    for (let i = 0; i < m; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cosM = new Float64Array(m >> 1 || 1);
    this.sinM = new Float64Array(m >> 1 || 1);
    for (let k = 0; k < m >> 1; k++) {
      this.cosM[k] = Math.cos((2 * Math.PI * k) / m);
      this.sinM[k] = Math.sin((2 * Math.PI * k) / m);
    }
    this.cosN = new Float64Array(m + 1);
    this.sinN = new Float64Array(m + 1);
    for (let k = 0; k <= m; k++) {
      this.cosN[k] = Math.cos((2 * Math.PI * k) / n);
      this.sinN[k] = Math.sin((2 * Math.PI * k) / n);
    }
  }

  /** Complex FFT of the packed signal held in this.re / this.im. */
  private transformPacked(): void {
    const m = this.m;
    const re = this.re;
    const im = this.im;
    const cosM = this.cosM;
    const sinM = this.sinM;
    for (let size = 2; size <= m; size <<= 1) {
      const half = size >> 1;
      const step = m / size;
      for (let start = 0; start < m; start += size) {
        for (let j = 0, t = 0; j < half; j++, t += step) {
          const a = start + j;
          const b = a + half;
          const wr = cosM[t];
          const wi = -sinM[t];
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }

  private load(x: ArrayLike<number>): void {
    const rev = this.rev;
    const re = this.re;
    const im = this.im;
    for (let k = 0; k < this.m; k++) {
      const r = rev[k];
      re[r] = x[2 * k];
      im[r] = x[2 * k + 1];
    }
  }

  /** Full spectrum: outRe/outIm must have length >= n/2 + 1. */
  forward(x: ArrayLike<number>, outRe: Float64Array | Float32Array, outIm: Float64Array | Float32Array): void {
    this.load(x);
    this.transformPacked();
    const m = this.m;
    const re = this.re;
    const im = this.im;
    for (let k = 0; k <= m; k++) {
      const k1 = k === m ? 0 : k;
      const k2 = k === 0 ? 0 : m - k;
      const a = re[k1];
      const b = im[k1];
      const c = re[k2];
      const d = im[k2];
      const er = 0.5 * (a + c);
      const ei = 0.5 * (b - d);
      const or = 0.5 * (b + d);
      const oi = -0.5 * (a - c);
      const cs = this.cosN[k];
      const sn = this.sinN[k];
      outRe[k] = er + cs * or + sn * oi;
      outIm[k] = ei + cs * oi - sn * or;
    }
  }

  /** Power spectrum |X[k]|^2 for k = 0..n/2 into out (length >= n/2 + 1). */
  power(x: ArrayLike<number>, out: Float64Array | Float32Array): void {
    this.load(x);
    this.transformPacked();
    const m = this.m;
    const re = this.re;
    const im = this.im;
    const cosN = this.cosN;
    const sinN = this.sinN;
    for (let k = 0; k <= m; k++) {
      const k1 = k === m ? 0 : k;
      const k2 = k === 0 ? 0 : m - k;
      const a = re[k1];
      const b = im[k1];
      const c = re[k2];
      const d = im[k2];
      const er = 0.5 * (a + c);
      const ei = 0.5 * (b - d);
      const or = 0.5 * (b + d);
      const oi = -0.5 * (a - c);
      const cs = cosN[k];
      const sn = sinN[k];
      const xr = er + cs * or + sn * oi;
      const xi = ei + cs * oi - sn * or;
      out[k] = xr * xr + xi * xi;
    }
  }
}

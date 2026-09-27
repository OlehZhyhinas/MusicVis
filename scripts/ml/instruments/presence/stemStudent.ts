// Plain synchronous TS forward pass of the presence/stem student (student.py), one step per
// 512-sample block at 44.1 kHz. Prototype only (lives under scripts/, not src/).
//
// Front end: mono = (L+R)/2 at 44.1 kHz; a 2048-sample periodic-Hann window ending at the end of the
// block; |FFT|^2; 64 triangular HTK-mel bands on the 21.5 Hz bins (identical bin frequencies to the
// training front end, which used 1024 samples at 22.05 kHz: the power is 4x larger, so log10(4) is
// subtracted); log10(p + 1e-7). (48 kHz input: resample to 44.1 kHz, or accept ~9% frequency skew.)
// Model: Linear+ReLU, 3 causal dilated convs (ring buffers), GRU, env head (dB rel. to mix) + act head.
// envDb (absolute) = envRel + mix dB, where mix dB is computed from the same mel power as in training.

export interface StudentWeights {
  meta: { C: number; H: number; dil: number[]; groups: string[]; stems: string[]; front: { n_mel: number; fmin: number; fmax: number; log_eps: number } };
  tensors: Record<string, [number[], string]>;
}

function b64f32(s: string): Float32Array {
  const b = Buffer.from(s, 'base64');
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

const N = 2048;
const SR = 44100;

export class StemStudent {
  readonly C: number;
  readonly H: number;
  readonly G: number;
  readonly S: number;
  readonly groups: string[];
  readonly stems: string[];
  private readonly W: Record<string, Float32Array> = {};
  private readonly nmel: number;
  // front end
  private readonly ring = new Float32Array(N);
  private rpos = 0;
  private readonly win = new Float32Array(N);
  private readonly re = new Float32Array(N);
  private readonly im = new Float32Array(N);
  private readonly rev = new Uint32Array(N);
  private readonly cosT = new Float32Array(N / 2);
  private readonly sinT = new Float32Array(N / 2);
  private readonly melLo: Int32Array;
  private readonly melW: Float32Array[];
  private readonly pw = new Float32Array(N / 2 + 1);
  private kMax = 0;
  readonly mel: Float32Array;
  mixDb = -100;
  // model state
  private readonly x0: Float32Array;
  private readonly hist: Float32Array[]; // per conv layer: ring of past inputs (C x (2*dil+1))
  private readonly hpos: number[];
  private readonly z: Float32Array;
  private readonly zt: Float32Array;
  readonly h: Float32Array;
  private readonly gates: Float32Array;
  readonly envRel: Float32Array;
  readonly envDb: Float32Array;
  readonly act: Float32Array;
  private readonly dil: number[];

  constructor(w: StudentWeights) {
    const m = w.meta;
    this.C = m.C; this.H = m.H; this.G = m.groups.length; this.S = m.stems.length;
    this.groups = m.groups; this.stems = m.stems; this.dil = m.dil;
    this.nmel = m.front.n_mel;
    for (const [k, [, b]] of Object.entries(w.tensors)) this.W[k] = b64f32(b);
    for (let i = 0; i < N; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
    const bits = Math.log2(N) - 1; // bit reversal for the N/2-point complex FFT
    for (let i = 0; i < N / 2; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    for (let i = 0; i < N / 2; i++) (this.cosT[i] = Math.cos((-2 * Math.PI * i) / N)), (this.sinT[i] = Math.sin((-2 * Math.PI * i) / N));
    // mel filterbank (same formula as features.py mel_fb on 22.05 kHz / 1024 -> same bin freqs)
    const hz2mel = (f: number) => 2595 * Math.log10(1 + f / 700);
    const mel2hz = (x: number) => 700 * (10 ** (x / 2595) - 1);
    const M = this.nmel, lo = hz2mel(m.front.fmin), hi = hz2mel(m.front.fmax);
    const pts = Array.from({ length: M + 2 }, (_, i) => mel2hz(lo + ((hi - lo) * i) / (M + 1)));
    const df = SR / N;
    this.melLo = new Int32Array(M);
    this.melW = [];
    for (let j = 0; j < M; j++) {
      const a = pts[j], c = pts[j + 1], b = pts[j + 2];
      const k0 = Math.ceil(a / df), k1 = Math.floor(b / df);
      const ws: number[] = [];
      for (let k = k0; k <= k1; k++) {
        const f = k * df;
        ws.push(Math.max(0, Math.min((f - a) / (c - a), (b - f) / (b - c))) * (2 / (b - a)));
      }
      this.melLo[j] = k0;
      this.kMax = Math.max(this.kMax, k1);
      this.melW.push(Float32Array.from(ws));
    }
    this.mel = new Float32Array(M);
    const C = this.C, H = this.H;
    this.x0 = new Float32Array(C);
    this.hist = this.dil.map((d) => new Float32Array(C * (2 * d + 1)));
    this.hpos = this.dil.map(() => 0);
    this.z = new Float32Array(C);
    this.zt = new Float32Array(C);
    this.h = new Float32Array(H);
    this.gates = new Float32Array(6 * H);
    this.envRel = new Float32Array(this.S);
    this.envDb = new Float32Array(this.S);
    this.act = new Float32Array(this.G);
  }

  /** Front end for one 512-sample mono block (44.1 kHz). Fills this.mel and this.mixDb. */
  frontEnd(block: Float32Array): void {
    for (let i = 0; i < block.length; i++) {
      this.ring[this.rpos] = block[i];
      this.rpos = (this.rpos + 1) & (N - 1);
    }
    // real FFT of N points via one N/2-point complex FFT (even samples -> re, odd -> im)
    const re = this.re, im = this.im, rev = this.rev, H2 = N >> 1;
    for (let n = 0; n < H2; n++) {
      const r = rev[n];
      re[r] = this.ring[(this.rpos + 2 * n) & (N - 1)] * this.win[2 * n];
      im[r] = this.ring[(this.rpos + 2 * n + 1) & (N - 1)] * this.win[2 * n + 1];
    }
    for (let size = 2; size <= H2; size <<= 1) {
      const half = size >> 1, step = N / size; // twiddle e^{-2 pi i k / size} = cosT[k * step]
      for (let s = 0; s < H2; s += size) {
        for (let k = 0; k < half; k++) {
          const c = this.cosT[k * step], sn = this.sinT[k * step];
          const a = s + k, b = a + half;
          const tr = re[b] * c - im[b] * sn, ti = re[b] * sn + im[b] * c;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
    const pw = this.pw;
    for (let k = 0; k <= this.kMax; k++) {
      const k2 = (H2 - k) & (H2 - 1);
      const zr = re[k], zi = im[k], cr = re[k2], ci = -im[k2];
      const er = 0.5 * (zr + cr), ei = 0.5 * (zi + ci); // even part
      const dr = 0.5 * (zr - cr), di = 0.5 * (zi - ci); // (Z - conj Z')/2, odd part = -i * that
      const or = di, oi = -dr;
      const c = this.cosT[k], sn = this.sinT[k];
      const xr = er + or * c - oi * sn, xi = ei + or * sn + oi * c;
      pw[k] = xr * xr + xi * xi;
    }
    let tot = 0;
    for (let j = 0; j < this.nmel; j++) {
      const w = this.melW[j], k0 = this.melLo[j];
      let p = 0;
      for (let q = 0; q < w.length; q++) {
        const k = k0 + q;
        p += w[q] * pw[k];
      }
      p *= 0.5; // 2048-sample 44.1 kHz window -> 1024-sample 22.05 kHz scale (x1/4), and ffmpeg -ac 1 downmix = (L+R)/sqrt2 (x2)
      tot += p;
      this.mel[j] = Math.log10(p + 1e-7);
    }
    this.mixDb = 10 * Math.log10(tot + 1e-10);
  }

  /** Model step on a feature frame (log-mel, length nmel). */
  step(f: Float32Array): void {
    const W = this.W, C = this.C, H = this.H, M = this.nmel;
    const mu = W['mu'], sd = W['sd'], wi = W['inp.weight'], bi = W['inp.bias'];
    const x = this.x0;
    for (let c = 0; c < C; c++) {
      let a = bi[c];
      const o = c * M;
      for (let j = 0; j < M; j++) a += wi[o + j] * ((f[j] - mu[j]) / sd[j]);
      x[c] = a > 0 ? a : 0;
    }
    let cur = x;
    for (let l = 0; l < this.dil.length; l++) {
      const d = this.dil[l], L = 2 * d + 1, hs = this.hist[l];
      // store cur at slot hpos; taps are cur(t-2d), cur(t-d), cur(t)
      const p = this.hpos[l];
      for (let c = 0; c < C; c++) hs[c * L + p] = cur[c];
      const p0 = (p - 2 * d + L) % L, p1 = (p - d + L) % L;
      this.hpos[l] = (p + 1) % L;
      const cw = W[`convs.${l}.weight`], cb = W[`convs.${l}.bias`]; // [C, C, 3]
      const out = l % 2 === 0 ? this.z : this.zt;
      for (let o = 0; o < C; o++) {
        let a = cb[o];
        const ob = o * C * 3;
        for (let i = 0; i < C; i++) {
          const r = i * L, wb = ob + i * 3;
          a += cw[wb] * hs[r + p0] + cw[wb + 1] * hs[r + p1] + cw[wb + 2] * hs[r + p];
        }
        const v = cur[o] + a;
        out[o] = v > 0 ? v : 0;
      }
      cur = out;
    }
    // GRU (PyTorch gate order r, z, n)
    const wih = W['gru.weight_ih_l0'], whh = W['gru.weight_hh_l0'], bih = W['gru.bias_ih_l0'], bhh = W['gru.bias_hh_l0'];
    const g = this.gates, h = this.h;
    for (let r = 0; r < 3 * H; r++) {
      let a = bih[r], b = bhh[r];
      const oi = r * C, oh = r * H;
      for (let i = 0; i < C; i++) a += wih[oi + i] * cur[i];
      for (let i = 0; i < H; i++) b += whh[oh + i] * h[i];
      g[r] = a; g[3 * H + r] = b;
    }
    for (let k = 0; k < H; k++) {
      const rg = 1 / (1 + Math.exp(-(g[k] + g[3 * H + k])));
      const zg = 1 / (1 + Math.exp(-(g[H + k] + g[4 * H + k])));
      const n = Math.tanh(g[2 * H + k] + rg * g[5 * H + k]);
      h[k] = (1 - zg) * n + zg * h[k];
    }
    const we = W['env.weight'], be = W['env.bias'];
    for (let s = 0; s < this.S; s++) {
      let a = be[s];
      const o = s * (C + H);
      for (let i = 0; i < C; i++) a += we[o + i] * cur[i];
      for (let i = 0; i < H; i++) a += we[o + C + i] * h[i];
      this.envRel[s] = a;
    }
    const wa = W['act.weight'], ba = W['act.bias'];
    for (let q = 0; q < this.G; q++) {
      let a = ba[q];
      const o = q * H;
      for (let i = 0; i < H; i++) a += wa[o + i] * h[i];
      this.act[q] = 1 / (1 + Math.exp(-a));
    }
  }

  /** One 512-sample mono block at 44.1 kHz -> envDb (absolute dB), envRel, act. */
  process(block: Float32Array): void {
    this.frontEnd(block);
    this.step(this.mel);
    for (let s = 0; s < this.S; s++) this.envDb[s] = this.envRel[s] + this.mixDb;
  }
}

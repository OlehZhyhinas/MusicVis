// Neural stem levels (and instrument activity) for the live path: a small causal network distilled
// from htdemucs_6s stem envelopes (offline teacher) that runs synchronously in plain TypeScript, one
// step per 512 samples at 44.1 kHz (~86 fps): no inference runtime, no audio delay.
//
// Front end: mono 44.1 kHz (other rates are linearly resampled), a 2048-sample periodic Hann window
// ENDING at the newest sample, power spectrum, 64 triangular mel bands (HTK mel, 30 Hz..11 kHz,
// area-normalised), log10(p + 1e-7). Network: Linear(64,C)+ReLU, three causal dilated convolutions
// (kernel 3, dilation 1/2/4, residual, ~170 ms receptive field), GRU(C,H); heads: per-stem level in
// dB RELATIVE TO THE MIX (drums, bass, vocals, guitar, piano, other) from the conv features and the
// GRU state (fast: ~25 ms onset response), and group activity (sigmoid) from the GRU state.
//
// Weights: public/models/stems.bin (~1 MB; scripts/ml/instruments/presence/student.py export-bin):
// u32 header length, JSON header, float32 tensors.

import { RealFFT } from './fft';

interface Arr { o: number; n: number }
interface HeaderJ {
  version: number;
  sampleRate: number;
  hop: number;
  frameSize: number;
  nMel: number;
  fmin: number;
  fmax: number;
  /** Power scale that maps this front end onto the training features (log10 domain offset folded in). */
  powScale: number;
  C: number;
  H: number;
  dil: number[];
  stems: string[];
  groups: string[];
  tensors: Record<string, Arr>;
}

export interface StemNetModel {
  header: HeaderJ;
  t: Record<string, Float32Array>;
  melStart: Int32Array;
  melW: Float32Array[];
}

/** Parse public/models/stems.bin. */
export function parseStemNet(buf: ArrayBuffer): StemNetModel {
  const hl = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as HeaderJ;
  const off = 4 + hl;
  const data = new Float32Array(buf.slice(off));
  const t: Record<string, Float32Array> = {};
  for (const [k, a] of Object.entries(header.tensors)) t[k] = data.subarray(a.o, a.o + a.n);
  // Mel filterbank on the frameSize-point spectrum at sampleRate (the same bin frequencies as training).
  const hz2mel = (f: number) => 2595 * Math.log10(1 + f / 700);
  const mel2hz = (x: number) => 700 * (10 ** (x / 2595) - 1);
  const M = header.nMel, lo = hz2mel(header.fmin), hi = hz2mel(header.fmax);
  const pts = Array.from({ length: M + 2 }, (_, i) => mel2hz(lo + ((hi - lo) * i) / (M + 1)));
  const df = header.sampleRate / header.frameSize;
  const melStart = new Int32Array(M);
  const melW: Float32Array[] = [];
  for (let j = 0; j < M; j++) {
    const a = pts[j], c = pts[j + 1], b = pts[j + 2];
    const k0 = Math.ceil(a / df), k1 = Math.floor(b / df);
    const w = new Float32Array(Math.max(0, k1 - k0 + 1));
    for (let k = k0; k <= k1; k++) {
      const f = k * df;
      w[k - k0] = Math.max(0, Math.min((f - a) / (c - a), (b - f) / (b - c))) * (2 / (b - a));
    }
    melStart[j] = k0;
    melW.push(w);
  }
  return { header, t, melStart, melW };
}

/** Fetch and parse the shipped weights (browser). */
export async function loadStemNet(url = 'models/stems.bin'): Promise<StemNetModel> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`stem net weights: HTTP ${r.status}`);
  return parseStemNet(await r.arrayBuffer());
}

/** Whether the live path uses the stem network by default (the live panel's "Neural stems" setting). */
export const DEFAULT_NEURAL_STEMS = true;

/**
 * One frame of neural stem levels. drums/bass/vocals/other are LINEAR power ratios against the mix
 * (other = other + guitar + piano, the 4-stem split the presets use); db holds the six stems'
 * levels in dB relative to the mix; act the group activities (0..1) in `groups` order.
 */
export interface StemLevels {
  drums: number;
  bass: number;
  vocals: number;
  other: number;
  readonly db: Float32Array;
  readonly act: Float32Array;
  readonly stems: readonly string[];
  readonly groups: readonly string[];
  /** Stream time (s) of the newest sample of the frame. */
  t: number;
}

/** The network with its streaming state. push() blocks; every 512 samples (44.1 kHz) one step runs. */
export class StemNet {
  readonly model: StemNetModel;
  private readonly fft: RealFFT;
  private readonly win: Float64Array;
  private readonly buf: Float64Array;
  private readonly pow: Float64Array;
  private readonly ring: Float32Array;
  private readonly mel: Float32Array;
  private readonly ratio: number;
  private written = 0;
  private sinceHop = 0;
  private resPos = 0;
  private lastIn = 0;
  /** Mix power of the latest frame (sum of the mel band powers, the training features' scale). */
  private mixP = 0;
  // network state
  private readonly x0: Float32Array;
  private readonly hist: Float32Array[];
  private readonly hpos: number[];
  private readonly za: Float32Array;
  private readonly zb: Float32Array;
  private readonly h: Float32Array;
  private readonly g: Float32Array;
  readonly out: StemLevels;
  private readonly iStem: Record<string, number>;

  constructor(model: StemNetModel, sampleRate: number) {
    this.model = model;
    const h = model.header;
    this.fft = new RealFFT(h.frameSize);
    this.win = new Float64Array(h.frameSize);
    for (let i = 0; i < h.frameSize; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / h.frameSize);
    this.buf = new Float64Array(h.frameSize);
    this.pow = new Float64Array(h.frameSize / 2 + 1);
    this.ring = new Float32Array(h.frameSize);
    this.mel = new Float32Array(h.nMel);
    this.ratio = sampleRate / h.sampleRate;
    const C = h.C;
    this.x0 = new Float32Array(C);
    this.hist = h.dil.map((d) => new Float32Array(C * (2 * d + 1)));
    this.hpos = h.dil.map(() => 0);
    this.za = new Float32Array(C);
    this.zb = new Float32Array(C);
    this.h = new Float32Array(h.H);
    this.g = new Float32Array(6 * h.H);
    this.iStem = Object.fromEntries(h.stems.map((s, i) => [s, i]));
    this.out = { drums: 0, bass: 0, vocals: 0, other: 0, db: new Float32Array(h.stems.length).fill(-60), act: new Float32Array(h.groups.length), stems: h.stems, groups: h.groups, t: 0 };
  }

  reset(): void {
    this.ring.fill(0);
    this.written = this.sinceHop = 0;
    this.resPos = this.lastIn = 0;
    for (const hs of this.hist) hs.fill(0);
    this.hpos.fill(0);
    this.h.fill(0);
    this.out.t = 0;
  }

  /** Push mono samples at the input rate; onFrame is called after every network step. */
  push(x: Float32Array, onFrame?: (s: StemLevels) => void): void {
    if (this.ratio === 1) {
      for (let i = 0; i < x.length; i++) this.sample(x[i], onFrame);
      return;
    }
    let p = this.resPos;
    let prev = this.lastIn;
    for (let i = 0; i < x.length; i++) {
      const cur = x[i];
      while (p <= i) {
        this.sample(prev + (cur - prev) * (p - (i - 1)), onFrame);
        p += this.ratio;
      }
      prev = cur;
    }
    this.resPos = p - x.length;
    this.lastIn = prev;
  }

  private sample(v: number, onFrame?: (s: StemLevels) => void): void {
    const N = this.ring.length;
    this.ring[this.written % N] = v;
    this.written++;
    if (++this.sinceHop === this.model.header.hop) {
      this.sinceHop = 0;
      this.frontEnd();
      this.step();
      this.out.t = this.written / this.model.header.sampleRate;
      onFrame?.(this.out);
    }
  }

  private frontEnd(): void {
    const h = this.model.header, N = h.frameSize;
    const start = this.written - N;
    for (let i = 0; i < N; i++) {
      const k = start + i;
      this.buf[i] = k >= 0 ? this.ring[k % N] * this.win[i] : 0;
    }
    this.fft.power(this.buf, this.pow);
    const { melStart, melW } = this.model;
    let mix = 0;
    for (let j = 0; j < h.nMel; j++) {
      const w = melW[j], k0 = melStart[j];
      let p = 0;
      for (let q = 0; q < w.length; q++) p += w[q] * this.pow[k0 + q];
      p = p * h.powScale + 1e-7;
      mix += p;
      this.mel[j] = Math.log10(p);
    }
    this.mixP = mix;
  }

  private step(): void {
    const hd = this.model.header, W = this.model.t, C = hd.C, H = hd.H, M = hd.nMel;
    const mu = W['mu'], sd = W['sd'], wi = W['inp.weight'], bi = W['inp.bias'];
    const f = this.mel, x = this.x0;
    const mixP = this.mixP;
    for (let c = 0; c < C; c++) {
      let a = bi[c];
      const o = c * M;
      for (let j = 0; j < M; j++) a += wi[o + j] * ((f[j] - mu[j]) / sd[j]);
      x[c] = a > 0 ? a : 0;
    }
    let cur = x;
    for (let l = 0; l < hd.dil.length; l++) {
      const d = hd.dil[l], L = 2 * d + 1, hs = this.hist[l], p = this.hpos[l];
      for (let c = 0; c < C; c++) hs[c * L + p] = cur[c];
      const p0 = (p - 2 * d + L) % L, p1 = (p - d + L) % L;
      this.hpos[l] = (p + 1) % L;
      const cw = W[`convs.${l}.weight`], cb = W[`convs.${l}.bias`];
      const out = l % 2 === 0 ? this.za : this.zb;
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
    const g = this.g, h = this.h;
    for (let r = 0; r < 3 * H; r++) {
      let a = bih[r], b = bhh[r];
      const oi = r * C, oh = r * H;
      for (let i = 0; i < C; i++) a += wih[oi + i] * cur[i];
      for (let i = 0; i < H; i++) b += whh[oh + i] * h[i];
      g[r] = a;
      g[3 * H + r] = b;
    }
    for (let k = 0; k < H; k++) {
      const rg = 1 / (1 + Math.exp(-(g[k] + g[3 * H + k])));
      const zg = 1 / (1 + Math.exp(-(g[H + k] + g[4 * H + k])));
      const n = Math.tanh(g[2 * H + k] + rg * g[5 * H + k]);
      h[k] = (1 - zg) * n + zg * h[k];
    }
    const out = this.out, S = hd.stems.length;
    const we = W['env.weight'], be = W['env.bias'];
    for (let s = 0; s < S; s++) {
      let a = be[s];
      const o = s * (C + H);
      for (let i = 0; i < C; i++) a += we[o + i] * cur[i];
      for (let i = 0; i < H; i++) a += we[o + C + i] * h[i];
      out.db[s] = mixP > 1e-9 ? a : -60;
    }
    const wa = W['act.weight'], ba = W['act.bias'];
    for (let q = 0; q < hd.groups.length; q++) {
      let a = ba[q];
      const o = q * H;
      for (let i = 0; i < H; i++) a += wa[o + i] * h[i];
      out.act[q] = 1 / (1 + Math.exp(-a));
    }
    const lin = (name: string) => (name in this.iStem ? 10 ** (out.db[this.iStem[name]] / 10) : 0);
    out.drums = lin('drums');
    out.bass = lin('bass');
    out.vocals = lin('vocals');
    out.other = lin('other') + lin('guitar') + lin('piano');
  }
}

/**
 * The stem network as RealtimeAnalyzer's stem source. Push each capture block before the analyzer
 * processes it; the analyzer then asks at(t) for each of its frames and gets the newest network frame
 * that ended by t (null before the first), so the levels stay causal and at the sound's time.
 */
export class StemNetSource {
  readonly net: StemNet;
  private readonly mono: Float32Array;
  private t0: number;
  private have = false;
  /** Latest frame, with its time shifted to the analyzer's stream clock. */
  private readonly latest: StemLevels;
  private readonly pendT: Float64Array;
  private readonly pend: Float32Array; // queued frames: [drums, bass, vocals, other, db..., act...]
  private readonly width: number;
  private qh = 0;
  private qn = 0;
  constructor(model: StemNetModel, sampleRate: number, t0 = 0) {
    this.net = new StemNet(model, sampleRate);
    this.t0 = t0;
    this.mono = new Float32Array(8192);
    const o = this.net.out;
    this.latest = { ...o, db: new Float32Array(o.db.length), act: new Float32Array(o.act.length) };
    this.width = 4 + o.db.length + o.act.length;
    this.pendT = new Float64Array(16);
    this.pend = new Float32Array(16 * this.width);
  }
  reset(t0 = 0): void {
    this.net.reset();
    this.t0 = t0;
    this.have = false;
    this.qh = this.qn = 0;
  }
  private readonly onFrame = (s: StemLevels) => {
    const cap = this.pendT.length;
    if (this.qn === cap) {
      this.qh = (this.qh + 1) % cap; // drop the oldest (the analyzer stopped asking)
      this.qn--;
    }
    const i = (this.qh + this.qn) % cap, w = this.width, b = i * w;
    this.pendT[i] = s.t + this.t0;
    this.pend[b] = s.drums;
    this.pend[b + 1] = s.bass;
    this.pend[b + 2] = s.vocals;
    this.pend[b + 3] = s.other;
    this.pend.set(s.db, b + 4);
    this.pend.set(s.act, b + 4 + s.db.length);
    this.qn++;
  };
  push(l: Float32Array, r: Float32Array): void {
    const n = Math.min(l.length, this.mono.length);
    for (let i = 0; i < n; i++) this.mono[i] = 0.5 * (l[i] + r[i]);
    this.net.push(this.mono.subarray(0, n), this.onFrame);
  }
  /** Stem levels for the analyzer frame at stream time t (the newest network frame ended by t). */
  readonly at = (t: number): StemLevels | null => {
    const cap = this.pendT.length, w = this.width, L = this.latest;
    while (this.qn > 0 && this.pendT[this.qh] <= t + 1e-6) {
      const b = this.qh * w;
      L.t = this.pendT[this.qh];
      L.drums = this.pend[b];
      L.bass = this.pend[b + 1];
      L.vocals = this.pend[b + 2];
      L.other = this.pend[b + 3];
      L.db.set(this.pend.subarray(b + 4, b + 4 + L.db.length));
      L.act.set(this.pend.subarray(b + 4 + L.db.length, b + w));
      this.have = true;
      this.qh = (this.qh + 1) % cap;
      this.qn--;
    }
    return this.have ? L : null;
  };
}

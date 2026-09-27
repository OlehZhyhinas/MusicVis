// Live downbeat evidence from madmom's downbeat RNN (Boeck, Krebs & Widmer 2016: 8 bidirectional
// 3x25 LSTM networks on a 3-resolution log-filtered spectrogram, outputs beat / downbeat
// probabilities), ported to plain TypeScript and run on a sliding window of PAST audio: every `hop`
// seconds the last `window` seconds are analysed and the downbeat probabilities of the frames
// `delay`..`delay + hop` seconds before the window's end are reported (they had `delay` seconds of
// right context). So evidence for a moment arrives delay..delay + hop seconds after it; the bar slot
// vote in BeatTracker does not mind.
//
// Features are computed once per frame as the audio arrives (not per window). The network work of a
// window is a generator stepped with a per-block time budget, so there is no spike on the main thread.
//
// Weights: public/models/downbeat-blstm.bin (scripts/ml/export-downbeat-blstm.py).

import { RealFFT } from './fft';

interface Arr { o: number; n: number }
interface GateJ { W: Arr; R: Arr; b: Arr; p: Arr | null }
interface LstmJ { in: number; units: number; input_gate: GateJ; forget_gate: GateJ; cell: GateJ; output_gate: GateJ }
interface ResJ { frameSize: number; bins: number; bands: number; fbStart: number[]; fbLen: number[]; fb: Arr; diffFrames: number }
interface HeaderJ {
  fps: number;
  hop: number;
  sampleRate: number;
  logMul: number;
  logAdd: number;
  resolutions: ResJ[];
  models: { layers: ({ type: 'bilstm'; fwd: LstmJ; bwd: LstmJ } | { type: 'dense'; in: number; out: number; W: Arr; b: Arr })[] }[];
}

/** An LSTM with its four gates packed row-major: row r = gate * n + unit (gates i, f, c, o). */
interface Lstm { nin: number; n: number; W: Float32Array; R: Float32Array; b: Float32Array; pi: Float32Array; pf: Float32Array; po: Float32Array }
interface Net { layers: { fwd: Lstm; bwd: Lstm }[]; W: Float32Array; b: Float32Array; nout: number }
interface Res { N: number; bands: number; fb: { start: number; w: Float32Array }[]; diff: number; fft: RealFFT; win: Float64Array }

export interface DownbeatModel {
  header: HeaderJ;
  nets: Net[];
  res: Res[];
  nfeat: number;
}

export function parseDownbeat(buf: ArrayBuffer): DownbeatModel {
  const hl = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as HeaderJ;
  const data = new Float32Array(buf, 4 + hl);
  const a = (x: Arr) => data.subarray(x.o, x.o + x.n);
  const pack = (x: LstmJ): Lstm => {
    const n = x.units, nin = x.in;
    const gates = [x.input_gate, x.forget_gate, x.cell, x.output_gate];
    const W = new Float32Array(4 * n * nin), R = new Float32Array(4 * n * n), b = new Float32Array(4 * n);
    gates.forEach((g, gi) => {
      const w = a(g.W), r = a(g.R), bb = a(g.b); // madmom: weights [in, n]
      for (let u = 0; u < n; u++) {
        const row = gi * n + u;
        for (let i = 0; i < nin; i++) W[row * nin + i] = w[i * n + u];
        for (let i = 0; i < n; i++) R[row * n + i] = r[i * n + u];
        b[row] = bb[u];
      }
    });
    const p = (g: GateJ) => (g.p ? Float32Array.from(a(g.p)) : new Float32Array(n));
    return { nin, n, W, R, b, pi: p(x.input_gate), pf: p(x.forget_gate), po: p(x.output_gate) };
  };
  const nets = header.models.map((m): Net => {
    const layers: Net['layers'] = [];
    let W = new Float32Array(0), b = new Float32Array(0), nout = 0;
    for (const L of m.layers) {
      if (L.type === 'bilstm') layers.push({ fwd: pack(L.fwd), bwd: pack(L.bwd) });
      else (W = a(L.W)), (b = a(L.b)), (nout = L.out);
    }
    return { layers, W, b, nout };
  });
  const res = header.resolutions.map((r): Res => {
    const fbw = a(r.fb);
    const fb: Res['fb'] = [];
    let o = 0;
    for (let k = 0; k < r.bands; k++) {
      fb.push({ start: r.fbStart[k], w: fbw.subarray(o, o + r.fbLen[k]) });
      o += r.fbLen[k];
    }
    const win = new Float64Array(r.frameSize);
    for (let i = 0; i < r.frameSize; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (r.frameSize - 1));
    return { N: r.frameSize, bands: r.bands, fb, diff: r.diffFrames, fft: new RealFFT(r.frameSize), win };
  });
  return { header, nets, res, nfeat: res.reduce((s, r) => s + 2 * r.bands, 0) };
}

const sigm = (x: number) => 1 / (1 + Math.exp(-x));

/** One LSTM step (madmom semantics: peepholes on i / f from the old state, on o from the new). */
function lstmStep(l: Lstm, x: Float32Array, xo: number, prev: Float32Array, st: Float32Array, g: Float32Array): void {
  const n = l.n, nin = l.nin, W = l.W, R = l.R;
  for (let r = 0; r < 4 * n; r++) {
    let s = l.b[r];
    const rw = r * nin;
    for (let i = 0; i < nin; i++) s += W[rw + i] * x[xo + i];
    const rr = r * n;
    for (let i = 0; i < n; i++) s += R[rr + i] * prev[i];
    g[r] = s;
  }
  for (let u = 0; u < n; u++) {
    const c = sigm(g[u] + st[u] * l.pi[u]) * Math.tanh(g[2 * n + u]) + st[u] * sigm(g[n + u] + st[u] * l.pf[u]);
    st[u] = c;
    prev[u] = Math.tanh(c) * sigm(g[3 * n + u] + c * l.po[u]);
  }
}

/** Spectrogram features, one frame at a time (frame centred on the given sample of `src`). */
class FrameFeatures {
  private readonly m: DownbeatModel;
  private readonly bufs: Float64Array[];
  private readonly pows: Float64Array[];
  /** Per resolution, the last (diff + 1) band spectra (ring). */
  private readonly hist: Float32Array[][];
  private frames = 0;
  constructor(m: DownbeatModel) {
    this.m = m;
    this.bufs = m.res.map((r) => new Float64Array(r.N));
    this.pows = m.res.map((r) => new Float64Array(r.N / 2 + 1));
    this.hist = m.res.map((r) => Array.from({ length: r.diff + 1 }, () => new Float32Array(r.bands)));
  }
  reset(): void {
    this.frames = 0;
  }
  /** at(k): sample k of the signal (0 outside it). Writes nfeat values to out at oo. */
  frame(at: (k: number) => number, centre: number, out: Float32Array, oo: number): void {
    const m = this.m;
    let col = 0;
    m.res.forEach((r, ri) => {
      const buf = this.bufs[ri], pow = this.pows[ri];
      const start = centre - (r.N >> 1);
      for (let j = 0; j < r.N; j++) buf[j] = at(start + j) * r.win[j];
      r.fft.power(buf, pow);
      const h = this.hist[ri];
      const cur = h[this.frames % h.length];
      for (let b = 0; b < r.bands; b++) {
        const { start: s0, w } = r.fb[b];
        let s = 0;
        for (let k = 0; k < w.length; k++) s += w[k] * Math.sqrt(pow[s0 + k]);
        cur[b] = Math.log10(m.header.logMul * s + m.header.logAdd);
      }
      const old = this.frames >= r.diff ? h[(this.frames - r.diff) % h.length] : null;
      for (let b = 0; b < r.bands; b++) {
        out[oo + col + b] = cur[b];
        out[oo + col + r.bands + b] = old ? Math.max(0, cur[b] - old[b]) : 0;
      }
      col += 2 * r.bands;
    });
    this.frames++;
  }
}

/** Features of `x` (mono, 44.1 kHz) in madmom's file mode: frame i centred on sample i * hop. */
export function downbeatFeatures(m: DownbeatModel, x: Float32Array): Float32Array {
  const hop = m.header.hop;
  const T = Math.ceil(x.length / hop);
  const out = new Float32Array(T * m.nfeat);
  const ff = new FrameFeatures(m);
  const at = (k: number) => (k >= 0 && k < x.length ? x[k] : 0);
  for (let i = 0; i < T; i++) ff.frame(at, i * hop, out, i * m.nfeat);
  return out;
}

/**
 * The networks over T frames of features. Returns T x 2 (beat, downbeat), averaged over the nets.
 * A generator: yields after every `step` LSTM steps so callers can slice the work.
 */
export function* runNets(m: DownbeatModel, feat: Float32Array, T: number, nets: number, step = 64): Generator<void, Float32Array> {
  const out = new Float32Array(T * 2);
  const use = m.nets.slice(0, nets);
  let steps = 0;
  for (const net of use) {
    let x = feat, nin = m.nfeat;
    for (const { fwd, bwd } of net.layers) {
      const n = fwd.n;
      const y = new Float32Array(T * 2 * n);
      const prev = new Float32Array(n), st = new Float32Array(n), g = new Float32Array(4 * n);
      for (let t = 0; t < T; t++) {
        lstmStep(fwd, x, t * nin, prev, st, g);
        y.set(prev, t * 2 * n);
        if (++steps % step === 0) yield;
      }
      prev.fill(0);
      st.fill(0);
      for (let t = T - 1; t >= 0; t--) {
        lstmStep(bwd, x, t * nin, prev, st, g);
        y.set(prev, t * 2 * n + n);
        if (++steps % step === 0) yield;
      }
      x = y;
      nin = 2 * n;
    }
    // Dense + softmax (no beat / beat / downbeat).
    for (let t = 0; t < T; t++) {
      let z0 = net.b[0], z1 = net.b[1], z2 = net.b[2];
      for (let i = 0; i < nin; i++) {
        const v = x[t * nin + i];
        z0 += v * net.W[i * 3];
        z1 += v * net.W[i * 3 + 1];
        z2 += v * net.W[i * 3 + 2];
      }
      const mx = Math.max(z0, z1, z2);
      const e0 = Math.exp(z0 - mx), e1 = Math.exp(z1 - mx), e2 = Math.exp(z2 - mx);
      const sum = e0 + e1 + e2;
      out[t * 2] += e1 / sum / use.length;
      out[t * 2 + 1] += e2 / sum / use.length;
    }
  }
  return out;
}

export interface DownbeatOptions {
  sampleRate: number;
  /** Networks of the ensemble (1..8). */
  nets?: number;
  /** Seconds of past audio per analysis. */
  window?: number;
  /** Seconds between analyses (= the span of frames reported per analysis). */
  hop?: number;
  /** Seconds of right context the reported frames get. */
  delay?: number;
}

/**
 * Streaming driver: push capture blocks, then call work(budgetMs) to advance the pending analysis.
 * Each finished analysis calls onEvidence(t, downbeat, beat) for every reported frame, t being the
 * stream time of the frame centre (t0 + seconds since the first pushed sample).
 */
export class DownbeatTracker {
  readonly model: DownbeatModel;
  readonly nets: number;
  readonly window: number;
  readonly hop: number;
  readonly delay: number;
  private readonly ratio: number;
  private readonly ring: Float32Array;
  private readonly ff: FrameFeatures;
  private readonly feats: Float32Array; // ring of frames x nfeat
  private readonly winFrames: number;
  private readonly hopFrames: number;
  private readonly lookSamples: number;
  private written = 0; // samples at 44.1 kHz
  private frames = 0; // feature frames computed
  private resPos = 0;
  private lastIn = 0;
  private nextAt: number; // frame count at which the next analysis starts
  private job: Generator<void, Float32Array> | null = null;
  private jobEnd = 0; // frame count at the job's start
  private jobT = 0;
  private readonly at = (k: number) => (k >= 0 && k < this.written && k >= this.written - this.ring.length ? this.ring[k % this.ring.length] : 0);
  onEvidence: ((t: number, down: number, beat: number) => void) | null = null;
  t0 = 0;
  /** Milliseconds spent in analyses (not features), for diagnostics. */
  workMs = 0;
  constructor(model: DownbeatModel, opts: DownbeatOptions) {
    this.model = model;
    const fps = model.header.fps;
    this.nets = Math.max(1, Math.min(model.nets.length, opts.nets ?? 2));
    this.window = opts.window ?? 5;
    this.hop = opts.hop ?? 1;
    this.delay = opts.delay ?? 1.5;
    this.ratio = opts.sampleRate / model.header.sampleRate;
    this.winFrames = Math.round(this.window * fps);
    this.hopFrames = Math.max(1, Math.round(this.hop * fps));
    this.lookSamples = Math.max(...model.res.map((r) => r.N >> 1));
    this.ring = new Float32Array(1 << 14);
    this.ff = new FrameFeatures(model);
    this.feats = new Float32Array((this.winFrames + this.hopFrames * 2 + 8) * model.nfeat);
    this.nextAt = this.winFrames;
  }
  reset(t0 = 0): void {
    this.written = 0;
    this.frames = 0;
    this.resPos = 0;
    this.lastIn = 0;
    this.job = null;
    this.ff.reset();
    this.nextAt = this.winFrames;
    this.t0 = t0;
  }
  push(x: Float32Array): void {
    if (this.ratio === 1) {
      for (let i = 0; i < x.length; i++) this.sample(x[i]);
    } else {
      let p = this.resPos, prev = this.lastIn;
      for (let i = 0; i < x.length; i++) {
        const cur = x[i];
        while (p <= i) {
          this.sample(prev + (cur - prev) * (p - (i - 1)));
          p += this.ratio;
        }
        prev = cur;
      }
      this.resPos = p - x.length;
      this.lastIn = prev;
    }
    // Feature frames whose windows are complete.
    const hop = this.model.header.hop, F = this.model.nfeat, cap = this.feats.length / F;
    while (this.frames * hop + this.lookSamples <= this.written) {
      this.ff.frame(this.at, this.frames * hop, this.feats, (this.frames % cap) * F);
      this.frames++;
      if (this.frames >= this.nextAt && !this.job) this.start();
    }
  }
  private sample(v: number): void {
    this.ring[this.written % this.ring.length] = v;
    this.written++;
  }
  private start(): void {
    const F = this.model.nfeat, cap = this.feats.length / F, T = this.winFrames;
    const feat = new Float32Array(T * F);
    for (let i = 0; i < T; i++) {
      const f = this.frames - T + i;
      feat.set(this.feats.subarray((f % cap) * F, (f % cap + 1) * F), i * F);
    }
    this.job = runNets(this.model, feat, T, this.nets);
    this.jobEnd = this.frames;
    this.jobT = T;
    this.nextAt = this.frames + this.hopFrames;
  }
  /** Advance the pending analysis for up to budgetMs (Infinity: finish it). */
  work(budgetMs: number): void {
    if (!this.job) return;
    const t0 = performance.now();
    for (;;) {
      const r = this.job.next();
      if (r.done) {
        this.finish(r.value);
        break;
      }
      if (performance.now() - t0 >= budgetMs) break;
    }
    this.workMs += performance.now() - t0;
  }
  private finish(act: Float32Array): void {
    this.job = null;
    const fps = this.model.header.fps;
    const first = this.jobEnd - this.jobT; // absolute frame index of the window's first frame
    const d = Math.round(this.delay * fps);
    for (let i = this.jobT - d - this.hopFrames; i < this.jobT - d; i++) {
      if (i < 0) continue;
      this.onEvidence?.(this.t0 + (first + i) / fps, act[i * 2 + 1], act[i * 2]);
    }
    // Fell behind (slow machine): start on the newest audio right away.
    if (this.frames >= this.nextAt) this.start();
  }
}

export async function loadDownbeat(url = 'models/downbeat-blstm.bin'): Promise<DownbeatModel> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`downbeat weights: HTTP ${r.status}`);
  return parseDownbeat(await r.arrayBuffer());
}

/** Whether the live path runs the downbeat network along with the beat RNN ("Neural beats"). */
export const DEFAULT_NEURAL_DOWNBEATS = true;

/** Settings of the live downbeat detector (see scripts/ml/downbeat-eval.ts for the measurements). */
export interface LiveDownbeatOptions extends Omit<DownbeatOptions, 'sampleRate'> {
  /** Weight of BeatTracker's own accents (kick / bass / snare, chord changes) once evidence arrives. */
  dspAccentWeight?: number;
  /** Weight of the network's evidence in the bar slot vote. */
  weight?: number;
}
export const LIVE_DOWNBEAT_DEFAULTS: Required<LiveDownbeatOptions> = { nets: 2, window: 5, hop: 1, delay: 1.5, dspAccentWeight: 0, weight: 1 };

/** The beat tracker side of the analyzer that the detector votes into. */
interface BeatSink {
  downbeatEvidence(t: number, v: number): void;
  dspAccentWeight: number;
  externalAccentWeight: number;
}

/**
 * The downbeat detector wired to a RealtimeAnalyzer: push each capture block, then work(budgetMs)
 * once per block; evidence goes to analyzer.beat (looked up on every call, so an analyzer reset is fine).
 */
export class LiveDownbeats {
  readonly tracker: DownbeatTracker;
  private readonly mono: Float32Array = new Float32Array(8192);
  constructor(model: DownbeatModel, sampleRate: number, analyzer: { beat: BeatSink; streamTime: number }, opts: LiveDownbeatOptions = {}) {
    const o = { ...LIVE_DOWNBEAT_DEFAULTS, ...opts };
    this.tracker = new DownbeatTracker(model, { sampleRate, nets: o.nets, window: o.window, hop: o.hop, delay: o.delay });
    this.tracker.t0 = analyzer.streamTime;
    this.tracker.onEvidence = (t, v) => {
      const bt = analyzer.beat;
      bt.dspAccentWeight = o.dspAccentWeight;
      bt.externalAccentWeight = o.weight;
      bt.downbeatEvidence(t, v);
    };
  }
  push(l: Float32Array, r: Float32Array): void {
    const n = Math.min(l.length, this.mono.length);
    for (let i = 0; i < n; i++) this.mono[i] = 0.5 * (l[i] + r[i]);
    this.tracker.push(this.mono.subarray(0, n));
  }
  work(budgetMs: number): void {
    this.tracker.work(budgetMs);
  }
}

// Online beat activation from madmom's causal beat RNN (Boeck, Krebs & Widmer: 8 uni-directional
// LSTM networks, 3 layers x 25 units with peepholes, averaged), ported to plain TypeScript so it runs
// synchronously in the live path: no inference runtime, no async hop, no audio delay.
//
// Frontend (as madmom RNNBeatProcessor(online=True)): 44.1 kHz mono, 2048-sample Hann frames at
// 100 fps, magnitude spectrum -> 81-band log filterbank (12 per octave, 30 Hz..17 kHz, normalised)
// -> log10(1 + x), plus its positive first difference; 162 inputs per frame. Frames END at the
// newest sample (causal), so an activation describes the sound up to its frame time.
//
// Weights and filterbank come from public/models/beat-lstm.bin (scripts/ml/export-beat-lstm.py,
// ~0.95 MB). Cost: ~0.24 M multiply-adds per frame for all 8 networks (100 frames/s).

import { RealFFT } from './fft';

interface Arr { o: number; n: number }
interface GateJ { W: Arr; R: Arr; b: Arr; p: Arr | null }
interface LayerJ {
  type: 'lstm' | 'dense';
  in: number;
  units?: number;
  out?: number;
  input_gate?: GateJ;
  forget_gate?: GateJ;
  cell?: GateJ;
  output_gate?: GateJ;
  W?: Arr;
  b?: Arr;
}
interface HeaderJ {
  fps: number;
  frameSize: number;
  hop: number;
  sampleRate: number;
  bins: number;
  bands: number;
  fbStart: number[];
  fbLen: number[];
  fb: Arr;
  logMul: number;
  logAdd: number;
  diffFrames: number;
  models: { layers: LayerJ[] }[];
}

interface Gate { W: Float32Array; R: Float32Array; b: Float32Array; p: Float32Array | null }
interface Lstm { kind: 'lstm'; nin: number; n: number; ig: Gate; fg: Gate; cg: Gate; og: Gate }
interface Dense { kind: 'dense'; nin: number; n: number; W: Float32Array; b: Float32Array }

export interface BeatRnnModel {
  header: HeaderJ;
  fb: { start: number; w: Float32Array }[];
  nets: (Lstm | Dense)[][];
}

/** Parse public/models/beat-lstm.bin. */
export function parseBeatRnn(buf: ArrayBuffer): BeatRnnModel {
  const hl = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))) as HeaderJ;
  const data = new Float32Array(buf, 4 + hl);
  const a = (x: Arr) => data.subarray(x.o, x.o + x.n);
  const g = (x: GateJ): Gate => ({ W: a(x.W), R: a(x.R), b: a(x.b), p: x.p ? a(x.p) : null });
  const fbw = a(header.fb);
  const fb: BeatRnnModel['fb'] = [];
  let o = 0;
  for (let b = 0; b < header.bands; b++) {
    fb.push({ start: header.fbStart[b], w: fbw.subarray(o, o + header.fbLen[b]) });
    o += header.fbLen[b];
  }
  const nets = header.models.map((m) =>
    m.layers.map((l): Lstm | Dense =>
      l.type === 'lstm'
        ? { kind: 'lstm', nin: l.in, n: l.units!, ig: g(l.input_gate!), fg: g(l.forget_gate!), cg: g(l.cell!), og: g(l.output_gate!) }
        : { kind: 'dense', nin: l.in, n: l.out!, W: a(l.W!), b: a(l.b!) },
    ),
  );
  return { header, fb, nets };
}

const sigm = (x: number) => 1 / (1 + Math.exp(-x));

/** One LSTM network's recurrent state. */
class NetState {
  readonly prev: Float32Array[];
  readonly state: Float32Array[];
  readonly bufs: Float32Array[];
  readonly layers: (Lstm | Dense)[];
  constructor(layers: (Lstm | Dense)[]) {
    this.layers = layers;
    this.prev = layers.map((l) => new Float32Array(l.n));
    this.state = layers.map((l) => new Float32Array(l.n));
    this.bufs = layers.map((l) => new Float32Array(l.n));
  }
  reset(): void {
    for (const p of this.prev) p.fill(0);
    for (const s of this.state) s.fill(0);
  }
  step(x: Float32Array): number {
    let inp = x;
    for (let li = 0; li < this.layers.length; li++) {
      const l = this.layers[li];
      const out = this.bufs[li];
      if (l.kind === 'dense') {
        for (let u = 0; u < l.n; u++) {
          let s = l.b[u];
          for (let i = 0; i < l.nin; i++) s += inp[i] * l.W[i * l.n + u];
          out[u] = sigm(s);
        }
      } else {
        const n = l.n, prev = this.prev[li], st = this.state[li];
        // Gate pre-activations: x.W + b + prev.R (+ peephole * state).
        for (let u = 0; u < n; u++) {
          let si = l.ig.b[u], sf = l.fg.b[u], sc = l.cg.b[u], so = l.og.b[u];
          for (let i = 0; i < l.nin; i++) {
            const v = inp[i];
            if (v === 0) continue;
            const k = i * n + u;
            si += v * l.ig.W[k];
            sf += v * l.fg.W[k];
            sc += v * l.cg.W[k];
            so += v * l.og.W[k];
          }
          for (let i = 0; i < n; i++) {
            const v = prev[i];
            const k = i * n + u;
            si += v * l.ig.R[k];
            sf += v * l.fg.R[k];
            sc += v * l.cg.R[k];
            so += v * l.og.R[k];
          }
          if (l.ig.p) si += st[u] * l.ig.p[u];
          if (l.fg.p) sf += st[u] * l.fg.p[u];
          const c = sigm(si) * Math.tanh(sc) + st[u] * sigm(sf);
          newState[u] = c;
          // The output gate's peephole sees the NEW state (madmom LSTMLayer.activate).
          out[u] = Math.tanh(c) * sigm(so + (l.og.p ? c * l.og.p[u] : 0));
        }
        st.set(newState.subarray(0, n));
        prev.set(out);
      }
      inp = out;
    }
    return inp[0];
  }
}
/** Scratch for the new cell states (every unit reads the old state until the layer is done). */
const newState = new Float32Array(256);

export interface BeatRnnOptions {
  /** Input sample rate (resampled linearly to 44.1 kHz when different). */
  sampleRate: number;
  /** Networks of the ensemble to run (1..8, default all 8). */
  nets?: number;
  /**
   * Samples of look-ahead per frame at 44.1 kHz: 0 = causal (frame ends at the newest sample, the
   * live default); 1023 = madmom's file mode (frames centred on i * hop), used to verify the port.
   */
  lookahead?: number;
}

export class BeatRnn {
  readonly model: BeatRnnModel;
  readonly fps: number;
  private readonly fft: RealFFT;
  private readonly win: Float64Array;
  private readonly buf: Float64Array;
  private readonly pow: Float64Array;
  private readonly ring: Float32Array;
  private readonly spec: Float32Array;
  private readonly prevSpec: Float32Array;
  readonly feat: Float32Array;
  private readonly nets: NetState[];
  private readonly ratio: number;
  private readonly la: number;
  /** Samples written (at 44.1 kHz). */
  private written = 0;
  private nextRef: number;
  private frames = 0;
  private resPos = 0;
  private lastIn = 0;
  /** Latest ensemble activation (0..1) and the stream time (s) of its frame's newest sample. */
  activation = 0;
  frameTime = 0;
  constructor(model: BeatRnnModel, opts: BeatRnnOptions) {
    this.model = model;
    const h = model.header;
    this.fps = h.fps;
    this.fft = new RealFFT(h.frameSize);
    this.win = new Float64Array(h.frameSize);
    // numpy.hanning (symmetric)
    for (let i = 0; i < h.frameSize; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (h.frameSize - 1));
    this.buf = new Float64Array(h.frameSize);
    this.pow = new Float64Array(h.frameSize / 2 + 1);
    this.ring = new Float32Array(h.frameSize * 2);
    this.spec = new Float32Array(h.bands);
    this.prevSpec = new Float32Array(h.bands);
    this.feat = new Float32Array(2 * h.bands);
    this.nets = model.nets.slice(0, Math.max(1, Math.min(model.nets.length, opts.nets ?? model.nets.length))).map((l) => new NetState(l));
    this.ratio = opts.sampleRate / h.sampleRate;
    this.la = opts.lookahead ?? 0;
    this.nextRef = 0;
  }

  reset(): void {
    for (const n of this.nets) n.reset();
    this.ring.fill(0);
    this.prevSpec.fill(0);
    this.written = 0;
    this.nextRef = 0;
    this.frames = 0;
    this.resPos = 0;
    this.lastIn = 0;
    this.activation = 0;
    this.frameTime = 0;
  }

  /**
   * Push mono samples at the input rate. For every completed 100 fps frame the ensemble runs and
   * onFrame(activation, frameTime) is called.
   */
  push(x: Float32Array, onFrame?: (act: number, t: number) => void): void {
    if (this.ratio === 1) {
      for (let i = 0; i < x.length; i++) this.sample(x[i], onFrame);
      return;
    }
    // Linear resampling to 44.1 kHz: output sample j sits at input position j * ratio.
    let p = this.resPos;
    let prev = this.lastIn;
    for (let i = 0; i < x.length; i++) {
      const cur = x[i];
      // positions in (i - 1, i] relative to this block
      while (p <= i) {
        const f = p - (i - 1);
        this.sample(prev + (cur - prev) * f, onFrame);
        p += this.ratio;
      }
      prev = cur;
    }
    this.resPos = p - x.length;
    this.lastIn = prev;
  }

  private sample(v: number, onFrame?: (act: number, t: number) => void): void {
    const R = this.ring.length;
    this.ring[this.written % R] = v;
    this.written++;
    // Frame with reference sample `ref` spans [ref - (N - 1) + la, ref + la].
    while (this.written > this.nextRef + this.la) {
      this.frame(this.nextRef);
      const act = this.run();
      this.activation = act;
      this.frameTime = (this.nextRef + 1) / this.model.header.sampleRate;
      onFrame?.(act, this.frameTime);
      this.nextRef += this.model.header.hop;
    }
  }

  private frame(ref: number): void {
    const h = this.model.header;
    const N = h.frameSize, R = this.ring.length;
    const start = ref - (N - 1) + this.la;
    for (let i = 0; i < N; i++) {
      const k = start + i;
      this.buf[i] = k >= 0 && k < this.written ? this.ring[k % R] * this.win[i] : 0;
    }
    this.fft.power(this.buf, this.pow);
    const B = h.bands;
    for (let b = 0; b < B; b++) {
      const { start: s0, w } = this.model.fb[b];
      let s = 0;
      for (let k = 0; k < w.length; k++) s += w[k] * Math.sqrt(this.pow[s0 + k]);
      const v = Math.log10(h.logMul * s + h.logAdd);
      this.feat[b] = v;
      this.feat[B + b] = this.frames >= h.diffFrames ? Math.max(0, v - this.prevSpec[b]) : 0;
      this.prevSpec[b] = v;
    }
    this.frames++;
  }

  private run(): number {
    let s = 0;
    for (const n of this.nets) s += n.step(this.feat);
    return s / this.nets.length;
  }
}

/** Fetch and parse the shipped weights (browser). */
export async function loadBeatRnn(url = 'models/beat-lstm.bin'): Promise<BeatRnnModel> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`beat RNN weights: HTTP ${r.status}`);
  return parseBeatRnn(await r.arrayBuffer());
}

/** Whether the live path uses the beat RNN by default (the live panel's "Neural beats" setting). */
export const DEFAULT_BEAT_RNN = true;

/** Scale of the activation handed to BeatTracker in place of the spectral-flux onset strength. */
const ONSET_GAIN = 4;

/**
 * The beat RNN as RealtimeAnalyzer's beat onset source. Push each capture block (before the
 * analyzer processes it); the analyzer then asks at(t) for every one of its frames, getting the
 * peak activation of the RNN frames that ended in (previous t, t]: nothing after t is used, so the
 * beat clock stays causal and at the sound's time.
 */
export class BeatRnnOnset {
  readonly rnn: BeatRnn;
  private readonly mono: Float32Array;
  private readonly qt: Float64Array;
  private readonly qa: Float32Array;
  private qh = 0;
  private qn = 0;
  private held = 0;
  /** Offset added to the RNN's own frame times (stream time of the first pushed sample). */
  private t0 = 0;
  /** t0: stream time of the first block pushed (the analyzer's streamTime when the source is attached). */
  constructor(model: BeatRnnModel, sampleRate: number, t0 = 0, nets?: number) {
    this.rnn = new BeatRnn(model, { sampleRate, nets });
    this.t0 = t0;
    this.mono = new Float32Array(8192);
    this.qt = new Float64Array(64);
    this.qa = new Float32Array(64);
  }
  reset(t0 = 0): void {
    this.rnn.reset();
    this.qh = this.qn = 0;
    this.held = 0;
    this.t0 = t0;
  }
  private readonly onFrame = (a: number, t: number) => {
    const cap = this.qt.length;
    if (this.qn === cap) {
      // Overflow (the analyzer stopped asking): fold the oldest into the held peak.
      this.held = Math.max(this.held, this.qa[this.qh]);
      this.qh = (this.qh + 1) % cap;
      this.qn--;
    }
    const i = (this.qh + this.qn) % cap;
    this.qt[i] = t + this.t0;
    this.qa[i] = a;
    this.qn++;
  };
  push(l: Float32Array, r: Float32Array): void {
    const n = Math.min(l.length, this.mono.length);
    for (let i = 0; i < n; i++) this.mono[i] = 0.5 * (l[i] + r[i]);
    this.rnn.push(this.mono.subarray(0, n), this.onFrame);
  }
  /** Onset strength for the analyzer frame at stream time t. */
  readonly at = (t: number): number => {
    let peak = this.held;
    this.held = 0;
    const cap = this.qt.length;
    while (this.qn > 0 && this.qt[this.qh] <= t + 1e-6) {
      peak = Math.max(peak, this.qa[this.qh]);
      this.qh = (this.qh + 1) % cap;
      this.qn--;
    }
    return ONSET_GAIN * peak;
  };
}

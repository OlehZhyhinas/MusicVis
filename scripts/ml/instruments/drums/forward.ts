// Plain synchronous TS forward pass for the causal drum-transcription student (2 causal conv1d
// layers with BatchNorm folded in, kernel 5, dilations 1/2, left-padding only; 1-layer GRU;
// linear + sigmoid head over 5 classes: kick, snare, hat, tom, cymbal). Streaming: one 512-sample
// hop's feature frame (160-dim, from features.ts's StreamFeatures) in, one activation frame out.
// No look-ahead here; the peak-picker (peakpick.ts) may add up to 2 frames on top.
//
// Weight layout: see export.py's header.json (conv0_w/b, conv1_w/b, gru_w*/b*, lin_w/b).

export interface DrumHeader {
  in_feat: number;
  conv_ch: [number, number];
  kernel: number;
  dilations: [number, number];
  gru_hidden: number;
  n_classes: number;
  classes: string[];
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
  total_bytes: number;
}

function sliceTensor(buf: Float32Array, header: DrumHeader, name: string): Float32Array {
  const t = header.tensors.find((x) => x.name === name)!;
  return buf.subarray(t.offset / 4, t.offset / 4 + t.count);
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const tanh = (x: number) => Math.tanh(x);

export class DrumStudentTS {
  private readonly h: DrumHeader;
  private readonly conv0w: Float32Array; // [C0, in_feat, K]
  private readonly conv0b: Float32Array; // [C0]
  private readonly conv1w: Float32Array; // [C1, C0, K]
  private readonly conv1b: Float32Array; // [C1]
  private readonly gruWih: Float32Array; // [3H, C1]
  private readonly gruWhh: Float32Array; // [3H, H]
  private readonly gruBih: Float32Array; // [3H]
  private readonly gruBhh: Float32Array; // [3H]
  private readonly linW: Float32Array; // [5, H]
  private readonly linB: Float32Array; // [5]

  private readonly featRing: Float32Array; // ring of last K frames of input (K = kernel), [K, in_feat]
  private readonly conv0Ring: Float32Array; // ring of last (K-1)*dil1+1 frames of conv0 out, [span1, C0]
  private hidden: Float32Array; // GRU hidden state [H]
  private filled = 0;

  readonly out = new Float32Array(0);
  readonly activation: Float32Array;

  constructor(buf: Float32Array, header: DrumHeader) {
    this.h = header;
    this.conv0w = sliceTensor(buf, header, 'conv0_w');
    this.conv0b = sliceTensor(buf, header, 'conv0_b');
    this.conv1w = sliceTensor(buf, header, 'conv1_w');
    this.conv1b = sliceTensor(buf, header, 'conv1_b');
    this.gruWih = sliceTensor(buf, header, 'gru_wih');
    this.gruWhh = sliceTensor(buf, header, 'gru_whh');
    this.gruBih = sliceTensor(buf, header, 'gru_bih');
    this.gruBhh = sliceTensor(buf, header, 'gru_bhh');
    this.linW = sliceTensor(buf, header, 'lin_w');
    this.linB = sliceTensor(buf, header, 'lin_b');

    const [C0, C1] = header.conv_ch;
    const K = header.kernel;
    const [d0, d1] = header.dilations;
    this.featRing = new Float32Array(K * header.in_feat);
    const span1 = (K - 1) * d1 + 1;
    this.conv0Ring = new Float32Array(span1 * C0);
    this.hidden = new Float32Array(header.gru_hidden);
    this.activation = new Float32Array(header.n_classes);
    (this as any)._span1 = span1;
    (this as any)._C0 = C0;
    (this as any)._C1 = C1;
    (this as any)._K = K;
    (this as any)._d0 = d0;
    (this as any)._d1 = d1;
  }

  /** Reset streaming state (call at the start of a new stream / song). */
  reset(): void {
    this.featRing.fill(0);
    this.conv0Ring.fill(0);
    this.hidden.fill(0);
    this.filled = 0;
  }

  /** Push one 160-dim feature frame; returns the 5 sigmoid activations (kick/snare/hat/tom/cymbal), reused buffer. */
  push(feat: Float32Array): Float32Array {
    const h = this.h;
    const inFeat = h.in_feat;
    const K = (this as any)._K as number;
    const C0 = (this as any)._C0 as number;
    const C1 = (this as any)._C1 as number;
    const d1 = (this as any)._d1 as number;
    const span1 = (this as any)._span1 as number;

    // shift feature ring (dilation of conv0 is 1, so taps are the last K raw frames, consecutive)
    this.featRing.copyWithin(0, inFeat);
    this.featRing.set(feat, (K - 1) * inFeat);

    // conv0: causal, dilation 1, taps = featRing[0..K-1] (oldest..newest)
    const conv0Out = new Float32Array(C0);
    for (let c = 0; c < C0; c++) {
      let s = this.conv0b[c];
      const wBase = c * inFeat * K;
      for (let k = 0; k < K; k++) {
        const fBase = k * inFeat;
        const wBase2 = wBase + k; // weight layout [C0, inFeat, K] -> index c*inFeat*K + f*K + k
        for (let f = 0; f < inFeat; f++) s += this.featRing[fBase + f] * this.conv0w[c * inFeat * K + f * K + k];
      }
      conv0Out[c] = Math.max(0, s); // ReLU
    }

    // shift conv0 output ring (span1 slots), push newest at the end
    this.conv0Ring.copyWithin(0, C0);
    this.conv0Ring.set(conv0Out, (span1 - 1) * C0);

    // conv1: causal, dilation d1, taps at ring positions (span1-1) - (K-1-k)*d1 for k=0..K-1
    const conv1Out = new Float32Array(C1);
    for (let c = 0; c < C1; c++) {
      let s = this.conv1b[c];
      for (let k = 0; k < K; k++) {
        const pos = (span1 - 1) - (K - 1 - k) * d1;
        const fBase = pos * C0;
        for (let f = 0; f < C0; f++) s += this.conv0Ring[fBase + f] * this.conv1w[c * C0 * K + f * K + k];
      }
      conv1Out[c] = Math.max(0, s);
    }

    // GRU cell (PyTorch gate order r, z, n)
    const H = h.gru_hidden;
    const hPrev = this.hidden;
    const hNext = new Float32Array(H);
    for (let i = 0; i < H; i++) {
      let ir = this.gruBih[i], iz = this.gruBih[H + i], inn = this.gruBih[2 * H + i];
      let hr = this.gruBhh[i], hz = this.gruBhh[H + i], hn = this.gruBhh[2 * H + i];
      for (let f = 0; f < C1; f++) {
        const x = conv1Out[f];
        ir += this.gruWih[i * C1 + f] * x;
        iz += this.gruWih[(H + i) * C1 + f] * x;
        inn += this.gruWih[(2 * H + i) * C1 + f] * x;
      }
      for (let j = 0; j < H; j++) {
        const hp = hPrev[j];
        hr += this.gruWhh[i * H + j] * hp;
        hz += this.gruWhh[(H + i) * H + j] * hp;
        hn += this.gruWhh[(2 * H + i) * H + j] * hp;
      }
      const r = sigmoid(ir + hr);
      const z = sigmoid(iz + hz);
      const n = tanh(inn + r * hn);
      hNext[i] = (1 - z) * n + z * hPrev[i];
    }
    this.hidden = hNext;

    // linear + sigmoid
    for (let c = 0; c < h.n_classes; c++) {
      let s = this.linB[c];
      for (let j = 0; j < H; j++) s += this.linW[c * H + j] * hNext[j];
      this.activation[c] = sigmoid(s);
    }
    this.filled++;
    return this.activation;
  }
}

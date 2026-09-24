// Real-time analysis of a live audio stream (microphone / audio interface /
// virtual loopback device). Pure code: no DOM or Web Audio APIs, so it runs
// on the main thread in the browser and in Node tests alike.
//
// Blocks of stereo PCM go in (any block size); per ~11.6 ms hop the analyzer
// updates the same kinds of features the offline pipeline produces:
// - stem envelopes / onsets (streaming version of stems.ts: causal
//   harmonic/percussive split + mid/side heuristics),
// - absolute stem presence and complexity with the calibration constants of
//   complexity.ts,
// - onset strength -> BeatTracker (tempo, PLL beat clock, downbeats),
// - chroma -> KeyTracker,
// - ~10 Hz summary -> StructureTracker (sections, builds, drops).

import { GrooveTracker } from './groove';
import type { StemName } from '../types';
import { STEM_NAMES } from '../types';
import { RealFFT } from './fft';
import { clamp01, hann } from './dsp';
import { stemBands, logBands, bandPower, type BandLayout } from './stft';
import { TIMBRE_BANDS } from './stems';
import {
  SILENCE_DB,
  GATE_DB,
  SHARE_LO,
  SHARE_HI,
  DRUM_HF_LO,
  DRUM_HF_HI,
  DRUM_FLAT_LO,
  DRUM_FLAT_HI,
  expand,
} from './complexity';
import { BoxAvg, History, StreamDecimator, medianInPlace } from './rtUtil';
import { BeatTracker } from './rtBeat';
import { KeyTracker } from './rtKey';
import { HarmonyTracker } from './harmony';
import { StructureTracker, STRUCT_RATE } from './rtStructure';

const TINY = 1e-20;
const RING = 4096; // decimated samples kept (power of two)
const CHROMA_EVERY = 4; // frames per chroma frame (as offline: chroma hop = 4 * hop)
const SONG_CX_MEMORY_S = 60;

function toDb(p: number): number {
  return 10 * Math.log10(p + TINY);
}

function aWeightPower(f: number): number {
  const f2 = f * f;
  const ra =
    (12194 * 12194 * f2 * f2) /
    ((f2 + 20.6 * 20.6) * Math.sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9)) * (f2 + 12194 * 12194));
  return ra * ra * 1.585;
}

function pow2Near(x: number): number {
  return Math.pow(2, Math.max(5, Math.round(Math.log2(Math.max(32, x)))));
}

function oddWidth(x: number, min = 3): number {
  let w = Math.max(min, Math.round(x));
  if (w % 2 === 0) w++;
  return w;
}

const zeroStems = (): Record<StemName, number> => ({ drums: 0, bass: 0, vocals: 0, other: 0 });

/** Causal onset picker on an absolute flux (streaming version of complexity.ts pickOnsets). */
class OnsetPicker {
  private readonly flux: History;
  private readonly base: History;
  private readonly gate: History;
  private readonly baseAvg: BoxAvg;
  private readonly h: number;
  private readonly minGap: number;
  private readonly delta: number;
  private sinceLast = 1e9;
  readonly rate: BoxAvg;
  private readonly fr: number;

  constructor(fr: number, delta: number) {
    this.fr = fr;
    this.delta = delta;
    this.h = Math.max(1, Math.round(0.03 * fr));
    this.minGap = Math.max(1, Math.round(0.06 * fr));
    this.flux = new History(2 * this.h + 2);
    this.base = new History(2 * this.h + 2);
    this.gate = new History(2 * this.h + 2);
    this.baseAvg = new BoxAvg(oddWidth(0.5 * fr));
    this.rate = new BoxAvg(oddWidth(2 * fr));
  }

  /** Returns onsets per second over the last ~2 s. */
  push(flux: number, gate: number): number {
    this.flux.push(flux);
    this.base.push(this.baseAvg.push(flux));
    this.gate.push(gate);
    this.sinceLast++;
    let onset = 0;
    const h = this.h;
    if (this.flux.count > 2 * h) {
      const v = this.flux.at(h);
      if (this.gate.at(h) >= 0.5 && v - this.base.at(h) >= this.delta) {
        let isMax = true;
        for (let a = 0; a <= 2 * h; a++) {
          if (a === h) continue;
          const u = this.flux.at(a);
          // Ties go to the earlier frame (larger age).
          if (u > v || (u === v && a > h)) {
            isMax = false;
            break;
          }
        }
        if (isMax && this.sinceLast - h >= this.minGap) {
          onset = 1;
          this.sinceLast = h;
        }
      }
    }
    return this.rate.push(onset) * this.fr;
  }
}

/** Running 0..1 normalization of a dB envelope against its own recent peak (stems.ts envelopeFromPower, causal). */
class DbEnvelope {
  private peak = -Infinity;
  private readonly sm: BoxAvg;
  private readonly decay: number; // dB per frame
  constructor(fr: number) {
    this.sm = new BoxAvg(oddWidth(0.03 * fr, 1));
    this.decay = 1.5 / fr; // peak memory falls 1.5 dB/s
  }
  push(p: number, top: number, floorRel: number, ceilRel: number): number {
    const db = this.sm.push(toDb(p));
    this.peak = Math.max(db, this.peak - this.decay);
    let hi = Math.max(this.peak - 2, top + ceilRel);
    const lo = Math.max(top + floorRel, hi - 36);
    if (hi < lo + 10) hi = lo + 10;
    return clamp01((db - lo) / (hi - lo));
  }
}

/** Running 0..1 normalization of an onset-strength signal (stems.ts normalizeOnset, causal). */
class OnsetNorm {
  private mean = 0;
  private peak = 0;
  private readonly kMean: number;
  private readonly kPeak: number;
  constructor(fr: number) {
    this.kMean = 1 - Math.exp(-1 / (3 * fr));
    this.kPeak = Math.exp(-1 / (8 * fr));
  }
  push(x: number): number {
    this.mean += (x - this.mean) * this.kMean;
    this.peak = Math.max(x, this.peak * this.kPeak);
    const lo = this.mean;
    const hi = Math.max(this.peak, lo + 4);
    return clamp01((x - lo) / (hi - lo));
  }
}

export class RealtimeAnalyzer {
  readonly inputRate: number;
  readonly factor: number;
  readonly sr: number; // analysis rate after decimation
  readonly hop: number;
  readonly n: number; // main FFT size
  readonly frameRate: number;

  // ---- Latest features (read by RealtimeSampler) ----
  /** Stream time (s) of the newest input sample. */
  streamTime = 0;
  /** Stream time (s) of the newest analysis frame's onset window center. */
  frameTime = 0;
  frames = 0;
  readonly stems = zeroStems();
  readonly stemOnsets = zeroStems();
  readonly stemPresence = zeroStems();
  loudness = 0;
  /** Complexity (forward asymmetric smoothing of the absolute measure), 0..1. */
  complexity = 0;
  /** Faster complexity used for drop detection. */
  complexityFast = 0;
  songComplexity = 0.5;
  /** Seconds of non-silent input since the last reset / new song. */
  musicSeconds = 0;
  readonly chroma = new Float32Array(12);
  /** Mix level of the analysis frame, dBFS. */
  dbfs = -120;
  /** 0..1 silence gate (0 = silent). */
  gate = 0;
  /** Input level for a meter: smoothed peak dBFS of the raw input. */
  levelDb = -120;
  /** Onset strength of the latest frame (normalized flux). */
  onset = 0;

  readonly beat: BeatTracker;
  /** Running timing-feel estimate against the beat clock (groove.ts). */
  readonly groove = new GrooveTracker();
  readonly key: KeyTracker;
  /** Realtime-lite chord tracking (harmony map). */
  readonly harmony = new HarmonyTracker();
  readonly structure: StructureTracker;

  // ---- internals ----
  private readonly decL: StreamDecimator;
  private readonly decR: StreamDecimator;
  private readonly ringM = new Float32Array(RING);
  private readonly ringS = new Float32Array(RING);
  private ringPos = 0;
  private decCount = 0;
  private stereo = false;

  private readonly fft: RealFFT;
  private readonly fftOn: RealFFT;
  private readonly fftCh: RealFFT;
  private readonly win: Float32Array;
  private readonly winOn: Float32Array;
  private readonly winCh: Float32Array;
  private readonly buf: Float64Array;
  private readonly bufOn: Float64Array;
  private readonly bufCh: Float64Array;
  private readonly pow: Float64Array;
  private readonly powOn: Float64Array;
  private readonly powCh: Float64Array;
  private readonly bands: BandLayout;
  private readonly onBands: BandLayout;
  private readonly midB: Float32Array;
  private readonly sideB: Float32Array;
  private readonly onDb: Float32Array;
  private readonly onPrev: Float32Array;
  private onTop = -Infinity;
  private onHave = false;
  private readonly onLocal: BoxAvg;
  private readonly onSq: BoxAvg;
  private onGlobalSq = 0;
  private onGlobalN = 0;

  // stems
  private readonly B: number;
  private readonly wT: number;
  private readonly xHist: Float32Array; // wT * B magnitudes
  private xHistPos = 0;
  private xHistCount = 0;
  private readonly medScratch: Float64Array;
  private readonly X: Float32Array;
  private readonly H: Float32Array;
  private readonly P: Float32Array;
  private readonly smM: Float32Array;
  private readonly smS: Float32Array;
  private readonly kRatio: number;
  private readonly center: Float32Array;
  private readonly prevPerc: Float32Array;
  private readonly prevHarm: Float32Array;
  private readonly prevCenter: Float32Array;
  private readonly prevOther: Float32Array;
  private readonly prevFull: Float32Array;
  private readonly wA: Float32Array;
  private readonly wHalf: Float32Array;
  private readonly timbreIdx: Int32Array;
  private readonly counts: { drum: number; bass: number; voc: number; other: number; kick: number; snare: number; flat: number; pf: number };
  private readonly timbreAcc = new Float64Array(TIMBRE_BANDS);
  readonly timbre = new Float32Array(TIMBRE_BANDS);
  private topDb = -Infinity;
  private readonly totSm: BoxAvg;
  private readonly env: Record<StemName, DbEnvelope>;
  private readonly loudEnv: DbEnvelope;
  private loudTop = -Infinity;
  private readonly onNorm: Record<StemName, OnsetNorm>;
  private readonly kickNorm: OnsetNorm;
  private readonly snareNorm: OnsetNorm;

  // complexity
  private readonly ref: number;
  private readonly sp: Record<StemName, BoxAvg>;
  private readonly totB: BoxAvg;
  private readonly pHighL: BoxAvg;
  private readonly pDrL: BoxAvg;
  private readonly drumsGroove: BoxAvg;
  private readonly restGroove: BoxAvg;
  private readonly hfFlatNum: BoxAvg;
  private readonly presSm: Record<StemName, BoxAvg>;
  private readonly flatB: BoxAvg;
  private readonly chromaBox: BoxAvg[];
  private readonly fluxPick: OnsetPicker;
  private readonly percPick: OnsetPicker;
  private onsetRate = 0;
  private readonly kickLow: BoxAvg;
  private readonly kickTot: BoxAvg;
  /** Share of the mix that is low percussive energy (kick drum) over ~0.5 s, 0..1. */
  kickShare = 0;
  private cxSongSum = 0;
  private cxSongN = 0;

  // chroma
  private readonly chK: number;
  private readonly chLo: number;
  private readonly pcA: Int32Array;
  private readonly pcB: Int32Array;
  private readonly cwA: Float32Array;
  private readonly cwB: Float32Array;
  private readonly chRaw = new Float64Array(12);
  private chGMax = 0;
  private chMaxE = 0;
  private chromaFresh = false;

  private structCountdown = 0;
  private lastNewSongs = 0;

  constructor(inputRate: number) {
    this.inputRate = inputRate > 0 ? inputRate : 44100;
    this.factor = Math.max(1, Math.round(this.inputRate / 22050));
    this.sr = this.inputRate / this.factor;
    this.hop = pow2Near(this.sr / 86);
    this.n = this.hop * 8;
    this.frameRate = this.sr / this.hop;
    const fr = this.frameRate;
    const n = this.n;

    this.decL = new StreamDecimator(this.factor);
    this.decR = new StreamDecimator(this.factor);

    this.fft = new RealFFT(n);
    this.fftOn = new RealFFT(n / 2);
    this.fftCh = new RealFFT(n * 2);
    this.win = hann(n);
    this.winOn = hann(n / 2);
    this.winCh = hann(n * 2);
    this.buf = new Float64Array(n);
    this.bufOn = new Float64Array(n / 2);
    this.bufCh = new Float64Array(n * 2);
    this.pow = new Float64Array(n / 2 + 1);
    this.powOn = new Float64Array(n / 4 + 1);
    this.powCh = new Float64Array(n + 1);

    this.bands = stemBands(n, this.sr);
    this.onBands = logBands(n / 2, this.sr, 30, Math.min(11000, this.sr * 0.48), 40);
    const B = this.bands.count;
    this.B = B;
    this.midB = new Float32Array(B);
    this.sideB = new Float32Array(B);
    this.onDb = new Float32Array(this.onBands.count);
    this.onPrev = new Float32Array(this.onBands.count);
    this.onLocal = new BoxAvg(oddWidth(0.5 * fr));
    this.onSq = new BoxAvg(oddWidth(4 * fr));

    this.wT = oddWidth(0.2 * fr, 5);
    this.xHist = new Float32Array(this.wT * B);
    this.medScratch = new Float64Array(Math.max(this.wT, 17) + 2);
    this.X = new Float32Array(B);
    this.H = new Float32Array(B);
    this.P = new Float32Array(B);
    this.smM = new Float32Array(B);
    this.smS = new Float32Array(B);
    this.kRatio = 1 - Math.exp(-1 / (0.05 * fr));
    this.center = new Float32Array(B);
    this.prevPerc = new Float32Array(B);
    this.prevHarm = new Float32Array(B);
    this.prevCenter = new Float32Array(B);
    this.prevOther = new Float32Array(B);
    this.prevFull = new Float32Array(B);
    this.wA = new Float32Array(B);
    this.wHalf = new Float32Array(B);
    this.timbreIdx = new Int32Array(B);
    const freq = this.bands.freq;
    const c = { drum: 0, bass: 0, voc: 0, other: 0, kick: 0, snare: 0, flat: 0, pf: 0 };
    for (let b = 0; b < B; b++) {
      const f = freq[b];
      this.wA[b] = aWeightPower(Math.max(f, 1));
      this.wHalf[b] = Math.sqrt(this.wA[b]);
      const lf = Math.log(Math.max(f, 30) / 30) / Math.log(11000 / 30);
      this.timbreIdx[b] = Math.min(TIMBRE_BANDS - 1, Math.max(0, Math.floor(lf * TIMBRE_BANDS)));
      if (f >= 40 && f <= 12000) c.drum++;
      if (f < 250) c.bass++;
      if (f >= 250 && f <= 4000) c.voc++;
      if (f >= 250) c.other++;
      if (f < 150) c.kick++;
      if (f >= 150 && f <= 5000) c.snare++;
      if (f >= 100 && f <= 8000) c.flat++;
      if (f >= 2000 && f <= 10000) c.pf++;
    }
    this.counts = c;
    this.totSm = new BoxAvg(oddWidth(0.1 * fr));
    this.env = { drums: new DbEnvelope(fr), bass: new DbEnvelope(fr), vocals: new DbEnvelope(fr), other: new DbEnvelope(fr) };
    this.loudEnv = new DbEnvelope(fr);
    this.onNorm = { drums: new OnsetNorm(fr), bass: new OnsetNorm(fr), vocals: new OnsetNorm(fr), other: new OnsetNorm(fr) };
    this.kickNorm = new OnsetNorm(fr);
    this.snareNorm = new OnsetNorm(fr);

    this.ref = 10 * Math.log10((3 * n * n) / 16);
    const wS = oddWidth(0.1 * fr);
    const wL = oddWidth(1.0 * fr);
    const wD = oddWidth(0.4 * fr);
    this.sp = { drums: new BoxAvg(wS), bass: new BoxAvg(wS), vocals: new BoxAvg(wS), other: new BoxAvg(wS) };
    this.totB = new BoxAvg(wS);
    this.pHighL = new BoxAvg(wL);
    this.pDrL = new BoxAvg(wL);
    this.drumsGroove = new BoxAvg(wD);
    this.restGroove = new BoxAvg(wD);
    this.hfFlatNum = new BoxAvg(wL);
    this.presSm = { drums: new BoxAvg(wS), bass: new BoxAvg(wS), vocals: new BoxAvg(wS), other: new BoxAvg(wS) };
    this.flatB = new BoxAvg(oddWidth(0.5 * fr));
    this.chromaBox = [];
    const wPoly = 2 * Math.max(1, Math.round(0.25 * fr)) + 1;
    for (let k = 0; k < 12; k++) this.chromaBox.push(new BoxAvg(wPoly));
    this.kickLow = new BoxAvg(oddWidth(0.5 * fr));
    this.kickTot = new BoxAvg(oddWidth(0.5 * fr));
    this.fluxPick = new OnsetPicker(fr, 1.2);
    this.percPick = new OnsetPicker(fr, 1.5);

    // Chroma bins (key.ts computeChroma): 55 Hz .. 5 kHz, two nearest pitch classes.
    const nCh = n * 2;
    const binHz = this.sr / nCh;
    const kLo = Math.max(1, Math.ceil(55 / binHz));
    const kHi = Math.min(nCh / 2, Math.floor(5000 / binHz));
    this.chLo = kLo;
    this.chK = Math.max(1, kHi - kLo + 1);
    this.pcA = new Int32Array(this.chK);
    this.pcB = new Int32Array(this.chK);
    this.cwA = new Float32Array(this.chK);
    this.cwB = new Float32Array(this.chK);
    for (let k = 0; k < this.chK; k++) {
      const f = (kLo + k) * binHz;
      const midi = 69 + 12 * Math.log2(f / 440);
      const lo = Math.floor(midi);
      const frac = midi - lo;
      this.pcA[k] = ((lo % 12) + 12) % 12;
      this.pcB[k] = (this.pcA[k] + 1) % 12;
      const ca = Math.cos((Math.PI / 2) * Math.min(1, frac * 2));
      const cb = Math.cos((Math.PI / 2) * Math.min(1, (1 - frac) * 2));
      this.cwA[k] = ca * ca;
      this.cwB[k] = cb * cb;
    }

    this.beat = new BeatTracker(fr);
    this.key = new KeyTracker();
    this.structure = new StructureTracker();
  }

  /** Forget everything (e.g. when switching input devices). */
  reset(): void {
    // Cheapest correct reset: rebuild via a fresh instance's state.
    const fresh = new RealtimeAnalyzer(this.inputRate);
    Object.assign(this, fresh);
  }

  /** Bar length in seconds from the current tempo. */
  get barSeconds(): number {
    return 4 * this.beat.period;
  }

  /**
   * Feed one block of input samples (any length). `right` may be the same
   * array as `left` (mono).
   */
  process(left: Float32Array, right: Float32Array): void {
    const len = Math.min(left.length, right.length);
    if (len <= 0) return;
    // Meter: block peak, fast attack / ~300 ms release.
    let peak = 0;
    for (let i = 0; i < len; i++) {
      const a = Math.abs(left[i]);
      const b = Math.abs(right[i]);
      const m = a > b ? a : b;
      if (m > peak) peak = m;
    }
    const pDb = peak > 1e-7 ? 20 * Math.log10(peak) : -120;
    const blockS = len / this.inputRate;
    this.levelDb = pDb > this.levelDb ? pDb : this.levelDb + (pDb - this.levelDb) * (1 - Math.exp(-blockS / 0.3));
    if (!Number.isFinite(this.levelDb)) this.levelDb = -120;

    const mono = left === right;
    const lIn = len === left.length ? left : left.subarray(0, len);
    const rIn = len === right.length ? right : right.subarray(0, len);
    if (mono) {
      this.decL.process(lIn, (v) => this.pushSample(v, v));
    } else {
      // Decimate both channels in lockstep: left first, then pair with right.
      this.decL.process(lIn, (v) => this.queueLeft(v));
      this.decR.process(rIn, (v) => this.pairRight(v));
    }
    this.streamTime += blockS;
  }

  private leftQueue: number[] = [];
  private queueLeft(v: number): void {
    this.leftQueue.push(v);
  }
  private pairRight(r: number): void {
    const l = this.leftQueue.length > 0 ? this.leftQueue.shift()! : r;
    if (!this.stereo && Math.abs(l - r) > 1e-6) this.stereo = true;
    this.pushSample(l, r);
  }

  private pushSample(l: number, r: number): void {
    const p = this.ringPos;
    this.ringM[p] = 0.5 * (l + r);
    this.ringS[p] = 0.5 * (l - r);
    this.ringPos = (p + 1) & (RING - 1);
    this.decCount++;
    if (this.decCount % this.hop === 0) this.frame();
  }

  /** Copy the last `len` decimated samples of a ring, windowed, into `out`. */
  private windowed(ring: Float32Array, len: number, win: Float32Array, out: Float64Array): void {
    const start = (this.ringPos - len + RING) & (RING - 1);
    for (let i = 0; i < len; i++) out[i] = ring[(start + i) & (RING - 1)] * win[i];
  }

  private frame(): void {
    const fr = this.frameRate;
    const n = this.n;
    this.frames++;
    // Onset window center (n/4 samples back), in stream time.
    this.frameTime = (this.decCount - n / 4) / this.sr;

    // ---------------- spectra ----------------
    this.windowed(this.ringM, n, this.win, this.buf);
    this.fft.power(this.buf, this.pow);
    bandPower(this.pow, this.bands, this.midB, 0);
    if (this.stereo) {
      this.windowed(this.ringS, n, this.win, this.buf);
      this.fft.power(this.buf, this.pow);
      bandPower(this.pow, this.bands, this.sideB, 0);
    } else this.sideB.fill(0);

    // ---------------- onset strength (beats.ts onsetEnvelope, causal) ----------------
    {
      this.windowed(this.ringM, n / 2, this.winOn, this.bufOn);
      this.fftOn.power(this.bufOn, this.powOn);
      const ob = this.onBands;
      const OB = ob.count;
      let top = -Infinity;
      for (let b = 0; b < OB; b++) {
        let s = 0;
        for (let k = ob.lo[b]; k < ob.hi[b]; k++) s += this.powOn[k];
        const d = 10 * Math.log10(s + 1e-20);
        this.onDb[b] = d;
        if (d > top) top = d;
      }
      this.onTop = Math.max(top, this.onTop - 0.5 / fr);
      const floor = this.onTop - 80;
      for (let b = 0; b < OB; b++) if (this.onDb[b] < floor) this.onDb[b] = floor;
      let raw = 0;
      if (this.onHave) {
        let acc = 0;
        for (let b = 0; b < OB; b++) {
          let ref = Math.max(this.onPrev[b], floor);
          if (b > 0 && this.onPrev[b - 1] > ref) ref = this.onPrev[b - 1];
          if (b + 1 < OB && this.onPrev[b + 1] > ref) ref = this.onPrev[b + 1];
          const d = this.onDb[b] - ref;
          if (d > 0) acc += d;
        }
        raw = acc / OB;
      }
      this.onPrev.set(this.onDb);
      this.onHave = true;
      const local = this.onLocal.push(raw);
      const d = raw > local ? raw - local : 0;
      const lrms = Math.sqrt(this.onSq.push(d * d));
      this.onGlobalN = Math.min(this.onGlobalN + 1, 30 * fr);
      this.onGlobalSq += (d * d - this.onGlobalSq) / this.onGlobalN;
      const grms = Math.sqrt(this.onGlobalSq);
      this.onset = grms > 1e-9 ? d / (lrms + 0.35 * grms) : 0;
    }

    // ---------------- stems (stems.ts, causal) ----------------
    const accents = this.stemFrame();

    // ---------------- chroma ----------------
    this.chromaFresh = false;
    if (this.frames % CHROMA_EVERY === 0 && this.decCount >= n) this.chromaFrame();

    // ---------------- complexity ----------------
    this.complexityFrame();

    // ---------------- beats ----------------
    const active = this.gate > 0.5;
    this.beat.push(this.onset, this.frameTime, active, accents, this.chroma);
    {
      const pos = this.beat.positionAt(this.frameTime);
      const fl = Math.floor(pos);
      const inBar = (((fl - this.beat.downbeatSlot) % 4) + 4) % 4;
      this.groove.push(Math.max(this.stemOnsets.drums, 0.6 * this.stemOnsets.bass, 0.5 * this.stemOnsets.other) * Math.pow(10, this.dbfs / 40), pos - fl, this.beat.period, inBar, 1 / this.frameRate, active ? this.beat.confidence : 0);
    }

    // ---------------- key ----------------
    if (this.chromaFresh) this.key.push(this.chroma, this.chromaWeight, CHROMA_EVERY / fr, this.frameTime);
    if (this.chromaFresh) {
      this.harmony.setKey(this.key.tonic, this.key.mode, this.key.key >= 0);
      this.harmony.push(this.chroma, this.chromaWeight, CHROMA_EVERY / fr);
    }

    // ---------------- structure (~10 Hz) ----------------
    if (--this.structCountdown <= 0) {
      this.structCountdown = Math.max(1, Math.round(fr / STRUCT_RATE));
      this.structure.update(
        {
          t: this.frameTime,
          db: this.dbfs,
          cx: this.complexityFast,
          pd: this.stemPresence.drums,
          pb: this.stemPresence.bass,
          kick: this.kickShare,
          rate: this.onsetRate,
          gate: this.gate,
          timbre: this.timbre,
        },
        this.barSeconds,
      );
      if (this.structure.newSongs !== this.lastNewSongs) {
        this.lastNewSongs = this.structure.newSongs;
        this.cxSongSum = 0;
        this.cxSongN = 0;
        this.musicSeconds = 0;
        this.key.reset();
        this.harmony.reset();
      }
    }
  }

  private chromaWeight = 0;

  private chromaFrame(): void {
    const n2 = this.n * 2;
    this.windowed(this.ringM, n2, this.winCh, this.bufCh);
    this.fftCh.power(this.bufCh, this.powCh);
    const raw = this.chRaw;
    raw.fill(0);
    let e = 0;
    let rMax = 0;
    for (let k = 0; k < this.chK; k++) {
      const p = this.powCh[this.chLo + k];
      raw[this.pcA[k]] += p * this.cwA[k];
      raw[this.pcB[k]] += p * this.cwB[k];
      e += p;
    }
    for (let k = 0; k < 12; k++) if (raw[k] > rMax) rMax = raw[k];
    // Running references (slow decay) stand in for the offline song-wide maxima.
    const decay = Math.exp(-CHROMA_EVERY / (this.frameRate * 30));
    this.chGMax = Math.max(rMax, this.chGMax * decay);
    this.chMaxE = Math.max(e, this.chMaxE * decay);
    this.chromaFresh = true;
    if (!(this.chMaxE > 0) || !(this.chGMax > 0) || this.gate <= 0) {
      this.chromaWeight = 0;
      return;
    }
    const db = 10 * Math.log10(e / this.chMaxE + 1e-12);
    const wgt = clamp01((db + 60) / 60);
    this.chromaWeight = wgt * this.gate;
    let m = 0;
    for (let k = 0; k < 12; k++) {
      const v = Math.log1p((1000 * raw[k]) / this.chGMax);
      this.chroma[k] = v;
      if (v > m) m = v;
    }
    if (m > 0) for (let k = 0; k < 12; k++) this.chroma[k] /= m;
  }

  // Raw per-frame stem features consumed by complexityFrame().
  private rw = { drums: 0, bass: 0, vocals: 0, other: 0, total: 0, percHigh: 0, percHiFlat: 0, flux: 0, percFlux: 0, flatness: 0, lowPerc: 0 };

  private stemFrame(): { kick: number; bass: number; snare: number } {
    const B = this.B;
    const freq = this.bands.freq;
    const fr = this.frameRate;
    const X = this.X;
    const H = this.H;
    const P = this.P;
    const c = this.counts;
    const inv = (k: number) => (k > 0 ? 1 / k : 0);

    let total = 0;
    for (let b = 0; b < B; b++) total += this.midB[b];
    const totalDb = toDb(this.totSm.push(total));
    if (total > 0) this.topDb = Number.isFinite(this.topDb) ? Math.max(totalDb, this.topDb - 0.1 / fr) : totalDb;
    const top = Number.isFinite(this.topDb) ? this.topDb : -150;
    const floorDb = top - 10 * Math.log10(B) - 75;
    if (this.frames === 1) {
      this.prevPerc.fill(floorDb);
      this.prevHarm.fill(floorDb);
      this.prevCenter.fill(floorDb);
      this.prevOther.fill(floorDb);
      this.prevFull.fill(floorDb);
    }

    // Magnitudes + causal temporal median (harmonic) + side/mid ratio.
    const o = this.xHistPos * B;
    for (let b = 0; b < B; b++) {
      const xm = Math.sqrt(this.midB[b]);
      X[b] = xm;
      this.xHist[o + b] = xm;
      this.smM[b] += (this.midB[b] - this.smM[b]) * this.kRatio;
      this.smS[b] += (this.sideB[b] - this.smS[b]) * this.kRatio;
    }
    this.xHistPos = (this.xHistPos + 1) % this.wT;
    this.xHistCount = Math.min(this.wT, this.xHistCount + 1);
    const cnt = this.xHistCount;
    const ms = this.medScratch;
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < cnt; i++) ms[i] = this.xHist[i * B + b];
      H[b] = medianInPlace(ms, cnt);
    }
    // Vertical (frequency) median, width 17, mirrored edges -> percussive-enhanced.
    {
      const wF = 17;
      const h = wF >> 1;
      for (let b = 0; b < B; b++) {
        let k = 0;
        for (let j = b - h; j <= b + h; j++) {
          let jj = j;
          if (jj < 0) jj = -jj;
          if (jj >= B) jj = 2 * (B - 1) - jj;
          ms[k++] = X[Math.max(0, Math.min(B - 1, jj))];
        }
        P[b] = medianInPlace(ms, k);
      }
    }

    let dr = 0,
      ba = 0,
      vo = 0,
      ot = 0,
      lo = 0;
    let fDr = 0,
      fBa = 0,
      fVo = 0,
      fOt = 0,
      fK = 0,
      fS = 0;
    this.timbreAcc.fill(0);
    let linSum = 0;
    let harmSum = 0;
    let fFull = 0;
    let pHi = 0;
    let flLog = 0;
    let flLin = 0;
    let pfLog = 0;
    let pfLin = 0;
    let wDr = 0,
      wBa = 0,
      wOt = 0,
      wVo = 0;
    let lowPerc = 0;
    const center = this.center;
    for (let b = 0; b < B; b++) {
      const xm = X[b];
      const x2 = xm * xm;
      const h = H[b];
      const p = P[b];
      const h2 = h * h;
      const p2 = p * p;
      const mh = h2 + p2 > 0 ? h2 / (h2 + p2) : 0.5;
      const harm = mh * x2;
      const perc = x2 - harm;
      const f = freq[b];
      lo += x2 * this.wA[b];
      this.timbreAcc[this.timbreIdx[b]] += x2;
      if (f >= 40 && f <= 12000) {
        const xDb = Math.max(floorDb, toDb(x2));
        const dx = xDb - this.prevFull[b];
        this.prevFull[b] = xDb;
        if (dx > 0) fFull += dx;
        if (f >= 2000) pHi += perc;
        if (f >= 100 && f <= 8000) {
          flLog += xDb;
          flLin += x2;
        }
      }
      const percDb = Math.max(floorDb, toDb(perc));
      const dPerc = percDb - this.prevPerc[b];
      this.prevPerc[b] = percDb;
      if (f >= 40 && f <= 12000) {
        dr += perc;
        wDr += perc * this.wHalf[b];
        if (f >= 2000 && f <= 10000) {
          pfLog += percDb;
          pfLin += perc;
        }
        if (dPerc > 0) fDr += dPerc;
      }
      if (f < 150) {
        lowPerc += perc;
        if (dPerc > 0) fK += dPerc;
      } else if (f <= 5000) {
        if (dPerc > 0) fS += dPerc;
      }
      if (f < 250) {
        ba += harm;
        wBa += harm * this.wHalf[b];
        const hDb = Math.max(floorDb, toDb(harm));
        const d = hDb - this.prevHarm[b];
        this.prevHarm[b] = hDb;
        if (d > 0) fBa += d;
        center[b] = 0;
      } else {
        const pk = x2 / (x2 + 12 * p2 + TINY);
        const ratio = this.smM[b] > 0 ? this.smS[b] / this.smM[b] : 1;
        const cw = 1 - 1.3 * ratio;
        let cc = cw > 0 ? harm * cw * pk : 0;
        if (f > 4000) cc = 0;
        center[b] = cc;
        if (f <= 4000) {
          linSum += cc;
          harmSum += harm;
        }
        const oth = harm - cc;
        ot += oth;
        wOt += oth * this.wHalf[b];
        const oDb = Math.max(floorDb, toDb(oth));
        const d = oDb - this.prevOther[b];
        this.prevOther[b] = oDb;
        if (d > 0) fOt += d;
      }
    }
    let tonal = 0;
    if (c.voc > 0 && linSum > 0 && harmSum > 0) {
      const fl = (1e-4 * harmSum) / c.voc + TINY;
      let logSum = 0;
      for (let b = 0; b < B; b++) {
        const f = freq[b];
        if (f >= 250 && f <= 4000) {
          const x2 = X[b] * X[b];
          const h = H[b];
          const p = P[b];
          const mh = h * h + p * p > 0 ? (h * h) / (h * h + p * p) : 0.5;
          logSum += Math.log(mh * x2 + fl);
        }
      }
      const flat = Math.exp(logSum / c.voc) / (harmSum / c.voc);
      tonal = clamp01((0.55 - flat) / 0.45);
    }
    for (let b = 0; b < B; b++) {
      const f = freq[b];
      if (f < 250 || f > 4000) continue;
      const cc = center[b] * tonal;
      vo += cc;
      wVo += cc * this.wHalf[b];
      wOt += (1 - tonal) * center[b] * this.wHalf[b];
      const cDb = Math.max(floorDb, toDb(cc));
      const d = cDb - this.prevCenter[b];
      this.prevCenter[b] = cDb;
      if (d > 0) fVo += d;
    }
    ot += (1 - tonal) * linSum;

    for (let k = 0; k < TIMBRE_BANDS; k++) this.timbre[k] = Math.max(floorDb, toDb(this.timbreAcc[k]));

    // Normalized envelopes / onsets (running references instead of song percentiles).
    const silent = !(top > -150);
    this.stems.drums = silent ? 0 : this.env.drums.push(dr, top, -60, -30);
    this.stems.bass = silent ? 0 : this.env.bass.push(ba, top, -60, -30);
    this.stems.vocals = silent ? 0 : this.env.vocals.push(vo, top, -60, -30);
    this.stems.other = silent ? 0 : this.env.other.push(ot, top, -60, -30);
    const lDb = toDb(lo);
    if (lo > 0) this.loudTop = Number.isFinite(this.loudTop) ? Math.max(lDb, this.loudTop - 0.1 / fr) : lDb;
    this.loudness = silent || !Number.isFinite(this.loudTop) ? 0 : this.loudEnv.push(lo, this.loudTop, -50, -12);
    this.stemOnsets.drums = this.onNorm.drums.push(fDr * inv(c.drum));
    this.stemOnsets.bass = this.onNorm.bass.push(fBa * inv(c.bass));
    this.stemOnsets.vocals = this.onNorm.vocals.push(fVo * inv(c.voc));
    this.stemOnsets.other = this.onNorm.other.push(fOt * inv(c.other));
    const kick = this.kickNorm.push(fK * inv(c.kick));
    const snare = this.snareNorm.push(fS * inv(c.snare));

    const rw = this.rw;
    rw.drums = wDr;
    rw.bass = wBa;
    rw.vocals = wVo;
    rw.other = wOt;
    rw.total = total;
    rw.percHigh = pHi;
    rw.percHiFlat = pfLin > 0 && c.pf > 0 ? clamp01(Math.pow(10, pfLog / c.pf / 10) / (pfLin / c.pf)) : 0;
    rw.flux = fFull * inv(c.drum);
    rw.percFlux = fDr * inv(c.drum);
    rw.lowPerc = lowPerc;
    rw.flatness = flLin > 0 && c.flat > 0 ? clamp01(Math.pow(10, flLog / c.flat / 10) / (flLin / c.flat)) : 0;
    return { kick, bass: this.stemOnsets.bass, snare };
  }

  private complexityFrame(): void {
    const fr = this.frameRate;
    const rw = this.rw;
    const sp = zeroStemsCache;
    sp.drums = this.sp.drums.push(rw.drums);
    sp.bass = this.sp.bass.push(rw.bass);
    sp.vocals = this.sp.vocals.push(rw.vocals);
    sp.other = this.sp.other.push(rw.other);
    const tot = this.totB.push(rw.total);
    const pHighL = this.pHighL.push(rw.percHigh);
    const pDrL = this.pDrL.push(rw.drums);
    const drumsGroove = this.drumsGroove.push(rw.drums);
    const restGroove = this.restGroove.push(rw.bass + rw.vocals + rw.other);
    const hfNum = this.hfFlatNum.push(rw.percHiFlat * rw.percHigh);
    const hfFlat = pHighL > TINY ? hfNum / pHighL : 0;

    const kl = this.kickLow.push(rw.lowPerc);
    const kt = this.kickTot.push(rw.total);
    this.kickShare = kt > TINY ? clamp01(kl / kt) : 0;
    const d = 10 * Math.log10(tot + TINY) - this.ref;
    this.dbfs = d;
    const gate = clamp01((d - SILENCE_DB) / GATE_DB);
    this.gate = gate;

    const onsetRate = this.fluxPick.push(rw.flux, gate);
    const percRate = this.percPick.push(rw.percFlux, gate);
    this.onsetRate = percRate;

    let sum = 0;
    for (const s of STEM_NAMES) sum += sp[s];
    for (const s of STEM_NAMES) {
      let p = 0;
      if (sum > TINY && gate > 0) {
        const share = s === 'drums' ? drumsGroove / (drumsGroove + restGroove + TINY) : sp[s] / sum;
        const sDb = d + 10 * Math.log10(share + TINY);
        const g = gate * clamp01((sDb - SILENCE_DB) / GATE_DB);
        p = clamp01((share - SHARE_LO) / (SHARE_HI - SHARE_LO)) * g;
        if (s === 'drums') {
          const hiFrac = pHighL / (pDrL + TINY);
          p *=
            clamp01((hiFrac - DRUM_HF_LO) / (DRUM_HF_HI - DRUM_HF_LO)) *
            clamp01((hfFlat - DRUM_FLAT_LO) / (DRUM_FLAT_HI - DRUM_FLAT_LO)) *
            clamp01((percRate - 1) / 2);
        }
      }
      const v = this.presSm[s].push(p);
      this.stemPresence[s] = v > 1 ? 1 : v;
    }

    // Polyphony: strong chroma peaks over ~0.5 s.
    let m = 0;
    const cm = polyScratch;
    for (let k = 0; k < 12; k++) {
      cm[k] = this.chromaBox[k].push(gate > 0 ? this.chroma[k] : 0);
      if (cm[k] > m) m = cm[k];
    }
    let poly = 0;
    if (m > 0) {
      for (let k = 0; k < 12; k++) {
        const v = cm[k] / m;
        const l = cm[(k + 11) % 12] / m;
        const r = cm[(k + 1) % 12] / m;
        if (v >= 0.8 && v >= l && v >= r) poly++;
      }
    }
    const flat = this.flatB.push(rw.flatness);

    const pr = this.stemPresence;
    const layers = pr.drums + pr.bass + 0.5 * (pr.vocals + pr.other);
    const cStems = clamp01((layers - 0.8) / 1.4);
    const cDrum = pr.drums;
    const cOnset = clamp01((onsetRate - 1) / 7);
    const cFlat = clamp01((flat - 0.05) / 0.3);
    const cPoly = clamp01((poly - 1) / 4);
    const cLoud = clamp01((d + 35) / 25);
    const v = 0.2 * cStems + 0.25 * cDrum + 0.15 * cOnset + 0.15 * cFlat + 0.1 * cPoly + 0.15 * cLoud;
    const rawC = expand(v) * gate;
    // Forward asymmetric smoothing (complexity.ts smoothAsym, 0.6 s up / 1.5 s down).
    const ku = 1 - Math.exp(-1 / (0.6 * fr));
    const kd = 1 - Math.exp(-1 / (1.5 * fr));
    this.complexity = clamp01(this.complexity + (rawC - this.complexity) * (rawC > this.complexity ? ku : kd));
    const kf = 1 - Math.exp(-1 / (0.12 * fr));
    this.complexityFast = clamp01(this.complexityFast + (rawC - this.complexityFast) * kf);

    if (d > SILENCE_DB) {
      this.musicSeconds += 1 / fr;
      // Running mean over the song so far; after a minute, an exponential mean.
      this.cxSongN = Math.min(this.cxSongN + 1, SONG_CX_MEMORY_S * fr);
      this.cxSongSum += (this.complexity - this.cxSongSum) / this.cxSongN;
      this.songComplexity = clamp01(this.cxSongSum);
    }
  }
}

const zeroStemsCache: Record<StemName, number> = { drums: 0, bass: 0, vocals: 0, other: 0 };
const polyScratch = new Float64Array(12);

// Offline stand-ins for the live audio path, computed from decoded PCM at an exact time
// instead of from a running AnalyserNode, so a render is a pure function of the song and
// the frame clock.
//
// OfflineLive reproduces src/audio/LiveAnalyser.ts on top of what an AnalyserNode does
// (fftSize 2048, Blackman window over the latest 2048 mono samples, magnitude smoothing
// 0.5, dB = 20 log10(|X| / N)), then the same band / running-average / Att math.
// MelodyProbe gives a crude melody pitch (harmonic product spectrum on an 8192 FFT,
// 150..1500 Hz) and a 64-band log spectrum for the timeline sheet.

import type { LiveAudioFrame } from '../../src/types';
import { RealFFT } from '../../src/analysis/fft';

const MIN_DB = -100;
const MAX_DB = -20;
const SLOW_AVG_TAU = 6;
const ATTACK_TAU = 0.3;
const SILENCE_ENERGY = 1e-7;

function downsampleAverage(src: Float32Array, outLen: number, out = new Float32Array(outLen)): Float32Array {
  const ratio = src.length / outLen;
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.max(start + 1, Math.floor((i + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let j = start; j < end && j < src.length; j++) {
      sum += src[j];
      count++;
    }
    out[i] = count > 0 ? sum / count : 0;
  }
  return out;
}

const alphaFor = (dt: number, tau: number) => (dt <= 0 ? 0 : 1 - Math.exp(-dt / tau));

/** Mono mix the way an AnalyserNode down-mixes (average of the channels). */
export function monoMix(left: Float32Array, right: Float32Array): Float32Array {
  if (left === right) return left;
  const out = new Float32Array(left.length);
  for (let i = 0; i < left.length; i++) out[i] = 0.5 * (left[i] + right[i]);
  return out;
}

export class OfflineLive {
  private readonly N = 2048;
  private fft = new RealFFT(2048);
  private win = new Float32Array(2048);
  private buf = new Float32Array(2048);
  private re = new Float32Array(1025);
  private im = new Float32Array(1025);
  private smooth = new Float32Array(1024);
  private freqDb = new Float32Array(1024);
  private bassAvg = 0;
  private midAvg = 0;
  private trebAvg = 0;
  private bassAttSm = 0;
  private midAttSm = 0;
  private trebAttSm = 0;
  private seeded = false;

  private pcm: Float32Array;
  private sr: number;

  constructor(pcm: Float32Array, sr: number) {
    this.pcm = pcm;
    this.sr = sr;
    // Blackman window as specified for AnalyserNode (alpha 0.16).
    const a = 0.16;
    const a0 = (1 - a) / 2;
    const a2 = a / 2;
    for (let i = 0; i < this.N; i++) this.win[i] = a0 - 0.5 * Math.cos((2 * Math.PI * i) / this.N) + a2 * Math.cos((4 * Math.PI * i) / this.N);
  }

  private bandEnergy(loHz: number, hiHz: number): number {
    const nyquist = this.sr / 2;
    const bins = this.freqDb.length;
    const lo = Math.max(0, Math.floor((loHz / nyquist) * bins));
    const hi = Math.min(bins - 1, Math.ceil((hiHz / nyquist) * bins));
    let sum = 0;
    for (let i = lo; i <= hi; i++) sum += Math.pow(10, Math.max(MIN_DB, this.freqDb[i]) / 10);
    return sum / (hi - lo + 1);
  }

  /** The LiveAudioFrame an AnalyserNode would give with the playhead at time t. */
  read(t: number, dt: number): LiveAudioFrame {
    const end = Math.round(t * this.sr);
    const pcm = this.pcm;
    for (let i = 0; i < this.N; i++) {
      const k = end - this.N + i;
      this.buf[i] = k >= 0 && k < pcm.length ? pcm[k] : 0;
    }
    const waveform = downsampleAverage(this.buf, 1024);
    const x = new Float32Array(this.N);
    for (let i = 0; i < this.N; i++) x[i] = this.buf[i] * this.win[i];
    this.fft.forward(x, this.re, this.im);
    for (let k = 0; k < 1024; k++) {
      const mag = Math.hypot(this.re[k], this.im[k]) / this.N;
      this.smooth[k] = 0.5 * this.smooth[k] + 0.5 * mag;
      this.freqDb[k] = this.smooth[k] > 0 ? 20 * Math.log10(this.smooth[k]) : -Infinity;
    }
    const spectrumDb = downsampleAverage(this.freqDb, 512);
    const spectrum = new Float32Array(512);
    for (let i = 0; i < 512; i++) spectrum[i] = Math.max(0, Math.min(1, (spectrumDb[i] - MIN_DB) / (MAX_DB - MIN_DB)));

    const bassE = this.bandEnergy(20, 250);
    const midE = this.bandEnergy(250, 2000);
    const trebE = this.bandEnergy(2000, 11000);
    if (!this.seeded) {
      this.bassAvg = bassE || SILENCE_ENERGY;
      this.midAvg = midE || SILENCE_ENERGY;
      this.trebAvg = trebE || SILENCE_ENERGY;
      this.seeded = true;
    }
    const slow = alphaFor(dt, SLOW_AVG_TAU);
    this.bassAvg += (bassE - this.bassAvg) * slow;
    this.midAvg += (midE - this.midAvg) * slow;
    this.trebAvg += (trebE - this.trebAvg) * slow;
    const bass = bassE < SILENCE_ENERGY || this.bassAvg < SILENCE_ENERGY ? 0 : bassE / this.bassAvg;
    const mid = midE < SILENCE_ENERGY || this.midAvg < SILENCE_ENERGY ? 0 : midE / this.midAvg;
    const treb = trebE < SILENCE_ENERGY || this.trebAvg < SILENCE_ENERGY ? 0 : trebE / this.trebAvg;
    const att = alphaFor(dt, ATTACK_TAU);
    this.bassAttSm += (bass - this.bassAttSm) * att;
    this.midAttSm += (mid - this.midAttSm) * att;
    this.trebAttSm += (treb - this.trebAttSm) * att;
    return { bass, mid, treb, bassAtt: this.bassAttSm, midAtt: this.midAttSm, trebAtt: this.trebAttSm, waveform, spectrum };
  }
}

export const SPEC_BANDS = 64;
const SPEC_LO = 40;
const SPEC_HI = 16000;

export interface MelodyFrame {
  /** Melody pitch as a MIDI note number (fractional), NaN when nothing pitched stands out. */
  midi: number;
  /** 0..1 how clearly that pitch stands out. */
  salience: number;
}

export class MelodyProbe {
  private readonly N = 8192;
  private fft = new RealFFT(8192);
  private win = new Float32Array(8192);
  private x = new Float32Array(8192);
  private pow = new Float32Array(4097);

  private pcm: Float32Array;
  private sr: number;

  constructor(pcm: Float32Array, sr: number) {
    this.pcm = pcm;
    this.sr = sr;
    for (let i = 0; i < this.N; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (this.N - 1));
  }

  /** Melody pitch at t (window centred on t), and the 64-band log spectrum (0..255, dB-scaled) into spec. */
  read(t: number, spec: Uint8Array): MelodyFrame {
    const c = Math.round(t * this.sr);
    for (let i = 0; i < this.N; i++) {
      const k = c - this.N / 2 + i;
      this.x[i] = (k >= 0 && k < this.pcm.length ? this.pcm[k] : 0) * this.win[i];
    }
    this.fft.power(this.x, this.pow);
    const hz = this.sr / this.N;
    // Log spectrum for the sheet.
    for (let b = 0; b < SPEC_BANDS; b++) {
      const f0 = SPEC_LO * Math.pow(SPEC_HI / SPEC_LO, b / SPEC_BANDS);
      const f1 = SPEC_LO * Math.pow(SPEC_HI / SPEC_LO, (b + 1) / SPEC_BANDS);
      const k0 = Math.max(1, Math.floor(f0 / hz));
      const k1 = Math.max(k0 + 1, Math.ceil(f1 / hz));
      let s = 0;
      for (let k = k0; k < k1 && k < this.pow.length; k++) s += this.pow[k];
      const db = 10 * Math.log10(s / (k1 - k0) + 1e-12);
      spec[b] = Math.max(0, Math.min(255, Math.round(((db + 20) / 80) * 255)));
    }
    // Harmonic product spectrum over 3 harmonics, candidate f0 in 150..1500 Hz.
    const lo = Math.floor(150 / hz);
    const hi = Math.ceil(1500 / hz);
    let best = -1;
    let bestV = 0;
    let sum = 0;
    let cnt = 0;
    for (let k = lo; k <= hi; k++) {
      if (3 * k + 1 >= this.pow.length) break;
      const a = Math.sqrt(this.pow[k] + 0.5 * (this.pow[k - 1] + this.pow[k + 1]));
      const b = Math.sqrt(Math.max(this.pow[2 * k - 1], this.pow[2 * k], this.pow[2 * k + 1]));
      const d = Math.sqrt(Math.max(this.pow[3 * k - 1], this.pow[3 * k], this.pow[3 * k + 1]));
      const v = Math.cbrt(a * b * d);
      sum += v;
      cnt++;
      if (v > bestV) {
        bestV = v;
        best = k;
      }
    }
    const meanV = cnt ? sum / cnt : 0;
    const ratio = meanV > 0 ? bestV / meanV : 0;
    const salience = Math.max(0, Math.min(1, (ratio - 2) / 6));
    if (best < 0 || salience <= 0) return { midi: NaN, salience: 0 };
    // Parabolic interpolation around the peak bin for sub-bin pitch.
    const p0 = this.pow[best - 1];
    const p1 = this.pow[best];
    const p2 = this.pow[best + 1];
    const den = p0 - 2 * p1 + p2;
    const off = den !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (p0 - p2)) / den)) : 0;
    const f = (best + off) * hz;
    return { midi: 69 + 12 * Math.log2(f / 440), salience };
  }
}

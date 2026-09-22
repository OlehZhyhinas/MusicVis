import type { LiveAudioFrame } from '../types';

// Wraps a Web Audio AnalyserNode and derives MilkDrop-compatible bass/mid/treb
// levels: each band's instantaneous energy divided by its own slow running
// average, so ~1.0 means "about as loud as usual" for that band in this song.

const MIN_DB = -100;
const MAX_DB = -20;

// Frequency band edges, Hz.
const BASS_LO = 20;
const BASS_HI = 250;
const MID_LO = 250;
const MID_HI = 2000;
const TREB_LO = 2000;
const TREB_HI = 11000;

// Time constants, seconds.
const SLOW_AVG_TAU = 6; // running average of "typical" band energy
const ATTACK_TAU = 0.3; // smoothed Att variants
const SILENCE_ENERGY = 1e-7; // below this, treat the band as silent

function downsampleAverage(src: Float32Array, outLen: number): Float32Array {
  const out = new Float32Array(outLen);
  const ratio = src.length / outLen;
  if (ratio <= 1) {
    // Fewer source samples than requested output: repeat/nearest-fill.
    for (let i = 0; i < outLen; i++) {
      out[i] = src[Math.min(src.length - 1, Math.floor(i * ratio))];
    }
    return out;
  }
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

function alphaFor(dt: number, tau: number): number {
  if (dt <= 0) return 0;
  return 1 - Math.exp(-dt / tau);
}

export class LiveAnalyser {
  private analyser: AnalyserNode;
  private sampleRate: number;
  private timeData: Float32Array<ArrayBuffer>;
  private freqData: Float32Array<ArrayBuffer>;

  private bassAvg = 0;
  private midAvg = 0;
  private trebAvg = 0;
  private bassAttSm = 0;
  private midAttSm = 0;
  private trebAttSm = 0;
  private seeded = false;

  constructor(ctx: AudioContext, input: AudioNode) {
    this.sampleRate = ctx.sampleRate;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.5;
    input.connect(this.analyser);
    this.timeData = new Float32Array(this.analyser.fftSize);
    this.freqData = new Float32Array(this.analyser.frequencyBinCount);
  }

  private bandEnergy(loHz: number, hiHz: number): number {
    const nyquist = this.sampleRate / 2;
    const binCount = this.freqData.length;
    const loBin = Math.max(0, Math.floor((loHz / nyquist) * binCount));
    const hiBin = Math.min(binCount - 1, Math.ceil((hiHz / nyquist) * binCount));
    let sum = 0;
    let count = 0;
    for (let i = loBin; i <= hiBin; i++) {
      const db = Math.max(MIN_DB, this.freqData[i]);
      // Convert dB (power) to a linear power value.
      sum += Math.pow(10, db / 10);
      count++;
    }
    return count > 0 ? sum / count : 0;
  }

  read(dt: number): LiveAudioFrame {
    this.analyser.getFloatTimeDomainData(this.timeData);
    this.analyser.getFloatFrequencyData(this.freqData);

    const waveform = downsampleAverage(this.timeData, 1024);

    const spectrumDb = downsampleAverage(this.freqData, 512);
    const spectrum = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      const db = spectrumDb[i];
      spectrum[i] = Math.max(0, Math.min(1, (db - MIN_DB) / (MAX_DB - MIN_DB)));
    }

    const bassE = this.bandEnergy(BASS_LO, BASS_HI);
    const midE = this.bandEnergy(MID_LO, MID_HI);
    const trebE = this.bandEnergy(TREB_LO, TREB_HI);

    if (!this.seeded) {
      // Seed the running averages on the first frame so we don't start
      // with a divide-by-near-zero spike.
      this.bassAvg = bassE || SILENCE_ENERGY;
      this.midAvg = midE || SILENCE_ENERGY;
      this.trebAvg = trebE || SILENCE_ENERGY;
      this.seeded = true;
    }

    const slowAlpha = alphaFor(dt, SLOW_AVG_TAU);
    this.bassAvg += (bassE - this.bassAvg) * slowAlpha;
    this.midAvg += (midE - this.midAvg) * slowAlpha;
    this.trebAvg += (trebE - this.trebAvg) * slowAlpha;

    const bass = bassE < SILENCE_ENERGY || this.bassAvg < SILENCE_ENERGY ? 0 : bassE / this.bassAvg;
    const mid = midE < SILENCE_ENERGY || this.midAvg < SILENCE_ENERGY ? 0 : midE / this.midAvg;
    const treb = trebE < SILENCE_ENERGY || this.trebAvg < SILENCE_ENERGY ? 0 : trebE / this.trebAvg;

    const attAlpha = alphaFor(dt, ATTACK_TAU);
    this.bassAttSm += (bass - this.bassAttSm) * attAlpha;
    this.midAttSm += (mid - this.midAttSm) * attAlpha;
    this.trebAttSm += (treb - this.trebAttSm) * attAlpha;

    return {
      bass,
      mid,
      treb,
      bassAtt: this.bassAttSm,
      midAtt: this.midAttSm,
      trebAtt: this.trebAttSm,
      waveform,
      spectrum,
    };
  }
}

// Timbre: what the sound is made of, per stem and for the whole mix, frame by frame.
//
//   bright   spectral centroid (log frequency within the stem's range): dull 0 .. brilliant 1
//   noise    spectral flatness: pure tones 0 .. noise 1 (breath, distortion, cymbals)
//   rough    sensory roughness: partials beating inside a critical band (Plomp-Levelt / Sethares
//            dissonance of the strongest spectral peaks): a pure sine 0, a detuned pair
//            or a rich buzzy saw higher, noise highest (plus richness and inharmonicity terms)
//   attack   how sharp the onsets are: the fastest level rise at each onset (a pluck or a hit near
//            1, a slow swell near 0), held for a moment and then relaxing
//
// "Per stem" follows the stem split of stems.ts by frequency region: bass 30-250 Hz, vocals
// 250 Hz-4 kHz, other 250 Hz-10 kHz; the drums are read from the spectral change (the positive
// power increase frame to frame, i.e. the transients) over 40 Hz-11 kHz. The mix covers 60 Hz-10 kHz.
// Values are absolute (not normalized per song), so a sine and a saw differ in every song.
//
// Offline, TimbreFrames is fed each frame of the main STFT that analyzePcm already computes (no extra
// FFT); realtime, the live analyzer feeds its main spectrum. Per frame it is one pass over the bins,
// a few dozen logs and one small peak-pair sum.

import type { StemName, TimbreStats, TimbreTrack } from '../types';

export const TIMBRE_KEYS = ['mix', 'drums', 'bass', 'vocals', 'other'] as const;
export type TimbreKey = (typeof TIMBRE_KEYS)[number];

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const TINY = 1e-20;
/** Peaks per region in the roughness sum. */
const PEAKS = 12;
/** Log bands per region for the flatness (keeps the log count small). */
const FLAT_BANDS = 24;

interface Region {
  lo: number; // first bin
  hi: number; // last bin (exclusive)
  logLo: number; // log2 of the edge frequencies (brightness range)
  logHi: number;
  flat: Int32Array; // band edges for the flatness
}

function region(fLo: number, fHi: number, binHz: number, nb: number): Region {
  const lo = Math.max(1, Math.round(fLo / binHz));
  const hi = Math.min(nb, Math.max(lo + 2, Math.round(fHi / binHz)));
  const edges: number[] = [lo];
  for (let i = 1; i <= FLAT_BANDS; i++) {
    const e = Math.round(lo * Math.pow(hi / lo, i / FLAT_BANDS));
    if (e > edges[edges.length - 1]) edges.push(Math.min(hi, e));
  }
  if (edges[edges.length - 1] < hi) edges.push(hi);
  return { lo, hi, logLo: Math.log2(fLo), logHi: Math.log2(fHi), flat: Int32Array.from(edges) };
}

export const neutralTimbre = (): TimbreStats => ({ bright: 0, noise: 0, rough: 0, attack: 0 });

/**
 * Per-frame timbre from power spectra (n/2 + 1 bins at sample rate sr). Feed frames in order with
 * push(); `out` holds the smoothed values per key after each frame.
 */
export class TimbreFrames {
  readonly out: Record<TimbreKey, TimbreStats> = {
    mix: neutralTimbre(), drums: neutralTimbre(), bass: neutralTimbre(), vocals: neutralTimbre(), other: neutralTimbre(),
  };
  private readonly binHz: number;
  private readonly regions: Record<Exclude<TimbreKey, 'drums'>, Region>;
  private readonly drumR: Region;
  private readonly prev: Float64Array;
  private readonly flux: Float64Array;
  private readonly hist: Record<TimbreKey, Float64Array>; // last 4 levels (dB) per key, ring
  private hpos = 0;
  private readonly top: Record<TimbreKey, number>;
  private readonly kSmooth: number;
  private readonly kHold: number;
  private readonly kTop: number;
  private readonly pk = new Int32Array(PEAKS);
  private readonly pv = new Float64Array(PEAKS);
  private frames = 0;
  private mixDb = -200;
  private readonly logf: Float64Array;

  constructor(n: number, sr: number, frameRate: number) {
    const nb = n / 2 + 1;
    this.binHz = sr / n;
    const hz = this.binHz;
    this.regions = {
      mix: region(60, Math.min(10000, sr * 0.45), hz, nb),
      bass: region(30, 250, hz, nb),
      vocals: region(250, 4000, hz, nb),
      other: region(250, Math.min(10000, sr * 0.45), hz, nb),
    };
    this.drumR = region(40, Math.min(11000, sr * 0.48), hz, nb);
    this.prev = new Float64Array(nb);
    this.logf = Float64Array.from({ length: nb }, (_, k) => Math.log2((k + 0.5) * hz));
    this.flux = new Float64Array(nb);
    const h = () => new Float64Array(4).fill(-200);
    this.hist = { mix: h(), drums: h(), bass: h(), vocals: h(), other: h() };
    this.top = { mix: -200, drums: -200, bass: -200, vocals: -200, other: -200 };
    this.kSmooth = 1 - Math.exp(-1 / (0.08 * frameRate));
    this.kHold = Math.exp(-1 / (0.6 * frameRate));
    this.kTop = 1 / (30 * frameRate); // the reference level sinks ~1 dB per 30 frames-seconds
  }

  push(pow: Float64Array): void {
    this.frames++;
    for (let k = 0; k < pow.length; k++) {
      const d = pow[k] - this.prev[k];
      this.flux[k] = d > 0 ? d : 0;
      this.prev[k] = pow[k];
    }
    const hp = this.hpos;
    this.hpos = (hp + 1) & 3;
    // Roughness (the costly part) every third frame, staggered; it is smoothed over ~80 ms anyway.
    // The other region shares the vocals' value (peaks are read below 6 kHz, where they overlap).
    const ph = this.frames % 3;
    this.update('mix', pow, this.regions.mix, hp, ph === 0);
    this.update('bass', pow, this.regions.bass, hp, ph === 1);
    this.update('vocals', pow, this.regions.vocals, hp, ph === 2);
    this.update('other', pow, this.regions.other, hp, false);
    this.out.other.rough = this.out.vocals.rough;
    // Drums: the transient spectrum (power increase), its roughness taken as the mix's.
    this.update('drums', this.flux, this.drumR, hp, false);
    this.out.drums.rough = this.out.mix.rough;
  }

  private update(key: TimbreKey, pow: Float64Array, r: Region, hp: number, rough: boolean): void {
    const o = this.out[key];
    let sum = 0;
    let lsum = 0;
    for (let k = r.lo; k < r.hi; k++) {
      const p = pow[k];
      sum += p;
      lsum += p * this.logf[k];
    }
    const db = 10 * Math.log10(sum + TINY);
    // Level history for the attack (dB rise over ~3 frames), and a slowly sinking reference level.
    const h = this.hist[key];
    const old = h[(hp + 1) & 3];
    h[hp] = db;
    this.top[key] = Math.max(db, this.top[key] - this.kTop);
    if (key === 'mix') this.mixDb = db;
    // Gated in silence (50 dB under the key's own recent top) and, for a stem region, when it is a
    // negligible share of the mix (spectral leakage of another region's sound).
    let gate = clamp01((db - (this.top[key] - 50)) / 10) * (sum > 1e-12 ? 1 : 0);
    if (key !== 'mix') gate *= clamp01((db - (this.mixDb - 30)) / 10);
    const rise = db - Math.max(old, this.top[key] - 60);
    const att = gate > 0.5 ? clamp01((rise - 6) / 24) : 0;
    o.attack = Math.max(att, o.attack * this.kHold);
    if (gate <= 0.01) return; // silence: hold the last values
    const k = this.kSmooth * gate;
    const c = lsum / sum;
    o.bright += (clamp01((c - r.logLo) / (r.logHi - r.logLo)) - o.bright) * k;
    // Flatness over log bands: geometric / arithmetic mean of the band power densities.
    const e = r.flat;
    let gl = 0, am = 0, nb = 0;
    const floor = sum / (r.hi - r.lo) * 1e-6 + TINY;
    for (let i = 0; i + 1 < e.length; i++) {
      let s = 0;
      for (let q = e[i]; q < e[i + 1]; q++) s += pow[q];
      const d = s / (e[i + 1] - e[i]) + floor;
      gl += Math.log(d);
      am += d;
      nb++;
    }
    const flat = nb ? Math.exp(gl / nb) / (am / nb) : 0;
    o.noise += (clamp01((flat - 0.05) / 0.6) - o.noise) * k;
    if (rough) o.rough += (this.roughness(pow, r) - o.rough) * k;
  }

  /** Sethares dissonance of the strongest peaks in the region, normalized by their energy. */
  private roughness(pow: Float64Array, r: Region): number {
    const pk = this.pk, pv = this.pv;
    let n = 0;
    let minI = 0;
    const hi = Math.min(r.hi, Math.round(6000 / this.binHz));
    for (let k = Math.max(r.lo, 1); k < hi - 1; k++) {
      const p = pow[k];
      if (!(p > pow[k - 1] && p >= pow[k + 1])) continue;
      if (n < PEAKS) {
        pk[n] = k;
        pv[n] = p;
        n++;
        if (n === PEAKS) for (let i = 1; i < n; i++) if (pv[i] < pv[minI]) minI = i;
      } else if (p > pv[minI]) {
        pk[minI] = k;
        pv[minI] = p;
        minI = 0;
        for (let i = 1; i < n; i++) if (pv[i] < pv[minI]) minI = i;
      }
    }
    if (n < 2) return 0;
    let e = 0, d = 0;
    for (let i = 0; i < n; i++) e += pv[i];
    // Peaks far below the strongest do not count (spectral leakage, noise floor).
    let mx = 0;
    for (let i = 0; i < n; i++) mx = Math.max(mx, pv[i]);
    // Richness (how many strong partials: a buzzy saw) and inharmonicity (partials off the harmonic
    // series of the lowest strong one: noise, bells, detuning).
    let f0 = Infinity, strong = 0;
    for (let i = 0; i < n; i++) if (pv[i] >= mx * 1e-3) {
      strong++;
      f0 = Math.min(f0, pk[i]);
    }
    let ih = 0, iw = 0;
    for (let i = 0; i < n; i++) if (pv[i] >= mx * 1e-3) {
      const q = pk[i] / f0;
      const a = Math.sqrt(pv[i]);
      ih += a * Math.abs(q - Math.round(q)) * 2;
      iw += a;
    }
    const rich = clamp01((strong - 1) / 10);
    const inharm = iw > 0 ? ih / iw : 0;
    for (let i = 0; i < n; i++) {
      if (pv[i] < mx * 1e-3) continue;
      const fi = pk[i] * this.binHz;
      const ai = Math.sqrt(pv[i]);
      for (let j = i + 1; j < n; j++) {
        if (pv[j] < mx * 1e-3) continue;
        const fj = pk[j] * this.binHz;
        const fmin = Math.min(fi, fj);
        const s = 0.24 / (0.021 * fmin + 19);
        const df = Math.abs(fi - fj) * s;
        d += ai * Math.sqrt(pv[j]) * (Math.exp(-3.5 * df) - Math.exp(-5.75 * df));
      }
    }
    return clamp01((d / (e + TINY)) * 2.5 + 0.3 * rich + 0.5 * inharm * rich);
  }
}

/** Offline: fills a TimbreTrack frame by frame (call frame() from the STFT callback). */
export class TimbreRecorder {
  readonly track: TimbreTrack;
  private readonly tf: TimbreFrames;

  constructor(n: number, sr: number, frameRate: number, numFrames: number) {
    this.tf = new TimbreFrames(n, sr, frameRate);
    const set = () => ({ bright: new Float32Array(numFrames), noise: new Float32Array(numFrames), rough: new Float32Array(numFrames), attack: new Float32Array(numFrames) });
    this.track = { mix: set(), drums: set(), bass: set(), vocals: set(), other: set() };
  }

  frame(f: number, pow: Float64Array): void {
    this.tf.push(pow);
    for (const key of TIMBRE_KEYS) {
      const o = this.tf.out[key];
      const t = this.track[key];
      t.bright[f] = o.bright;
      t.noise[f] = o.noise;
      t.rough[f] = o.rough;
      t.attack[f] = o.attack;
    }
  }
}

/** The stems by name plus the mix, in TimbreTrack order. */
export const TIMBRE_STEMS: readonly StemName[] = ['drums', 'bass', 'vocals', 'other'];

// Phenotype fingerprints: what a preset looks and moves like when rendered,
// as opposed to what its genes say. Gene distance is not visual distance (a
// small fractal parameter can change everything; different genomes can look
// the same), so novelty, duplicate rejection, the map and the similarity fit
// all work on this vector.
//
// Every preset is rendered on the same fixed reference clip (ReferenceClip:
// calm, build and drop sections of a deterministic 128 bpm track), low-res
// frames are sampled, and FingerprintAcc reduces them to named features in
// five groups: colour, detail, structure (symmetry, rings, blobs), motion
// (block-matched flow) and music response (correlations with bass, drums and
// the beat, and the lift into the build and the drop).
//
// Pure logic (no DOM, no GL) so it runs in the Node tests; the GPU side is
// fingerprintRender.ts.

import type { MusicState, Section, StemName } from '../types';

/** Bump when the clip or the features change: stored fingerprints of another version are recomputed. */
export const FP_VERSION = 1;

export type FeatureGroup = 'colour' | 'detail' | 'structure' | 'motion' | 'response';
export const GROUPS: FeatureGroup[] = ['colour', 'detail', 'structure', 'motion', 'response'];

export interface FeatureSpec {
  key: string;
  group: FeatureGroup;
  /** Smallest scale used for the z-score (so a feature nearly constant across the archive can't blow up). */
  floor: number;
  label: string;
}

const F = (key: string, group: FeatureGroup, floor: number, label: string): FeatureSpec => ({ key, group, floor, label });

export const FEATURES: FeatureSpec[] = [
  F('lum', 'colour', 0.03, 'brightness'),
  F('lumStd', 'colour', 0.02, 'contrast'),
  F('cover', 'colour', 0.05, 'coverage'),
  F('peak', 'colour', 0.05, 'highlights'),
  F('sat', 'colour', 0.05, 'saturation'),
  F('colorful', 'colour', 0.02, 'colourfulness'),
  F('hueX', 'colour', 0.1, 'hue (x)'),
  F('hueY', 'colour', 0.1, 'hue (y)'),
  F('hueEnt', 'colour', 0.05, 'hue spread'),
  F('warm', 'colour', 0.02, 'warmth'),
  F('fine', 'detail', 0.03, 'fine detail'),
  F('mid', 'detail', 0.03, 'medium detail'),
  F('coarse', 'detail', 0.03, 'coarse detail'),
  F('grad', 'detail', 0.05, 'edges'),
  F('lines', 'detail', 0.03, 'linework'),
  F('symX', 'structure', 0.05, 'mirror symmetry'),
  F('symY', 'structure', 0.05, 'vertical symmetry'),
  F('rot', 'structure', 0.05, 'rotational symmetry'),
  F('kfold', 'structure', 0.02, 'n-fold symmetry'),
  F('rings', 'structure', 0.03, 'rings'),
  F('centre', 'structure', 0.02, 'centre weight'),
  F('blobs', 'structure', 0.1, 'blob count'),
  F('blobSize', 'structure', 0.01, 'blob size'),
  F('motion', 'motion', 0.003, 'change'),
  F('evolve', 'motion', 0.03, 'evolution'),
  F('strobe', 'motion', 0.02, 'flicker'),
  F('flowMag', 'motion', 0.05, 'flow speed'),
  F('zoom', 'motion', 0.02, 'zoom'),
  F('spin', 'motion', 0.02, 'rotation'),
  F('trans', 'motion', 0.03, 'drift'),
  F('coh', 'motion', 0.05, 'flow coherence'),
  F('bassLum', 'response', 0.05, 'brightness with bass'),
  F('drumLum', 'response', 0.05, 'brightness with drums'),
  F('beatLum', 'response', 0.05, 'flash on the beat'),
  F('beatMot', 'response', 0.05, 'motion on the beat'),
  F('bassMot', 'response', 0.05, 'motion with bass'),
  F('dropLum', 'response', 0.05, 'drop brightness lift'),
  F('dropMot', 'response', 0.05, 'drop motion lift'),
  F('buildMot', 'response', 0.05, 'build motion lift'),
];
export const FEATURE_COUNT = FEATURES.length;
export const GROUP_INDEX: Record<FeatureGroup, number[]> = Object.fromEntries(
  GROUPS.map((g) => [g, FEATURES.map((f, i) => (f.group === g ? i : -1)).filter((i) => i >= 0)]),
) as Record<FeatureGroup, number[]>;

// ------------------------------------------------------- reference clip

const BPM = 128;
const BEAT = 60 / BPM;
const STEMS: StemName[] = ['drums', 'bass', 'vocals', 'other'];

/** Section boundaries of the reference clip, in beats: calm, build, drop. */
export const CLIP = {
  calmEnd: 8 * BEAT,
  buildEnd: 16 * BEAT,
  end: 26 * BEAT,
  fps: 60,
};

export type ClipPart = 'calm' | 'build' | 'drop';

const SECTIONS: Record<ClipPart, Section> = {
  calm: { start: 0, end: CLIP.calmEnd, label: 'verse', energy: 0.3 },
  build: { start: CLIP.calmEnd, end: CLIP.buildEnd, label: 'build', energy: 0.6 },
  drop: { start: CLIP.buildEnd, end: 1e9, label: 'drop', energy: 1 },
};

export function clipPart(t: number): ClipPart {
  return t < CLIP.calmEnd ? 'calm' : t < CLIP.buildEnd ? 'build' : 'drop';
}

/** Audio levels at a clip time, for the music-response correlations. */
export interface ClipAudio {
  bass: number;
  drums: number;
  beat: number;
}

/**
 * A deterministic track (same for every preset, so fingerprints compare):
 * calm (pads, a quiet bass line, no drums), build (a snare roll that speeds
 * up, rising bass and buildIntensity), drop (hard kick on every beat, heavy
 * sidechained bass, dropPulse at the start). next() is stateful only through
 * the frame counter, so reset() replays it exactly.
 */
export class ReferenceClip {
  private wave = new Float32Array(1024);
  private spec = new Float32Array(512);
  private chroma = new Float32Array(12);
  private frame = 0;
  private lastPart: ClipPart = 'calm';

  reset(): void {
    this.frame = 0;
    this.lastPart = 'calm';
  }

  get time(): number {
    return this.frame / CLIP.fps;
  }

  static audio(t: number): ClipAudio {
    const part = clipPart(t);
    const bp = t / BEAT;
    const phase = bp - Math.floor(bp);
    if (part === 'calm') return { bass: 0.25 + 0.1 * Math.sin(t * 0.9), drums: 0, beat: 0 };
    if (part === 'build') {
      const k = (t - CLIP.calmEnd) / (CLIP.buildEnd - CLIP.calmEnd);
      // Snare roll: quarters, then eighths, then sixteenths.
      const div = k < 0.5 ? 1 : k < 0.8 ? 2 : 4;
      const sub = (bp * div) % 1;
      const snare = Math.exp(-sub * 10) * (0.4 + 0.6 * k);
      return { bass: 0.35 + 0.4 * k, drums: snare, beat: snare * 0.6 };
    }
    const kick = Math.exp(-phase * 9);
    return { bass: 0.6 + 0.4 * (1 - Math.exp(-phase * 3)), drums: Math.max(kick, 0.5 * Math.exp(-((bp * 2) % 1) * 14)), beat: kick };
  }

  next(): MusicState {
    this.frame++;
    const dt = 1 / CLIP.fps;
    const t = this.frame / CLIP.fps;
    const part = clipPart(t);
    const changed = part !== this.lastPart;
    this.lastPart = part;
    const a = ReferenceClip.audio(t);
    const bp = t / BEAT;
    const beatIndex = Math.floor(bp);
    const beatPhase = bp - beatIndex;
    const barIndex = Math.floor(bp / 4);
    const barPhase = bp / 4 - barIndex;
    const buildK = part === 'build' ? (t - CLIP.calmEnd) / (CLIP.buildEnd - CLIP.calmEnd) : 0;
    const drop = part === 'drop';
    const loud = part === 'calm' ? 0.3 : part === 'build' ? 0.4 + 0.3 * buildK : 0.75 + 0.2 * a.beat;
    const voc = part === 'calm' ? 0.5 + 0.2 * Math.sin(t * 0.7) : part === 'build' ? 0.4 : 0.55;
    const other = part === 'calm' ? 0.45 : 0.5 + 0.3 * buildK;
    for (let i = 0; i < 1024; i++) {
      const x = i / 1024;
      this.wave[i] = loud * (0.4 * Math.sin(x * 2 * Math.PI * 4 + t * 2) + 0.25 * voc * Math.sin(x * 2 * Math.PI * 15 - t * 5) + 0.35 * a.drums * Math.sin(x * 2 * Math.PI * 2));
    }
    for (let i = 0; i < 512; i++) {
      const f = i / 512;
      const base = loud * (0.7 * Math.exp(-f * 3.5) + 0.06 * Math.sin(f * 40 + t * 2));
      this.spec[i] = Math.max(0, Math.min(1, base + 0.45 * a.bass * Math.exp(-f * 14) + 0.3 * a.drums * Math.exp(-(f - 0.55) * (f - 0.55) * 25) + 0.15 * voc * Math.exp(-(f - 0.2) * (f - 0.2) * 80)));
    }
    this.chroma.fill(0.1);
    this.chroma[9] = this.chroma[0] = this.chroma[4] = 1;
    this.chroma[(Math.floor(t / 1.875) * 5) % 12] = 0.7;
    const lv: Record<StemName, number> = { drums: a.drums, bass: a.bass, vocals: voc, other };
    const on: Record<StemName, number> = { drums: a.drums, bass: drop ? a.beat * 0.7 : 0, vocals: 0, other: part === 'build' ? a.drums * 0.4 : 0 };
    const pres: Record<StemName, number> = { drums: part === 'calm' ? 0 : 0.7, bass: 0.6, vocals: 0.5, other: 0.5 };
    const stems = {} as Record<StemName, number>;
    for (const k of STEMS) stems[k] = lv[k];
    const sec = SECTIONS[part];
    return {
      time: t, dt, playing: true,
      bass: 0.6 + 1.2 * a.bass * (drop ? 1 : 0.7) + 0.5 * a.beat, mid: 0.7 + 0.4 * loud, treb: 0.7 + 0.6 * a.drums * (drop ? 0.6 : 1),
      bassAtt: 0.6 + a.bass, midAtt: 0.7 + 0.4 * loud, trebAtt: 0.8 + 0.3 * buildK,
      waveform: this.wave, spectrum: this.spec,
      bpm: BPM, beatIndex, barIndex, beatPhase, barPhase,
      beatPulse: a.beat, barPulse: drop ? Math.exp(-barPhase * 6) : 0,
      onBeat: false, onBar: false,
      stems, stemOnsets: on, stemPresence: pres,
      loudness: loud, complexity: part === 'calm' ? 0.3 : part === 'build' ? 0.45 + 0.3 * buildK : 0.8, songComplexity: 0.6,
      chroma: this.chroma, keyTonic: 9, keyMode: 'minor', keyHue: 0.62, keyChangePulse: 0,
      section: sec, sectionIndex: part === 'calm' ? 0 : part === 'build' ? 1 : 2,
      sectionProgress: Math.min(1, (t - sec.start) / Math.max(0.01, Math.min(sec.end, CLIP.end) - sec.start)),
      sectionChanged: changed,
      dropPulse: drop ? Math.exp(-(t - CLIP.buildEnd) * 1.5) : 0,
      buildIntensity: buildK,
      timeToDrop: t < CLIP.buildEnd ? CLIP.buildEnd - t : Infinity,
      sinceDrop: drop ? t - CLIP.buildEnd : Infinity,
      barSeconds: BEAT * 4,
    };
  }
}

// ---------------------------------------------------------- image maths

/** Box blur by running sums (edge-clamped), O(n) per pass whatever the radius. */
function blur(src: Float32Array, W: number, H: number, rad: number, tmp: Float32Array, out: Float32Array): Float32Array {
  const n = 2 * rad + 1;
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let s = 0;
    for (let d = -rad; d <= rad; d++) s += src[row + Math.min(W - 1, Math.max(0, d))];
    for (let x = 0; x < W; x++) {
      tmp[row + x] = s / n;
      s += src[row + Math.min(W - 1, x + rad + 1)] - src[row + Math.max(0, x - rad)];
    }
  }
  for (let x = 0; x < W; x++) {
    let s = 0;
    for (let d = -rad; d <= rad; d++) s += tmp[Math.min(H - 1, Math.max(0, d)) * W + x];
    for (let y = 0; y < H; y++) {
      out[y * W + x] = s / n;
      s += tmp[Math.min(H - 1, y + rad + 1) * W + x] - tmp[Math.max(0, y - rad) * W + x];
    }
  }
  return out;
}

export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  if (n < 3) return 0;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    sab += x * y;
    saa += x * x;
    sbb += y * y;
  }
  return saa > 1e-10 && sbb > 1e-10 ? sab / Math.sqrt(saa * sbb) : 0;
}

const NA = 64, NR = 24, KMAX = 16;
const COS = new Float32Array(KMAX * NA + NA);
const SIN = new Float32Array(KMAX * NA + NA);
for (let k = 1; k <= KMAX; k++) for (let a = 0; a < NA; a++) {
  COS[k * NA + a] = Math.cos((2 * Math.PI * k * a) / NA);
  SIN[k * NA + a] = Math.sin((2 * Math.PI * k * a) / NA);
}

type Frame = Record<string, number>;

/** Still-image traits of one RGBA8 frame. */
export function frameFeatures(px: Uint8Array, W: number, H: number, scratch?: Scratch): Frame {
  const s = scratch ?? new Scratch(W, H);
  const n = W * H;
  const L = s.L;
  let sum = 0, sq = 0, cover = 0, satW = 0, sv = 0, hx = 0, hy = 0, hw = 0, warm = 0;
  const hueBins = new Float32Array(12);
  let rgS = 0, rgQ = 0, ybS = 0, ybQ = 0;
  const hist = new Uint32Array(100);
  for (let i = 0; i < n; i++) {
    const r = px[i * 4] / 255, g = px[i * 4 + 1] / 255, b = px[i * 4 + 2] / 255;
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    L[i] = l;
    sum += l;
    sq += l * l;
    if (l > 0.08) cover++;
    hist[Math.min(99, Math.floor(l * 100))]++;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx > 1e-3 ? (mx - mn) / mx : 0;
    satW += sat * mx;
    sv += mx;
    warm += r - b;
    if (mx - mn > 0.02) {
      let h: number;
      if (mx === r) h = ((g - b) / (mx - mn) + 6) % 6;
      else if (mx === g) h = (b - r) / (mx - mn) + 2;
      else h = (r - g) / (mx - mn) + 4;
      h /= 6;
      const w = sat * mx;
      hx += Math.cos(h * 2 * Math.PI) * w;
      hy += Math.sin(h * 2 * Math.PI) * w;
      hw += w;
      hueBins[Math.floor(h * 12) % 12] += w;
    }
    const rg = r - g, yb = 0.5 * (r + g) - b;
    rgS += rg;
    rgQ += rg * rg;
    ybS += yb;
    ybQ += yb * yb;
  }
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sq / n - mean * mean));
  let acc = 0, peak = 0.99;
  for (let b = 99; b >= 0; b--) {
    acc += hist[b];
    if (acc > n * 0.005) {
      peak = b / 100;
      break;
    }
  }
  const rgM = rgS / n, ybM = ybS / n;
  const colorful = Math.sqrt(Math.max(0, rgQ / n - rgM * rgM) + Math.max(0, ybQ / n - ybM * ybM)) + 0.3 * Math.sqrt(rgM * rgM + ybM * ybM);
  let hent = 0;
  if (hw > 1e-6) for (const v of hueBins) if (v > 0) {
    const p = v / hw;
    hent -= p * Math.log(p);
  }
  hent /= Math.log(12);
  // Hue as a vector whose length is how dominant it is (grey pictures sit near 0).
  const hueLen = sv > 1e-6 ? hw / sv : 0;
  const hueX = hw > 1e-6 ? (hx / hw) * hueLen : 0;
  const hueY = hw > 1e-6 ? (hy / hw) * hueLen : 0;

  // Multi-scale detail: energy in three bands of the luma pyramid.
  const b1 = blur(L, W, H, 1, s.tmp, s.b1), b4 = blur(L, W, H, 4, s.tmp, s.b4), b16 = blur(L, W, H, 16, s.tmp, s.b16);
  let e0 = 0, e1 = 0, e2 = 0, grad = 0, lines = 0, bright = 0;
  for (let i = 0; i < n; i++) {
    e0 += (L[i] - b1[i]) ** 2;
    e1 += (b1[i] - b4[i]) ** 2;
    e2 += (b4[i] - b16[i]) ** 2;
    if (L[i] > 0.15) {
      bright++;
      // Thin structures: much brighter than the 3x3 neighbourhood.
      if (L[i] - b1[i] > 0.06) lines++;
    }
  }
  for (let y = 1; y < H; y++) for (let x = 1; x < W; x++) {
    const i = y * W + x;
    grad += Math.abs(L[i] - L[i - 1]) + Math.abs(L[i] - L[i - W]);
  }
  const et = e0 + e1 + e2 + 1e-9;

  // Symmetry: correlation with the mirrored / rotated picture.
  const fx = s.fx, fy = s.fy, fr = s.fr;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    fx[y * W + x] = L[y * W + (W - 1 - x)];
    fy[y * W + x] = L[(H - 1 - y) * W + x];
    fr[y * W + x] = L[(H - 1 - y) * W + (W - 1 - x)];
  }
  const symX = pearson(L, fx), symY = pearson(L, fy), rot = pearson(L, fr);

  // Polar resampling around the centre: n-fold periodicity and the radial profile.
  const polar = s.polar;
  const rMax = Math.min(W, H) * 0.47;
  for (let ri = 0; ri < NR; ri++) {
    const rr = 1 + (ri * (rMax - 1)) / NR;
    for (let a = 0; a < NA; a++) {
      const th = (a / NA) * 2 * Math.PI;
      const x = Math.round(W / 2 + rr * Math.cos(th)), y = Math.round(H / 2 + rr * Math.sin(th));
      polar[ri * NA + a] = L[Math.min(H - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))];
    }
  }
  const kpow = new Float32Array(KMAX + 1);
  let kall = 1e-9;
  const prof = new Float32Array(NR);
  for (let ri = 0; ri < NR; ri++) {
    let m = 0;
    for (let a = 0; a < NA; a++) m += polar[ri * NA + a];
    m /= NA;
    prof[ri] = m;
    for (let k = 1; k <= KMAX; k++) {
      let c = 0, sn = 0;
      for (let a = 0; a < NA; a++) {
        const v = polar[ri * NA + a] - m;
        c += v * COS[k * NA + a];
        sn += v * SIN[k * NA + a];
      }
      const p = c * c + sn * sn;
      kpow[k] += p;
      kall += p;
    }
  }
  let kmax = 0;
  for (let k = 3; k <= KMAX; k++) if (kpow[k] > kmax) kmax = kpow[k];
  const kfold = kmax / kall;
  let pm = 0;
  for (const v of prof) pm += v;
  pm /= NR;
  let pv = 0, osc = 0;
  for (let i = 0; i < NR; i++) {
    pv += (prof[i] - pm) ** 2;
    if (i > 0 && i < NR - 1) osc += (prof[i] - 0.5 * (prof[i - 1] + prof[i + 1])) ** 2;
  }
  const rings = osc / (pv + 1e-4);
  const centre = (prof[0] + prof[1] + prof[2]) / 3 - (prof[NR - 3] + prof[NR - 2] + prof[NR - 1]) / 3;

  // Blobs / cells: connected components above mean + 0.5 std, at half resolution.
  const w2 = W >> 1, h2 = H >> 1;
  const bm = s.bm;
  const thr = mean + 0.5 * std;
  for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    const i = 2 * y * W + 2 * x;
    bm[y * w2 + x] = (L[i] + L[i + 1] + L[i + W] + L[i + W + 1]) / 4 > thr && std > 0.02 ? 1 : 0;
  }
  let comps = 0, compArea = 0;
  const stack = s.stack;
  for (let i = 0; i < w2 * h2; i++) {
    if (bm[i] !== 1) continue;
    let area = 0, sp = 0;
    stack[sp++] = i;
    bm[i] = 2;
    while (sp) {
      const j = stack[--sp];
      area++;
      const x = j % w2, y = (j / w2) | 0;
      if (x > 0 && bm[j - 1] === 1) { bm[j - 1] = 2; stack[sp++] = j - 1; }
      if (x < w2 - 1 && bm[j + 1] === 1) { bm[j + 1] = 2; stack[sp++] = j + 1; }
      if (y > 0 && bm[j - w2] === 1) { bm[j - w2] = 2; stack[sp++] = j - w2; }
      if (y < h2 - 1 && bm[j + w2] === 1) { bm[j + w2] = 2; stack[sp++] = j + w2; }
    }
    if (area >= 3) {
      comps++;
      compArea += area;
    }
  }

  return {
    lum: mean, lumStd: std, cover: cover / n, peak,
    sat: sv > 1e-6 ? satW / sv : 0, colorful, hueX, hueY, hueEnt: hent, warm: warm / n,
    fine: e0 / et, mid: e1 / et, coarse: e2 / et, grad: grad / n / (mean + 0.03), lines: bright ? lines / bright : 0,
    symX, symY, rot, kfold, rings: Math.min(3, rings), centre,
    blobs: Math.log(1 + comps), blobSize: comps ? compArea / comps / (w2 * h2) : 0,
  };
}

/** Reusable buffers for frameFeatures at one resolution. */
export class Scratch {
  L: Float32Array; tmp: Float32Array; b1: Float32Array; b4: Float32Array; b16: Float32Array;
  fx: Float32Array; fy: Float32Array; fr: Float32Array; polar = new Float32Array(NA * NR);
  bm: Uint8Array; stack: Int32Array;
  constructor(W: number, H: number) {
    const n = W * H;
    this.L = new Float32Array(n);
    this.tmp = new Float32Array(n);
    this.b1 = new Float32Array(n);
    this.b4 = new Float32Array(n);
    this.b16 = new Float32Array(n);
    this.fx = new Float32Array(n);
    this.fy = new Float32Array(n);
    this.fr = new Float32Array(n);
    this.bm = new Uint8Array((W >> 1) * (H >> 1));
    this.stack = new Int32Array((W >> 1) * (H >> 1) + 4);
  }
}

function lumaOf(px: Uint8Array, W: number, H: number, out?: Float32Array): Float32Array {
  const L = out ?? new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) L[i] = (0.2126 * px[i * 4] + 0.7152 * px[i * 4 + 1] + 0.0722 * px[i * 4 + 2]) / 255;
  return L;
}

interface Flow { mag: number; div: number; curl: number; trans: number; coh: number }

/** Block-matching flow between two consecutive frames' luma: speed, zoom (divergence), spin (curl), drift. */
export function blockFlow(a: Float32Array, b: Float32Array, W: number, H: number): Flow | null {
  const BS = 8, STEP = 10, SR = 3;
  let n = 0, mag = 0, div = 0, curl = 0, sx = 0, sy = 0, rr = 0;
  for (let by = SR; by + BS + SR <= H; by += STEP) for (let bx = SR; bx + BS + SR <= W; bx += STEP) {
    let m = 0, q = 0;
    for (let y = 0; y < BS; y++) for (let x = 0; x < BS; x++) {
      const v = a[(by + y) * W + bx + x];
      m += v;
      q += v * v;
    }
    m /= BS * BS;
    if (q / (BS * BS) - m * m < 0.0006) continue; // flat block: no texture to match
    let best = Infinity, bdx = 0, bdy = 0, zero = 0;
    for (let dy = -SR; dy <= SR; dy++) for (let dx = -SR; dx <= SR; dx++) {
      let e = 0;
      for (let y = 0; y < BS; y++) {
        const ra = (by + y) * W + bx, rb = (by + y + dy) * W + bx + dx;
        for (let x = 0; x < BS; x++) e += Math.abs(a[ra + x] - b[rb + x]);
      }
      if (dx === 0 && dy === 0) zero = e;
      if (e < best - 1e-6) {
        best = e;
        bdx = dx;
        bdy = dy;
      }
    }
    if (zero - best < 0.001 * BS * BS) {
      bdx = 0;
      bdy = 0;
    }
    const cx = bx + BS / 2 - W / 2, cy = by + BS / 2 - H / 2;
    n++;
    mag += Math.hypot(bdx, bdy);
    sx += bdx;
    sy += bdy;
    div += bdx * cx + bdy * cy;
    curl += cx * bdy - cy * bdx;
    rr += cx * cx + cy * cy;
  }
  if (!n) return null;
  return { mag: mag / n, div: (div / (rr + 1e-9)) * 100, curl: (curl / (rr + 1e-9)) * 100, trans: Math.hypot(sx / n, sy / n), coh: mag > 0 ? Math.hypot(sx, sy) / mag : 0 };
}

/** Log ratio, damped near zero. */
const lift = (hi: number, lo: number, eps: number) => Math.log((hi + eps) / (lo + eps));

/**
 * Accumulates sampled frames of one reference-clip render into a fingerprint.
 * sample(): every sampled frame (cheap: luma, 40x22 grid for change); still(): the
 * frames to analyse fully; pair(): two consecutive frames for flow.
 */
export class FingerprintAcc {
  private stills: Frame[] = [];
  private small: Float32Array[] = [];
  private t: number[] = [];
  private lum: number[] = [];
  private flows: Flow[] = [];
  private scratch: Scratch;
  private W: number;
  private H: number;

  constructor(W: number, H: number) {
    this.W = W;
    this.H = H;
    this.scratch = new Scratch(W, H);
  }

  /** A sampled frame at clip time t. full: also compute the still-image traits. */
  sample(px: Uint8Array, t: number, full: boolean): void {
    const W = this.W, H = this.H;
    const SW = W >> 2, SH = H >> 2;
    const s = new Float32Array(SW * SH);
    let sum = 0;
    for (let y = 0; y < SH * 4; y++) for (let x = 0; x < SW * 4; x++) {
      const i = (y * W + x) * 4;
      const l = (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255;
      s[(y >> 2) * SW + (x >> 2)] += l / 16;
      sum += l;
    }
    this.small.push(s);
    this.t.push(t);
    this.lum.push(sum / (SW * SH * 16));
    if (full) this.stills.push(frameFeatures(px, W, H, this.scratch));
  }

  pair(a: Uint8Array, b: Uint8Array): void {
    const f = blockFlow(lumaOf(a, this.W, this.H), lumaOf(b, this.W, this.H), this.W, this.H);
    if (f) this.flows.push(f);
  }

  get samples(): number {
    return this.small.length;
  }

  /** The fingerprint (FEATURES order), rounded to 4 decimals. */
  finish(): number[] {
    const out: Record<string, number> = {};
    const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
    // Still traits averaged over the calm-to-drop frames (a preset is what it looks like throughout).
    for (const f of FEATURES) if (f.group !== 'motion' && f.group !== 'response') out[f.key] = avg(this.stills.map((s) => s[f.key] ?? 0));
    // Change between consecutive samples, on the 4x4-averaged grid (dither doesn't count).
    const mot: number[] = [0];
    for (let i = 1; i < this.small.length; i++) {
      let d = 0;
      const a = this.small[i], b = this.small[i - 1];
      for (let j = 0; j < a.length; j++) d += Math.abs(a[j] - b[j]);
      mot.push(d / a.length);
    }
    if (mot.length > 1) mot[0] = mot[1];
    out.motion = avg(mot);
    // Evolution: 1 - correlation with the picture ~1 s earlier.
    const lag = Math.max(1, Math.round(this.small.length / Math.max(1e-3, this.t[this.t.length - 1] ?? 1)));
    const ev: number[] = [];
    for (let i = lag; i < this.small.length; i++) ev.push(1 - pearson(this.small[i], this.small[i - lag]));
    out.evolve = avg(ev);
    const lm = avg(this.lum);
    out.strobe = Math.sqrt(avg(this.lum.map((l) => (l - lm) ** 2))) / (lm + 0.03);
    const fl = (k: keyof Flow) => avg(this.flows.map((f) => f[k]));
    out.flowMag = fl('mag');
    out.zoom = fl('div');
    // Spin direction and drift direction don't change the look much: magnitudes.
    out.spin = avg(this.flows.map((f) => Math.abs(f.curl)));
    out.trans = fl('trans');
    out.coh = fl('coh');
    // Music response.
    const audio = this.t.map((t) => ReferenceClip.audio(t));
    const bass = audio.map((a) => a.bass), drums = audio.map((a) => a.drums), beat = audio.map((a) => a.beat);
    out.bassLum = pearson(this.lum, bass);
    out.drumLum = pearson(this.lum, drums);
    const inDrop = this.t.map((t) => clipPart(t) === 'drop');
    const pick = (a: number[]) => a.filter((_, i) => inDrop[i]);
    out.beatLum = Math.abs(pearson(pick(this.lum), pick(beat)));
    out.beatMot = Math.abs(pearson(pick(mot), pick(beat)));
    out.bassMot = pearson(mot, bass);
    const part = (p: ClipPart, a: number[]) => avg(a.filter((_, i) => clipPart(this.t[i]) === p));
    out.dropLum = lift(part('drop', this.lum), part('calm', this.lum), 0.03);
    out.dropMot = lift(part('drop', mot), part('calm', mot), 0.004);
    out.buildMot = lift(part('build', mot), part('calm', mot), 0.004);
    return FEATURES.map((f) => {
      const v = out[f.key];
      return Number.isFinite(v) ? Math.round(v * 10000) / 10000 : 0;
    });
  }
}

export function validFingerprint(v: unknown): v is number[] {
  return Array.isArray(v) && v.length === FEATURE_COUNT && v.every((x) => typeof x === 'number' && Number.isFinite(x));
}

// ------------------------------------------------------ normalisation

const Z_CLIP = 4;

function median(a: number[]): number {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/**
 * Robust z-scores: (x - median) / max(1.4826 * MAD, floor), clipped to ±4.
 * Fit on the archive (every fingerprint seen), refit as it grows.
 */
export class FeatureNorm {
  center: Float64Array;
  scale: Float64Array;
  n: number;

  constructor(center: ArrayLike<number>, scale: ArrayLike<number>, n = 0) {
    this.center = Float64Array.from(center);
    this.scale = Float64Array.from(scale);
    this.n = n;
  }

  static identity(): FeatureNorm {
    return new FeatureNorm(FEATURES.map(() => 0), FEATURES.map((f) => f.floor * 4), 0);
  }

  static fit(fps: number[][]): FeatureNorm {
    const ok = fps.filter(validFingerprint);
    if (ok.length < 3) return FeatureNorm.identity();
    const c: number[] = [], s: number[] = [];
    for (let i = 0; i < FEATURE_COUNT; i++) {
      const col = ok.map((f) => f[i]);
      const m = median(col);
      const mad = median(col.map((x) => Math.abs(x - m)));
      c.push(m);
      s.push(Math.max(1.4826 * mad, FEATURES[i].floor));
    }
    return new FeatureNorm(c, s, ok.length);
  }

  z(fp: number[]): Float32Array {
    const z = new Float32Array(FEATURE_COUNT);
    for (let i = 0; i < FEATURE_COUNT; i++) z[i] = Math.max(-Z_CLIP, Math.min(Z_CLIP, (fp[i] - this.center[i]) / this.scale[i]));
    return z;
  }
}

export type GroupWeights = Record<FeatureGroup, number>;
export const EQUAL_WEIGHTS: GroupWeights = { colour: 1, detail: 1, structure: 1, motion: 1, response: 1 };

/** Per-group mean squared z difference (the terms the similarity fit weights). */
export function groupTerms(za: Float32Array, zb: Float32Array): number[] {
  return GROUPS.map((g) => {
    const idx = GROUP_INDEX[g];
    let s = 0;
    for (const i of idx) s += (za[i] - zb[i]) ** 2;
    return s / idx.length;
  });
}

/** Phenotype distance between two z-scored fingerprints: weighted RMS over groups. */
export function zDistance(za: Float32Array, zb: Float32Array, w: GroupWeights = EQUAL_WEIGHTS): number {
  const t = groupTerms(za, zb);
  let s = 0, ws = 0;
  GROUPS.forEach((g, i) => {
    s += w[g] * t[i];
    ws += w[g];
  });
  return ws > 0 ? Math.sqrt(s / ws) : 0;
}

export function fpDistance(a: number[], b: number[], norm: FeatureNorm, w: GroupWeights = EQUAL_WEIGHTS): number {
  return zDistance(norm.z(a), norm.z(b), w);
}

/**
 * Distance below which a child counts as a visual duplicate. In z units: two
 * renders of the same genome land at 0-0.16 (particle bursts are random);
 * 0.3 is well under the closest pair of seeds (~0.67).
 */
export const DUP_FP_DIST = 0.3;

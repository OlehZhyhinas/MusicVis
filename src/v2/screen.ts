// Screening before a child is kept, plus still thumbnails. Both render the
// genome offscreen on the main GL context against a synthetic 128 bpm
// MusicState, a few frames per animation frame so the main view never stalls.
//
// Rejections: compile error, too slow (cost model or measured), nearly all
// black / white, blank for long stretches (a longer, coarse run after the music
// run: shapes that drift out of a fold's wedge on two unsynced clocks), frozen,
// pure noise, and unreactive (moves the same with and without music).

import type { MusicState, NoteMark, NoteStats, Section, StemName } from '../types';
import { COST_BUDGET_MS, estimateCost, type Genome } from './genome';
import type { Engine, Stage } from './engine';
import { Stage as StageClass } from './engine';
import { mulberry32 } from './ops';

// --------------------------------------------------------- synthetic music

const BPM = 128;
const SECTION: Section = { start: 0, end: 1e9, label: 'chorus', energy: 0.7 };
const STEMS: StemName[] = ['drums', 'bass', 'vocals', 'other'];

// A simple melody for previews and screening: bars alternate between a staccato riff (eighth-note
// "tu tu tu") and a legato phrase (one gliding note per beat), so note-driven genes draw both.
const MELODY = [0, 4, 7, 12, 11, 7, 4, 2];
function syntheticNotes(t: number, beatLen: number, s: number): NoteStats | undefined {
  if (!s) return undefined;
  const noteAt = (i: number) => {
    const bar = Math.floor(i / 8);
    const staccato = bar % 2 === 0;
    const start = staccato ? i * (beatLen / 2) : i * (beatLen / 2);
    const len = staccato ? beatLen * 0.22 : beatLen * 0.48;
    const pitch = 64 + MELODY[i % 8] + (staccato ? 0 : 0.4 * Math.sin(i));
    return { start, len, pitch, legato: staccato ? 0.15 : 0.9 };
  };
  const cur = Math.floor(t / (beatLen / 2));
  const recent: NoteMark[] = [];
  for (let i = Math.max(0, cur - 11); i <= cur; i++) {
    const n = noteAt(i);
    const age = t - n.start;
    if (age < 0) continue;
    recent.push({ age, len: Math.min(age, n.len), ended: age > n.len, height: (n.pitch - 60) / 20, strength: 0.8 });
  }
  const n = noteAt(cur);
  const age = t - n.start;
  const held = age < n.len ? 0.8 : 0;
  const vib = n.legato > 0.5 ? 0.25 * Math.sin(t * 2 * Math.PI * 5.5) : 0;
  return {
    on: 0.8 * Math.exp(-age / 0.15), held, legato: n.legato, glide: n.legato > 0.5 ? 1.5 : 0, vibrato: n.legato > 0.5 ? 0.25 : 0,
    pitch: n.pitch + vib, height: (n.pitch + vib - 60) / 20, voice: n.legato > 0.5 ? 0.5 : 0.05, recent,
  };
}

/** Notes held flat for the metronome lane: the synthetic melody's mean articulation, one frozen snapshot of the recent marks. */
function flatNotesOf(n: NoteStats | undefined): NoteStats | undefined {
  if (!n) return undefined;
  return { on: 0.12, held: 0.45, legato: 0.5, glide: 0, vibrato: 0.12, pitch: 70, height: 0.5, voice: 0.3, recent: n.recent.map((r) => ({ ...r })) };
}

/** A 128 bpm groove: kick on beats, hats on eighths, bass, a vocal line. silent=true gives the same clock with no sound. */
export class SyntheticMusic {
  private wave = new Float32Array(1024);
  private spec = new Float32Array(512);
  private chroma = new Float32Array(12);
  private t = 0;

  constructor(private silent = false) {}

  /** Music time runs `shift` seconds ahead of the clock (desync counterfactual); state.time stays on the clock. */
  private shift = 0;
  /** Metronome: the same beat / bar clock, every audio value held at its mean (clock-lock counterfactual). */
  private flat = false;
  /** The melody under the metronome: articulation at its mean, the recent marks frozen. */
  private flatNotes: NoteStats | undefined;

  reset(silent = this.silent, shift = 0, flat = false): void {
    this.t = 0;
    this.silent = silent;
    this.shift = shift;
    this.flat = flat;
    this.flatNotes = undefined;
  }

  get time(): number {
    return this.t;
  }

  next(dt: number): MusicState {
    this.t += dt;
    const t = this.t + this.shift;
    const beatLen = 60 / BPM;
    const bp = t / beatLen;
    const beatIndex = Math.floor(bp);
    const beatPhase = bp - beatIndex;
    const barIndex = Math.floor(bp / 4);
    const barPhase = bp / 4 - barIndex;
    const eighth = (bp * 2) % 1;
    const s = this.silent ? 0 : 1;
    const flat = this.flat;
    const tw = flat ? 0 : t; // waveform / spectrum animation freezes under the metronome
    const pulse = Math.exp(-beatPhase * 9); // the clock's beat pulse (kept under the metronome)
    const kick = (flat ? 0.111 : pulse) * s;
    const hat = (flat ? 0.071 : Math.exp(-eighth * 14)) * s;
    const bass = (0.55 + (flat ? 0 : 0.35 * Math.sin(t * 1.3))) * s;
    const voc = (0.45 + (flat ? 0 : 0.4 * Math.sin(t * 0.7 + 1))) * s;
    const other = (0.5 + (flat ? 0 : 0.3 * Math.sin(t * 0.45 + 2))) * s;
    for (let i = 0; i < 1024; i++) {
      const x = i / 1024;
      this.wave[i] = s * (0.35 * Math.sin(x * 2 * Math.PI * 6 + tw * 3) + 0.2 * Math.sin(x * 2 * Math.PI * 17 - tw * 5) * voc + 0.25 * kick * Math.sin(x * 2 * Math.PI * 2));
    }
    for (let i = 0; i < 512; i++) {
      const f = i / 512;
      const base = 0.62 * Math.exp(-f * 3.2) + 0.12 * Math.sin(f * 40 + tw * 2) * 0.5;
      this.spec[i] = s * Math.max(0, Math.min(1, base + 0.3 * kick * Math.exp(-f * 10) + 0.25 * hat * Math.exp(-(f - 0.6) * (f - 0.6) * 30) + 0.18 * voc * Math.exp(-(f - 0.2) * (f - 0.2) * 80)));
    }
    this.chroma.fill(0.1);
    this.chroma[0] = this.chroma[4] = this.chroma[7] = s;
    this.chroma[(Math.floor(tw / 2) * 5) % 12] = 0.8 * s;
    const lv: Record<StemName, number> = { drums: Math.max(kick, hat * 0.6), bass, vocals: voc, other };
    const on: Record<StemName, number> = { drums: Math.max(kick, hat * 0.7), bass: kick * 0.6, vocals: 0, other: hat * 0.3 };
    const pres: Record<StemName, number> = { drums: s * 0.7, bass: s * 0.6, vocals: s * 0.5, other: s * 0.5 };
    const stems = {} as Record<StemName, number>;
    for (const k of STEMS) stems[k] = lv[k];
    return {
      time: this.t, dt, playing: true,
      bass: 1 + kick * s, mid: 1, treb: 1 + hat * 0.5, bassAtt: 1, midAtt: 1, trebAtt: 1,
      waveform: this.wave, spectrum: this.spec,
      bpm: BPM, beatIndex, barIndex, beatPhase, barPhase,
      beatPulse: pulse * s, barPulse: Math.exp(-barPhase * 6) * s,
      onBeat: false, onBar: false,
      stems, stemOnsets: on, stemPresence: pres,
      loudness: s * (0.55 + 0.25 * kick), complexity: this.silent ? 0.3 : 0.62, songComplexity: 0.6,
      chroma: this.chroma, keyTonic: 0, keyMode: 'major', keyHue: 0.58, keyChangePulse: 0,
      section: SECTION, sectionIndex: 1, sectionProgress: 0.3, sectionChanged: false, dropPulse: 0, buildIntensity: 0,
      notes: flat ? (this.flatNotes ??= flatNotesOf(syntheticNotes(t, beatLen, s))) : syntheticNotes(t, beatLen, s),
    };
  }
}

// ----------------------------------------------------------- job runner

type Step = () => boolean; // returns true when finished

/** Runs offscreen work a few milliseconds per animation frame. */
export class JobRunner {
  private jobs: Step[] = [];
  enqueue(step: Step): void {
    this.jobs.push(step);
  }
  get busy(): boolean {
    return this.jobs.length > 0;
  }
  get queued(): number {
    return this.jobs.length;
  }
  pump(budgetMs: number): void {
    const t0 = performance.now();
    while (this.jobs.length && performance.now() - t0 < budgetMs) {
      const done = this.jobs[0]();
      if (done) this.jobs.shift();
    }
  }
}

// ------------------------------------------------------------ screening

export const SCREEN_W = 160;
export const SCREEN_H = 90;
const MUSIC_SECS = 3;
const SILENT_SECS = 1.5;
const FPS = 60;
const SAMPLE_EVERY = 2; // read pixels every 2nd frame (30 Hz)
/**
 * Long run after the music run (same slot, carrying on): seconds, frame rate (a coarser clock dims
 * presets whose trails build up per frame), frames per readback, and the window: a window is blank
 * when every frame read in it is near-black, so a picture that flares on the beat is never blank.
 */
const LONG_SECS = 12;
const LONG_FPS = 20;
const LONG_READ_EVERY = 2;
const LONG_WINDOW_S = 0.5;

export interface ScreenMetrics {
  mean: number; // mean sRGB luma 0..1
  peak: number; // 99.7th percentile sRGB luma
  coverage: number; // fraction of pixels above 0.08
  motion: number; // mean abs luma change between 30 Hz samples
  motionSilent: number;
  beatCorr: number; // |corr| of luma / motion with the beat
  noise: number; // fraction of samples that look like noise
  cost: number; // estimated ms at 1440p
  msPerFrame: number; // measured wall time per screening frame
  /**
   * Audio-event drive (the AV harness's metronome counterfactual, cheap version): how differently
   * the picture moves frame to frame when the same beat clock plays with every audio value held
   * flat, relative to its own frame-to-frame change. ~0: the motion is the clock's (a constant
   * spin, a bar-locked sweep) or the preset drifts on its own.
   */
  events: number;
  /** Onset hit lift: share of kicks with a motion peak from 45 ms before to 125 ms after, above chance (-1..1). */
  hitLift: number;
  /** 0..1 cheap reactivity score from events and hitLift (for breeding / fitness). */
  reactivity: number;
  /** Share of the long run's half-second windows in which every frame is near-black (mean and peak luma under the black thresholds). */
  blank: number;
}

export interface ScreenResult {
  ok: boolean;
  reason?: string;
  metrics: ScreenMetrics;
  descriptor: number[];
}

export const THRESH = {
  black: 0.02, // mean luma
  blackPeak: 0.12, // 99.7th percentile luma: thin bright lines on black are not "black"
  white: 0.86,
  frozen: 0.0005, // 4x4-averaged luma change per 1/30 s; the output dither alone is ~0.0004
  frozenCorr: 0.3,
  noiseTemporal: 0.16,
  noiseSpatial: 0.14,
  noiseFrac: 0.6,
  reactRatio: 1.15,
  reactCorr: 0.12,
  /** Clock-locked / drifting: motion ignores audio events (events) and lands no kicks (hitLift). */
  driftEvents: 0.06,
  driftHit: 0.1,
  msPerFrame: 10,
  /** Reject when more than this share of the long run is near-black. */
  blankShare: 0.25,
};

/** Seconds of the music run rendered in lockstep with the metronome lane. */
const LOCKSTEP_SECS = 2;

/** Cheap reactivity score (0..1) from the screener's audio-event drive and onset hit lift. */
export function reactivityScore(events: number, hitLift: number): number {
  const c = (x: number) => (Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : 0);
  return c(0.6 * c(events / 0.5) + 0.4 * c(hitLift / 0.5));
}

/**
 * Onset hit lift of a 30 Hz motion series against beat frames: a hit is a motion peak (local max
 * above median + 1.5 MAD of the series) from 1 sample before to 3 after the beat (-45..+125 ms at
 * 30 Hz, the audiovisual sync window); chance is the same count at every other offset.
 */
export function hitLiftOf(motion: number[], beatIdx: number[]): number {
  const n = motion.length;
  if (n < 10 || !beatIdx.length) return 0;
  const sorted = [...motion].sort((a, b) => a - b);
  const med = sorted[n >> 1];
  const mad = [...motion].map((x) => Math.abs(x - med)).sort((a, b) => a - b)[n >> 1];
  const peak = motion.map((x, i) => i > 0 && i + 1 < n && x >= motion[i - 1] && x > motion[i + 1] && x > med + 1.5 * mad + 1e-5);
  const hitsAt = (off: number) => {
    let h = 0, tot = 0;
    for (const b of beatIdx) {
      const e = b + off;
      if (e - 1 < 0 || e + 3 >= n) continue;
      tot++;
      if (peak.slice(e - 1, e + 4).some(Boolean)) h++;
    }
    return tot ? h / tot : 0;
  };
  const hit = hitsAt(0);
  let ch = 0, cn = 0;
  for (let off = 5; off < 14; off++) {
    ch += hitsAt(off);
    cn++;
  }
  const chance = cn ? ch / cn : 0;
  return chance < 1 ? (hit - chance) / (1 - chance) : 0;
}

const srgbToLin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const LIN = new Float32Array(256).map((_, i) => srgbToLin(i / 255));

interface Run {
  frames: number;
  sample: number;
  lumas: number[];
  motions: number[];
  beats: number[];
  noiseHits: number;
  samples: number;
  prev: Float32Array | null;
  ms: number;
  cover: number[];
  peaks: number[];
  /** 4x4-averaged luma grid per sample after the settle (for the metronome comparison). */
  grids: Float32Array[];
  /** Sample indices (after the settle) where a beat started. */
  beatIdx: number[];
  lastBeat: number;
}

function newRun(): Run {
  return { frames: 0, sample: 0, lumas: [], motions: [], beats: [], noiseHits: 0, samples: 0, prev: null, ms: 0, cover: [], peaks: [], grids: [], beatIdx: [], lastBeat: -1 };
}

function corr(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 4) return 0;
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
  return saa > 1e-12 && sbb > 1e-12 ? sab / Math.sqrt(saa * sbb) : 0;
}

export class Screener {
  readonly runner = new JobRunner();
  private stage: Stage;
  /** Second stage rendering the metronome counterfactual in lockstep with the music run. */
  private stageB: Stage;
  private thumbStage: Stage;
  private px = new Uint8Array(SCREEN_W * SCREEN_H * 4);
  private pxB = new Uint8Array(SCREEN_W * SCREEN_H * 4);
  private music = new SyntheticMusic();
  private musicB = new SyntheticMusic();

  constructor(private eng: Engine) {
    this.stage = new StageClass(eng, { offscreen: true, particleCap: 16384, flameCap: 32768 });
    this.stage.resize(SCREEN_W, SCREEN_H);
    this.stageB = new StageClass(eng, { offscreen: true, particleCap: 16384, flameCap: 32768 });
    this.stageB.resize(SCREEN_W, SCREEN_H);
    this.thumbStage = new StageClass(eng, { offscreen: true, particleCap: 32768, flameCap: 131072 });
    this.thumbStage.resize(320, 180);
  }

  /** Wait for the genome's programs (parallel compile), resolving null on failure. */
  private whenCompiled(g: Genome, resolve: (ok: boolean) => void): Step {
    let waited = 0;
    return () => {
      const p = this.eng.cache.get(g, waited > 240);
      if (p) {
        resolve(true);
        return true;
      }
      if (this.eng.cache.failed(g)) {
        resolve(false);
        return true;
      }
      waited++;
      return false;
    };
  }

  screen(g: Genome): Promise<ScreenResult> {
    return new Promise((resolve) => {
      const cost = estimateCost(g);
      const metrics: ScreenMetrics = { mean: 0, peak: 0, coverage: 0, motion: 0, motionSilent: 0, beatCorr: 0, noise: 0, cost, msPerFrame: 0, events: NaN, hitLift: 0, reactivity: 0, blank: 0 };
      const fail = (reason: string, descriptor: number[] = []) => resolve({ ok: false, reason, metrics, descriptor });
      if (cost > COST_BUDGET_MS) {
        fail(`too slow (estimated ${cost.toFixed(1)} ms)`);
        return;
      }
      let compiled: boolean | null = null;
      const waitStep = this.whenCompiled(g, (ok) => (compiled = ok));
      let phase: 'compile' | 'music' | 'long' | 'silent' | 'done' = 'compile';
      const longFrames = LONG_SECS * LONG_FPS;
      let longN = 0;
      let longBlank = 0;
      let longWins = 0;
      let winAllBlack = true;
      const winFrames = Math.max(1, Math.round(LONG_WINDOW_S * LONG_FPS));
      const st = this.stage;
      let slot: ReturnType<Stage['makeSlot']> | null = null;
      let slotB: ReturnType<Stage['makeSlot']> | null = null;
      const metroRun = newRun();
      const lockstepFrames = LOCKSTEP_SECS * FPS;
      // Both lanes draw the same random stream, so only the music differs between them.
      let randA = mulberry32(9);
      let randB = mulberry32(9);
      const music = newRun();
      const silent = newRun();
      const musicFrames = MUSIC_SECS * FPS;
      const silentFrames = SILENT_SECS * FPS;
      const lastLin = new Float32Array(SCREEN_W * SCREEN_H);
      const meanRGB = [0, 0, 0];
      let rgbN = 0;
      let lastPx: Uint8Array | null = null;

      const start = (silentRun: boolean) => {
        if (slot) st.disposeSlot(slot);
        const progs = this.eng.cache.get(g)!;
        slot = st.makeSlot(g, progs);
        st.slots = [slot];
        st.resetHistory();
        this.music.reset(silentRun);
        randA = mulberry32(9);
        if (!silentRun) {
          if (slotB) this.stageB.disposeSlot(slotB);
          slotB = this.stageB.makeSlot(g, progs);
          this.stageB.slots = [slotB];
          this.stageB.resetHistory();
          this.musicB.reset(false, 0, true);
          randB = mulberry32(9);
        }
      };

      const withRandom = (r: () => number, f: () => void) => {
        const save = Math.random;
        Math.random = r;
        try {
          f();
        } finally {
          Math.random = save;
        }
      };

      const frame = (run: Run, isMusic: boolean) => {
        const t0 = performance.now();
        const state = this.music.next(1 / FPS);
        withRandom(randA, () => st.render(state, 1 / FPS, 'out'));
        run.frames++;
        const lockstep = isMusic && run.frames <= lockstepFrames;
        if (lockstep) {
          const sb = this.musicB.next(1 / FPS);
          withRandom(randB, () => this.stageB.render(sb, 1 / FPS, 'out'));
          metroRun.frames++;
        }
        if (run.frames % SAMPLE_EVERY === 0) {
          st.readPixels(this.px);
          this.analyse(run, state.beatPulse, isMusic, lastLin, meanRGB, () => rgbN++, state.beatIndex);
          if (lockstep && run.frames >= 30) {
            this.stageB.readPixels(this.pxB);
            metroRun.grids.push(this.smallGrid(this.pxB));
          }
        }
        run.ms += performance.now() - t0;
      };

      this.runner.enqueue(() => {
        if (phase === 'compile') {
          if (!waitStep()) return false;
          if (!compiled) {
            phase = 'done';
            fail(`compile error: ${(this.eng.cache.failed(g) ?? '').split('\n').find((l) => /ERROR/.test(l)) ?? 'shader failed'}`.slice(0, 160));
            return true;
          }
          phase = 'music';
          start(false);
          return false;
        }
        if (phase === 'music') {
          frame(music, true);
          if (music.frames < musicFrames) return false;
          lastPx = this.px.slice();
          phase = 'long';
          return false;
        }
        if (phase === 'long') {
          // Carry on with the same slot at a coarse clock and look only for near-black frames.
          const state = this.music.next(1 / LONG_FPS);
          withRandom(randA, () => st.render(state, 1 / LONG_FPS, 'out'));
          longN++;
          if (longN % LONG_READ_EVERY === 0) {
            st.readPixels(this.px);
            if (!this.nearBlack(this.px)) winAllBlack = false;
          }
          if (longN % winFrames === 0) {
            longWins++;
            if (winAllBlack) longBlank++;
            winAllBlack = true;
          }
          if (longN < longFrames) return false;
          metrics.blank = longBlank / Math.max(1, longWins);
          phase = 'silent';
          start(true);
          return false;
        }
        if (phase === 'silent') {
          frame(silent, false);
          if (silent.frames < silentFrames) return false;
          phase = 'done';
          if (slot) st.disposeSlot(slot);
          slot = null;
          if (slotB) this.stageB.disposeSlot(slotB);
          slotB = null;
          this.reactivity(music, metroRun, metrics);
          const res = this.verdict(g, music, silent, metrics, meanRGB, rgbN, lastPx);
          resolve(res);
          return true;
        }
        return true;
      });
    });
  }

  /** Whether a readback is near-black: mean sRGB luma and its 99.7th percentile under the black thresholds (sampled on a 2 px grid). */
  private nearBlack(px: Uint8Array): boolean {
    const W = SCREEN_W, H = SCREEN_H;
    const hist = new Uint32Array(64);
    let sum = 0;
    let n = 0;
    for (let y = 0; y < H; y += 2) {
      for (let x = 0; x < W; x += 2) {
        const i = (y * W + x) * 4;
        const s = (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255;
        sum += s;
        hist[Math.min(63, Math.floor(s * 64))]++;
        n++;
      }
    }
    if (sum / n >= THRESH.black) return false;
    let acc = 0;
    for (let b = 63; b >= 0; b--) {
      acc += hist[b];
      if (acc > n * 0.003) return b / 64 < THRESH.blackPeak;
    }
    return true;
  }

  /** Per-sample statistics (luma, motion, noise). */
  /** 4x4-averaged sRGB luma grid of a readback. */
  private smallGrid(px: Uint8Array): Float32Array {
    const W = SCREEN_W;
    const SW = SCREEN_W >> 2, SH = SCREEN_H >> 2;
    const small = new Float32Array(SW * SH);
    for (let y = 0; y < SH * 4; y++) {
      for (let x = 0; x < SW * 4; x++) {
        const i = (y * W + x) * 4;
        small[(y >> 2) * SW + (x >> 2)] += (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255 / 16;
      }
    }
    return small;
  }

  /** Audio-event drive (music lane vs metronome lane) and onset hit lift of the music run (see ScreenMetrics). */
  private reactivity(music: Run, metro: Run, m: ScreenMetrics): void {
    const A = music.grids;
    const B = metro.grids;
    const diff = (a: Float32Array, b: Float32Array) => {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
      return s / a.length;
    };
    let d = 0, self = 0;
    const n = Math.min(A.length, B.length);
    for (let i = 1; i < n; i++) {
      const sa = diff(A[i], A[i - 1]);
      const sb = diff(B[i], B[i - 1]);
      d += Math.abs(sa - sb);
      self += sa;
    }
    m.events = self > 1e-5 ? d / self : d > 1e-4 ? 1 : 0;
    m.hitLift = hitLiftOf(music.motions, music.beatIdx);
    m.reactivity = reactivityScore(m.events, m.hitLift);
  }

  private analyse(run: Run, beat: number, isMusic: boolean, lastLin: Float32Array, meanRGB: number[], bump: () => void, beatIndex = -1): void {
    const W = SCREEN_W, H = SCREEN_H;
    const px = this.px;
    const lin = new Float32Array(W * H);
    let sum = 0;
    let cover = 0;
    let spatial = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const L = 0.2126 * LIN[px[i]] + 0.7152 * LIN[px[i + 1]] + 0.0722 * LIN[px[i + 2]];
        lin[y * W + x] = L;
        const s = (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255;
        sum += s;
        if (s > 0.08) cover++;
        if (x > 0) spatial += Math.abs(s - (0.2126 * px[i - 4] + 0.7152 * px[i - 3] + 0.0722 * px[i - 2]) / 255);
      }
    }
    const n = W * H;
    const mean = sum / n;
    // Motion on 4x4-averaged perceptual luminance, so the output dither does not count as movement.
    const SW = W >> 2, SH = H >> 2;
    const small = new Float32Array(SW * SH);
    for (let y = 0; y < SH * 4; y++) {
      for (let x = 0; x < SW * 4; x++) {
        const i = (y * W + x) * 4;
        small[(y >> 2) * SW + (x >> 2)] += (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255 / 16;
      }
    }
    let motion = 0;
    if (run.prev) {
      let d = 0;
      for (let i = 0; i < small.length; i++) d += Math.abs(small[i] - run.prev[i]);
      motion = d / small.length;
    }
    run.prev = small;
    run.samples++;
    // Skip the first 0.5 s (feedback and exposure settle).
    if (run.frames < 30) return;
    run.lumas.push(mean);
    run.motions.push(motion);
    run.beats.push(beat);
    run.grids.push(small);
    if (beatIndex !== run.lastBeat) {
      if (run.lastBeat >= 0) run.beatIdx.push(run.motions.length - 1);
      run.lastBeat = beatIndex;
    }
    if (motion > THRESH.noiseTemporal && spatial / n > THRESH.noiseSpatial) run.noiseHits++;
    run.cover.push(cover / n);
    // 99.7th percentile of luma from a coarse histogram.
    const hist = new Uint32Array(64);
    for (let i = 0; i < n; i++) hist[Math.min(63, Math.floor(((0.2126 * px[i * 4] + 0.7152 * px[i * 4 + 1] + 0.0722 * px[i * 4 + 2]) / 255) * 64))]++;
    let acc = 0;
    let pk = 63;
    for (let b = 63; b >= 0; b--) {
      acc += hist[b];
      if (acc > n * 0.003) {
        pk = b;
        break;
      }
    }
    run.peaks.push(pk / 64);
    if (isMusic) {
      for (let i = 0; i < n; i++) lastLin[i] = lin[i];
      if (run.frames > 60) {
        let r = 0, g = 0, b = 0;
        for (let i = 0; i < n; i++) {
          r += px[i * 4];
          g += px[i * 4 + 1];
          b += px[i * 4 + 2];
        }
        meanRGB[0] += r / n / 255;
        meanRGB[1] += g / n / 255;
        meanRGB[2] += b / n / 255;
        bump();
      }
    }
  }

  private verdict(g: Genome, music: Run, silent: Run, m: ScreenMetrics, meanRGB: number[], rgbN: number, lastPx: Uint8Array | null): ScreenResult {
    const avg = (a: number[]) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
    const lastHalf = <T>(a: T[]) => a.slice(Math.floor(a.length / 2));
    m.peak = avg(lastHalf(music.peaks));
    m.mean = avg(lastHalf(music.lumas));
    m.coverage = avg(lastHalf(music.cover));
    m.motion = avg(music.motions);
    m.motionSilent = avg(silent.motions);
    m.beatCorr = Math.max(Math.abs(corr(music.lumas, music.beats)), Math.abs(corr(music.motions, music.beats)));
    m.noise = music.lumas.length ? music.noiseHits / music.lumas.length : 0;
    m.msPerFrame = music.ms / Math.max(1, music.frames);
    const descriptor = this.descriptor(meanRGB, rgbN, m, lastPx);
    const r = (reason: string): ScreenResult => ({ ok: false, reason, metrics: m, descriptor });
    if (m.msPerFrame > THRESH.msPerFrame) return r(`too slow (${m.msPerFrame.toFixed(1)} ms per screening frame)`);
    if (m.mean < THRESH.black && m.peak < THRESH.blackPeak) return r('nearly all black');
    if (m.mean > THRESH.white) return r('nearly all white');
    if (m.blank > THRESH.blankShare) return r(`blank for ${Math.round(m.blank * 100)}% of a ${LONG_SECS} s run`);
    if (m.noise > THRESH.noiseFrac) return r('pure noise');
    const recent = lastHalf(music.motions);
    if (avg(recent) < THRESH.frozen && Math.max(...recent) < THRESH.frozen * 3 && m.beatCorr < THRESH.frozenCorr) return r('frozen');
    if (m.motion < m.motionSilent * THRESH.reactRatio && m.beatCorr < THRESH.reactCorr) return r('unreactive (same motion without music)');
    if (m.events < THRESH.driftEvents && m.hitLift < THRESH.driftHit) return r(`clock-locked or drifting (motion ignores the audio: events ${m.events.toFixed(2)}, hit lift ${m.hitLift.toFixed(2)})`);
    void g;
    return { ok: true, metrics: m, descriptor };
  }

  /** Mean colour, motion, mirror symmetry, radial symmetry, detail, coverage. */
  private descriptor(meanRGB: number[], n: number, m: ScreenMetrics, px: Uint8Array | null): number[] {
    const W = SCREEN_W, H = SCREEN_H;
    let symX = 0, symR = 0, detail = 0;
    if (px) {
      const L = (x: number, y: number) => {
        const i = (y * W + x) * 4;
        return (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255;
      };
      let dx = 0, dr = 0, dd = 0, tot = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const v = L(x, y);
          tot += v;
          dx += Math.abs(v - L(W - 1 - x, y));
          dr += Math.abs(v - L(W - 1 - x, H - 1 - y));
          if (x > 0 && y > 0) dd += Math.abs(v - L(x - 1, y)) + Math.abs(v - L(x, y - 1));
        }
      }
      const norm = Math.max(tot, 1e-3);
      symX = 1 - Math.min(1, dx / norm / 2);
      symR = 1 - Math.min(1, dr / norm / 2);
      detail = Math.min(1, dd / norm / 2);
    }
    const k = Math.max(1, n);
    return [meanRGB[0] / k, meanRGB[1] / k, meanRGB[2] / k, Math.min(1, m.motion * 12), symX, symR, detail, m.coverage].map(
      (v) => Math.round(v * 1000) / 1000,
    );
  }

  /** Render a still at 320x180 after ~1.5 s of synthetic music; resolves a JPEG data URL (or '' on failure). */
  thumbnail(g: Genome, secs = 1.6): Promise<string> {
    return new Promise((resolve) => {
      let compiled: boolean | null = null;
      const waitStep = this.whenCompiled(g, (ok) => (compiled = ok));
      const st = this.thumbStage;
      let frames = 0;
      let started = false;
      const music = new SyntheticMusic();
      let slot: ReturnType<Stage['makeSlot']> | null = null;
      this.runner.enqueue(() => {
        if (!started) {
          if (!waitStep()) return false;
          if (!compiled) {
            resolve('');
            return true;
          }
          slot = st.makeSlot(g, this.eng.cache.get(g)!);
          st.slots = [slot];
          st.resetHistory();
          started = true;
          return false;
        }
        st.render(music.next(1 / FPS), 1 / FPS, 'out');
        if (++frames < secs * FPS) return false;
        const w = st.w, h = st.h;
        const buf = new Uint8Array(w * h * 4);
        st.readPixels(buf);
        st.disposeSlot(slot!);
        slot = null;
        const c = document.createElement('canvas');
        c.width = w;
        c.height = h;
        const ctx = c.getContext('2d');
        if (!ctx) {
          resolve('');
          return true;
        }
        const img = ctx.createImageData(w, h);
        for (let y = 0; y < h; y++) img.data.set(buf.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
        ctx.putImageData(img, 0, 0);
        resolve(c.toDataURL('image/jpeg', 0.82));
        return true;
      });
    });
  }
}

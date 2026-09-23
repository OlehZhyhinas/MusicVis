// Screening before a child is kept, plus still thumbnails. Both render the
// genome offscreen on the main GL context against a synthetic 128 bpm
// MusicState, a few frames per animation frame so the main view never stalls.
//
// Rejections: compile error, too slow (cost model or measured), nearly all
// black / white, frozen, pure noise, flashing (photosensitive safety: more
// than 3 large-area luminance flashes in any second, after WCAG 2.3.1's
// general flash definition), and unreactive (moves the same with and without
// music).

import type { MusicState, Section, StemName } from '../types';
import { COST_BUDGET_MS, estimateCost, type Genome } from './genome';
import type { Engine, Stage } from './engine';
import { Stage as StageClass } from './engine';

// --------------------------------------------------------- synthetic music

const BPM = 128;
const SECTION: Section = { start: 0, end: 1e9, label: 'chorus', energy: 0.7 };
const STEMS: StemName[] = ['drums', 'bass', 'vocals', 'other'];

/** A 128 bpm groove: kick on beats, hats on eighths, bass, a vocal line. silent=true gives the same clock with no sound. */
export class SyntheticMusic {
  private wave = new Float32Array(1024);
  private spec = new Float32Array(512);
  private chroma = new Float32Array(12);
  private t = 0;

  constructor(private silent = false) {}

  reset(silent = this.silent): void {
    this.t = 0;
    this.silent = silent;
  }

  get time(): number {
    return this.t;
  }

  next(dt: number): MusicState {
    this.t += dt;
    const t = this.t;
    const beatLen = 60 / BPM;
    const bp = t / beatLen;
    const beatIndex = Math.floor(bp);
    const beatPhase = bp - beatIndex;
    const barIndex = Math.floor(bp / 4);
    const barPhase = bp / 4 - barIndex;
    const eighth = (bp * 2) % 1;
    const s = this.silent ? 0 : 1;
    const kick = Math.exp(-beatPhase * 9) * s;
    const hat = Math.exp(-eighth * 14) * s;
    const bass = (0.55 + 0.35 * Math.sin(t * 1.3)) * s;
    const voc = (0.45 + 0.4 * Math.sin(t * 0.7 + 1)) * s;
    const other = (0.5 + 0.3 * Math.sin(t * 0.45 + 2)) * s;
    for (let i = 0; i < 1024; i++) {
      const x = i / 1024;
      this.wave[i] = s * (0.35 * Math.sin(x * 2 * Math.PI * 6 + t * 3) + 0.2 * Math.sin(x * 2 * Math.PI * 17 - t * 5) * voc + 0.25 * kick * Math.sin(x * 2 * Math.PI * 2));
    }
    for (let i = 0; i < 512; i++) {
      const f = i / 512;
      const base = 0.62 * Math.exp(-f * 3.2) + 0.12 * Math.sin(f * 40 + t * 2) * 0.5;
      this.spec[i] = s * Math.max(0, Math.min(1, base + 0.3 * kick * Math.exp(-f * 10) + 0.25 * hat * Math.exp(-(f - 0.6) * (f - 0.6) * 30) + 0.18 * voc * Math.exp(-(f - 0.2) * (f - 0.2) * 80)));
    }
    this.chroma.fill(0.1);
    this.chroma[0] = this.chroma[4] = this.chroma[7] = s;
    this.chroma[(Math.floor(t / 2) * 5) % 12] = 0.8 * s;
    const lv: Record<StemName, number> = { drums: Math.max(kick, hat * 0.6), bass, vocals: voc, other };
    const on: Record<StemName, number> = { drums: Math.max(kick, hat * 0.7), bass: kick * 0.6, vocals: 0, other: hat * 0.3 };
    const pres: Record<StemName, number> = { drums: s * 0.7, bass: s * 0.6, vocals: s * 0.5, other: s * 0.5 };
    const stems = {} as Record<StemName, number>;
    for (const k of STEMS) stems[k] = lv[k];
    return {
      time: t, dt, playing: true,
      bass: 1 + kick * s, mid: 1, treb: 1 + hat * 0.5, bassAtt: 1, midAtt: 1, trebAtt: 1,
      waveform: this.wave, spectrum: this.spec,
      bpm: BPM, beatIndex, barIndex, beatPhase, barPhase,
      beatPulse: kick, barPulse: Math.exp(-barPhase * 6) * s,
      onBeat: false, onBar: false,
      stems, stemOnsets: on, stemPresence: pres,
      loudness: s * (0.55 + 0.25 * kick), complexity: this.silent ? 0.3 : 0.62, songComplexity: 0.6,
      chroma: this.chroma, keyTonic: 0, keyMode: 'major', keyHue: 0.58, keyChangePulse: 0,
      section: SECTION, sectionIndex: 1, sectionProgress: 0.3, sectionChanged: false, dropPulse: 0, buildIntensity: 0,
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
const BX = 16;
const BY = 9;

export interface ScreenMetrics {
  mean: number; // mean sRGB luma 0..1
  peak: number; // 99.7th percentile sRGB luma
  coverage: number; // fraction of pixels above 0.08
  motion: number; // mean abs luma change between 30 Hz samples
  motionSilent: number;
  beatCorr: number; // |corr| of luma / motion with the beat
  noise: number; // fraction of samples that look like noise
  flashArea: number; // max screen fraction flashing > 3 Hz
  cost: number; // estimated ms at 1440p
  msPerFrame: number; // measured wall time per screening frame
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
  flashDelta: 0.1, // relative (linear) luminance change counted as a transition
  flashDark: 0.8, // the darker state must be below this
  flashAreaMax: 0.03, // screen fraction; conservative so large monitors stay within WCAG 2.3.1
  flashPerSec: 3,
  reactRatio: 1.15,
  reactCorr: 0.12,
  msPerFrame: 10,
};

const srgbToLin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const LIN = new Float32Array(256).map((_, i) => srgbToLin(i / 255));

interface Run {
  frames: number;
  sample: number;
  blocks: Float32Array[]; // linear luminance per block per sample
  lumas: number[];
  motions: number[];
  beats: number[];
  noiseHits: number;
  samples: number;
  prev: Float32Array | null;
  ms: number;
  cover: number[];
  peaks: number[];
}

function newRun(): Run {
  return { frames: 0, sample: 0, blocks: [], lumas: [], motions: [], beats: [], noiseHits: 0, samples: 0, prev: null, ms: 0, cover: [], peaks: [] };
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

/** Count luminance transitions (with hysteresis) inside any 1 s window, per block. */
export function flashArea(series: Float32Array[], rate: number, delta = THRESH.flashDelta, dark = THRESH.flashDark, maxPerSec = THRESH.flashPerSec): number {
  if (series.length < 3) return 0;
  const nb = series[0].length;
  const win = Math.round(rate);
  let worst = 0;
  const flashing = new Uint8Array(nb);
  for (let b = 0; b < nb; b++) {
    const times: number[] = [];
    let lo = series[0][b], hi = series[0][b];
    let dir = 0; // 1 rising, -1 falling
    for (let i = 1; i < series.length; i++) {
      const v = series[i][b];
      if (dir >= 0) {
        if (v > hi) hi = v;
        if (hi - v >= delta && Math.min(v, hi) < dark) {
          times.push(i);
          dir = -1;
          lo = v;
        }
      }
      if (dir <= 0) {
        if (v < lo) lo = v;
        if (v - lo >= delta && lo < dark) {
          times.push(i);
          dir = 1;
          hi = v;
        }
      }
    }
    // More than maxPerSec flashes (2 transitions each) inside any 1 s window.
    for (let i = 0; i + maxPerSec * 2 < times.length; i++) {
      if (times[i + maxPerSec * 2] - times[i] < win) {
        flashing[b] = 1;
        break;
      }
    }
  }
  let n = 0;
  for (let b = 0; b < nb; b++) n += flashing[b];
  worst = n / nb;
  return worst;
}

export class Screener {
  readonly runner = new JobRunner();
  private stage: Stage;
  private thumbStage: Stage;
  private px = new Uint8Array(SCREEN_W * SCREEN_H * 4);
  private music = new SyntheticMusic();

  constructor(private eng: Engine) {
    this.stage = new StageClass(eng, { offscreen: true, particleCap: 16384, flameCap: 32768 });
    this.stage.resize(SCREEN_W, SCREEN_H);
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
      const metrics: ScreenMetrics = { mean: 0, peak: 0, coverage: 0, motion: 0, motionSilent: 0, beatCorr: 0, noise: 0, flashArea: 0, cost, msPerFrame: 0 };
      const fail = (reason: string, descriptor: number[] = []) => resolve({ ok: false, reason, metrics, descriptor });
      if (cost > COST_BUDGET_MS) {
        fail(`too slow (estimated ${cost.toFixed(1)} ms)`);
        return;
      }
      let compiled: boolean | null = null;
      const waitStep = this.whenCompiled(g, (ok) => (compiled = ok));
      let phase: 'compile' | 'music' | 'silent' | 'done' = 'compile';
      const st = this.stage;
      let slot: ReturnType<Stage['makeSlot']> | null = null;
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
      };

      const frame = (run: Run, isMusic: boolean) => {
        const t0 = performance.now();
        const state = this.music.next(1 / FPS);
        st.render(state, 1 / FPS, 'out');
        run.frames++;
        if (run.frames % SAMPLE_EVERY === 0) {
          st.readPixels(this.px);
          this.analyse(run, state.beatPulse, isMusic, lastLin, meanRGB, () => rgbN++);
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
          const res = this.verdict(g, music, silent, metrics, meanRGB, rgbN, lastPx);
          resolve(res);
          return true;
        }
        return true;
      });
    });
  }

  /** Per-sample statistics (luma, block luminance, motion, noise). */
  private analyse(run: Run, beat: number, isMusic: boolean, lastLin: Float32Array, meanRGB: number[], bump: () => void): void {
    const W = SCREEN_W, H = SCREEN_H;
    const px = this.px;
    const blocks = new Float32Array(BX * BY);
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
        blocks[Math.floor((y * BY) / H) * BX + Math.floor((x * BX) / W)] += L;
        if (x > 0) spatial += Math.abs(s - (0.2126 * px[i - 4] + 0.7152 * px[i - 3] + 0.0722 * px[i - 2]) / 255);
      }
    }
    const perBlock = (W / BX) * (H / BY);
    for (let b = 0; b < blocks.length; b++) blocks[b] /= perBlock;
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
    run.blocks.push(blocks);
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
    m.flashArea = flashArea(music.blocks, FPS / SAMPLE_EVERY);
    m.msPerFrame = music.ms / Math.max(1, music.frames);
    const descriptor = this.descriptor(meanRGB, rgbN, m, lastPx);
    const r = (reason: string): ScreenResult => ({ ok: false, reason, metrics: m, descriptor });
    if (m.msPerFrame > THRESH.msPerFrame) return r(`too slow (${m.msPerFrame.toFixed(1)} ms per screening frame)`);
    if (m.mean < THRESH.black && m.peak < THRESH.blackPeak) return r('nearly all black');
    if (m.mean > THRESH.white) return r('nearly all white');
    if (m.noise > THRESH.noiseFrac) return r('pure noise');
    if (m.flashArea > THRESH.flashAreaMax) return r(`flashing (${Math.round(m.flashArea * 100)}% of the screen above 3 flashes/s)`);
    const recent = lastHalf(music.motions);
    if (avg(recent) < THRESH.frozen && Math.max(...recent) < THRESH.frozen * 3 && m.beatCorr < THRESH.frozenCorr) return r('frozen');
    if (m.motion < m.motionSilent * THRESH.reactRatio && m.beatCorr < THRESH.reactCorr) return r('unreactive (same motion without music)');
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

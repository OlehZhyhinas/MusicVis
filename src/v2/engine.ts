// V2 renderer: draws any genome. A Stage owns a resolution (feedback pool,
// HDR scene, bloom, exposure, the shared fluid / particle / flame sims) and a
// Signals object (music -> per-frame values and analysis textures). The main
// view is one Stage with up to two Slots crossfading; screening and
// thumbnails use a second, small offscreen Stage on the same GL context so
// compiled programs are shared.

import type { MusicState, StemName } from '../types';
import { Bloom } from '../render/bloom';
import { Flame, type FlameSpec } from '../render/flame';
import { Fluid } from '../render/fluid';
import { Fullscreen, GL, PendingProgram, PingPong, Program, Target, TexFormat, canRenderTo, createTexture, formats } from '../render/gl';
import { Particles, type ParticleUpdate } from '../render/particles';
import { EXPOSURE_FS, FINAL_FS, FULLSCREEN_VS, SCALE_FS } from '../render/shaders';
import {
  CARRIER_SCHEMA, COLOR_SCHEMA, EMITTER_SCHEMAS, OP_SCHEMAS, clampParam, structuralKey,
  type EmitterGene, type FlameVar, type Genome, type Params, type Scheme, type Schema, type Signal,
} from './genome';
import { WAVE_FS, WAVE_VS, buildSources } from './glsl';

const TAU = Math.PI * 2;
const SPIN_WRAP = TAU * 16;
const WAVE_N = 512;
const SPEC_N = 128;
const STEMS: StemName[] = ['drums', 'bass', 'vocals', 'other'];
const MAX_SIDE = 2560;

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const smooth01 = (t: number) => {
  t = clamp01(t);
  return t * t * (3 - 2 * t);
};
const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const approach = (cur: number, target: number, rate: number, dt: number) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
function h11(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

// ------------------------------------------------------------- palette

const SCHEME_OFFSETS: Record<Scheme, number[]> = {
  analogous: [0, 0.13, -0.11, 1, 0.9, 1],
  complementary: [0, 0.5, 0.06, 1, 0.9, 1],
  triad: [0, 0.333, 0.667, 1, 0.9, 0.9],
  split: [0, 0.42, 0.58, 1, 0.9, 1],
  mono: [0, 0.03, -0.03, 1, 0.6, 1.2],
};
function hsvLin(h: number, s: number, v: number, out: Float32Array, o: number): void {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const rgb = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  for (let k = 0; k < 3; k++) out[o + k] = Math.pow(rgb[k], 2.2);
}
function paletteColors(scheme: Scheme, hue: number, sat: number, out: Float32Array): void {
  const s = SCHEME_OFFSETS[scheme];
  const sa = clamp01(sat);
  hsvLin(hue + s[0], sa, s[3], out, 0);
  hsvLin(hue + s[1], sa * (scheme === 'mono' ? 0.5 : 1), s[4], out, 3);
  hsvLin(hue + s[2], sa * 0.85, s[5], out, 6);
}

// ------------------------------------------------------------- signals

export interface Frame {
  time: number; dt: number; phase: number; act: number; cx: number; speed: number; spin: number;
  bars: number; beats: number; barIndex: number; beatIndex: number; barPhase: number; beatPhase: number;
  beatPulse: number; onBeat: boolean; stem: Float32Array; onset: Float32Array; gate: Float32Array;
  loud: number; melody: number; build: number; drop: number; keyTonic: number; minor: boolean;
  sectionIndex: number; aspect: number; hit: number; hitPulse: number; dropStart: boolean; keyHue: number; keyPulse: number;
}

/** Music state -> per-frame values and the waveform / spectrum textures (one per Stage). */
export class Signals {
  readonly F: Frame = {
    time: 0, dt: 0, phase: 0, act: 0.5, cx: 0.5, speed: 1, spin: 0, bars: 0, beats: 0, barIndex: 0, beatIndex: 0,
    barPhase: 0, beatPhase: 0, beatPulse: 0, onBeat: false, stem: new Float32Array(4), onset: new Float32Array(4),
    gate: new Float32Array(4), loud: 0, melody: 0.5, build: 0, drop: 0, keyTonic: 0, minor: false, sectionIndex: 0,
    aspect: 1, hit: 0, hitPulse: 0, dropStart: false, keyHue: 0, keyPulse: 0,
  };
  spinStep = 0;
  clock = 0;
  songCx: number | null = null;
  readonly waveTex: WebGLTexture;
  readonly specTex: WebGLTexture;
  readonly chroma = new Float32Array(12);
  private wave = new Float32Array(WAVE_N);
  private waveTmp = new Float32Array(WAVE_N);
  private spec = new Float32Array(SPEC_N);
  private rms = 0.1;
  private prevBarPhase = 0;
  private prevDrumOnset = 0;
  private hitCooldown = 0;

  constructor(private gl: GL) {
    const f = formats(gl);
    this.waveTex = createTexture(gl, WAVE_N, 1, f.r16f, gl.LINEAR);
    this.specTex = createTexture(gl, SPEC_N, 1, f.r16f, gl.LINEAR);
  }

  reset(): void {
    this.F.act = 0.5;
    this.F.phase = 0;
    this.F.spin = 0;
    this.clock = 0;
    this.wave.fill(0);
    this.spec.fill(0);
  }

  update(state: MusicState, sdt: number, aspect: number): void {
    const F = this.F;
    const dt = sdt;
    this.clock += dt;
    F.time = this.clock;
    F.dt = dt;
    F.aspect = aspect;
    F.cx = num(state.complexity, 0.5);
    const song = this.songCx ?? num(state.songComplexity, 0.5);
    for (let i = 0; i < 4; i++) {
      const n = STEMS[i];
      const g = smooth01((num(state.stemPresence?.[n], 0.5) - 0.03) / 0.25);
      F.gate[i] = g;
      F.stem[i] = Math.min(1, num(state.stems?.[n], 0)) * g;
      F.onset[i] = Math.min(1, num(state.stemOnsets?.[n], 0)) * g;
    }
    const target = clamp01((0.75 * F.cx + 0.25 * song - 0.08) / 0.72);
    F.act += (target - F.act) * (1 - Math.exp(-dt * 1.2));
    F.loud = num(state.loudness, 0);
    F.build = num(state.buildIntensity, 0);
    F.drop = num(state.dropPulse, 0);
    F.beatPulse = num(state.beatPulse, 0);
    F.onBeat = !!state.onBeat;
    F.keyTonic = num(state.keyTonic, 0);
    F.minor = state.keyMode === 'minor';
    F.keyHue = num(state.keyHue, 0);
    F.keyPulse = num(state.keyChangePulse, 0);
    F.sectionIndex = Math.max(0, num(state.sectionIndex, 0));
    F.dropStart = !!state.sectionChanged && state.section?.label === 'drop';
    F.speed = (0.4 + 0.75 * F.act) * (1 + 0.5 * F.build) + 0.4 * F.drop * F.act;
    F.phase = (F.phase + dt * F.speed) % 4096;

    const bpm = num(state.bpm, 120) || 120;
    F.barIndex = num(state.barIndex, -1);
    F.beatIndex = num(state.beatIndex, -1);
    F.barPhase = num(state.barPhase, 0);
    F.beatPhase = num(state.beatPhase, 0);
    if (F.barIndex >= 0) F.bars = (F.barIndex + F.barPhase) % 48;
    else F.bars = (F.bars + (dt * bpm) / 240) % 48;
    if (F.beatIndex >= 0) F.beats = (F.beatIndex + F.beatPhase) % 256;
    else F.beats = (F.beats + (dt * bpm) / 60) % 256;

    let dBar = F.barPhase - this.prevBarPhase;
    if (dBar < 0) dBar += 1;
    if (!state.playing || dBar > 0.5) dBar = (bpm / 240) * dt * (state.playing ? 1 : 0.3);
    this.prevBarPhase = F.barPhase;
    const spinMul = 0.5 + 0.5 * smooth01((F.act - 0.2) / 0.3);
    this.spinStep = TAU * dBar * spinMul;
    F.spin = (F.spin + this.spinStep) % SPIN_WRAP;

    this.processAudio(state, dt);

    F.hit = 0;
    this.hitCooldown -= dt;
    const on = F.onset[0];
    const edge = on > 0.35 && this.prevDrumOnset <= 0.35;
    this.prevDrumOnset = on;
    if (state.playing && this.hitCooldown <= 0 && F.gate[0] > 0.2 && (edge || (F.onBeat && F.stem[0] > 0.25))) {
      F.hit = Math.max(on, 0.3) * (0.4 + F.stem[0]);
      this.hitCooldown = 0.1;
    }
    F.hitPulse = Math.max(F.hit, F.hitPulse * Math.exp(-dt * 8));
  }

  private processAudio(state: MusicState, dt: number): void {
    const gl = this.gl;
    const wf = state.waveform;
    if (wf && wf.length >= WAVE_N * 2) {
      const half = wf.length - WAVE_N;
      let start = 0;
      for (let i = 1; i < half; i++) {
        if (wf[i - 1] <= 0 && wf[i] > 0) {
          start = i;
          break;
        }
      }
      let e = 0;
      for (let j = 0; j < WAVE_N; j++) {
        const v = wf[start + j];
        this.waveTmp[j] = v;
        e += v * v;
      }
      this.rms += (Math.sqrt(e / WAVE_N) - this.rms) * 0.05;
      const gain = Math.pow(Math.min(4, 0.2 / Math.max(this.rms, 0.012)), 0.6);
      for (let j = 0; j < WAVE_N; j++) {
        const a = this.waveTmp[Math.max(0, j - 1)];
        const b = this.waveTmp[j];
        const c = this.waveTmp[Math.min(WAVE_N - 1, j + 1)];
        this.wave[j] += ((a + 2 * b + c) * 0.25 * gain - this.wave[j]) * 0.5;
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.waveTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WAVE_N, 1, gl.RED, gl.FLOAT, this.wave);

    const sp = state.spectrum;
    if (sp && sp.length >= 64) {
      const n = sp.length;
      const top = Math.min(n - 1, Math.round(n * 0.78));
      for (let k = 0; k < SPEC_N; k++) {
        const a = Math.floor(Math.pow(top, k / SPEC_N));
        const b = Math.max(a + 1, Math.floor(Math.pow(top, (k + 1) / SPEC_N)));
        let s = 0;
        for (let i = a; i < b; i++) s += sp[i];
        const v = clamp01((s / (b - a) - 0.28) / 0.6);
        const cur = this.spec[k];
        this.spec[k] = cur + (v - cur) * (v > cur ? 0.5 : 0.12);
      }
      const lo = Math.max(2, Math.round(n * 0.006));
      const hi = Math.max(lo + 4, Math.round(n * 0.12));
      let wsum = 0;
      let lsum = 0;
      for (let i = lo; i < hi; i++) {
        const w = Math.max(0, sp[i] - 0.3);
        const w3 = w * w * w;
        wsum += w3;
        lsum += w3 * Math.log2(i);
      }
      if (wsum > 1e-5) {
        const m = (lsum / wsum - Math.log2(lo)) / (Math.log2(hi) - Math.log2(lo));
        this.F.melody += (clamp01(m) - this.F.melody) * (1 - Math.exp(-dt * 5));
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.specTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SPEC_N, 1, gl.RED, gl.FLOAT, this.spec);
    if (state.chroma && state.chroma.length >= 12) for (let i = 0; i < 12; i++) this.chroma[i] = num(state.chroma[i], 0);
  }

  signal(s: Signal): number {
    const F = this.F;
    switch (s) {
      case 'drums': return F.stem[0];
      case 'bass': return F.stem[1];
      case 'vocals': return F.stem[2];
      case 'other': return F.stem[3];
      case 'hit': return F.hitPulse;
      case 'beat': return F.beatPulse * Math.max(0.3, F.gate[0]);
      case 'bar': return 0.5 + 0.5 * Math.sin(TAU * F.barPhase);
      case 'complexity': return F.act;
      case 'drop': return F.drop;
      case 'loud': return F.loud;
      case 'melody': return F.melody;
      case 'build': return F.build;
    }
  }

  melodic(): number {
    const F = this.F;
    return Math.max(F.stem[2], F.stem[3] * 0.85, F.loud * 0.35);
  }

  dispose(): void {
    this.gl.deleteTexture(this.waveTex);
    this.gl.deleteTexture(this.specTex);
  }
}

// ------------------------------------------------------- program cache

export interface GenomePrograms {
  feedback: Program;
  composite: Program;
}
interface CacheEntry {
  pending: [PendingProgram, PendingProgram];
  done: GenomePrograms | null;
  failed: string | null;
  used: number;
}

export class ProgramCache {
  private entries = new Map<string, CacheEntry>();
  private parallel: boolean;
  private tick = 0;

  constructor(private gl: GL) {
    this.parallel = !!gl.getExtension('KHR_parallel_shader_compile');
  }

  request(g: Genome): string {
    const key = structuralKey(g);
    const e = this.entries.get(key);
    if (e) {
      e.used = ++this.tick;
      return key;
    }
    const src = buildSources(g);
    this.entries.set(key, {
      pending: [
        new PendingProgram(this.gl, FULLSCREEN_VS, src.feedback, `v2-fb ${key}`, this.parallel),
        new PendingProgram(this.gl, FULLSCREEN_VS, src.composite, `v2-comp ${key}`, this.parallel),
      ],
      done: null,
      failed: null,
      used: ++this.tick,
    });
    this.evict();
    return key;
  }

  /** Programs if linked, null while compiling or failed. */
  get(g: Genome, now = false): GenomePrograms | null {
    const key = this.request(g);
    const e = this.entries.get(key)!;
    if (e.done || e.failed) return e.done;
    const a = e.pending[0].poll(now);
    const b = e.pending[1].poll(now);
    if (!a || !b) return null;
    const err = e.pending[0].error ?? e.pending[1].error;
    if (err) {
      console.error(err);
      e.failed = err;
      e.pending[0].program?.dispose();
      e.pending[1].program?.dispose();
      return null;
    }
    e.done = { feedback: e.pending[0].program!, composite: e.pending[1].program! };
    return e.done;
  }

  failed(g: Genome): string | null {
    return this.entries.get(structuralKey(g))?.failed ?? null;
  }

  get size(): number {
    return this.entries.size;
  }

  private evict(): void {
    if (this.entries.size <= 80) return;
    const list = [...this.entries.entries()].sort((a, b) => a[1].used - b[1].used);
    for (const [k, e] of list.slice(0, this.entries.size - 64)) {
      if (!e.done && !e.failed) continue;
      e.done?.feedback.dispose();
      e.done?.composite.dispose();
      this.entries.delete(k);
    }
  }

  /** Keep entries in use from being evicted. */
  touch(g: Genome): void {
    const e = this.entries.get(structuralKey(g));
    if (e) e.used = ++this.tick;
  }

  dispose(): void {
    for (const e of this.entries.values()) {
      e.done?.feedback.dispose();
      e.done?.composite.dispose();
    }
    this.entries.clear();
  }
}

// ---------------------------------------------------------------- slots

/** One genome running at a Stage: its feedback buffer and JS-side state. */
export class Slot {
  weight = 1;
  mem: Record<string, number> = {};
  readonly cols = new Float32Array(9);
  readonly opA = new Float32Array(24);
  readonly opB = new Float32Array(24);
  readonly em = new Float32Array(48);
  readonly ink = new Float32Array(24);
  readonly blob = new Float32Array(24);
  readonly seg = new Float32Array(48 * 4);
  readonly segZ = new Float32Array(48);
  segN = 0;
  readonly waveU = new Float32Array(12);
  waveBright = 0;
  waveThick = 1.5;
  waveN = 512;
  rem: number[] = [];
  shift = [0, 0];
  decay = 0.9;
  delta = new Map<string, number>();
  flameSpec: FlameSpec | null = null;
  flameMorph = 0;

  constructor(
    readonly genome: Genome,
    readonly progs: GenomePrograms,
    readonly fb: PingPong,
  ) {}

  /** Parameter with this frame's reactions applied, clamped to its spec. */
  P(group: 'op' | 'em' | 'car' | 'col', i: number, p: Params, k: string, schema: Schema): number {
    const base = p[k];
    const d = this.delta.get(`${group}${i}.${k}`);
    if (d === undefined) return base;
    const s = schema[k];
    return clampParam(base + d * (s.max - s.min) * 0.5, s);
  }
}

// ---------------------------------------------------------------- solids

type Solid = { v: number[][]; e: [number, number][] };
function edgesByLength(v: number[][]): [number, number][] {
  let min = Infinity;
  const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) min = Math.min(min, d(v[i], v[j]));
  const e: [number, number][] = [];
  for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) if (d(v[i], v[j]) < min * 1.01) e.push([i, j]);
  return e;
}
const PHI = (1 + Math.sqrt(5)) / 2;
const SOLIDS: Solid[] = (() => {
  const nrm = (v: number[][]) => v.map((p) => {
    const l = Math.hypot(p[0], p[1], p[2]);
    return [p[0] / l, p[1] / l, p[2] / l];
  });
  const cube: number[][] = [];
  for (let i = 0; i < 8; i++) cube.push([i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1]);
  const oct = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const ico: number[][] = [];
  for (const a of [-1, 1]) for (const b of [-PHI, PHI]) ico.push([0, a, b], [a, b, 0], [b, 0, a]);
  const tet = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
  return [tet, cube, oct, ico].map((v) => {
    const n = nrm(v);
    return { v: n, e: edgesByLength(n) };
  });
})();
const solidTmp = new Float32Array(3 * 12);

function projectSolid(s: Solid, sl: Slot, start: number, ang: number, tilt: number, scale: number, weight: number): number {
  const ca = Math.cos(ang), sa = Math.sin(ang), ct = Math.cos(tilt), st = Math.sin(tilt);
  for (let i = 0; i < s.v.length; i++) {
    const [x0, y0, z0] = s.v[i];
    const x1 = x0 * ca + z0 * sa;
    const z1 = -x0 * sa + z0 * ca;
    const y2 = y0 * ct - z1 * st;
    const z2 = y0 * st + z1 * ct;
    const k = 1 / (1 + z2 * 0.3);
    solidTmp[i * 3] = x1 * scale * k;
    solidTmp[i * 3 + 1] = y2 * scale * k;
    solidTmp[i * 3 + 2] = z2;
  }
  let n = start;
  for (const [a, b] of s.e) {
    if (n >= 48) break;
    sl.seg[n * 4] = solidTmp[a * 3];
    sl.seg[n * 4 + 1] = solidTmp[a * 3 + 1];
    sl.seg[n * 4 + 2] = solidTmp[b * 3];
    sl.seg[n * 4 + 3] = solidTmp[b * 3 + 1];
    sl.segZ[n] = weight * (0.35 + 0.65 * clamp01(0.5 - (solidTmp[a * 3 + 2] + solidTmp[b * 3 + 2]) * 0.25));
    n++;
  }
  return n;
}

function polygonSegs(sl: Slot, start: number, sides: number, r: number, ang: number, weight: number): number {
  let n = start;
  for (let i = 0; i < sides && n < 48; i++, n++) {
    const a0 = ang + (i / sides) * TAU;
    const a1 = ang + ((i + 1) / sides) * TAU;
    sl.seg[n * 4] = Math.cos(a0) * r;
    sl.seg[n * 4 + 1] = Math.sin(a0) * r;
    sl.seg[n * 4 + 2] = Math.cos(a1) * r;
    sl.seg[n * 4 + 3] = Math.sin(a1) * r;
    sl.segZ[n] = weight;
  }
  return n;
}

// ---------------------------------------------------------------- stage

export interface StageOptions {
  /** Render the final image into an RGBA8 target instead of the canvas. */
  offscreen: boolean;
  particleCap: number;
  flameCap: number;
}

export interface PostParams {
  bloom: number;
  exposure: number;
  vignette: number;
  adapt: number;
  ca: number;
  contrast: number;
}

export class Stage {
  w = 0;
  h = 0;
  readonly sig: Signals;
  slots: Slot[] = [];
  scene: Target | null = null;
  out: Target | null = null;
  private bloom: Bloom;
  private avgLum: PingPong;
  fluid: Fluid | null = null;
  particles: Particles | null = null;
  flame: Flame | null = null;
  private pu: ParticleUpdate;
  flash = 0;
  frame = 0;

  constructor(private eng: Engine, readonly opts: StageOptions) {
    const gl = eng.gl;
    this.sig = new Signals(gl);
    this.bloom = new Bloom(gl, eng.fs);
    this.avgLum = new PingPong(gl, 1, 1, [eng.hdr], gl.NEAREST);
    this.pu = {
      dt: 0, time: 0, aspect: 1, velocity: eng.black, simTexelX: 0, simTexelY: 0, wave: this.sig.waveTex,
      fluidAmt: 0, curl: 0, zoomFlow: 0, rotFlow: 0, converge: 0, drag: 3, lifeRate: 0.3, speed: 0.5,
      spawnFrom: 0, spawnTo: 0, spawnMix: 0, emitCount: 3, emitAngle: 0, emitRadius: 0.3,
      burst: 0, burstSeed: 0, burstSpeed: 0, liftX: 0, liftY: 0, spread: 0.01,
    };
  }

  resize(w: number, h: number): void {
    w = Math.max(2, Math.round(w));
    h = Math.max(2, Math.round(h));
    if (w === this.w && h === this.h && this.scene) return;
    const gl = this.eng.gl;
    this.w = w;
    this.h = h;
    this.scene?.dispose();
    this.scene = new Target(gl, w, h, [this.eng.hdr], gl.LINEAR);
    this.bloom.resize(w, h, this.eng.hdr);
    if (this.opts.offscreen) {
      this.out?.dispose();
      this.out = new Target(gl, w, h, [formats(gl).rgba8], gl.LINEAR);
    }
    this.fluid?.resize(w / h);
    for (const s of this.slots) {
      s.fb.dispose();
      (s as { fb: PingPong }).fb = this.makeFb();
    }
  }

  makeFb(): PingPong {
    return new PingPong(this.eng.gl, this.w, this.h, [this.eng.hdr], this.eng.gl.LINEAR);
  }

  /** New slot for a genome (programs must be ready). */
  makeSlot(g: Genome, progs: GenomePrograms): Slot {
    const s = new Slot(g, progs, this.makeFb());
    return s;
  }

  disposeSlot(s: Slot): void {
    s.fb.dispose();
    this.slots = this.slots.filter((x) => x !== s);
  }

  /** Copy the current scene (scaled) into a slot's feedback so it morphs out of it. */
  seedFrom(s: Slot, value: number): void {
    const gl = this.eng.gl;
    if (!this.scene || s.genome.carrier.kind === 'none') return;
    gl.disable(gl.BLEND);
    s.fb.read.bind();
    this.eng.pSeed.use().tex('uTex', this.scene.t).f1('uValue', value);
    this.eng.fs.draw();
  }

  // ------------------------------------------------------------ frame

  render(state: MusicState, dt: number, target: 'canvas' | 'out' = 'canvas'): void {
    const eng = this.eng;
    const gl = eng.gl;
    const sdt = state.playing ? dt : dt * 0.25;
    this.frame++;
    this.sig.update(state, sdt, this.w / this.h);
    const F = this.sig.F;
    this.flash *= Math.exp(-sdt * 3);
    if (F.dropStart) this.flash = 0.4 + 0.6 * F.act;

    const slots = this.slots.filter((s) => s.weight > 0.001);
    for (const s of slots) this.tick(s, sdt);
    const dom = slots.reduce<Slot | null>((a, b) => (!a || b.weight > a.weight ? b : a), null);

    // Fluid, particles, flame: owned by the heaviest slot that uses them.
    const fluidSlot = slots.filter((s) => s.genome.carrier.kind === 'fluid').sort((a, b) => b.weight - a.weight)[0];
    if (fluidSlot && eng.hq) {
      if (!this.fluid) {
        this.fluid = new Fluid(gl, eng.fs);
        this.fluid.resize(this.w / this.h);
      }
      const cp = fluidSlot.genome.carrier.p;
      this.fluid.dissipation = 0.6;
      this.fluid.step(sdt, this.sig.clock, cp.fnoise * (0.3 + 0.7 * F.act) * (0.3 + 0.7 * F.stem[3]) * 0.5 * sdt * 60, cp.vort, F.aspect);
    }
    const partSlot = slots.filter((s) => s.genome.emitters.some((e) => e.kind === 'particles')).sort((a, b) => b.weight - a.weight)[0] ?? null;
    if (partSlot && eng.hq) this.updateParticles(partSlot, sdt, !!fluidSlot);
    const flameSlot = slots.filter((s) => s.flameSpec).sort((a, b) => b.weight - a.weight)[0] ?? null;
    if (flameSlot && eng.hq) {
      if (!this.flame) this.flame = new Flame(gl, eng.fs);
      const spec = flameSlot.flameSpec!;
      const n = Math.min(spec.count, this.opts.flameCap);
      if (Math.abs(this.flame.count - n) > n * 0.1) this.flame.setCount(n);
      const fe = flameSlot.genome.emitters.find((e) => e.kind === 'flame')!;
      this.flame.configure(spec, {
        spin: F.spin, bass: F.stem[1], vocals: F.stem[2], morph: flameSlot.flameMorph, hue: fe.p.hue, bars: F.bars, beat: F.beatPulse,
      });
    }

    for (const s of slots) this.feedbackPass(s, sdt, s === partSlot, s === flameSlot);

    // Composite every live slot into the HDR scene.
    this.scene!.bind();
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (const s of slots) {
      const g = s.genome;
      const p = s.progs.composite.use();
      this.setCommon(p, s, sdt, false);
      p.tex('uFb', s.fb.read.t)
        .f1('uWeight', s.weight)
        .f1('uSat', 1 - 0.45 * F.build)
        .f1('uSweep', F.keyPulse)
        .f1('uReflectY', g.color.p.reflectY);
      eng.fs.draw();
    }
    for (const s of slots) {
      for (const e of s.genome.emitters) {
        if (e.layer !== 'top') continue;
        if (e.kind === 'wave') this.drawWave(s, s.weight, sdt, false);
        if (e.kind === 'particles' && s === partSlot) this.drawParticles(s, e, s.weight);
      }
    }
    gl.disable(gl.BLEND);

    // Post parameters crossfade with the slot weights.
    const post: PostParams = { bloom: 0, exposure: 0, vignette: 0, adapt: 0, ca: 0, contrast: 0 };
    let wsum = 0;
    for (const s of slots) wsum += s.weight;
    for (const s of slots) {
      const c = s.genome.color.p;
      const w = s.weight / Math.max(wsum, 1e-3);
      post.bloom += s.P('col', 0, c, 'bloom', COLOR_SCHEMA) * w;
      post.exposure += s.P('col', 0, c, 'exposure', COLOR_SCHEMA) * w;
      post.vignette += s.P('col', 0, c, 'vignette', COLOR_SCHEMA) * w;
      post.adapt += c.adapt * w;
      post.ca += s.P('col', 0, c, 'ca', COLOR_SCHEMA) * w;
      post.contrast += c.contrast * w;
    }
    if (!slots.length) Object.assign(post, { bloom: 1, exposure: 1, vignette: 0.45, adapt: 0.3, ca: 0, contrast: 0.03 });
    void dom;
    this.post(post, target, state.dt);
  }

  private tick(s: Slot, sdt: number): void {
    const g = s.genome;
    const F = this.sig.F;
    const sig = this.sig;
    const f60 = sdt * 60;
    const A = F.aspect * 0.5;

    // Reactions -> per-parameter deltas for this frame.
    s.delta.clear();
    for (const r of g.reactions) {
      const key = `${r.g}${r.i}.${r.k}`;
      s.delta.set(key, (s.delta.get(key) ?? 0) + r.gain * sig.signal(r.src));
    }

    // Palette
    const cp = g.color.p;
    const sat = s.P('col', 0, cp, 'sat', COLOR_SCHEMA) * (0.78 + 0.3 * F.stem[2]) * (1 - 0.5 * F.build);
    paletteColors(g.color.scheme, F.keyHue + s.P('col', 0, cp, 'hue', COLOR_SCHEMA), sat, s.cols);

    // Carrier decay
    const car = g.carrier;
    const half = car.p.halfLife;
    s.decay = car.kind === 'none' ? 0 : Math.pow(0.5, sdt / half);

    // Chain ops
    s.shift[0] = s.shift[1] = 0;
    g.chain.forEach((o, i) => {
      const sch = OP_SCHEMAS[o.op];
      const P = (k: string) => s.P('op', i, o.p, k, sch);
      const a = s.opA;
      const j = i * 4;
      a[j] = a[j + 1] = a[j + 2] = a[j + 3] = 0;
      const wander = (w: number) => {
        // Every wandering op follows the same bar-locked Lissajous path.
        a[j] = P('cx') + w * A * 0.64 * Math.sin((TAU * F.bars) / 8);
        a[j + 1] = P('cy') + w * 0.63 * Math.sin((TAU * F.bars) / 6 + 1);
      };
      switch (o.op) {
        case 'zoom':
          wander(P('wander'));
          a[j + 2] = P('rate') * o.w * f60 * F.speed;
          a[j + 3] = P('radial');
          break;
        case 'rotate': {
          a[j] = P('cx');
          a[j + 1] = P('cy');
          const sign = o.p.alt > 0.5 && F.barIndex % 2 === 1 ? -1 : 1;
          a[j + 2] = -(o.p.lock * sig.spinStep + P('rate') * o.w * f60 * F.speed) * sign;
          break;
        }
        case 'translate': {
          const v = [P('vx'), P('vy')];
          while (s.rem.length < 12) s.rem.push(0);
          for (let k = 0; k < 2; k++) {
            const px = -v[k] * o.w * F.speed * sdt * this.h + s.rem[i * 2 + k];
            const n = Math.round(px);
            s.rem[i * 2 + k] = px - n;
            a[j + k] = n / this.h;
            s.shift[k] += n / this.h;
          }
          break;
        }
        case 'swirl':
          wander(P('wander'));
          a[j + 2] = P('amt') * o.w * f60 * F.speed;
          a[j + 3] = P('k');
          break;
        case 'twist':
          a[j] = P('cx');
          a[j + 1] = P('cy');
          a[j + 2] = P('amt') * o.w * f60 * F.speed;
          break;
        case 'ripple': {
          const ph = (s.mem['rp' + i] = ((s.mem['rp' + i] ?? 0) + sdt * P('speed') * F.speed * TAU) % 4096);
          a[j] = P('amp') * o.w * f60;
          a[j + 1] = P('freq');
          a[j + 2] = ph;
          a[j + 3] = o.p.radial;
          break;
        }
        case 'noise': {
          const ph = (s.mem['np' + i] = ((s.mem['np' + i] ?? 0) + sdt * P('speed') * F.speed) % 4096);
          a[j] = P('amp') * o.w * f60 * F.speed;
          a[j + 1] = P('scale');
          a[j + 2] = ph;
          break;
        }
        case 'mirror':
          a[j] = o.p.axis;
          break;
        case 'tile':
          a[j] = P('n');
          break;
        case 'polar':
          a[j] = P('scale');
          a[j + 1] = F.spin * o.p.lock;
          break;
        case 'kaleido':
          a[j] = o.p.n;
          a[j + 1] = F.spin * o.p.lock;
          break;
        default: // flame variations
          a[j] = Math.min(1, o.w * (o.stage === 'warp' ? 0.1 * f60 : 1));
          a[j + 1] = P('s');
      }
    });

    // Emitters
    s.flameSpec = null;
    s.waveBright = 0;
    g.emitters.forEach((e, slot) => this.tickEmitter(s, e, slot, sdt));
  }

  private tickEmitter(s: Slot, e: EmitterGene, slot: number, sdt: number): void {
    const F = this.sig.F;
    const sig = this.sig;
    const sch = EMITTER_SCHEMAS[e.kind];
    const P = (k: string) => s.P('em', slot, e.p, k, sch);
    const m = s.mem;
    const E = s.em;
    const o = slot * 16;
    E.fill(0, o, o + 16);
    const gain = P('gain');
    E[o] = gain;
    E[o + 1] = P('hue');
    const A = F.aspect * 0.5;
    const key = (k: string) => `${e.kind}.${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);

    switch (e.kind) {
      case 'spectrum':
        E[o + 2] = P('x'); E[o + 3] = P('y');
        E[o + 4] = e.p.mode; E[o + 5] = e.p.bins; E[o + 6] = P('radius'); E[o + 7] = P('len');
        E[o + 8] = P('fill');
        break;
      case 'stars': {
        m[key('drift')] = (mm('drift') + sdt * F.speed * P('drift') * 6) % 1000;
        E[o + 2] = m[key('drift')];
        E[o + 3] = Math.max(sig.melodic(), 0.4 * F.stem[1]) * 1.2;
        E[o + 4] = P('density'); E[o + 5] = P('scale'); E[o + 6] = P('links'); E[o + 7] = P('twinkle');
        break;
      }
      case 'ink': {
        const n = e.p.count;
        E[o + 2] = n;
        const orbit = P('orbit');
        const R = P('radius');
        const wv = P('wander');
        const size = P('size');
        const force = P('force');
        const stations = [[-0.55 * A, -0.2], [0, -0.28], [0, 0.18], [0.55 * A, -0.05], [-0.3 * A, 0.25], [0.3 * A, 0.25]];
        const fl = this.fluid && s.genome.carrier.kind === 'fluid' ? this.fluid : null;
        for (let i = 0; i < n; i++) {
          const st = stations[i % 6];
          const row = P('row');
          const rx = (i - (n - 1) / 2) * (1.2 / Math.max(1, n)) * A + 0.05 * Math.sin(F.phase * 0.4 + i * 2.1);
          const sx = st[0] + (rx - st[0]) * row + wv * Math.sin(F.phase * 0.23 + i * 1.9);
          const sy = st[1] + (-0.5 - st[1]) * row + wv * 0.7 * Math.sin(F.phase * 0.31 + i * 2.7) * (1 - row);
          const ang = F.spin * 0.5 + (i / n) * TAU;
          const ox = R * Math.cos(ang), oy = R * Math.sin(ang);
          const x = sx + (ox - sx) * orbit;
          const y = sy + (oy - sy) * orbit;
          const lvl = i % 4 === 0 ? Math.max(F.onset[0], F.stem[0] * 0.5) : F.stem[i % 4];
          const strength = lvl * (1 - orbit) + (Math.max(F.stem[3], F.stem[2]) * 0.8 + 0.2 * F.loud) * orbit;
          s.ink[i * 4] = x;
          s.ink[i * 4 + 1] = y;
          s.ink[i * 4 + 2] = strength;
          s.ink[i * 4 + 3] = size * (0.7 + 0.8 * strength);
          if (fl && force > 0 && strength > 0.04 && (i % 4 > 0 || F.hit > 0)) {
            const a = F.spin * 0.5 + i * (TAU / 4);
            const Fm = (i % 4 === 0 ? 1400 * F.hit : 420 * strength) * (0.4 + 0.6 * F.act) * force;
            fl.splat(x / F.aspect + 0.5, y + 0.5, Math.cos(a) * Fm, Math.sin(a) * Fm, 0.002, 0);
          }
        }
        break;
      }
      case 'wire': {
        E[o + 2] = P('thick');
        const solid = e.p.solid === 5 ? ((F.sectionIndex % 4) + 4) % 4 : e.p.solid;
        const beat = F.beatPulse * F.gate[0];
        const scale = P('scale') + 0.07 * F.stem[1] + 0.02 * beat;
        const ang = F.spin * e.p.lock * 4;
        let n = 0;
        if (solid === 4) {
          n = polygonSegs(s, 0, e.p.sides, scale * 0.5, ang, 1);
        } else {
          const tilt = P('tilt') + 0.25 * Math.sin(F.phase * 0.07);
          n = projectSolid(SOLIDS[solid], s, 0, ang, tilt, scale, 1);
          const inner = clamp01((F.act - 0.45) / 0.25) * P('inner');
          if (inner > 0.01) n = projectSolid(SOLIDS[(solid + 2) % 4], s, n, -ang * 0.5, -tilt, scale * 0.45, inner * 0.7);
        }
        s.segN = n;
        break;
      }
      case 'plasma': {
        m[key('t')] = (mm('t') + sdt * F.speed * P('speed') * 0.66) % 256;
        m[key('l')] = (mm('l') + sdt * (0.15 + 1.6 * F.beatPulse * F.gate[0]) * F.speed) % 256;
        E[o + 2] = m[key('t')]; E[o + 3] = m[key('l')];
        E[o + 4] = P('scale'); E[o + 5] = P('warp'); E[o + 6] = P('bands'); E[o + 7] = P('lines');
        break;
      }
      case 'aurora': {
        m[key('t')] = (mm('t') + sdt * F.speed * 0.15) % 512;
        E[o + 2] = Math.max(F.stem[2], F.stem[3] * 0.7 * (1 - F.gate[2])) + 0.08 * F.loud;
        E[o + 3] = m[key('t')];
        E[o + 4] = P('y'); E[o + 5] = P('fall'); E[o + 6] = P('rays'); E[o + 7] = P('wav');
        break;
      }
      case 'blobs': {
        const n = e.p.count;
        E[o + 2] = n;
        E[o + 3] = P('chrome');
        const src = [1, 1, 0, 2, 3, 3];
        const size = P('size');
        const spread = P('spread');
        const speed = P('speed');
        m[key('ph')] = (mm('ph') + sdt * F.speed) % 4096;
        const ph = m[key('ph')];
        for (let i = 0; i < n; i++) {
          const t = ph * (speed * 0.9 + 0.03 * i);
          const lv = src[i] === 0 ? F.onset[0] * 0.6 + F.stem[0] * 0.4 : F.stem[src[i]];
          const tr = size * (0.75 + 1.25 * lv) + 0.015 * F.beatPulse * F.gate[0];
          const rr = approach(mm('r' + i, tr), tr, 8, sdt);
          m[key('r' + i)] = rr;
          s.blob[i * 4] = spread * A * Math.sin(t + i * 1.7) * (0.6 + 0.4 * Math.sin(t * 0.37 + i));
          s.blob[i * 4 + 1] = 0.26 * (spread / 0.55) * Math.sin(t * 1.31 + i * 2.3);
          s.blob[i * 4 + 2] = rr;
        }
        break;
      }
      case 'edge': {
        const side = e.p.side;
        const perp = side === 0 || side === 3 ? Math.abs(s.shift[0]) : Math.abs(s.shift[1]);
        E[o + 2] = perp + 2 / this.h;
        E[o + 4] = e.p.mode; E[o + 5] = side; E[o + 6] = P('base'); E[o + 7] = P('height');
        E[o + 8] = P('density');
        if (e.p.mode === 0) {
          const b = F.beatIndex >= 0 ? F.beatIndex : Math.floor(F.time * 2);
          let start = mm('start', b);
          let wide = mm('wide', 1);
          if (b >= start + wide || b < start) {
            start = m[key('start')] = b;
            wide = m[key('wide')] = h11(b * 1.7) < 0.3 ? 2 : 1;
            const lv = 0.6 * F.loud + 0.4 * Math.max(F.stem[1], F.stem[3]);
            m[key('h')] = (0.04 + P('height') * Math.pow(h11(b * 3.3), 1.6) + 0.12 * lv) * (0.6 + 0.4 * F.act);
            m[key('cols')] = 3 + Math.floor(h11(b * 5.1) * 3) * wide;
            m[key('lit')] = (0.08 + 0.3 * Math.max(F.stem[3], F.stem[2]) + 0.15 * F.act) * (0.5 + P('density'));
          }
          const frac = (b - start + (F.beatIndex >= 0 ? F.beatPhase : (F.time * 2) % 1)) / wide;
          E[o + 9] = mm('cols', 4);
          E[o + 12] = mm('h', 0.1); E[o + 13] = start; E[o + 14] = mm('lit', 0.3); E[o + 15] = frac;
        } else if (e.p.mode === 1) {
          const y = (F.melody - 0.5) * 0.62 + P('base') * 0.3;
          const prev = mm('y', y);
          const yy = approach(prev, y, 6, sdt);
          m[key('y')] = yy;
          E[o] = gain * sig.melodic() * 1.2;
          E[o + 10] = yy; E[o + 11] = prev;
        } else if (e.p.mode === 3) {
          const tgt = clamp01(0.6 * F.loud + 0.4 * F.stem[1] + 0.15 * Math.sin(F.phase * 0.4));
          m[key('th')] = approach(mm('th', tgt), tgt, 3, sdt);
          E[o + 10] = m[key('th')];
        }
        break;
      }
      case 'tiles':
        E[o + 2] = Math.max(sig.melodic(), 0.3 * F.stem[1]);
        E[o + 3] = F.spin * e.p.lock;
        E[o + 4] = e.p.shape; E[o + 5] = P('scale'); E[o + 6] = P('lit'); E[o + 7] = P('edges');
        break;
      case 'horizon': {
        const prev = mm('lb', F.beats);
        let d = F.beats - prev;
        if (d < 0) d += 256;
        if (d > 2) d = sdt * 2;
        m[key('lb')] = F.beats;
        m[key('sc')] = (mm('sc') + d * P('speed') * (0.5 + 0.5 * F.act)) % 1;
        E[o + 2] = m[key('sc')];
        E[o + 4] = P('y'); E[o + 5] = P('density'); E[o + 6] = P('peaks');
        break;
      }
      case 'orb': {
        m[key('halo')] = approach(mm('halo'), Math.max(F.loud, sig.melodic()), 3, sdt);
        E[o + 2] = P('x'); E[o + 3] = P('y') + 0.03 * Math.sin(F.phase * 0.01);
        E[o + 4] = P('radius') * (1 + 0.08 * F.stem[1]); E[o + 5] = P('halo'); E[o + 6] = P('stripes'); E[o + 7] = P('craters');
        E[o + 8] = m[key('halo')];
        break;
      }
      case 'wave': {
        const u = s.waveU;
        m[key('ph')] = (mm('ph') + sdt * F.speed * 0.07) % 4096;
        u[0] = e.p.shape; u[1] = P('amp'); u[2] = P('x'); u[3] = P('y');
        u[4] = P('radius'); u[5] = P('turns'); u[6] = e.p.ra + 0.004 * Math.sin(F.phase * 0.05); u[7] = e.p.rb * (F.minor ? 0.75 : 1) - 0.005;
        u[8] = m[key('ph')]; u[9] = F.spin * 0.0625; u[10] = P('hue'); u[11] = 0;
        s.waveBright = gain * 0.5 * (0.3 + 0.9 * Math.max(F.loud, 0.6 * sig.melodic()));
        s.waveThick = P('thick');
        s.waveN = e.p.shape === 3 ? 1024 : 512;
        break;
      }
      case 'particles':
        break;
      case 'flame': {
        const xs = e.xforms ?? [];
        // A drop reverses the vars <-> alt morph (as in V1).
        if (F.dropStart) m[key('mt')] = mm('mt') > 0.5 ? 0 : 1;
        m[key('m')] = mm('m') + (mm('mt') - mm('m')) * (1 - Math.exp(-sdt * 0.8));
        s.flameMorph = m[key('m')];
        s.flameSpec = {
          xforms: xs.map((x) => ({
            aff: x.aff.slice(0, 6) as [number, number, number, number, number, number],
            weight: x.weight,
            color: x.color,
            vars: x.vars as Partial<Record<FlameVar, number>>,
            alt: x.alt as Partial<Record<FlameVar, number>> | undefined,
            spin: x.spin,
            bass: x.bass,
            drift: x.drift[0] || x.drift[1] ? [x.drift[0], x.drift[1]] : undefined,
            pulse: x.pulse,
          })),
          count: e.p.count,
          iters: 4,
          rounds: e.p.rounds,
          zoom: P('zoom'),
          offset: [P('ox'), P('oy')],
          camSpin: e.p.camSpin,
          gain,
          flow: P('flow'),
          breathe: P('breathe'),
        };
        break;
      }
    }
  }

  private updateParticles(s: Slot, sdt: number, fluid: boolean): void {
    const eng = this.eng;
    if (!this.particles) this.particles = new Particles(eng.gl, eng.fs);
    const slot = s.genome.emitters.findIndex((e) => e.kind === 'particles');
    const e = s.genome.emitters[slot];
    const sch = EMITTER_SCHEMAS.particles;
    const P = (k: string) => s.P('em', slot, e.p, k, sch);
    const n = Math.min(e.p.count, this.opts.particleCap);
    if (Math.abs(this.particles.count - n) > n * 0.15) this.particles.setCount(n);
    const F = this.sig.F;
    const pu = this.pu;
    const fl = this.fluid;
    pu.dt = sdt;
    pu.time = this.sig.clock;
    pu.aspect = F.aspect;
    pu.wave = this.sig.waveTex;
    pu.velocity = fl && fluid ? fl.velocityTex : eng.black;
    pu.simTexelX = fl ? fl.texelX : 0;
    pu.simTexelY = fl ? fl.texelY : 0;
    pu.fluidAmt = fl && fluid ? 1 : 0;
    const flow = s.genome.carrier.kind === 'flow' ? s.genome.carrier.p.famt * 300 : 0;
    pu.curl = (P('curl') + flow) * (0.5 + F.stem[3]);
    pu.zoomFlow = P('zoomFlow') * F.speed * (1 + 0.8 * F.stem[1]);
    // Particles follow the chain's rotation too, so a vortex carries its stars.
    let rot = 0;
    for (const o of s.genome.chain) if (o.op === 'rotate') rot += o.p.lock * 4 * (this.sig.spinStep / Math.max(sdt, 1e-3)) * 0.25;
    pu.rotFlow = -rot;
    pu.converge = F.build * 1.5;
    pu.drag = P('drag');
    pu.lifeRate = P('life');
    pu.speed = P('speed') * F.speed;
    pu.spawnFrom = pu.spawnTo = e.p.spawn;
    pu.spawnMix = 0;
    pu.emitAngle = TAU * F.barPhase;
    pu.burst = F.hit > 0 ? 0.02 * F.act : 0;
    pu.burstSeed = Math.random() * 1000;
    pu.burstSpeed = 0.3 + F.stem[0] + F.drop;
    pu.liftX = 0;
    pu.liftY = P('lift') * F.speed;
    pu.spread = P('spread');
    this.particles.update(pu);
  }

  private drawParticles(s: Slot, e: EmitterGene, weight: number): void {
    if (!this.particles) return;
    const F = this.sig.F;
    const slot = s.genome.emitters.indexOf(e);
    const sch = EMITTER_SCHEMAS.particles;
    const size = Math.max(1, s.P('em', slot, e.p, 'size', sch) * (this.h / 1080));
    const alive = 0.35 + 0.65 * F.act;
    const bright = s.P('em', slot, e.p, 'gain', sch) * 1.4 * weight * (e.layer === 'fb' ? 0.5 : 1) * (0.7 + 0.6 * F.beatPulse * F.gate[0]);
    this.particles.draw(size, bright, alive, s.cols);
  }

  private drawWave(s: Slot, weight: number, sdt: number, fb: boolean): void {
    const eng = this.eng;
    const gl = eng.gl;
    const p = eng.pWave.use();
    this.setCommon(p, s, sdt, fb);
    // At least a pixel wide; below 1080p the energy is scaled down so small renders match.
    const want = s.waveThick * (this.h / 1080);
    const thick = Math.max(1, want);
    p.f1('uN', s.waveN)
      .f1('uThick', thick)
      .f1('uBright', s.waveBright * weight * (fb ? 1 : 3) * Math.sqrt(Math.min(1, want / thick)))
      .f4v('uW', s.waveU);
    gl.bindVertexArray(eng.lineVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, s.waveN * 2);
  }

  private feedbackPass(s: Slot, sdt: number, partOwner: boolean, flameOwner: boolean): void {
    const eng = this.eng;
    const gl = eng.gl;
    const g = s.genome;
    const cp = g.carrier.p;
    const fl = this.fluid;
    s.fb.write.bind();
    gl.disable(gl.BLEND);
    const p = s.progs.feedback.use();
    this.setCommon(p, s, sdt, true);
    p.tex('uPrev', s.fb.read.t)
      .tex('uVel', fl && g.carrier.kind === 'fluid' ? fl.velocityTex : eng.black)
      .f2('uSimTexel', fl ? fl.texelX : 0, fl ? fl.texelY : 0)
      .f1('uFluidAmt', s.P('car', 0, cp, 'amount', CARRIER_SCHEMA))
      .f1('uBlur', cp.blur)
      .f1('uDecaySub', (eng.hq ? 0.0015 : 1.5 / 255) * cp.floor * sdt * 60)
      .f1('uFlowAmt', s.P('car', 0, cp, 'famt', CARRIER_SCHEMA) * sdt * 60 * this.sig.F.speed)
      .f1('uFlowScale', cp.fscale);
    eng.fs.draw();

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (const e of g.emitters) {
      if (e.layer !== 'fb') continue;
      if (e.kind === 'wave') this.drawWave(s, 1, sdt, true);
      if (e.kind === 'particles' && partOwner) this.drawParticles(s, e, 1);
    }
    if (flameOwner && this.flame && s.flameSpec) {
      const spec = s.flameSpec;
      const fill = 1 - Math.max(0, Math.min(0.995, s.decay));
      const w = (spec.gain * 0.06 * fill * this.w * this.h) / (this.flame.count * spec.rounds);
      for (let i = 0; i < spec.rounds; i++) {
        this.flame.update();
        s.fb.write.bind();
        this.flame.draw(w * (0.8 + 0.4 * this.sig.F.loud), this.sig.F.aspect, s.cols);
      }
    }
    gl.disable(gl.BLEND);
    s.fb.swap();
  }

  private setCommon(p: Program, s: Slot, sdt: number, fbPass: boolean): void {
    const F = this.sig.F;
    const c = s.cols;
    p.f2('uRes', this.w, this.h)
      .f1('uAspect', F.aspect)
      .f1('uTime', this.sig.clock % 4096)
      .f1('uPhase', F.phase)
      .f1('uDt', sdt)
      .f1('uF60', sdt * 60)
      .f1('uSpeed', F.speed)
      .f1('uBeat', F.beatPhase)
      .f1('uBar', F.barPhase)
      .f1('uBars', F.bars)
      .f1('uBeats', F.beats)
      .f1('uBeatPulse', F.beatPulse)
      .f1('uSpin', F.spin)
      .f1('uSpinStep', this.sig.spinStep)
      .f4('uStem', F.stem[0], F.stem[1], F.stem[2], F.stem[3])
      .f4('uOnset', F.onset[0], F.onset[1], F.onset[2], F.onset[3])
      .f4('uPres', F.gate[0], F.gate[1], F.gate[2], F.gate[3])
      .f1('uAct', F.act)
      .f1('uBuild', F.build)
      .f1('uDrop', F.drop)
      .f1('uLoud', F.loud)
      .f1('uMelody', F.melody)
      .f1('uKeyHue', F.keyHue)
      .f3('uColA', c[0], c[1], c[2])
      .f3('uColB', c[3], c[4], c[5])
      .f3('uColC', c[6], c[7], c[8])
      .f1('uDecay', s.decay)
      .f1('uLayerK', fbPass ? Math.max(0.02, 1 - s.decay) : 1)
      .f1('uAccum', fbPass ? 1 : 6)
      .f4v('uOpA', s.opA)
      .f4v('uOpB', s.opB)
      .f4v('uEm', s.em)
      .f4v('uInk', s.ink)
      .f4v('uBlob', s.blob)
      .f4v('uSeg', s.seg)
      .f4v('uSegZ', s.segZ)
      .i1('uSegN', s.segN)
      .f1v('uChroma', this.sig.chroma)
      .tex('uWave', this.sig.waveTex)
      .tex('uSpec', this.sig.specTex);
  }

  private post(pp: PostParams, target: 'canvas' | 'out', rawDt: number): void {
    const eng = this.eng;
    const gl = eng.gl;
    const F = this.sig.F;
    this.bloom.run(this.scene!, 0.9 - F.build * 0.3, 0.5);
    this.avgLum.write.bind();
    eng.pExposure
      .use()
      .tex('uScene', this.scene!.t)
      .tex('uPrev', this.avgLum.read.t)
      .f1('uRate', 1 - Math.exp(-Math.min(rawDt || 0.016, 0.1) * 1.5))
      .f1('uFrame', this.frame % 997);
    eng.fs.draw();
    this.avgLum.swap();
    if (target === 'out' && this.out) this.out.bind();
    else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.w, this.h);
    }
    const actB = 0.55 + 0.6 * F.act;
    const bloomStr = (pp.bloom * actB * (0.32 + 0.3 * F.build + this.flash * 0.8 + F.drop * 0.6)) / this.bloom.levels * 1.6;
    const exposure = pp.exposure * (0.78 + 0.12 * F.beatPulse * F.act + 0.2 * F.drop + 0.15 * F.build + this.flash * 0.25);
    eng.pFinal
      .use()
      .tex('uScene', this.scene!.t)
      .tex('uBloom', this.bloom.output)
      .tex('uAvg', this.avgLum.read.t)
      .f1('uKey', 0.12)
      .f1('uAdapt', pp.adapt)
      .f1('uContrast', pp.contrast)
      .f1('uBloomStr', bloomStr)
      .f1('uExposure', exposure)
      .f1('uCA', pp.ca + F.drop * 0.015 * F.act + this.flash * 0.005)
      .f1('uVignette', pp.vignette)
      .f1('uTonemap', 1)
      .f1('uFrame', this.frame % 1024);
    eng.fs.draw();
  }

  /** Reset exposure history and signals (fresh screening / thumbnail run). */
  resetHistory(): void {
    const gl = this.eng.gl;
    for (const t of [this.avgLum.read, this.avgLum.write]) {
      t.bind();
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    this.sig.reset();
    this.flash = 0;
    this.frame = 0;
  }

  readPixels(buf: Uint8Array): void {
    const gl = this.eng.gl;
    if (!this.out) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.out.fbo);
    gl.readPixels(0, 0, this.w, this.h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose(): void {
    for (const s of this.slots) s.fb.dispose();
    this.slots = [];
    this.scene?.dispose();
    this.out?.dispose();
    this.bloom.dispose();
    this.avgLum.dispose();
    this.fluid?.dispose();
    this.particles?.dispose();
    this.flame?.dispose();
    this.sig.dispose();
  }
}

// --------------------------------------------------------------- engine

export interface RenderStats {
  frameMs: number;
  cpuMs: number;
  gpuMs: number;
  width: number;
  height: number;
  scale: number;
}

/** Owns the GL context, shared programs and the main (on-canvas) stage with crossfades. */
export class Engine {
  readonly gl: GL;
  readonly fs: Fullscreen;
  readonly hq: boolean;
  readonly hdr: TexFormat;
  readonly cache: ProgramCache;
  readonly pFinal: Program;
  readonly pExposure: Program;
  readonly pSeed: Program;
  readonly pWave: Program;
  readonly black: WebGLTexture;
  readonly lineVao: WebGLVertexArrayObject;
  readonly main: Stage;
  readonly stats: RenderStats = { frameMs: 16.7, cpuMs: 0, gpuMs: 0, width: 0, height: 0, scale: 1 };

  private css = { w: 1, h: 1, dpr: 1 };
  private renderScale = 1;
  private slowTime = 0;
  private fastTime = 0;
  private lastNow = 0;
  private from: Slot | null = null;
  private to: Slot | null = null;
  private blendT = 1;
  private blendDur = 1;
  private pending: { g: Genome; secs: number } | null = null;
  private timerExt: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null = null;
  private queries: WebGLQuery[] = [];
  private pendingQueries: WebGLQuery[] = [];
  onSwitched: ((g: Genome) => void) | null = null;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false, premultipliedAlpha: false,
      preserveDrawingBuffer: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    const f = formats(gl);
    const cbf = gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('EXT_float_blend');
    gl.getExtension('OES_texture_float_linear');
    this.hq = !!cbf && canRenderTo(gl, f.rgba16f) && canRenderTo(gl, f.rgba32f);
    this.hdr = this.hq ? f.rgba16f : f.rgba8;
    this.fs = new Fullscreen(gl);
    this.cache = new ProgramCache(gl);
    this.pFinal = new Program(gl, FULLSCREEN_VS, FINAL_FS, 'v2-final');
    this.pExposure = new Program(gl, FULLSCREEN_VS, EXPOSURE_FS, 'v2-exposure');
    this.pSeed = new Program(gl, FULLSCREEN_VS, SCALE_FS, 'v2-seed');
    this.pWave = new Program(gl, WAVE_VS, WAVE_FS, 'v2-wave');
    this.black = createTexture(gl, 1, 1, f.rgba8, gl.NEAREST, new Uint8Array([0, 0, 0, 255]));
    const vao = gl.createVertexArray();
    if (!vao) throw new Error('cannot create VAO');
    this.lineVao = vao;
    this.timerExt = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    this.main = new Stage(this, { offscreen: false, particleCap: 262144, flameCap: 524288 });
    const r = canvas.getBoundingClientRect();
    this.resize(r.width || 1280, r.height || 720, window.devicePixelRatio || 1);
  }

  resize(cssW: number, cssH: number, dpr: number): void {
    this.css = { w: cssW, h: cssH, dpr };
    let w = Math.max(1, cssW * dpr * this.renderScale);
    let h = Math.max(1, cssH * dpr * this.renderScale);
    const s = Math.min(1, MAX_SIDE / Math.max(w, h));
    w = Math.round(w * s);
    h = Math.round(h * s);
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    this.main.resize(w, h);
    this.stats.width = w;
    this.stats.height = h;
  }

  /** Crossfade the main view to a genome once its programs are linked. */
  show(g: Genome, secs = 1.5): void {
    this.pending = { g, secs };
    this.cache.request(g);
  }

  current(): Genome | null {
    return this.pending?.g ?? this.to?.genome ?? null;
  }

  playing(): Genome | null {
    return this.to?.genome ?? null;
  }

  failed(g: Genome): string | null {
    return this.cache.failed(g);
  }

  setSongComplexity(cx: number): void {
    this.main.sig.songCx = cx;
  }

  render(state: MusicState): void {
    const t0 = performance.now();
    if (this.lastNow) this.stats.frameMs += (t0 - this.lastNow - this.stats.frameMs) * 0.05;
    this.lastNow = t0;
    this.pollTimer();
    const q = this.beginTimer();
    const dt = state.dt > 0 ? Math.min(state.dt, 0.1) : 1 / 60;
    this.adapt(dt);

    if (this.pending) {
      const pg = this.cache.get(this.pending.g, !this.to);
      if (pg) this.doSwitch(this.pending.g, pg, this.pending.secs);
      else if (this.cache.failed(this.pending.g)) this.pending = null;
    }
    const st = this.main;
    if (this.from && this.to) {
      this.blendT = Math.min(1, this.blendT + dt / this.blendDur);
      const w = smooth01(this.blendT);
      this.to.weight = w;
      this.from.weight = 1 - w;
      if (this.blendT >= 1) {
        st.disposeSlot(this.from);
        this.from = null;
        this.to.weight = 1;
      }
    }
    for (const s of st.slots) this.cache.touch(s.genome);
    st.render(state, dt, 'canvas');
    this.endTimer(q);
    this.stats.cpuMs += (performance.now() - t0 - this.stats.cpuMs) * 0.05;
  }

  private doSwitch(g: Genome, progs: GenomePrograms, secs: number): void {
    this.pending = null;
    const st = this.main;
    if (this.from) {
      // Already blending: drop the fainter one.
      const drop = this.from.weight < this.to!.weight ? this.from : this.to!;
      const keep = drop === this.from ? this.to! : this.from;
      st.disposeSlot(drop);
      this.from = keep;
    } else this.from = this.to;
    const s = st.makeSlot(g, progs);
    st.slots.push(s);
    if (this.from && secs > 0.05) {
      st.seedFrom(s, 0.5);
      s.weight = 0;
      this.to = s;
      this.blendT = 0;
      this.blendDur = secs;
    } else {
      if (this.from) st.disposeSlot(this.from);
      this.from = null;
      this.to = s;
      s.weight = 1;
      this.blendT = 1;
    }
    this.onSwitched?.(g);
  }

  private adapt(dt: number): void {
    if (typeof document !== 'undefined' && document.hidden) return;
    if (this.stats.frameMs > 21) {
      this.slowTime += dt;
      this.fastTime = 0;
      if (this.slowTime > 4 && this.renderScale > 0.5) {
        this.renderScale = Math.max(0.5, this.renderScale * 0.8);
        this.slowTime = 0;
        this.stats.scale = this.renderScale;
        console.info(`[v2] frame time ${this.stats.frameMs.toFixed(1)} ms, render scale ${this.renderScale.toFixed(2)}`);
        this.resize(this.css.w, this.css.h, this.css.dpr);
      }
    } else if (this.stats.frameMs < 14 && this.renderScale < 1) {
      this.fastTime += dt;
      if (this.fastTime > 12) {
        this.renderScale = Math.min(1, this.renderScale / 0.8);
        this.fastTime = 0;
        this.stats.scale = this.renderScale;
        this.resize(this.css.w, this.css.h, this.css.dpr);
      }
    } else {
      this.slowTime = Math.max(0, this.slowTime - dt);
    }
  }

  private beginTimer(): WebGLQuery | null {
    const ext = this.timerExt;
    if (!ext || this.pendingQueries.length > 3) return null;
    const q = this.queries.pop() ?? this.gl.createQuery();
    if (!q) return null;
    this.gl.beginQuery(ext.TIME_ELAPSED_EXT, q);
    return q;
  }

  private endTimer(q: WebGLQuery | null): void {
    if (!q || !this.timerExt) return;
    this.gl.endQuery(this.timerExt.TIME_ELAPSED_EXT);
    this.pendingQueries.push(q);
  }

  private pollTimer(): void {
    const ext = this.timerExt;
    if (!ext) return;
    const gl = this.gl;
    const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
    while (this.pendingQueries.length) {
      const q = this.pendingQueries[0];
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number;
      if (!disjoint) this.stats.gpuMs += (ns / 1e6 - this.stats.gpuMs) * 0.1;
      this.pendingQueries.shift();
      this.queries.push(q);
    }
  }
}

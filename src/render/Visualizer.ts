// Rendering engine: enhanced (per-preset GLSL feedback + composite, optional
// fluid / particles / line geometry, bloom), classic (butterchurn) and hybrid
// (butterchurn injected into the enhanced feedback).
//
// Enhanced frame: shared analysis textures -> per-preset JS hook -> fluid and
// particle simulation (only when a live preset uses them) -> each live
// preset's feedback pass into its own buffer -> each preset's composite,
// crossfaded additively into the HDR scene -> top layers -> bloom, auto
// exposure, tonemap. During a blend both presets run; otherwise only one.

import type { IVisualizer, MusicState, StemName, VisualMode, VisualizerOptions } from '../types';
import { Bloom } from './bloom';
import { Classic } from './classic';
import { Flame } from './flame';
import { Fluid, SPLAT_RADIAL } from './fluid';
import { Fullscreen, GL, PingPong, Program, Target, TexFormat, canRenderTo, createTexture, formats } from './gl';
import { Particles, ParticleUpdate } from './particles';
import { Effects, Frame, PRESETS, Preset, Runtime, makeRuntime, paletteColors } from './presets';
import { PresetPrograms, ProgramCache } from './programs';
import { CLASSIC_SCENE_FS, EXPOSURE_FS, FINAL_FS, FULLSCREEN_VS, SCALE_FS } from './shaders';

const TAU = Math.PI * 2;
const SPIN_WRAP = TAU * 16; // presets scale uSpin by multiples of 1/16
const MAX_SIDE = 2560;
const WAVE_N = 512;
const SPEC_N = 128;
const MAX_RINGS = 8;
const PARTICLE_STEPS = [65536, 262144, 1048576];
const STEMS: StemName[] = ['drums', 'bass', 'vocals', 'other'];

export interface RenderStats {
  frameMs: number; // smoothed interval between render() calls
  cpuMs: number; // smoothed CPU time spent inside render()
  gpuMs: number; // smoothed GPU time (0 when timer queries are unavailable)
  particles: number;
  width: number;
  height: number;
  hq: boolean;
}

const DEFAULT_OPTS: VisualizerOptions = { particleCount: 1048576, renderScale: 1 };

function smooth01(t: number): number {
  t = Math.min(1, Math.max(0, t));
  return t * t * (3 - 2 * t);
}
function num(v: unknown, d: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : d;
}

interface Active {
  preset: Preset;
  idx: number;
  progs: PresetPrograms;
  buf: number; // index into the feedback pool
  rt: Runtime;
  rem: [number, number];
  shift: [number, number];
  cols: Float32Array;
  weight: number;
}

export class Visualizer implements IVisualizer {
  private gl: GL;
  private fs: Fullscreen;
  private hq: boolean;
  private hdr: TexFormat;
  private opts: VisualizerOptions;
  private mode: VisualMode = 'enhanced';
  private disposed = false;

  private cache: ProgramCache;
  private pClassic: Program;
  private pFinal: Program;
  private pExposure: Program;
  private pSeed: Program;
  private avgLum: PingPong;
  private fluid: Fluid | null = null;
  private particles: Particles | null = null;
  private bloom: Bloom;
  private classic: Classic | null = null;

  private pool: PingPong[] = [];
  private scene: Target | null = null;
  private waveTex: WebGLTexture;
  private specTex: WebGLTexture;
  private black: WebGLTexture;
  private lineVao: WebGLVertexArrayObject;
  private wave = new Float32Array(WAVE_N);
  private waveTmp = new Float32Array(WAVE_N);
  private spec = new Float32Array(SPEC_N);
  private rms = 0.1;

  private width = 1;
  private height = 1;
  private css = { w: 1, h: 1, dpr: 1 };

  // Presets and blending
  private from: Active | null = null;
  private to!: Active;
  private blendT = 1;
  private blendDur = 1;
  private pending: { idx: number; secs: number } | null = null;
  private history: number[] = [];
  private autoSwitch = true;
  private songCx: number | null = null;
  private warmNext = 0;
  private warmTimer = 0;

  // Per-frame music state shared with preset hooks
  private F: Frame = {
    time: 0, dt: 0, phase: 0, act: 0.5, cx: 0.5, speed: 1, spin: 0, bars: 0, beats: 0, barIndex: 0, beatIndex: 0,
    barPhase: 0, beatPhase: 0, beatPulse: 0, onBeat: false, onBar: false,
    stem: new Float32Array(4), onset: new Float32Array(4), gate: new Float32Array(4),
    loud: 0, melody: 0.5, build: 0, drop: 0, keyTonic: 0, minor: false, sectionIndex: 0, aspect: 1, hit: 0,
    dropStart: false,
  };
  private spinStep = 0;
  private live = new Float32Array(6); // bass, mid, treb, keyHue, keyChangePulse, barPulse
  private fx: Effects;

  // Rings (drum shockwaves) and particles
  private rings = new Float32Array(MAX_RINGS * 4);
  private ringAge = new Float32Array(MAX_RINGS).fill(99);
  private ringStr = new Float32Array(MAX_RINGS);
  private ringX = new Float32Array(MAX_RINGS);
  private ringY = new Float32Array(MAX_RINGS);
  private ringNext = 0;
  private chroma = new Float32Array(12);
  private pu: ParticleUpdate;
  private particleCap: number;
  private partOwner: Active | null = null;
  private flame: Flame | null = null;
  private flameOwner: Active | null = null;

  // Timing / rhythm tracking
  private clock = 0;
  private frame = 0;
  private prevBarPhase = 0;
  private prevDrumOnset = 0;
  private hitCooldown = 0;
  private flash = 0;
  private lastNow = 0;
  private slowTime = 0;
  readonly stats: RenderStats = { frameMs: 16.7, cpuMs: 0, gpuMs: 0, particles: 0, width: 0, height: 0, hq: true };

  // GPU timer
  private timerExt: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null = null;
  private queries: WebGLQuery[] = [];
  private pendingQueries: WebGLQuery[] = [];

  constructor(
    private canvas: HTMLCanvasElement,
    private audio: { context: AudioContext; source: AudioNode },
    opts?: Partial<VisualizerOptions>,
  ) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser. The visualizer needs WebGL2 (Chrome, Safari 15+, Firefox).');
    this.gl = gl;
    this.opts = { ...DEFAULT_OPTS, ...opts };
    this.particleCap = this.opts.particleCount;

    const f = formats(gl);
    const cbf = gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('EXT_float_blend');
    gl.getExtension('OES_texture_float_linear');
    this.hq = !!cbf && canRenderTo(gl, f.rgba16f) && canRenderTo(gl, f.rgba32f);
    this.hdr = this.hq ? f.rgba16f : f.rgba8;
    if (!this.hq) console.warn('[render] float render targets unavailable; running reduced quality (no fluid/particles)');
    this.stats.hq = this.hq;

    this.fs = new Fullscreen(gl);
    this.cache = new ProgramCache(gl);
    this.pClassic = new Program(gl, FULLSCREEN_VS, CLASSIC_SCENE_FS, 'classic-scene');
    this.pFinal = new Program(gl, FULLSCREEN_VS, FINAL_FS, 'final');
    this.pExposure = new Program(gl, FULLSCREEN_VS, EXPOSURE_FS, 'exposure');
    this.pSeed = new Program(gl, FULLSCREEN_VS, SCALE_FS, 'seed');
    this.avgLum = new PingPong(gl, 1, 1, [this.hdr], gl.NEAREST);
    this.bloom = new Bloom(gl, this.fs);
    if (this.hq) {
      this.fluid = new Fluid(gl, this.fs);
      this.particles = new Particles(gl, this.fs);
    }

    this.waveTex = createTexture(gl, WAVE_N, 1, f.r16f, gl.LINEAR);
    this.specTex = createTexture(gl, SPEC_N, 1, f.r16f, gl.LINEAR);
    this.black = createTexture(gl, 1, 1, f.rgba8, gl.NEAREST, new Uint8Array([0, 0, 0, 255]));
    const vao = gl.createVertexArray();
    if (!vao) throw new Error('[render] cannot create VAO');
    this.lineVao = vao;

    this.timerExt = gl.getExtension('EXT_disjoint_timer_query_webgl2');

    this.fx = {
      splat: (x, y, fx, fy, radius, type) => this.fluid?.splat(x, y, fx, fy, radius, type),
    };
    this.pu = {
      dt: 0, time: 0, aspect: 1, velocity: this.black, simTexelX: 0, simTexelY: 0, wave: this.waveTex,
      fluidAmt: 0, curl: 0, zoomFlow: 0, rotFlow: 0, converge: 0, drag: 3, lifeRate: 0.3, speed: 0.5,
      spawnFrom: 0, spawnTo: 0, spawnMix: 0, emitCount: 3, emitAngle: 0, emitRadius: 0.3,
      burst: 0, burstSeed: 0, burstSpeed: 0, liftX: 0, liftY: 0, spread: 0.01,
    };

    const r = canvas.getBoundingClientRect();
    this.resize(r.width || canvas.width || 1280, r.height || canvas.height || 720, window.devicePixelRatio || 1);

    const first = this.pickPreset(0.5, false);
    this.to = this.makeActive(first, this.compileNow(first), 0);
    this.history.push(first);
  }

  // ------------------------------------------------------------ public API

  resize(cssWidth: number, cssHeight: number, dpr: number): void {
    this.css = { w: cssWidth, h: cssHeight, dpr };
    let w = Math.max(1, cssWidth * dpr * this.opts.renderScale);
    let h = Math.max(1, cssHeight * dpr * this.opts.renderScale);
    const s = Math.min(1, MAX_SIDE / Math.max(w, h));
    w = Math.round(w * s);
    h = Math.round(h * s);
    // The visualizer owns the canvas backing size; always enforce it, even when
    // the render targets can be kept, so it never drifts from the GL viewport.
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    if (w === this.width && h === this.height && this.pool.length) return;
    this.width = w;
    this.height = h;
    this.stats.width = w;
    this.stats.height = h;
    const gl = this.gl;
    for (const p of this.pool) p.dispose();
    this.scene?.dispose();
    this.pool = [new PingPong(gl, w, h, [this.hdr], gl.LINEAR), new PingPong(gl, w, h, [this.hdr], gl.LINEAR)];
    this.scene = new Target(gl, w, h, [this.hdr], gl.LINEAR);
    this.bloom.resize(w, h, this.hdr);
    this.fluid?.resize(w / h);
    this.resizeClassic();
  }

  render(state: MusicState): void {
    if (this.disposed) return;
    const t0 = performance.now();
    if (this.lastNow) this.stats.frameMs += (t0 - this.lastNow - this.stats.frameMs) * 0.05;
    this.lastNow = t0;
    this.pollTimer();
    const q = this.beginTimer();

    const dt = state.dt > 0 ? Math.min(state.dt, 0.1) : 1 / 60;
    const sdt = state.playing ? dt : dt * 0.25;
    this.clock += sdt;
    this.frame++;
    this.flash *= Math.exp(-sdt * 3);
    this.adaptPerformance(dt);

    const wantClassic = this.mode === 'classic' || this.mode === 'hybrid';
    if (wantClassic) this.ensureClassic();
    const classicReady = !!this.classic?.ready;
    if (classicReady && this.autoSwitch) this.classic!.autoSwitch(state);

    this.computeFrame(state, sdt);
    this.updatePresets(state, sdt);

    if (this.mode === 'classic' && classicReady && this.classic!.render()) {
      this.scene!.bind();
      this.gl.disable(this.gl.BLEND);
      this.pClassic.use().tex('uA', this.classic!.texture);
      this.fs.draw();
      this.post(state, true);
    } else {
      const hybrid = this.mode === 'hybrid' && classicReady && this.classic!.render();
      this.simulate(state, sdt, hybrid);
      this.compose(state, hybrid);
      this.post(state, false);
    }

    this.endTimer(q);
    this.stats.cpuMs += (performance.now() - t0 - this.stats.cpuMs) * 0.05;
  }

  setMode(mode: VisualMode): void {
    this.mode = mode;
    if (mode !== 'enhanced') {
      this.ensureClassic();
      this.resizeClassic();
    }
  }

  getMode(): VisualMode {
    return this.mode;
  }

  nextPreset(): void {
    if (this.mode === 'classic' && this.classic?.ready) {
      this.classic.pick(2);
      return;
    }
    this.requestSwitch((this.target().idx + 1) % PRESETS.length, 1.5);
  }

  newSong(songComplexity: number): void {
    this.songCx = num(songComplexity, 0.5);
    this.requestSwitch(this.pickPreset(this.songCx, false), 1.5);
    if (this.classic?.ready && this.mode !== 'enhanced') this.classic.pick(1.5);
  }

  /** Jump to a specific enhanced preset by index or name. */
  selectPreset(which: number | string, blendSeconds = 1): void {
    const idx = typeof which === 'number' ? which : PRESETS.findIndex((p) => p.name === which || p.id === which);
    if (idx >= 0 && idx < PRESETS.length) this.requestSwitch(idx, blendSeconds);
  }

  /** Select by stable ID: "E07" (enhanced) or "C-3fa2" (classic / hybrid butterchurn layer). */
  selectPresetById(id: string): boolean {
    const key = id.trim();
    if (/^C-/i.test(key)) {
      if (this.mode === 'enhanced' || !this.classic?.ready) return false;
      return this.classic.pickById('C-' + key.slice(2).toLowerCase(), 1.5);
    }
    if (this.mode === 'classic') return false;
    const idx = PRESETS.findIndex((p) => p.id === key.toUpperCase());
    if (idx < 0) return false;
    this.requestSwitch(idx, 1.5);
    return true;
  }

  /** Disable drop-triggered switching (dev harness preset lock). */
  setAutoSwitch(on: boolean): void {
    this.autoSwitch = on;
  }

  getPresetName(): string {
    const c = this.classic?.ready ? `${this.classic.presetId} · ${this.classic.presetName}` : '';
    if (this.mode === 'classic' && c) return c;
    const p = this.target().preset;
    const base = `${p.id} · ${p.name}`;
    return this.mode === 'hybrid' && c ? `${base} + ${c}` : base;
  }

  setOptions(opts: Partial<VisualizerOptions>): void {
    const prevScale = this.opts.renderScale;
    this.opts = { ...this.opts, ...opts };
    this.opts.renderScale = Math.min(1, Math.max(0.25, this.opts.renderScale));
    if (opts.particleCount !== undefined) {
      this.particleCap = opts.particleCount;
      this.applyParticleCount();
      this.slowTime = 0;
      this.stats.frameMs = 16.7;
      this.lastNow = 0;
    }
    if (opts.renderScale !== undefined && opts.renderScale !== prevScale) {
      for (const p of this.pool) p.dispose();
      this.pool = [];
      this.resize(this.css.w, this.css.h, this.css.dpr);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    for (const p of this.pool) p.dispose();
    this.pool = [];
    this.scene?.dispose();
    this.scene = null;
    this.bloom.dispose();
    this.fluid?.dispose();
    this.particles?.dispose();
    this.flame?.dispose();
    this.classic?.dispose();
    this.classic = null;
    this.cache.dispose();
    this.pClassic.dispose();
    this.pFinal.dispose();
    this.pExposure.dispose();
    this.pSeed.dispose();
    this.avgLum.dispose();
    this.fs.dispose();
    gl.deleteTexture(this.waveTex);
    gl.deleteTexture(this.specTex);
    gl.deleteTexture(this.black);
    gl.deleteVertexArray(this.lineVao);
    for (const q of this.queries) gl.deleteQuery(q);
    for (const q of this.pendingQueries) gl.deleteQuery(q);
    this.queries.length = 0;
    this.pendingQueries.length = 0;
  }

  // ------------------------------------------------------------- presets

  private compileNow(idx: number): PresetPrograms {
    const p = this.cache.get(PRESETS[idx], true);
    if (p) return p;
    // A broken preset must never take the engine down: fall back to any preset that compiles.
    for (let i = 0; i < PRESETS.length; i++) {
      const q = this.cache.get(PRESETS[i], true);
      if (q) return q;
    }
    throw new Error('[render] no enhanced preset compiled');
  }

  private makeActive(idx: number, progs: PresetPrograms, buf: number): Active {
    return {
      preset: PRESETS[idx], idx, progs, buf, rt: makeRuntime(), rem: [0, 0], shift: [0, 0],
      cols: new Float32Array(9), weight: 1,
    };
  }

  /** The preset we are at or blending towards. */
  private target(): Active {
    return this.to;
  }

  private requestSwitch(idx: number, secs: number): void {
    this.pending = { idx, secs };
    this.cache.request(PRESETS[idx]);
  }

  private doSwitch(idx: number, progs: PresetPrograms, secs: number): void {
    const t = smooth01(this.blendT);
    const keep = this.from && t < 0.5 ? this.from : this.to;
    if (keep.idx === idx && this.blendT >= 1) return;
    const a = this.makeActive(idx, progs, 1 - keep.buf);
    // Seed the incoming feedback with the current picture so it morphs out of it.
    if (a.preset.feedback !== false) {
      const gl = this.gl;
      const pp = this.pool[a.buf];
      gl.disable(gl.BLEND);
      pp.write.bind();
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      pp.read.bind();
      if (this.scene && this.frame > 1 && secs >= 0.1) {
        this.pSeed.use().tex('uTex', this.scene.t).f1('uValue', 0.5);
        this.fs.draw();
      } else {
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }
    this.from = keep;
    this.to = a;
    this.blendT = 0;
    this.blendDur = Math.max(0.05, secs);
    this.history.push(idx);
    if (this.history.length > 6) this.history.shift();
    this.applyParticleCount();
  }

  /** Random preset whose energy range suits `target`, avoiding recent ones. */
  private pickPreset(target: number, high: boolean): number {
    const cur = this.to ? this.to.idx : -1;
    const suited: number[] = [];
    let best = -1;
    let bestScore = Infinity;
    for (let i = 0; i < PRESETS.length; i++) {
      const p = PRESETS[i];
      if (i === cur || this.cache.failed(p)) continue;
      const [lo, hi] = p.energy;
      const dist = target < lo ? lo - target : target > hi ? target - hi : 0;
      const recent = this.history.includes(i);
      if (dist === 0 && !recent && (!high || hi >= 0.8)) suited.push(i);
      const score = dist + (recent ? 0.5 : 0) + (high && hi < 0.8 ? 0.3 : 0) + Math.random() * 0.05;
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
    if (suited.length) return suited[Math.floor(Math.random() * suited.length)];
    return best >= 0 ? best : (cur + 1) % PRESETS.length;
  }

  private updatePresets(state: MusicState, sdt: number): void {
    // Switch only on drops (plus newSong() and manual requests).
    if (state.sectionChanged && state.section?.label === 'drop') {
      const cx = Math.max(this.songCx ?? this.F.cx, this.F.cx);
      if (this.autoSwitch) this.requestSwitch(this.pickPreset(Math.min(1, Math.max(0.55, cx + 0.3)), true), 0.35);
      this.flash = 0.4 + 0.6 * this.F.act;
      this.bigSplat(0.4 + 0.6 * this.F.act);
    }
    if (this.pending) {
      const p = PRESETS[this.pending.idx];
      const progs = this.cache.get(p);
      if (progs) {
        this.doSwitch(this.pending.idx, progs, this.pending.secs);
        this.pending = null;
      } else if (this.cache.failed(p)) {
        this.pending = null;
      }
    }
    this.blendT = Math.min(1, this.blendT + sdt / this.blendDur);
    if (this.blendT >= 1 && this.from) {
      this.from = null;
      this.applyParticleCount();
    }
    // Warm the cache in the background when the driver compiles in parallel.
    this.warmTimer -= sdt;
    if (this.warmTimer <= 0 && this.warmNext < PRESETS.length && this.blendT >= 1) {
      this.warmTimer = 0.3;
      const p = PRESETS[this.warmNext++];
      if (!this.cache.has(p) && this.gl.getExtension('KHR_parallel_shader_compile')) this.cache.request(p);
    }
  }

  // ------------------------------------------------------------ analysis

  private computeFrame(state: MusicState, sdt: number): void {
    const F = this.F;
    const dt = sdt;
    F.time = this.clock;
    F.dt = dt;
    F.aspect = this.width / this.height;
    F.cx = num(state.complexity, 0.5);
    const song = this.songCx ?? num(state.songComplexity, 0.5);
    for (let i = 0; i < 4; i++) {
      const n = STEMS[i];
      const pres = num(state.stemPresence?.[n], 0.5);
      const g = smooth01((pres - 0.03) / 0.25);
      F.gate[i] = g;
      F.stem[i] = Math.min(1, num(state.stems?.[n], 0)) * g;
      F.onset[i] = Math.min(1, num(state.stemOnsets?.[n], 0)) * g;
    }
    // Activity budget: absolute musical density, leaning on the song mean.
    const target = Math.min(1, Math.max(0, (0.75 * F.cx + 0.25 * song - 0.08) / 0.72));
    F.act += (target - F.act) * (1 - Math.exp(-dt * 1.2));
    F.loud = num(state.loudness, 0);
    F.build = num(state.buildIntensity, 0);
    F.drop = num(state.dropPulse, 0);
    F.beatPulse = num(state.beatPulse, 0);
    F.onBeat = !!state.onBeat;
    F.onBar = !!state.onBar;
    F.keyTonic = num(state.keyTonic, 0);
    F.minor = state.keyMode === 'minor';
    F.sectionIndex = Math.max(0, num(state.sectionIndex, 0));
    F.dropStart = !!state.sectionChanged && state.section?.label === 'drop';
    const L = this.live;
    L[0] = Math.min(3, num(state.bass, 1));
    L[1] = Math.min(3, num(state.mid, 1));
    L[2] = Math.min(3, num(state.treb, 1));
    L[3] = num(state.keyHue, 0);
    L[4] = num(state.keyChangePulse, 0);
    L[5] = num(state.barPulse, 0);
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

    // Bar-locked spin: one turn per bar, half-time when the music is sparse.
    let dBar = F.barPhase - this.prevBarPhase;
    if (dBar < 0) dBar += 1;
    if (!state.playing || dBar > 0.5) dBar = (bpm / 240) * dt * (state.playing ? 1 : 0.3);
    this.prevBarPhase = F.barPhase;
    const spinMul = 0.5 + 0.5 * smooth01((F.act - 0.2) / 0.3);
    this.spinStep = TAU * dBar * spinMul;
    F.spin = (F.spin + this.spinStep) % SPIN_WRAP;

    this.processAudio(state, dt);

    // Drum hits (only when a drum stem is actually present).
    F.hit = 0;
    this.hitCooldown -= dt;
    const on = F.onset[0];
    const edge = on > 0.35 && this.prevDrumOnset <= 0.35;
    this.prevDrumOnset = on;
    if (state.playing && this.hitCooldown <= 0 && F.gate[0] > 0.2 && (edge || (F.onBeat && F.stem[0] > 0.25))) {
      F.hit = Math.max(on, 0.3) * (0.4 + F.stem[0]);
      this.hitCooldown = 0.1;
    }
  }

  private processAudio(state: MusicState, dt: number): void {
    const gl = this.gl;
    // Waveform: trigger on a rising zero crossing so the line holds still.
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
        const s = ((a + 2 * b + c) * 0.25) * gain;
        this.wave[j] += (s - this.wave[j]) * 0.5;
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.waveTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WAVE_N, 1, gl.RED, gl.FLOAT, this.wave);

    // Spectrum: log-spaced bins with a noise floor removed; also the melody height.
    const sp = state.spectrum;
    if (sp && sp.length >= 64) {
      const n = sp.length;
      const top = Math.min(n - 1, Math.round(n * 0.78));
      for (let k = 0; k < SPEC_N; k++) {
        const a = Math.floor(Math.pow(top, k / SPEC_N));
        const b = Math.max(a + 1, Math.floor(Math.pow(top, (k + 1) / SPEC_N)));
        let s = 0;
        for (let i = a; i < b; i++) s += sp[i];
        const v = Math.min(1, Math.max(0, (s / (b - a) - 0.28) / 0.6));
        const cur = this.spec[k];
        this.spec[k] = cur + (v - cur) * (v > cur ? 0.5 : 0.12);
      }
      const lo = Math.max(2, Math.round(n * 0.006));
      const hi = Math.max(lo + 4, Math.round(n * 0.12));
      let wsum = 0;
      let lsum = 0;
      for (let i = lo; i < hi; i++) {
        const w = Math.max(0, sp[i] - 0.3);
        const w2 = w * w * w;
        wsum += w2;
        lsum += w2 * Math.log2(i);
      }
      if (wsum > 1e-5) {
        const m = (lsum / wsum - Math.log2(lo)) / (Math.log2(hi) - Math.log2(lo));
        this.F.melody += (Math.min(1, Math.max(0, m)) - this.F.melody) * (1 - Math.exp(-dt * 5));
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.specTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SPEC_N, 1, gl.RED, gl.FLOAT, this.spec);

    if (state.chroma && state.chroma.length >= 12) for (let i = 0; i < 12; i++) this.chroma[i] = num(state.chroma[i], 0);
  }

  // ------------------------------------------------------------ simulate

  private bigSplat(k: number): void {
    const fl = this.fluid;
    if (!fl || !this.anyFluid()) return;
    const n = 8;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * TAU + this.F.spin;
      const r = 0.06;
      const c = Math.cos(a);
      const s = Math.sin(a);
      fl.splat(0.5 + (c * r * this.height) / this.width, 0.5 + s * r, c * 1500 * k, s * 1500 * k, 0.004, 0);
    }
    fl.splat(0.5, 0.5, 1200 * k, 0, 0.02, SPLAT_RADIAL);
  }

  private anyFluid(): boolean {
    return !!(this.to.preset.fluid || this.from?.preset.fluid);
  }

  private spawnRing(x: number, y: number, strength: number): void {
    const i = this.ringNext;
    this.ringNext = (i + 1) % MAX_RINGS;
    this.ringAge[i] = 0;
    this.ringStr[i] = strength;
    this.ringX[i] = x;
    this.ringY[i] = y;
  }

  private actives(): Active[] {
    return this.from ? [this.from, this.to] : [this.to];
  }

  private applyParticleCount(): void {
    const parts = this.particles;
    if (!parts) return;
    let want = 0;
    for (const a of this.actives()) if (a.preset.particles) want = Math.max(want, a.preset.particles.count);
    if (!want) return;
    const n = Math.min(want, this.particleCap);
    if (Math.abs(parts.count - n) > n * 0.1) parts.setCount(n);
  }

  private simulate(state: MusicState, sdt: number, hybrid: boolean): void {
    const F = this.F;
    const f60 = sdt * 60;
    const w = smooth01(this.blendT);
    this.to.weight = this.from ? w : 1;
    if (this.from) this.from.weight = 1 - w;
    const list = this.actives();
    const dom = this.from && w < 0.5 ? this.from : this.to;

    // Rings
    if (F.hit > 0) this.spawnRing(dom.rt.ringCenter[0], dom.rt.ringCenter[1], 0.35 + 0.9 * F.hit);
    for (let i = 0; i < MAX_RINGS; i++) {
      const age = (this.ringAge[i] += sdt);
      const o = i * 4;
      if (age > 1.6) {
        this.rings[o + 3] = 0;
        continue;
      }
      this.rings[o] = this.ringX[i];
      this.rings[o + 1] = this.ringY[i];
      this.rings[o + 2] = age * (0.35 + 0.35 * this.ringStr[i]);
      this.rings[o + 3] = this.ringStr[i] * Math.exp(-age * 3.2) * 0.55 * Math.min(1, f60 * 1.2);
    }

    // Per-preset hooks, palette and scroll
    const hue0 = num(state.keyHue, 0);
    for (const a of list) {
      const p = a.preset;
      a.rt.curveBright = 1;
      a.rt.particleFlow = 1;
      p.js?.(F, a.rt, this.fx);
      const sat = (p.sat ?? 1) * (0.78 + 0.3 * F.stem[2]) * (1 - 0.5 * F.build);
      paletteColors(p.palette, hue0 + (p.hue ?? 0), sat, a.cols, 0);
      a.shift[0] = a.shift[1] = 0;
      if (p.scroll) {
        for (let k = 0; k < 2; k++) {
          const v = -p.scroll[k] * F.speed * sdt * this.height + a.rem[k];
          const n = Math.round(v);
          a.rem[k] = v - n;
          a.shift[k] = n / this.height;
        }
      }
    }

    // Fluid (only when a live preset uses it)
    const fl = this.fluid;
    const fluidP = list.find((a) => a.preset.fluid)?.preset.fluid;
    if (fl && fluidP) {
      fl.dissipation = 0.6;
      fl.step(sdt, this.clock, fluidP.noise * (0.3 + 0.7 * F.act) * (0.3 + 0.7 * F.stem[3]) * 0.5 * f60, fluidP.vorticity, F.aspect);
    }

    // Particles (only when a live preset opts in)
    this.partOwner = null;
    for (const a of list) if (a.preset.particles && (!this.partOwner || a.weight > this.partOwner.weight)) this.partOwner = a;
    const parts = this.particles;
    this.stats.particles = this.partOwner && parts ? parts.count : 0;
    if (parts && this.partOwner) {
      const s = this.partOwner.preset.particles!;
      const pu = this.pu;
      pu.dt = sdt;
      pu.time = this.clock;
      pu.aspect = F.aspect;
      pu.velocity = fl && fluidP ? fl.velocityTex : this.black;
      pu.simTexelX = fl ? fl.texelX : 0;
      pu.simTexelY = fl ? fl.texelY : 0;
      pu.fluidAmt = fl && fluidP ? s.fluid ?? 0 : 0;
      pu.curl = s.curl * (0.5 + F.stem[3]);
      const flow = this.partOwner.rt.particleFlow;
      pu.zoomFlow = (s.zoomFlow ?? 0) * F.speed * (1 + 0.8 * F.stem[1]) * flow;
      pu.rotFlow = 0;
      pu.converge = F.build * 1.5;
      pu.drag = s.drag ?? 2.5;
      pu.lifeRate = s.life;
      pu.speed = s.speed * F.speed * flow;
      pu.spawnFrom = pu.spawnTo = s.spawn;
      pu.spawnMix = 0;
      pu.emitAngle = TAU * F.barPhase;
      pu.burst = F.hit > 0 ? 0.02 * F.act : 0;
      pu.burstSeed = Math.random() * 1000;
      pu.burstSpeed = 0.3 + F.stem[0] + F.drop;
      pu.liftX = s.lift ? s.lift[0] * F.speed : 0;
      pu.liftY = s.lift ? s.lift[1] * F.speed : 0;
      pu.spread = s.spread ?? 0.01;
      parts.update(pu);
    }

    // Fractal flame (only when a live preset is a flame)
    this.flameOwner = null;
    for (const a of list) if (a.preset.flame && (!this.flameOwner || a.weight > this.flameOwner.weight)) this.flameOwner = a;
    if (this.flameOwner && this.hq) {
      const o = this.flameOwner;
      const spec = o.preset.flame!;
      if (!this.flame) this.flame = new Flame(this.gl, this.fs);
      const n = Math.min(spec.count, this.particleCap);
      if (Math.abs(this.flame.count - n) > n * 0.1) this.flame.setCount(n);
      const rt = o.rt;
      if (F.dropStart) rt.mem.mt = (rt.mem.mt ?? 0) > 0.5 ? 0 : 1;
      rt.mem.m = (rt.mem.m ?? 0) + ((rt.mem.mt ?? 0) - (rt.mem.m ?? 0)) * (1 - Math.exp(-sdt * 0.8));
      this.flame.configure(spec, { spin: F.spin, bass: F.stem[1], vocals: F.stem[2], morph: rt.mem.m, hue: 0, bars: F.bars, beat: F.beatPulse });
      this.stats.particles += this.flame.count;
    }

    // Feedback passes
    for (const a of list) if (a.progs.feedback) this.feedbackPass(a, hybrid, sdt);
  }

  /** Uniforms shared by every per-preset program (see LIB in shaders.ts). */
  private setCommon(p: Program, a: Active, sdt: number): void {
    const F = this.F;
    const c = a.cols;
    const pr = a.preset;
    p.f2('uRes', this.width, this.height)
      .f1('uAspect', F.aspect)
      .f1('uTime', this.clock % 4096)
      .f1('uPhase', F.phase)
      .f1('uDt', sdt)
      .f1('uF60', sdt * 60)
      .f1('uSpeed', F.speed)
      .f1('uBeat', F.beatPhase)
      .f1('uBar', F.barPhase)
      .f1('uBars', F.bars)
      .f1('uBeats', F.beats)
      .f1('uBeatPulse', F.beatPulse)
      .f1('uBarPulse', this.live[5])
      .f1('uSpin', F.spin)
      .f1('uSpinStep', this.spinStep)
      .f3('uBands', this.live[0], this.live[1], this.live[2])
      .f4('uStem', F.stem[0], F.stem[1], F.stem[2], F.stem[3])
      .f4('uOnset', F.onset[0], F.onset[1], F.onset[2], F.onset[3])
      .f4('uPres', F.gate[0], F.gate[1], F.gate[2], F.gate[3])
      .f1('uCx', F.cx)
      .f1('uAct', F.act)
      .f1('uBuild', F.build)
      .f1('uDrop', F.drop)
      .f1('uFlash', this.flash)
      .f1('uLoud', F.loud)
      .f1('uMelody', F.melody)
      .f1('uKeyHue', this.live[3])
      .f1('uKeyPulse', this.live[4])
      .f1('uMinor', F.minor ? 1 : 0)
      .f3('uColA', c[0], c[1], c[2])
      .f3('uColB', c[3], c[4], c[5])
      .f3('uColC', c[6], c[7], c[8])
      .f2('uShift', a.shift[0], a.shift[1])
      .f1('uDecay', Math.pow(pr.decay ?? 0.96, sdt * 60))
      .f4v('uV', a.rt.v)
      .f4v('uSeg', a.rt.seg)
      .f4v('uSegZ', a.rt.segZ)
      .i1('uSegN', a.rt.segN)
      .f1v('uChroma', this.chroma)
      .tex('uWave', this.waveTex)
      .tex('uSpec', this.specTex);
  }

  private feedbackPass(a: Active, hybrid: boolean, sdt: number): void {
    const gl = this.gl;
    const F = this.F;
    const pp = this.pool[a.buf];
    const pr = a.preset;
    const fl = this.fluid;
    pp.write.bind();
    gl.disable(gl.BLEND);
    const p = a.progs.feedback!.use();
    this.setCommon(p, a, sdt);
    p.tex('uPrev', pp.read.t)
      .tex('uVel', fl && pr.fluid ? fl.velocityTex : this.black)
      .tex('uBC', hybrid ? this.classic!.texture : this.black)
      .f2('uSimTexel', fl ? fl.texelX : 0, fl ? fl.texelY : 0)
      .f1('uFluidAmt', pr.fluid ? pr.fluid.amount : 0)
      .f1('uBlur', pr.blur ?? 0)
      .f1('uDecaySub', (this.hq ? 0.0015 : 1.5 / 255) * (pr.floor ?? 1) * sdt * 60)
      .f1('uBCMix', hybrid ? 0.12 * (1 + F.beatPulse) : 0)
      .f1('uRingW', (pr.rings ?? 0) * F.gate[0] * (0.4 + 0.6 * F.act))
      .f4v('uRings', this.rings);
    this.fs.draw();

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    if (a.progs.curve && pr.curve && pr.curve.target !== 'top') this.drawCurve(a, 1, sdt);
    if (this.partOwner === a && pr.particles?.target === 'fb') this.drawParticles(a, 1);
    if (this.flameOwner === a && this.flame && pr.flame) {
      const spec = pr.flame;
      const fill = 1 - Math.pow(pr.decay ?? 0.9, sdt * 60);
      const w = (spec.gain * 0.06 * fill * this.width * this.height) / (this.flame.count * spec.rounds);
      for (let i = 0; i < spec.rounds; i++) {
        this.flame.update();
        pp.write.bind();
        this.flame.draw(w * (0.8 + 0.4 * F.loud), F.aspect, a.cols);
      }
    }
    gl.disable(gl.BLEND);
    pp.swap();
  }

  private drawCurve(a: Active, weight: number, sdt: number): void {
    const gl = this.gl;
    const c = a.preset.curve!;
    const p = a.progs.curve!.use();
    this.setCommon(p, a, sdt);
    p.f1('uN', c.n)
      .f1('uThick', c.thick * (this.height / 1080))
      .f1('uBright', c.bright * a.rt.curveBright * weight);
    gl.bindVertexArray(this.lineVao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, c.n * 2, c.instances ?? 1);
  }

  private drawParticles(a: Active, weight: number): void {
    const parts = this.particles!;
    const s = a.preset.particles!;
    const F = this.F;
    const size = Math.max(1, s.size * (this.height / 1080));
    const alive = s.minAct !== undefined ? smooth01((F.act - s.minAct) / 0.3) : 0.35 + 0.65 * F.act;
    if (alive <= 0.001) return;
    const bright = s.bright * weight * (s.target === 'fb' ? 0.5 : 1) * (0.7 + 0.6 * F.beatPulse * F.gate[0]);
    parts.draw(size, bright, alive, a.cols);
  }

  // ------------------------------------------------------------- compose

  private compose(state: MusicState, hybrid: boolean): void {
    const gl = this.gl;
    const F = this.F;
    const dt = this.F.dt;
    this.scene!.bind();
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const sat = 1 - 0.45 * F.build;
    for (const a of this.actives()) {
      if (a.weight < 0.001) continue;
      const p = a.progs.composite.use();
      this.setCommon(p, a, dt);
      p.tex('uFb', a.progs.feedback ? this.pool[a.buf].read.t : this.black)
        .tex('uBC', hybrid ? this.classic!.texture : this.black)
        .f1('uWeight', a.weight)
        .f1('uSat', sat)
        .f1('uSweep', num(state.keyChangePulse, 0))
        .f1('uBCMix', hybrid ? 0.25 : 0);
      this.fs.draw();
    }
    // Top layers (not part of any feedback)
    for (const a of this.actives()) {
      if (a.weight < 0.001) continue;
      if (a.progs.curve && a.preset.curve?.target === 'top') this.drawCurve(a, a.weight, dt);
      if (this.partOwner === a && a.preset.particles?.target === 'top') this.drawParticles(a, a.weight);
    }
    gl.disable(gl.BLEND);
  }

  private post(state: MusicState, classic: boolean): void {
    const gl = this.gl;
    const F = this.F;
    const beat = F.beatPulse;
    const drop = F.drop;
    const build = F.build;
    // Post parameters crossfade with the preset blend.
    let bloomP = 0;
    let expP = 0;
    let vigP = 0;
    let adaptP = 0;
    for (const a of this.actives()) {
      const w = this.from ? a.weight : 1;
      bloomP += (a.preset.bloom ?? 1) * w;
      expP += (a.preset.exposure ?? 1) * w;
      vigP += (a.preset.vignette ?? 0.45) * w;
      adaptP += (a.preset.adapt ?? 0.6) * w;
    }
    this.bloom.run(this.scene!, classic ? 0.55 : 0.9 - build * 0.3, 0.5);
    this.avgLum.write.bind();
    this.pExposure
      .use()
      .tex('uScene', this.scene!.t)
      .tex('uPrev', this.avgLum.read.t)
      .f1('uRate', 1 - Math.exp(-Math.min(state.dt || 0.016, 0.1) * 1.5))
      .f1('uFrame', this.frame % 997);
    this.fs.draw();
    this.avgLum.swap();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    const actB = 0.55 + 0.6 * F.act;
    const bloomStr = classic ? 0.25 : (bloomP * actB * (0.32 + 0.3 * build + this.flash * 0.8 + drop * 0.6)) / this.bloom.levels * 1.6;
    const exposure = classic ? 1 : expP * (0.78 + 0.12 * beat * F.act + 0.2 * drop + 0.15 * build + this.flash * 0.25);
    this.pFinal
      .use()
      .tex('uScene', this.scene!.t)
      .tex('uBloom', this.bloom.output)
      .tex('uAvg', this.avgLum.read.t)
      .f1('uKey', classic ? 1 : 0.12)
      .f1('uAdapt', classic ? 0 : adaptP)
      .f1('uContrast', classic ? 0 : 0.03)
      .f1('uBloomStr', bloomStr)
      .f1('uExposure', exposure)
      .f1('uCA', classic ? 0 : 0.0015 + drop * 0.015 * F.act + this.flash * 0.005)
      .f1('uVignette', classic ? 0.25 : vigP)
      .f1('uTonemap', classic ? 0 : 1)
      .f1('uFrame', this.frame % 1024);
    this.fs.draw();
  }

  // ------------------------------------------------------------- helpers

  private ensureClassic(): void {
    if (this.classic || !this.pool.length) return;
    this.classic = new Classic(this.gl, this.audio);
    const [w, h] = this.classicSize();
    void this.classic.load(w, h);
  }

  private classicSize(): [number, number] {
    const cap = this.mode === 'hybrid' ? 960 : 1920;
    const s = Math.min(1, cap / Math.max(this.width, this.height));
    return [Math.max(2, Math.round(this.width * s)), Math.max(2, Math.round(this.height * s))];
  }

  private resizeClassic(): void {
    if (!this.classic) return;
    const [w, h] = this.classicSize();
    this.classic.setSize(w, h);
  }

  private adaptPerformance(dt: number): void {
    const parts = this.particles;
    if (!parts || !this.partOwner) return;
    if (typeof document !== 'undefined' && document.hidden) return;
    if (this.stats.frameMs > 20 && parts.count > 32768 && this.frame > 120) {
      this.slowTime += dt;
      if (this.slowTime > 3) {
        this.particleCap = Math.max(16384, Math.floor(parts.count / 4));
        this.applyParticleCount();
        this.slowTime = 0;
        console.info(`[render] frame time above 20 ms, reducing particles to ${parts.count}`);
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

export { PRESETS, PARTICLE_STEPS };

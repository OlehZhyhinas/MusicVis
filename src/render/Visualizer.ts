// Rendering engine: enhanced (fluid + coupled feedback + particles + bloom),
// classic (butterchurn) and hybrid (butterchurn injected into the feedback).

import type { IVisualizer, MusicState, VisualMode, VisualizerOptions } from '../types';
import { Bloom } from './bloom';
import { Classic } from './classic';
import { Fluid, SPLAT_DIR, SPLAT_RADIAL, SPLAT_SWIRL } from './fluid';
import { Fullscreen, GL, PingPong, Program, Target, TexFormat, canRenderTo, createTexture, formats } from './gl';
import { Particles, ParticleUpdate } from './particles';
import { NUM_KEYS, NumParams, PRESETS, Preset, lerpParams, makeParams, paletteColors } from './presets';
import { EXPOSURE_FS, FEEDBACK_FS, FINAL_FS, FULLSCREEN_VS, SCENE_FS, WAVE_FS, WAVE_VS } from './shaders';

const TAU = Math.PI * 2;
const MAX_SIDE = 2560;
const WAVE_N = 1024;
const MAX_RINGS = 8;
const PARTICLE_STEPS = [65536, 262144, 1048576];

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

export class Visualizer implements IVisualizer {
  private gl: GL;
  private fs: Fullscreen;
  private hq: boolean;
  private hdr: TexFormat;
  private opts: VisualizerOptions;
  private mode: VisualMode = 'enhanced';
  private disposed = false;

  private pFeedback: Program;
  private pWave: Program;
  private pScene: Program;
  private pFinal: Program;
  private pExposure: Program;
  private avgLum: PingPong;
  private fluid: Fluid | null = null;
  private particles: Particles | null = null;
  private bloom: Bloom;
  private classic: Classic | null = null;

  private fb: PingPong | null = null;
  private scene: Target | null = null;
  private waveTex: WebGLTexture;
  private black: WebGLTexture;
  private waveVao: WebGLVertexArrayObject;
  private wave = new Float32Array(WAVE_N);

  private width = 1;
  private height = 1;
  private css = { w: 1, h: 1, dpr: 1 };

  // Presets and blending
  private fromP: Preset;
  private toP: Preset;
  private presetIdx = 0;
  private blendT = 1;
  private blendDur = 1;
  private lastSwitch = 0;
  private P: NumParams = makeParams();

  // Per-frame scratch (no allocations in render)
  private cols = new Float32Array(9);
  private colTmp = new Float32Array(18);
  private rings = new Float32Array(MAX_RINGS * 4);
  private ringAge = new Float32Array(MAX_RINGS).fill(99);
  private ringStr = new Float32Array(MAX_RINGS);
  private ringX = new Float32Array(MAX_RINGS);
  private ringY = new Float32Array(MAX_RINGS);
  private ringNext = 0;
  private chroma = new Float32Array(12);
  private pu: ParticleUpdate;

  // Timing / rhythm tracking
  private clock = 0;
  private frame = 0;
  private prevBarPhase = 0;
  private kalAngle = 0;
  private prevDrumOnset = 0;
  private hitCooldown = 0;
  private swirlSign = 1;
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

    const f = formats(gl);
    const cbf = gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('EXT_float_blend');
    gl.getExtension('OES_texture_float_linear');
    this.hq = !!cbf && canRenderTo(gl, f.rgba16f) && canRenderTo(gl, f.rgba32f);
    this.hdr = this.hq ? f.rgba16f : f.rgba8;
    if (!this.hq) console.warn('[render] float render targets unavailable; running reduced quality (no fluid/particles)');
    this.stats.hq = this.hq;

    this.fs = new Fullscreen(gl);
    this.pFeedback = new Program(gl, FULLSCREEN_VS, FEEDBACK_FS, 'feedback');
    this.pWave = new Program(gl, WAVE_VS, WAVE_FS, 'waveform');
    this.pScene = new Program(gl, FULLSCREEN_VS, SCENE_FS, 'scene');
    this.pFinal = new Program(gl, FULLSCREEN_VS, FINAL_FS, 'final');
    this.pExposure = new Program(gl, FULLSCREEN_VS, EXPOSURE_FS, 'exposure');
    this.avgLum = new PingPong(gl, 1, 1, [this.hdr], gl.NEAREST);
    this.bloom = new Bloom(gl, this.fs);
    if (this.hq) {
      this.fluid = new Fluid(gl, this.fs);
      this.particles = new Particles(gl, this.fs);
      this.particles.setCount(this.opts.particleCount);
    }

    this.waveTex = createTexture(gl, WAVE_N, 1, f.r16f, gl.LINEAR);
    this.black = createTexture(gl, 1, 1, f.rgba8, gl.NEAREST, new Uint8Array([0, 0, 0, 255]));
    const vao = gl.createVertexArray();
    if (!vao) throw new Error('[render] cannot create VAO');
    this.waveVao = vao;

    this.timerExt = gl.getExtension('EXT_disjoint_timer_query_webgl2');

    this.presetIdx = Math.floor(Math.random() * PRESETS.length);
    this.fromP = PRESETS[this.presetIdx];
    this.toP = this.fromP;
    lerpParams(this.P, this.fromP, this.toP, 1);

    this.pu = {
      dt: 0, time: 0, aspect: 1, velocity: this.black, simTexelX: 0, simTexelY: 0, wave: this.waveTex,
      fluidAmt: 0, curl: 0, zoomFlow: 0, rotFlow: 0, converge: 0, drag: 3, lifeRate: 0.3, speed: 0.5,
      spawnFrom: 0, spawnTo: 0, spawnMix: 0, emitCount: 3, emitAngle: 0, emitRadius: 0.3,
      burst: 0, burstSeed: 0, burstSpeed: 0,
    };

    const r = canvas.getBoundingClientRect();
    this.resize(r.width || canvas.width || 1280, r.height || canvas.height || 720, window.devicePixelRatio || 1);
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
    if (w === this.width && h === this.height && this.fb) return;
    this.width = w;
    this.height = h;
    this.stats.width = w;
    this.stats.height = h;
    const gl = this.gl;
    this.fb?.dispose();
    this.scene?.dispose();
    this.fb = new PingPong(gl, w, h, [this.hdr, this.hdr], gl.LINEAR);
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
    if (classicReady) this.classic!.autoSwitch(state, this.clock);

    this.updatePresets(state, sdt);

    if (this.mode === 'classic' && classicReady && this.classic!.render()) {
      this.drawScene(state, true);
      this.post(state, true);
    } else {
      const hybrid = this.mode === 'hybrid' && classicReady && this.classic!.render();
      this.simulate(state, sdt, hybrid);
      this.drawScene(state, false);
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
      this.classic.pick(2, this.clock);
      return;
    }
    this.switchTo((this.presetIdx + 1) % PRESETS.length, 1.5);
  }

  /** Jump to a specific enhanced preset by index or name. */
  selectPreset(which: number | string, blendSeconds = 1): void {
    const idx = typeof which === 'number' ? which : PRESETS.findIndex((p) => p.name === which);
    if (idx >= 0 && idx < PRESETS.length) this.switchTo(idx, blendSeconds);
  }

  getPresetName(): string {
    if (this.mode === 'classic' && this.classic?.ready) return this.classic.presetName;
    const base = PRESETS[this.presetIdx].name;
    return this.mode === 'hybrid' && this.classic?.ready ? `${base} + ${this.classic.presetName}` : base;
  }

  setOptions(opts: Partial<VisualizerOptions>): void {
    const prevScale = this.opts.renderScale;
    this.opts = { ...this.opts, ...opts };
    this.opts.renderScale = Math.min(1, Math.max(0.25, this.opts.renderScale));
    if (opts.particleCount !== undefined && this.particles) {
      this.particles.setCount(opts.particleCount);
      this.slowTime = 0;
      this.stats.frameMs = 16.7;
      this.lastNow = 0;
    }
    if (opts.renderScale !== undefined && opts.renderScale !== prevScale) {
      this.fb?.dispose();
      this.fb = null;
      this.resize(this.css.w, this.css.h, this.css.dpr);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    this.fb?.dispose();
    this.scene?.dispose();
    this.fb = null;
    this.scene = null;
    this.bloom.dispose();
    this.fluid?.dispose();
    this.particles?.dispose();
    this.classic?.dispose();
    this.classic = null;
    this.pFeedback.dispose();
    this.pWave.dispose();
    this.pScene.dispose();
    this.pFinal.dispose();
    this.pExposure.dispose();
    this.avgLum.dispose();
    this.fs.dispose();
    gl.deleteTexture(this.waveTex);
    gl.deleteTexture(this.black);
    gl.deleteVertexArray(this.waveVao);
    for (const q of this.queries) gl.deleteQuery(q);
    for (const q of this.pendingQueries) gl.deleteQuery(q);
    this.queries.length = 0;
    this.pendingQueries.length = 0;
  }

  // ------------------------------------------------------------- presets

  private switchTo(idx: number, seconds: number): void {
    // Snapshot the current blended state as the new "from" preset.
    const t = smooth01(this.blendT);
    const dominant = t > 0.5 ? this.toP : this.fromP;
    const snap: Preset = { ...dominant };
    for (let i = 0; i < NUM_KEYS.length; i++) snap[NUM_KEYS[i]] = this.P[NUM_KEYS[i]];
    this.fromP = snap;
    this.presetIdx = idx;
    this.toP = PRESETS[idx];
    this.blendT = 0;
    this.blendDur = Math.max(0.05, seconds);
    this.lastSwitch = this.clock;
  }

  private pickPreset(energy: 'high' | 'calm'): number {
    const pool: number[] = [];
    for (let i = 0; i < PRESETS.length; i++) if (PRESETS[i].energy === energy && i !== this.presetIdx) pool.push(i);
    if (!pool.length) return (this.presetIdx + 1) % PRESETS.length;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  private updatePresets(state: MusicState, sdt: number): void {
    const label = state.section?.label;
    if (state.sectionChanged && label) {
      if (label === 'drop' || label === 'chorus') {
        this.switchTo(this.pickPreset('high'), label === 'drop' ? 0.3 : 0.5);
        this.flash = 1;
        this.bigSplat(label === 'drop' ? 1 : 0.6);
      } else {
        this.switchTo(this.pickPreset('calm'), 3);
      }
    } else if (this.clock - this.lastSwitch > 45) {
      const high = label === 'drop' || label === 'chorus';
      this.switchTo(this.pickPreset(high ? 'high' : 'calm'), 3);
    }
    this.blendT = Math.min(1, this.blendT + sdt / this.blendDur);
    lerpParams(this.P, this.fromP, this.toP, smooth01(this.blendT));
  }

  // ------------------------------------------------------------ simulate

  private bigSplat(k: number): void {
    const fl = this.fluid;
    if (!fl) return;
    const n = 10;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * TAU + this.kalAngle;
      const r = 0.06;
      const c = Math.cos(a);
      const s = Math.sin(a);
      fl.splat(0.5 + (c * r * this.height) / this.width, 0.5 + s * r, c * 1800 * k, s * 1800 * k, 0.004, SPLAT_DIR);
    }
    fl.splat(0.5, 0.5, 1400 * k, 0, 0.02, SPLAT_RADIAL);
    this.spawnRing(0, 0, 1.6 * k);
  }

  private spawnRing(x: number, y: number, strength: number): void {
    const i = this.ringNext;
    this.ringNext = (i + 1) % MAX_RINGS;
    this.ringAge[i] = 0;
    this.ringStr[i] = strength;
    this.ringX[i] = x;
    this.ringY[i] = y;
  }

  private drumHit(state: MusicState, strength: number, barAngle: number): void {
    const P = this.P;
    const fl = this.fluid;
    const aspect = this.width / this.height;
    const F = 520 * P.splat * strength;
    const pattern = smooth01(this.blendT) > 0.5 ? this.toP.emitter : this.fromP.emitter;
    const count = Math.round(smooth01(this.blendT) > 0.5 ? this.toP.emitterCount : this.fromP.emitterCount);
    let rx = 0;
    let ry = 0;
    if (fl) {
      switch (pattern) {
        case 'radial':
          fl.splat(0.5 + (Math.random() - 0.5) * 0.05, 0.5 + (Math.random() - 0.5) * 0.05, F * 1.2, 0, 0.012, SPLAT_RADIAL);
          break;
        case 'swirl':
          this.swirlSign = state.barIndex % 2 === 0 ? 1 : -1;
          fl.splat(0.5, 0.5, F * this.swirlSign, 0, 0.02, SPLAT_SWIRL);
          break;
        case 'orbit':
          for (let i = 0; i < count; i++) {
            const a = barAngle + (i / count) * TAU;
            const c = Math.cos(a);
            const s = Math.sin(a);
            const r = 0.28;
            fl.splat(0.5 + (c * r) / aspect, 0.5 + s * r, -s * F, c * F, 0.003, SPLAT_DIR);
          }
          rx = Math.cos(barAngle) * 0.28;
          ry = Math.sin(barAngle) * 0.28;
          break;
        case 'jets':
          for (let i = 0; i < 2; i++) {
            const a = barAngle + i * Math.PI;
            const c = Math.cos(a);
            const s = Math.sin(a);
            const r = 0.38;
            fl.splat(0.5 + (c * r) / aspect, 0.5 + s * r, -c * F * 1.3, -s * F * 1.3, 0.004, SPLAT_DIR);
          }
          break;
        default:
          for (let i = 0; i < 2; i++) {
            const a = Math.random() * TAU;
            fl.splat(0.2 + Math.random() * 0.6, 0.2 + Math.random() * 0.6, Math.cos(a) * F, Math.sin(a) * F, 0.004, SPLAT_DIR);
          }
      }
    }
    this.spawnRing(rx, ry, 0.35 + 0.9 * strength);
  }

  private simulate(state: MusicState, sdt: number, hybrid: boolean): void {
    const gl = this.gl;
    const P = this.P;
    const fb = this.fb!;
    const w = this.width;
    const h = this.height;
    const aspect = w / h;
    const f60 = sdt * 60;
    const t = smooth01(this.blendT);
    const stems = state.stems;
    const bass = stems?.bass ?? 0;
    const drums = stems?.drums ?? 0;
    const voc = stems?.vocals ?? 0;
    const oth = stems?.other ?? 0;
    const build = state.buildIntensity || 0;
    const beat = state.beatPulse || 0;
    const drop = state.dropPulse || 0;

    // Rhythm: rotation locked to bar progress.
    let dBar = state.barPhase - this.prevBarPhase;
    if (dBar < 0) dBar += 1;
    if (!state.playing || dBar > 0.5) dBar = ((state.bpm || 120) / 60 / 4) * sdt * (state.playing ? 1 : 0.3);
    this.prevBarPhase = state.barPhase;
    const flipSrc = t > 0.5 ? this.toP : this.fromP;
    const sign = flipSrc.rotFlip && state.barIndex % 2 === 1 ? -1 : 1;
    const rotRad = TAU * dBar * P.rot * sign;
    this.kalAngle = (this.kalAngle + TAU * dBar * P.kalRot) % TAU;
    const barAngle = TAU * state.barPhase;

    // Zoom: bass + beat kick + accelerating build + drop release.
    const zoomRate = (P.zoom - 1) + P.zoomBass * bass + P.zoomBeat * beat + build * build * 0.035 + drop * 0.03;
    const zoom = 1 + zoomRate * f60;
    const decayA = Math.pow(P.decayA, f60);
    const decayB = Math.pow(P.decayB, f60);
    const acc = 1 - decayA;

    // Drums: onsets and beats trigger splats and rings.
    this.hitCooldown -= sdt;
    const on = state.stemOnsets?.drums ?? 0;
    const onsetEdge = on > 0.35 && this.prevDrumOnset <= 0.35;
    this.prevDrumOnset = on;
    if (state.playing && this.hitCooldown <= 0 && (onsetEdge || state.onBeat)) {
      this.drumHit(state, Math.max(on, 0.3) * (0.4 + drums), barAngle);
      this.hitCooldown = 0.1;
    }

    // Rings
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
      this.rings[o + 3] = this.ringStr[i] * Math.exp(-age * 3.2) * P.drumsW * 0.55 * Math.min(1, f60 * 1.2);
    }

    // Palette
    const hue = (state.keyHue || 0) + P.hueOffset;
    const sat = P.sat * (0.72 + 0.4 * voc) * (1 - 0.55 * build);
    paletteColors(this.fromP.palette, hue, sat, this.colTmp, 0);
    paletteColors(this.toP.palette, hue, sat, this.colTmp, 9);
    for (let i = 0; i < 9; i++) this.cols[i] = this.colTmp[i] + (this.colTmp[i + 9] - this.colTmp[i]) * t;
    const c = this.cols;

    // Waveform (temporal + spatial smoothing)
    const wf = state.waveform;
    if (wf && wf.length) {
      const n = Math.min(wf.length, WAVE_N);
      let prev = wf[0];
      for (let i = 0; i < n; i++) {
        const cur = wf[i];
        const nx = wf[Math.min(n - 1, i + 1)];
        const s = (prev + cur * 2 + nx) * 0.25;
        this.wave[i] += (s - this.wave[i]) * 0.6;
        prev = cur;
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.waveTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WAVE_N, 1, gl.RED, gl.FLOAT, this.wave);

    // Fluid
    const fl = this.fluid;
    if (fl) {
      fl.dissipation = 0.6;
      fl.step(sdt, this.clock, P.noise * (0.25 + 0.9 * oth) * 0.55 * f60, P.vorticity, aspect);
    }

    // Particles (update)
    const parts = this.particles;
    if (parts && fl) {
      const pu = this.pu;
      pu.dt = sdt;
      pu.time = this.clock;
      pu.aspect = aspect;
      pu.velocity = fl.velocityTex;
      pu.simTexelX = fl.texelX;
      pu.simTexelY = fl.texelY;
      pu.fluidAmt = P.partFluid;
      pu.curl = P.partCurl * (0.5 + oth);
      pu.zoomFlow = (zoom - 1) / Math.max(sdt, 1e-3);
      pu.rotFlow = rotRad / Math.max(sdt, 1e-3);
      pu.converge = build * 2.2;
      pu.drag = 2.5;
      pu.lifeRate = P.partLife;
      pu.speed = P.partSpeed * (0.6 + drums);
      pu.spawnFrom = this.fromP.spawn;
      pu.spawnTo = this.toP.spawn;
      pu.spawnMix = t;
      pu.emitCount = t > 0.5 ? this.toP.emitterCount : this.fromP.emitterCount;
      pu.emitAngle = barAngle;
      pu.emitRadius = 0.28;
      pu.burst = state.playing && state.onBeat ? 0.03 + 0.05 * drums : 0;
      if (state.sectionChanged && state.section?.label === 'drop') pu.burst = 0.5;
      pu.burstSeed = Math.random() * 1000;
      pu.burstSpeed = 0.5 + drums + drop * 1.5;
      parts.update(pu);
    }

    // Feedback pass (MRT: A and B)
    fb.write.bind();
    const fp = this.pFeedback.use();
    fp.f2('uTexel', 0, 0)
      .tex('uPrevA', fb.read.tex[0])
      .tex('uPrevB', fb.read.tex[1])
      .tex('uVel', fl ? fl.velocityTex : this.black)
      .tex('uBC', hybrid ? this.classic!.texture : this.black)
      .f2('uRes', w, h)
      .f2('uSimTexel', fl ? fl.texelX : 0, fl ? fl.texelY : 0)
      .f1('uAspect', aspect)
      .f1('uDt', sdt)
      .f1('uTime', this.clock)
      .f1('uZoom', zoom)
      .f1('uZoomExp', P.zoomExp)
      .f1('uRot', rotRad)
      .f1('uBZoom', P.bZoom)
      .f1('uBRot', P.bRot)
      .f2('uTrans', P.trans * 0.0015 * Math.sin(this.clock * 0.31) * f60, P.trans * 0.0015 * Math.cos(this.clock * 0.23) * f60)
      .f4('uWarp0', this.fromP.warpFn, P.warpAmt * (1 + build * 0.6) * f60, P.warpSpeed, P.warpScale)
      .f4('uWarp1', this.toP.warpFn, P.warpAmt * (1 + build * 0.6) * f60, P.warpSpeed, P.warpScale)
      .f1('uWarpMix', t)
      .f1('uFluid', fl ? P.fluid : 0)
      .f1('uCouple', P.couple)
      .f1('uBlur', P.blur)
      .f1('uDecaySub', this.hq ? 0.004 * f60 : 1.5 / 255)
      .f1('uHueDrift', P.hueDrift * f60)
      .f2('uDecay', decayA, decayB)
      .f3('uColA', c[0], c[1], c[2])
      .f3('uColB', c[3], c[4], c[5])
      .f3('uColC', c[6], c[7], c[8])
      .f4v('uRings', this.rings);
    const sides = t > 0.5 ? this.toP.bassSides : this.fromP.bassSides;
    fp.f4(
      'uBass',
      0.07 + 0.15 * bass * P.bassSize + 0.02 * beat,
      sides,
      barAngle * (state.barIndex % 2 === 1 && flipSrc.rotFlip ? -1 : 1),
      P.bassW * (0.15 + bass * 1.2) * acc * 1.4,
    );
    fp.f4('uVocal', P.vocalsW * Math.pow(voc, 1.4) * acc * 2, 0.006 + 0.016 * voc, this.clock * 0.9, 0);
    if (state.chroma && state.chroma.length >= 12) for (let i = 0; i < 12; i++) this.chroma[i] = state.chroma[i];
    fp.f1v('uChroma', this.chroma);
    fp.f4('uChromaP', 0.36, barAngle, P.chromaW * (0.3 + oth) * acc * 3.5, P.sparkle * oth * 0.002);
    fp.f1('uKeyHue', state.keyHue || 0);
    fp.f1('uBCMix', hybrid ? 0.12 * (1 + beat) : 0);
    fp.f1('uBuild', build);
    gl.disable(gl.BLEND);
    this.fs.draw();

    // Waveform
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const waveBright = P.waveW * (0.35 + 0.6 * (state.loudness || 0) + 2.5 * beat + drop * 3) * acc * 3.5;
    const thick = P.waveThick * (h / 1080) * (1 + beat * 0.6);
    const wp = this.pWave.use();
    wp.tex('uWave', this.waveTex)
      .f1('uAmp', P.waveAmp * (0.8 + 0.6 * (state.mid || 0)))
      .f1('uThick', thick)
      .f1('uAspect', aspect)
      .f1('uAngle', barAngle)
      .f1('uN', WAVE_N)
      .f1('uMirrorW', P.waveMirror)
      .f2('uRes', w, h)
      .f3('uColA', c[0], c[1], c[2])
      .f3('uColC', c[6], c[7], c[8])
      .f2('uRoute', 1, 0.3);
    gl.bindVertexArray(this.waveVao);
    if (this.fromP.waveStyle === this.toP.waveStyle) {
      wp.f1('uStyle', this.toP.waveStyle).f1('uBright', waveBright);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, WAVE_N * 2, 2);
    } else {
      wp.f1('uStyle', this.fromP.waveStyle).f1('uBright', waveBright * (1 - t));
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, WAVE_N * 2, 2);
      wp.f1('uStyle', this.toP.waveStyle).f1('uBright', waveBright * t);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, WAVE_N * 2, 2);
    }

    // Particles into the feedback so they leave trails.
    if (parts) {
      const size = Math.max(1, P.partSize * (h / 1080));
      const density = (parts.count * size * size) / (w * h);
      const bright = (P.partW * 0.035 * acc * (0.6 + 0.8 * oth + 0.6 * beat)) / Math.max(density, 0.02);
      parts.draw(size, bright, c, 0.75, 0.35);
    }
    gl.disable(gl.BLEND);
    fb.swap();
  }

  // ------------------------------------------------------------- compose

  private drawScene(state: MusicState, classic: boolean): void {
    const P = this.P;
    const t = smooth01(this.blendT);
    const voc = state.stems?.vocals ?? 0;
    const build = state.buildIntensity || 0;
    this.scene!.bind();
    const sp = this.pScene.use();
    if (classic) {
      sp.tex('uA', this.classic!.texture).tex('uB', this.black).f1('uLinearize', 1).f3('uKal', 0, 0, 0).f1('uMirror', 0);
      sp.f1('uSat', 1).f1('uHueShift', 0);
    } else {
      const kf = this.fromP.kaleido;
      const kt = this.toP.kaleido;
      sp.tex('uA', this.fb!.read.tex[0])
        .tex('uB', this.fb!.read.tex[1])
        .f1('uLinearize', 0)
        .f3('uKal', kf, kt, kf === kt ? 0 : t)
        .f1('uKalRot', this.kalAngle)
        .f1('uMirror', (t > 0.5 ? this.toP.mirror : this.fromP.mirror) ? 1 : 0)
        .f1('uSat', 1 + 0.2 * voc - 0.45 * build)
        .f1('uHueShift', voc * 0.3 * Math.sin(this.clock * 0.37));
    }
    sp.f1('uAspect', this.width / this.height)
      .f1('uBMix', P.bMix).f1('uSweep', state.keyChangePulse || 0);
    this.fs.draw();
  }

  private post(state: MusicState, classic: boolean): void {
    const gl = this.gl;
    const P = this.P;
    const beat = state.beatPulse || 0;
    const drop = state.dropPulse || 0;
    const build = state.buildIntensity || 0;
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
    const bloomStr = classic ? 0.25 : P.bloom * (0.32 + 0.3 * build + this.flash * 0.8 + drop * 0.6) / this.bloom.levels * 1.6;
    const exposure = classic ? 1 : P.exposure * (0.72 + 0.15 * beat + 0.2 * drop + 0.15 * build + this.flash * 0.25);
    this.pFinal
      .use()
      .tex('uScene', this.scene!.t)
      .tex('uBloom', this.bloom.output)
      .tex('uAvg', this.avgLum.read.t)
      .f1('uKey', classic ? 1 : 0.12)
      .f1('uAdapt', classic ? 0 : 0.75)
      .f1('uContrast', classic ? 0 : 0.035)
      .f1('uBloomStr', bloomStr)
      .f1('uExposure', exposure)
      .f1('uCA', classic ? 0 : 0.002 + drop * 0.02 + this.flash * 0.006)
      .f1('uVignette', classic ? 0.25 : P.vignette)
      .f1('uTonemap', classic ? 0 : 1)
      .f1('uFrame', this.frame % 1024);
    this.fs.draw();
  }

  // ------------------------------------------------------------- helpers

  private ensureClassic(): void {
    if (this.classic || !this.fb) return;
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
    if (!parts) return;
    this.stats.particles = parts.count;
    if (typeof document !== 'undefined' && document.hidden) return;
    if (this.stats.frameMs > 20 && parts.count > 262144 && this.frame > 120) {
      this.slowTime += dt;
      if (this.slowTime > 3) {
        parts.setCount(262144);
        this.slowTime = 0;
        console.info('[render] frame time above 20 ms, reducing particles to 262144');
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

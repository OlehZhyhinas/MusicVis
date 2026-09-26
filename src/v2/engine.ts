// V2 renderer: draws any genome. A Stage owns a resolution (feedback pool,
// HDR scene, bloom, exposure, the shared fluid / particle / flame sims) and a
// Signals object (music -> per-frame values and analysis textures). The main
// view is one Stage with up to two Slots crossfading; screening and
// thumbnails use a second, small offscreen Stage on the same GL context so
// compiled programs are shared.

import { packSuperscope } from './genes/superscope';
import { resetCurveBlend, setCurveBlend } from './genes/blend';
import { Water } from './genes/waterSim';
import { packMosaic } from './genes/mosaic';
import { packTunnel } from './genes/tunnel';
import { hueMapUniforms } from './genes/huemap';
import { reliefUniforms } from './genes/relief';
import type { AnalysisResult, MusicState, Section, StemName } from '../types';
import { grooveClock, grooveOffset, type GrooveOffset } from './genes/groove';
import type { GrooveStats, NoteStats, TimbreStats } from '../types';
import { timbreLook, timbreSource, timbreTone, timbreUniforms, type TimbreLook } from './genes/timbre';
import { Bloom } from '../render/bloom';
import { Flame, type FlameSpec } from '../render/flame';
import { Fluid } from '../render/fluid';
import { Fullscreen, GL, PendingProgram, PingPong, Program, Target, TexFormat, canRenderTo, createTexture, formats } from '../render/gl';
import { Particles, type ParticleUpdate } from '../render/particles';
import { EXPOSURE_FS, FINAL_FS, FULLSCREEN_VS, SCALE_FS } from '../render/shaders';
import { IDENTITY_POSE, blendPoses, cameraUniforms, choreoPose, cueOf, type ChoreoPose } from './genes/choreo';
import { accentPlan, applyAccents, type AccentInput, type AccentPlan } from './genes/accent';
import { HarmonyMotor, IDLE_HARMONY, type HarmonyInputs, type HarmonyOut } from './genes/harmony';
import { DejaVuBank } from './genes/dejavuGpu';
import { NEUTRAL_NUDGE, easeNudge, lineKick, lyricTarget, type LyricNudge } from './genes/lyrics';
import { CaptionLayer } from './genes/lyricsGpu';
import {
  CARRIER_SCHEMA, TONE_SCHEMA, PALETTE_SCHEMAS, MAPPING_SCHEMAS, MAPPING_KINDS, DEFORM_SCHEMAS, EMIT_SCHEMAS, FUSE_SCHEMA, MATERIAL_SCHEMAS, MOTION_SCHEMAS, OP_SCHEMAS,
  PLACE_SCHEMAS, SHAPE_CLASS, SHAPE_SCHEMAS, MAX_DRAW, MAX_REACTIONS, clampParam, cloneGenome, drawOpId, schemaFor, structuralKey,
  type BodyGene, type FlameVar, type GeneGroup, type Genome, type OpGene, type PaletteGene, type Params, type Scheme, type Schema,
  type ShapeGene, type Signal,
  paletteHue,
} from './genome';
import { BODY_VEC4, COPY_SLOTS, WAVE_FS, WAVE_VS, buildSources } from './glsl';
import { Physarum } from './genes/physarumGpu';
import { Boids } from './genes/boidsGpu';
import { FLOCK_GAIN, FLOCK_OVERLAY_GAIN, flockOverlaySize } from './genes/boids';
import { SLIME_GAIN, slimeDisplayScale } from './genes/physarum';
import { Ecosystem } from './genes/ecosystemGpu';
import { ecoCuts, ecoFieldScale } from './genes/ecosystem';
import { packCells } from './genes/cells';
import { packBeams } from './genes/beams';
import { SCENE_VEC4, packScene } from './genes/raymarch';
import { LAND_VEC4, packLandscape } from './genes/landscape';
import { packTonnetz } from './genes/tonnetz';
import { NOTE_W, NoteHistory, packNotes } from './genes/notes';
import { LandWorld } from './genes/landscapeGpu';
import { packCymatics } from './genes/cymatics';
import { DriftDriver } from './genes/driftPlay';

/**
 * A body's musical clock this frame (from its feel gene): mul scales its periodic motion, s = div / 4
 * scales its event spacing; bars / spin / phases are the body's own time.
 */
interface BodyClock {
  mul: number;
  s: number;
  lock: boolean;
  bars: number;
  spin: number;
  barPhase: number;
  beatPhase: number;
}

/** One copy of a body this frame: position, angle, scale, level (brightness), hue offset, previous position. */
interface Copy {
  x: number;
  y: number;
  a: number;
  s: number;
  level: number;
  hue: number;
  px?: number;
  py?: number;
}

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
const WAVE_RATIOS: [number, number][] = [[2, 3], [3, 4], [3, 5], [4, 5], [2, 5], [5, 6]];
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
/** The palette's three slots around `hue` (the key hue plus the palette's offset). */
function paletteColors(pal: PaletteGene, hue: number, sat: number, out: Float32Array): void {
  const sa = clamp01(sat);
  if (pal.kind === 'free') {
    hsvLin(hue, sa, 1, out, 0);
    hsvLin(hue + pal.p.s1, sa, 0.9, out, 3);
    hsvLin(hue + pal.p.s2, sa * 0.85, 1, out, 6);
    return;
  }
  const s = SCHEME_OFFSETS[pal.kind];
  const k = pal.p.spread;
  hsvLin(hue + s[0], sa, s[3], out, 0);
  hsvLin(hue + s[1] * k, sa * (pal.kind === 'mono' ? 0.5 : 1), s[4], out, 3);
  hsvLin(hue + s[2] * k, sa * 0.85, s[5], out, 6);
}

// ------------------------------------------------------------- signals

export interface Frame {
  time: number; dt: number; phase: number; act: number; cx: number; speed: number; spin: number;
  bars: number; beats: number; barIndex: number; beatIndex: number; barPhase: number; beatPhase: number;
  beatPulse: number; onBeat: boolean; stem: Float32Array; onset: Float32Array; gate: Float32Array;
  loud: number; melody: number; build: number; drop: number; keyTonic: number; minor: boolean;
  sectionIndex: number; aspect: number; hit: number; hitPulse: number; dropStart: boolean; keyHue: number; keyPulse: number;
  barPulse: number; bpm: number;
  /** Harmony map: tension 0..1 and the resolve / chord change / modulation pulses. */
  tension: number; resolve: number; chordPulse: number; modPulse: number;
  /** Harmony map: the current chord (-1 none) and its Tonnetz position relative to the tonic. */
  chord: number; tonnetzX: number; tonnetzY: number;
  /** Timing feel (swing, push, humanity, syncopation; neutral when not analysed). */
  groove: GrooveStats;
  /** Timbre per stem and the mix (undefined when not analysed). */
  timbre: Record<'mix' | StemName, TimbreStats> | undefined;
  /** Melody notes and articulation (undefined when not analysed). */
  notes: NoteStats | undefined;
  /** Beat surge envelope: fast attack, slow ease; cruises with loudness, jumps on drops (0..~3). */
  surge: number;
  /** Lyrics: new-line pulse, and the words' valence / arousal (the music's own mood where no words are sung). */
  line: number; valence: number; arousal: number;
  /** Hooks: inside a repeat (0/1), phase through it, the repeat-start and motif-note pulses, the note index (-1 outside), the hook id (-1 outside). */
  hookOn: number; hookPhase: number; hookPulse: number; hookNotePulse: number; hookNote: number; hookId: number;
}

/** Music state -> per-frame values and the waveform / spectrum textures (one per Stage). */
export class Signals {
  readonly F: Frame = {
    time: 0, dt: 0, phase: 0, act: 0.5, cx: 0.5, speed: 1, spin: 0, bars: 0, beats: 0, barIndex: 0, beatIndex: 0,
    barPhase: 0, beatPhase: 0, beatPulse: 0, onBeat: false, stem: new Float32Array(4), onset: new Float32Array(4),
    gate: new Float32Array(4), loud: 0, melody: 0.5, build: 0, drop: 0, keyTonic: 0, minor: false, sectionIndex: 0,
    aspect: 1, hit: 0, hitPulse: 0, dropStart: false, keyHue: 0, keyPulse: 0, barPulse: 0, bpm: 120, surge: 0,
    tension: 0, resolve: 0, chordPulse: 0, modPulse: 0, chord: -1, tonnetzX: 0.5, tonnetzY: 0.2887, line: 0, valence: 0.5, arousal: 0.5,
    hookOn: 0, hookPhase: 0, hookPulse: 0, hookNotePulse: 0, hookNote: -1, hookId: -1,
    groove: { swing: 0, push: 0, humanity: 0, synco: 0 },
    timbre: undefined,
    notes: undefined,
  };
  /** Seconds the vocals have been silent (a new line's fallback pulse without lyrics). */
  private vocalRest = 0;
  spinStep = 0;
  clock = 0;
  songCx: number | null = null;
  readonly waveTex: WebGLTexture;
  readonly specTex: WebGLTexture;
  /** Melody history and recent notes for the notes shape (genes/notes.ts). */
  readonly noteTex: WebGLTexture;
  private readonly noteHist = new NoteHistory();
  readonly chroma = new Float32Array(12);
  private wave = new Float32Array(WAVE_N);
  private waveTmp = new Float32Array(WAVE_N);
  readonly spec = new Float32Array(SPEC_N);
  private rms = 0.1;
  private prevBarPhase = 0;
  private prevDrumOnset = 0;
  private hitCooldown = 0;
  private lastSection = -1;
  /** 1 on a section change, easing back to 0 over about a bar. */
  sectionPulse = 0;

  constructor(private gl: GL) {
    const f = formats(gl);
    this.waveTex = createTexture(gl, WAVE_N, 1, f.r16f, gl.LINEAR);
    this.specTex = createTexture(gl, SPEC_N, 1, f.r16f, gl.LINEAR);
    this.noteTex = createTexture(gl, NOTE_W, 2, f.rgba16f, gl.LINEAR);
  }

  reset(): void {
    this.F.act = 0.5;
    this.F.phase = 0;
    this.F.spin = 0;
    this.F.surge = 0;
    this.clock = 0;
    this.wave.fill(0);
    this.spec.fill(0);
    this.noteHist.reset();
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
    F.barPulse = num(state.barPulse, 0);
    F.tension = num(state.tension, 0);
    F.resolve = num(state.resolvePulse, 0);
    F.chordPulse = num(state.chordPulse, 0);
    F.modPulse = num(state.modulationPulse, 0);
    F.chord = num(state.chord, -1);
    F.tonnetzX = num(state.tonnetzX, 0.5);
    F.tonnetzY = num(state.tonnetzY, 0.2887);
    const gr = state.groove;
    F.groove.swing = num(gr?.swing, 0);
    F.groove.push = num(gr?.push, 0);
    F.groove.humanity = num(gr?.humanity, 0);
    F.groove.synco = num(gr?.synco, 0);
    F.timbre = state.timbre;
    F.notes = state.notes;
    F.hookOn = num(state.hookOn, 0);
    F.hookPhase = num(state.hookPhase, 0);
    F.hookPulse = num(state.hookPulse, 0);
    F.hookNotePulse = num(state.hookNotePulse, 0);
    F.hookNote = num(state.hookNote, -1);
    F.hookId = num(state.hookId, -1);
    F.sectionIndex = Math.max(0, num(state.sectionIndex, 0));
    this.sectionPulse *= Math.exp(-dt * 1.5);
    if (this.lastSection >= 0 && F.sectionIndex !== this.lastSection) this.sectionPulse = 1;
    this.lastSection = F.sectionIndex;
    F.dropStart = !!state.sectionChanged && state.section?.label === 'drop';
    F.speed = (0.4 + 0.75 * F.act) * (1 + 0.5 * F.build) + 0.4 * F.drop * F.act;
    F.phase = (F.phase + dt * F.speed) % 4096;

    const bpm = num(state.bpm, 120) || 120;
    F.bpm = bpm;
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

    const surge = 1.3 * F.beatPulse * F.gate[0] + 0.6 * F.loud + 2.5 * F.drop;
    F.surge = approach(F.surge, surge, surge > F.surge ? 18 : 3, dt);
    this.updateLyrics(state, dt);
  }

  /**
   * Lyric signals: with lyrics, the words' mood and a pulse per line; without (or between lines),
   * the music's mood (major / minor key, activity) and a pulse when the vocals come back in.
   */
  private updateLyrics(state: MusicState, dt: number): void {
    const F = this.F;
    const pres = clamp01(num(state.lyricPresence, 0));
    const musV = (F.minor ? 0.38 : 0.62) - 0.15 * (F.tension - 0.3);
    // The words lead the mood while they are sung, the music keeps a share (arousal is mostly heard).
    F.valence = clamp01(musV + (num(state.lyricValence, 0.5) - musV) * pres * 0.7);
    F.arousal = clamp01(F.act + (num(state.lyricArousal, 0.5) - F.act) * pres * 0.5);
    if (state.lyricPulse !== undefined) F.line = num(state.lyricPulse, 0);
    else {
      F.line *= Math.exp(-dt * 4);
      if (F.gate[2] > 0.5) {
        if (this.vocalRest > 1.2 && state.playing) F.line = 1;
        this.vocalRest = 0;
      } else this.vocalRest += dt;
    }
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
    this.noteHist.push(this.F.notes, dt);
    gl.bindTexture(gl.TEXTURE_2D, this.noteTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, NOTE_W, 2, gl.RGBA, gl.FLOAT, this.noteHist.data);
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
      case 'surge': return Math.min(1.5, F.surge);
      case 'barpulse': return F.barPulse;
      case 'section': return this.sectionPulse;
      case 'tension': return F.tension;
      case 'resolve': return F.resolve;
      case 'chordchange': return F.chordPulse;
      case 'modulation': return F.modPulse;
      case 'swing': return F.groove.swing;
      case 'push': return Math.abs(F.groove.push);
      case 'humanity': return F.groove.humanity;
      case 'synco': return F.groove.synco;
      case 'bright': return num(F.timbre?.mix.bright, 0);
      case 'noisy': return num(F.timbre?.mix.noise, 0);
      case 'rough': return num(F.timbre?.mix.rough, 0);
      case 'attack': return num(F.timbre?.mix.attack, 0);
      case 'noteon': return num(F.notes?.on, 0);
      case 'held': return num(F.notes?.held, 0);
      case 'legato': return num(F.notes?.legato, 0.5);
      case 'glide': return Math.min(1, Math.abs(num(F.notes?.glide, 0)) / 12);
      case 'vibrato': return Math.min(1, num(F.notes?.vibrato, 0) / 0.6);
      case 'voice': return num(F.notes?.voice, 0);
      case 'hook': return Math.max(F.hookPulse, 0.8 * F.hookNotePulse);
      case 'hookphase': return F.hookPhase;
      case 'hookon': return F.hookOn;
      case 'line': return F.line;
      case 'valence': return F.valence;
      case 'arousal': return F.arousal;
    }
  }

  melodic(): number {
    const F = this.F;
    return Math.max(F.stem[2], F.stem[3] * 0.85, F.loud * 0.35);
  }

  dispose(): void {
    this.gl.deleteTexture(this.waveTex);
    this.gl.deleteTexture(this.specTex);
    this.gl.deleteTexture(this.noteTex);
  }
}

// ------------------------------------------------------- program cache

export interface GenomePrograms {
  feedback: Program;
  composite: Program;
  /** The ray-marched scene pass (genomes with a 'scene' body). */
  scene?: Program;
  /** The landscape pass (genomes with a 'landscape' body). */
  land?: Program;
}
interface CacheEntry {
  /** feedback, composite and (optionally) the scene pass. */
  pending: PendingProgram[];
  /** Which optional passes follow feedback and composite in `pending`, in order. */
  extra: ('scene' | 'land')[];
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
    const extra = (['scene', 'land'] as const).filter((k) => src[k]);
    this.entries.set(key, {
      pending: [
        new PendingProgram(this.gl, FULLSCREEN_VS, src.feedback, `v2-fb ${key}`, this.parallel),
        new PendingProgram(this.gl, FULLSCREEN_VS, src.composite, `v2-comp ${key}`, this.parallel),
        ...extra.map((k) => new PendingProgram(this.gl, FULLSCREEN_VS, src[k]!, `v2-${k} ${key}`, this.parallel)),
      ],
      extra,
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
    const ready = e.pending.map((p) => p.poll(now));
    if (ready.includes(false)) return null;
    const err = e.pending.find((p) => p.error)?.error;
    if (err) {
      console.error(err);
      e.failed = err;
      for (const p of e.pending) p.program?.dispose();
      return null;
    }
    e.done = { feedback: e.pending[0].program!, composite: e.pending[1].program! };
    e.extra.forEach((k, i) => (e.done![k] = e.pending[2 + i].program!));
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
      e.done?.scene?.dispose();
      e.done?.land?.dispose();
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
      e.done?.scene?.dispose();
      e.done?.land?.dispose();
    }
    this.entries.clear();
  }
}

// ---------------------------------------------------------------- slots

/** How one curve body is drawn this frame (one draw per copy). */
interface CurveDraw {
  body: number;
  top: boolean;
  bright: number;
  thick: number;
  n: number;
  soft: number;
  dash: number;
  /** Copies: x, y, angle, scale, flipX, flipY. */
  copies: number[][];
  deform: number;
  dp: [number, number, number, number];
  /** Blend mode (genes/blend.ts). */
  blend: number;
}

/** One genome running at a Stage: its feedback buffer and JS-side state. */
export class Slot {
  weight = 1;
  mem: Record<string, number> = {};
  readonly cols = new Float32Array(9);
  readonly opA = new Float32Array(24);
  readonly opB = new Float32Array(24);
  readonly nb: number;
  /** Per-body uniform slots (BODY_VEC4 vec4 each), copies, curve slots and draw ops. */
  readonly bd: Float32Array;
  readonly cp: Float32Array;
  readonly cq: Float32Array;
  readonly wv: Float32Array;
  readonly drA: Float32Array;
  readonly drB: Float32Array;
  readonly seg = new Float32Array(48 * 4);
  readonly segZ = new Float32Array(48);
  segN = 0;
  curves: CurveDraw[] = [];
  rem: number[] = [];
  shift = [0, 0];
  decay = 0.9;
  delta = new Map<string, number>();
  flameSpec: FlameSpec | null = null;
  flameMorph = 0;
  flameHue = 0;
  flameClock = { spin: 0, bars: 0 };
  /** The body throwing sparks (-1: none) and its spawn settings. */
  sparks = -1;
  spawn = { mode: 4, count: 3, angle: 0, radius: 0.3 };
  /** The body growing a physarum network (-1: none). */
  slime = -1;
  /** The body leading a flock of boids (-1: none). */
  flock = -1;
  /** The body running the stem ecosystem (-1: none). */
  eco = -1;
  /** Per reaction this frame: the source signal and the response after its curve (live meters). */
  readonly meters = new Float32Array(MAX_REACTIONS * 2);
  /** Ray-marched scene: its pass uniforms and its reduced-resolution target. */
  readonly scn = new Float32Array(SCENE_VEC4 * 4);
  sceneT: Target | null = null;
  /** Landscape: its pass uniforms and its reduced-resolution target. */
  readonly lsu = new Float32Array(LAND_VEC4 * 4);
  landT: Target | null = null;

  /** The genome the slot stands for (saved, edited, shown in the editor); `genome` is what renders (a drift may differ). */
  home: Genome;
  /** structuralKey(genome). */
  key: string;

  constructor(
    public genome: Genome,
    readonly progs: GenomePrograms,
    readonly fb: PingPong,
  ) {
    this.home = genome;
    this.key = structuralKey(genome);
    this.nb = genome.bodies.length;
    this.bd = new Float32Array(this.nb * BODY_VEC4 * 4);
    this.cp = new Float32Array(this.nb * COPY_SLOTS * 4);
    this.cq = new Float32Array(this.nb * COPY_SLOTS * 4);
    this.wv = new Float32Array(this.nb * 16);
    this.drA = new Float32Array(this.nb * 12);
    this.drB = new Float32Array(this.nb * 12);
    this.sparks = genome.bodies.findIndex((b) => b.emit.kind === 'sparks');
    this.slime = genome.bodies.findIndex((b) => b.emit.kind === 'slime');
    this.flock = genome.bodies.findIndex((b) => b.emit.kind === 'flock');
    this.eco = genome.bodies.findIndex((b) => b.emit.kind === 'ecosystem');
  }

  /** Swaps in a genome with the same structure (same programs, same uniform layout): parameters only. */
  retarget(g: Genome, key?: string): void {
    this.genome = g;
    if (key !== undefined) this.key = key;
    this.sparks = g.bodies.findIndex((b) => b.emit.kind === 'sparks');
    this.slime = g.bodies.findIndex((b) => b.emit.kind === 'slime');
    this.flock = g.bodies.findIndex((b) => b.emit.kind === 'flock');
    this.eco = g.bodies.findIndex((b) => b.emit.kind === 'ecosystem');
  }

  /** Parameter with this frame's reactions applied, clamped to its spec. */
  P(group: GeneGroup, i: number, p: Params, k: string, schema: Schema): number {
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
  water: Water | null = null;
  private waterBeat = -1;
  particles: Particles | null = null;
  slime: Physarum | null = null;
  /** The song's world map for landscape bodies (created with the first one). */
  land: LandWorld | null = null;
  private slimeOwner: Slot | null = null;
  private readonly slimeCopies = new Float32Array(COPY_SLOTS * 3);
  flock: Boids | null = null;
  private flockOwner: Slot | null = null;
  eco: Ecosystem | null = null;
  private ecoOwner: Slot | null = null;
  flame: Flame | null = null;
  private pu: ParticleUpdate;
  flash = 0;
  frame = 0;
  /** Choreography: each slot's pose this frame, their blend, and the final pass camera. */
  private poses = new WeakMap<Slot, ChoreoPose>();
  private pose: ChoreoPose = { ...IDENTITY_POSE };
  /** Accents (genes/accent.ts): the slot's resolved parts and this frame's music input. */
  private accPlan = {} as AccentPlan;
  private accIn: AccentInput | null = null;
  private cam = new Float32Array(6);
  /** Harmony gene: each slot's spring state and effect this frame, and the final pass warp. */
  private motors = new WeakMap<Slot, HarmonyMotor>();
  private harmOut = new WeakMap<Slot, HarmonyOut>();
  private harmIn: HarmonyInputs = { tension: 0, resolve: 0, chordPulse: 0, modPulse: 0, tonnetzX: 0.5, tonnetzY: 0.2887, keyWalk: 0 };
  private harmWarp = new Float32Array(4);
  private harmSeed = 0;
  /** Visual deja vu: each slot's snapshots of returning sections (genes/dejavu.ts). */
  readonly dejavu: DejaVuBank;
  /** Lyrics gene: each slot's eased nudges from the words (genes/lyrics.ts); temporary, never saved. */
  private lyricNudges = new WeakMap<Slot, LyricNudge>();
  private lyricTgt: LyricNudge = { ...NEUTRAL_NUDGE };
  /** The sung line drawn into the feedback of slots whose lyrics gene smears it (main stage only). */
  caption: CaptionLayer | null = null;

  constructor(private eng: Engine, readonly opts: StageOptions) {
    const gl = eng.gl;
    this.sig = new Signals(gl);
    this.bloom = new Bloom(gl, eng.fs);
    this.avgLum = new PingPong(gl, 1, 1, [eng.hdr], gl.NEAREST);
    this.dejavu = new DejaVuBank(gl, eng.fs, eng.hdr);
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
    this.slime?.resize(w, h);
    this.eco?.resize(w, h);
    this.water?.resize(w / h);
    for (const s of this.slots) {
      s.fb.dispose();
      (s as { fb: PingPong }).fb = this.makeFb();
      s.sceneT?.dispose();
      s.sceneT = null;
      s.landT?.dispose();
      s.landT = null;
    }
  }

  makeFb(): PingPong {
    return new PingPong(this.eng.gl, this.w, this.h, [this.eng.hdr], this.eng.gl.LINEAR);
  }

  /** New slot for a genome (programs must be ready). */
  makeSlot(g: Genome, progs: GenomePrograms): Slot {
    return new Slot(g, progs, this.makeFb());
  }

  disposeSlot(s: Slot): void {
    this.dejavu.drop(s);
    s.fb.dispose();
    s.sceneT?.dispose();
    s.sceneT = null;
    s.landT?.dispose();
    s.landT = null;
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

    // Choreography over the song timeline (look-ahead from the offline analysis).
    const cue = cueOf(state);
    const acc = (this.accIn ??= { cue, hookOn: 0, hookPulse: 0, hookNotePulse: 0, hookNote: -1, hookId: -1, hit: 0 });
    acc.cue = cue;
    acc.hookOn = F.hookOn;
    acc.hookPulse = F.hookPulse;
    acc.hookNotePulse = F.hookNotePulse;
    acc.hookNote = F.hookNote;
    acc.hookId = F.hookId;
    acc.hit = F.hitPulse;
    for (const s of slots) {
      let q = this.poses.get(s);
      if (!q) this.poses.set(s, (q = { ...IDENTITY_POSE }));
      choreoPose(s.genome.choreo, cue, q);
      this.harmonize(s, state, sdt, q);
      // Deja vu: a returning section pulls the framing, colours and phases back to its first appearance.
      this.dejavu.update(s, s.genome.dejavu, state, sdt, { pose: q, hue: paletteHue(s.genome.palette.p, F.keyHue), mem: s.mem });
      this.lyricize(s, state, sdt, q);
      // Accents: the hook gesture, section look and drum kick every preset gets unless its genome turns them off.
      applyAccents(accentPlan(s.genome, this.accPlan), acc, q);
    }
    this.harmonyWarp(slots, state);
    blendPoses(slots.map((s) => this.poses.get(s)!), slots.map((s) => s.weight), this.pose);
    cameraUniforms(this.pose, this.w / this.h, this.cam);

    if (slots.some((s) => s.progs.land)) (this.land ??= new LandWorld(gl)).update(state, sdt);
    // The lyrics gene sets each slot's pace (F.speed is shared, so it is set per slot and restored).
    const speed0 = F.speed;
    for (const s of slots) {
      F.speed = speed0 * (this.lyricNudges.get(s)?.speed ?? 1);
      this.tick(s, sdt);
    }
    F.speed = speed0;

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
    // Water ripples (AVS Water Bump): drops on the beats, a big one on each drop, damped by the trail length.
    const waterSlot = slots.filter((s) => s.genome.carrier.kind !== 'none' && s.genome.carrier.p.water > 0.001).sort((a, b) => b.weight - a.weight)[0];
    if (waterSlot && eng.hq) {
      if (!this.water) {
        this.water = new Water(gl, eng.fs);
        this.water.resize(this.w / this.h);
      }
      const cp = waterSlot.genome.carrier.p;
      const r = waterSlot.P('car', 0, cp, 'wsize', CARRIER_SCHEMA);
      if (F.beatIndex !== this.waterBeat) {
        this.waterBeat = F.beatIndex;
        const hx = Math.sin(F.beatIndex * 12.9898 + 1.7) * 43758.5453;
        const hy = Math.sin(F.beatIndex * 78.233 + 4.1) * 12543.1234;
        this.water.drop(0.15 + 0.7 * (hx - Math.floor(hx)), 0.15 + 0.7 * (hy - Math.floor(hy)), r * (0.8 + 0.6 * F.stem[1]), 0.4 + 1.2 * F.stem[1] + 0.4 * F.stem[0]);
      }
      if (F.dropStart) this.water.drop(0.5, 0.5, r * 3, 2.5);
      this.water.step(Math.min(0.994, 0.972 + 0.006 * Math.log2(1 + cp.halfLife * 4)), F.aspect);
    }
    const partSlot = slots.filter((s) => s.sparks >= 0).sort((a, b) => b.weight - a.weight)[0] ?? null;
    if (partSlot && eng.hq) this.updateParticles(partSlot, sdt, !!fluidSlot);
    const flameSlot = slots.filter((s) => s.flameSpec).sort((a, b) => b.weight - a.weight)[0] ?? null;
    if (flameSlot && eng.hq) {
      if (!this.flame) this.flame = new Flame(gl, eng.fs);
      const spec = flameSlot.flameSpec!;
      const n = Math.min(spec.count, this.opts.flameCap);
      if (Math.abs(this.flame.count - n) > n * 0.1) this.flame.setCount(n);
      this.flame.configure(spec, {
        spin: flameSlot.flameClock.spin, bass: F.stem[1], vocals: F.stem[2], morph: flameSlot.flameMorph, hue: flameSlot.flameHue, bars: flameSlot.flameClock.bars, beat: F.beatPulse,
      });
    }

    const slimeSlot = slots.filter((s) => s.slime >= 0).sort((a, b) => b.weight - a.weight)[0] ?? null;
    if (slimeSlot && eng.hq) this.updateSlime(slimeSlot, sdt);
    const ecoSlot = slots.filter((s) => s.eco >= 0).sort((a, b) => b.weight - a.weight)[0] ?? null;
    if (ecoSlot && eng.hq) this.updateEco(ecoSlot, sdt);

    for (const s of slots) if (s.progs.scene) this.scenePass(s, sdt);
    const flockSlot = slots.filter((s) => s.flock >= 0).sort((a, b) => b.weight - a.weight)[0] ?? null;
    if (flockSlot && eng.hq) this.updateFlock(flockSlot, sdt);

    for (const s of slots) if (s.progs.land) this.landPass(s, sdt);
    for (const s of slots) if (s.genome.carrier.kind !== 'none') this.dejavu.recall(s, s.fb.read);
    for (const s of slots) this.feedbackPass(s, sdt, s === partSlot, s === flameSlot, s === slimeSlot, s === flockSlot, s === ecoSlot);
    for (const s of slots) this.dejavu.snapshot(s, s.fb.read);

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
        .f1('uSat', (1 - 0.45 * F.build) * this.poses.get(s)!.sat)
        .f1('uSweep', F.keyPulse)
        .f1('uReflectY', g.tone.p.reflectY);
      const tget = (k: string) => s.P('col', 0, g.tone.p, k, TONE_SCHEMA);
      // Timbre: a rough / noisy sound embosses the picture, glossier and more metallic when bright.
      const tmbG = g.timbre;
      const tmbT = tmbG ? timbreSource(tmbG, F.timbre) : null;
      const rl = reliefUniforms(tmbG && tmbT ? (k) => timbreTone(tmbG, tmbT, tget, k) : tget);
      p.f4('uRelief', rl.v[0], rl.v[1], rl.v[2], rl.v[3]).f1('uMetal', rl.metal);
      // Hue-map drift runs on the bar clock, accumulated per slot so the 48-bar wrap never jumps.
      const db = (F.bars - (s.mem['hm.b'] ?? F.bars) + 48) % 48;
      s.mem['hm.b'] = F.bars;
      s.mem['hm.p'] = ((s.mem['hm.p'] ?? 0) + (db < 4 ? db : 0)) % 4096;
      const hm = hueMapUniforms((k) => s.P('col', 0, g.tone.p, k, TONE_SCHEMA), s.mem['hm.p']);
      p.f4('uHueMap', hm.v[0], hm.v[1], hm.v[2], hm.v[3]).f1('uPoster', hm.poster);
      eng.fs.draw();
    }
    for (const s of slots) {
      for (const c of s.curves) if (c.top) this.drawCurve(s, c, s.weight, sdt, false);
      if (s === partSlot && s.genome.bodies[s.sparks].emit.p.top > 0.5) this.drawParticles(s, s.weight, false);
      if (s === this.flockOwner && s.flock >= 0) this.drawFlockOverlay(s);
    }
    gl.disable(gl.BLEND);

    // Post parameters crossfade with the slot weights.
    const post: PostParams = { bloom: 0, exposure: 0, vignette: 0, adapt: 0, ca: 0, contrast: 0 };
    let wsum = 0;
    for (const s of slots) wsum += s.weight;
    for (const s of slots) {
      const c = s.genome.tone.p;
      const w = s.weight / Math.max(wsum, 1e-3);
      post.bloom += s.P('col', 0, c, 'bloom', TONE_SCHEMA) * w;
      post.exposure += s.P('col', 0, c, 'exposure', TONE_SCHEMA) * w;
      post.vignette += s.P('col', 0, c, 'vignette', TONE_SCHEMA) * w;
      post.adapt += c.adapt * w;
      post.ca += s.P('col', 0, c, 'ca', TONE_SCHEMA) * w;
      post.contrast += c.contrast * w;
    }
    if (!slots.length) Object.assign(post, { bloom: 1, exposure: 1, vignette: 0.45, adapt: 0.3, ca: 0, contrast: 0.03 });
    this.post(post, target, state.dt);
  }

  private tick(s: Slot, sdt: number): void {
    const g = s.genome;
    const F = this.sig.F;
    const sig = this.sig;
    const f60 = sdt * 60;
    const A = F.aspect * 0.5;

    // Reactions -> per-parameter deltas for this frame.
    // Reactions -> per-parameter deltas: signal -> response curve (threshold, quantise, attack / release) -> gain.
    s.delta.clear();
    g.reactions.forEach((r, j) => {
      const key = `${r.g}${r.i}.${r.k}`;
      let x = sig.signal(r.src);
      if (j < MAX_REACTIONS) s.meters[j * 2] = x;
      if (r.thr > 0) x = Math.max(0, x - r.thr) / (1 - r.thr);
      if (r.q > 0.5) {
        const idx = Math.floor(F.beats / r.div);
        if (idx !== s.mem['rq' + j]) {
          s.mem['rq' + j] = idx;
          s.mem['rh' + j] = x;
        }
        x = s.mem['rh' + j] ?? x;
      }
      if (r.atk > 0.006 || r.rel > 0.006) {
        const e = s.mem['re' + j] ?? x;
        x = s.mem['re' + j] = approach(e, x, 1 / (x > e ? r.atk : r.rel), sdt);
      }
      if (j < MAX_REACTIONS) s.meters[j * 2 + 1] = x;
      s.delta.set(key, (s.delta.get(key) ?? 0) + r.gain * x);
    });
    const ln = this.lyricNudges.get(s);
    if (ln) this.lyricDeltas(s, ln);

    // Palette
    const tp = g.tone.p;
    const sat = s.P('col', 0, tp, 'sat', TONE_SCHEMA) * (0.78 + 0.3 * F.stem[2]) * (1 - 0.5 * F.build);
    paletteColors(g.palette, paletteHue(g.palette.p, F.keyHue, s.P('pal', 0, g.palette.p, 'hue', PALETTE_SCHEMAS[g.palette.kind])) + (this.poses.get(s)?.hue ?? 0), sat, s.cols);

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
      const b = s.opB;
      const j = i * 4;
      a[j] = a[j + 1] = a[j + 2] = a[j + 3] = 0;
      b[j] = b[j + 1] = b[j + 2] = b[j + 3] = 0;
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
          wander(P('wander'));
          const sign = o.p.alt > 0.5 && F.barIndex % 2 === 1 ? -1 : 1;
          a[j + 2] = -(o.p.lock * sig.spinStep + P('rate') * o.w * f60 * F.speed) * sign;
          break;
        }
        case 'translate': {
          const v = [P('vx'), P('vy')];
          const lanes = o.p.lanes > 1 ? o.p.lanes : 0;
          a[j + 2] = lanes;
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
        case 'quad':
          wander(P('wander'));
          a[j + 2] = P('amt') * o.w * Math.min(2, f60);
          a[j + 3] = o.p.turn * TAU;
          break;
        case 'push':
          a[j] = P('amt') * o.w * f60 * F.speed;
          a[j + 1] = o.p.axis;
          break;
        case 'stretch': {
          const k = o.stage === 'warp' ? Math.min(1, 0.04 * f60) : 1;
          a[j] = P('base');
          a[j + 1] = P('amt') * k;
          a[j + 2] = P('beat') * k;
          a[j + 3] = o.p.strips;
          b[j] = o.stage === 'view' ? P('win') : 0;
          b[j + 1] = o.stage === 'view' ? P('sky') : 0;
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
          this.loosen(s, b, j, i, o.stage);
          break;
        case 'tile':
          a[j] = P('n');
          this.loosen(s, b, j, i, o.stage);
          break;
        case 'polar':
          a[j] = P('scale');
          a[j + 1] = F.spin * o.p.lock;
          this.loosen(s, b, j, i, o.stage);
          break;
        case 'kaleido':
          a[j] = o.p.n;
          a[j + 1] = F.spin * o.p.lock;
          this.loosen(s, b, j, i, o.stage);
          break;
        case 'mosaic':
          packMosaic(a, j, P, o.p, F.stem[1], F.spin);
          break;
        case 'tunnel':
          packTunnel(a, b, j, P, o.p, s.mem, `op${i}.tz`, sdt, F.speed, F.beatPulse, F.spin);
          break;
        default: // flame variations
          a[j] = Math.min(1, o.w * (o.stage === 'warp' ? 0.1 * f60 : 1));
          a[j + 1] = P('s');
      }
    });

    s.flameSpec = null;
    s.curves = [];
    s.segN = 0;
    g.bodies.forEach((b, bi) => this.tickBody(s, b, bi, sdt));
  }

  // ------------------------------------------------------------ bodies

  /** Instrument voices: 0 drums, 1 bass, 2 vocals (or the melody), 3 other, 4 loudness. */
  private voice(k: number): number {
    const F = this.sig.F;
    switch (k % 5) {
      case 0: return Math.max(F.onset[0], F.stem[0] * 0.5);
      case 1: return F.stem[1];
      case 2: return Math.max(F.stem[2], 0.7 * this.sig.melodic());
      case 3: return F.stem[3];
      default: return F.loud;
    }
  }

  /**
   * One body's frame: material, shape state, copies (placement then motion),
   * deformation, emission and fuse, all packed into the slot's uniform arrays.
   */
  private tickBody(s: Slot, b: BodyGene, bi: number, sdt: number): void {
    const F = this.sig.F;
    const sig = this.sig;
    const E = s.bd;
    const o = bi * BODY_VEC4 * 4;
    E.fill(0, o, o + BODY_VEC4 * 4);
    const m = s.mem;
    const key = (k: string) => `b${bi}.${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);
    const set = (k: string, v: number) => (m[key(k)] = v);
    const A = F.aspect * 0.5;
    const cls = SHAPE_CLASS[b.shape.kind];
    const PS = (k: string) => s.P('sh', bi, b.shape.p, k, SHAPE_SCHEMAS[b.shape.kind]);
    const PP = (k: string) => s.P('pl', bi, b.place.p, k, PLACE_SCHEMAS[b.place.kind]);
    const PM = (k: string) => s.P('mo', bi, b.motion.p, k, MOTION_SCHEMAS[b.motion.kind]);
    const PD = (k: string) => s.P('de', bi, b.deform.p, k, DEFORM_SCHEMAS[b.deform.kind]);
    const PA = (k: string) => s.P('ma', bi, b.material.p, k, MATERIAL_SCHEMAS[b.material.kind]);
    const PE = (k: string) => s.P('em', bi, b.emit.p, k, EMIT_SCHEMAS[b.emit.kind]);
    const gain = PA('gain');
    this.clk = this.bodyClock(s, b, bi, sdt);
    // Groove: a grid-locked clock takes the music's timing feel (swing, lean, crisp ticks).
    if (s.genome.groove && this.clk.lock) grooveClock(s.genome.groove, F.groove, this.clk);
    // Colour mapping: the body's base hue (plus the age drift) and slot 20 (kind, detail, height, amount).
    const cm = b.color;
    const PC = (k: string) => s.P('cm', bi, cm.p, k, MAPPING_SCHEMAS[cm.kind]);
    const hue = PC('hue') + (cm.kind === 'age' ? (this.clk.bars * cm.p.rate) % 1 : 0);
    this.bodyHue = hue;
    E[o + 80] = MAPPING_KINDS.indexOf(cm.kind);
    E[o + 81] = PC('detail');
    E[o + 82] = cm.kind === 'height' ? PC('amount') : 0;
    E[o + 83] = cm.kind === 'instrument' ? PC('amount') : 1;

    // Material: slot 0 (gain, hue, a, b), slot 1.
    const halo = (m[key('halo')] = approach(mm('halo'), Math.max(F.loud, sig.melodic()), 3, sdt));
    E[o] = gain;
    E[o + 1] = hue;
    switch (b.material.kind) {
      case 'line': E[o + 2] = PA('width'); E[o + 3] = PA('halo'); break;
      case 'fill': E[o + 2] = PA('soft'); E[o + 3] = PA('halo'); E[o + 4] = PA('outline'); E[o + 5] = PA('core'); E[o + 6] = PA('clip'); E[o + 7] = halo; break;
      case 'glow': E[o + 2] = PA('width'); E[o + 3] = PA('base'); E[o + 4] = PA('halo'); break;
      case 'dots': E[o + 2] = PA('spacing'); E[o + 3] = PA('size'); break;
      case 'textured': E[o + 2] = PA('amount'); E[o + 3] = PA('halo'); E[o + 4] = b.material.p.tex; E[o + 5] = PA('clip'); E[o + 6] = halo; break;
      case 'chrome': E[o + 2] = PA('chrome'); break;
    }

    // Copies: placement, then motion.
    const copies = this.placeCopies(s, b, bi, sdt);
    this.applyMotion(s, b, bi, copies, sdt);
    if (s.genome.timbre) {
      // Timbre as material: the surface amounts for this body (smoothed per slot).
      const tg = s.genome.timbre;
      let st = this.tmbState.get(s);
      if (!st || st.u.length !== s.nb * 8) this.tmbState.set(s, (st = { u: new Float32Array(s.nb * 8), m: new Float32Array(s.nb * 5) }));
      timbreUniforms(tg, timbreLook(tg, timbreSource(tg, F.timbre), this.tmbLook), st.m, bi, F.time, sdt, st.u);
    }
    if (s.genome.groove) {
      // Groove: sway, off-beat pulse, crisp ticks, onset nudges and syncopated accents.
      const go = grooveOffset(s.genome.groove, F.groove, { beat: F.beats, period: 60 / Math.max(1, F.bpm), dt: sdt, onset: F.onset[0] }, s.mem, `b${bi}.g`, this.grooveOff);
      for (const c of copies) {
        c.x += go.dx;
        c.y += go.dy;
        c.a += go.da;
        c.s *= go.s;
      }
    }
    this.mapHues(s, b, bi, copies, sdt);
    const n = Math.min(COPY_SLOTS, copies.length);
    for (let i = 0; i < COPY_SLOTS; i++) {
      const c = copies[Math.min(i, copies.length - 1)];
      const j = (bi * COPY_SLOTS + i) * 4;
      s.cp[j] = c.x; s.cp[j + 1] = c.y; s.cp[j + 2] = c.a; s.cp[j + 3] = c.s;
      s.cq[j] = c.level; s.cq[j + 1] = c.hue; s.cq[j + 2] = c.px ?? c.x; s.cq[j + 3] = c.py ?? c.y;
    }

    // Shape: slots 2-3 (and the curve / segment / field / flame state).
    const R = this.packShape(s, b.shape, bi, o + 8, sdt, false, copies);
    E[o + 16] = n;
    E[o + 17] = b.place.p.fuse ?? 0;
    E[o + 19] = R;

    // Fold placements: slots 5-6.
    if (b.place.kind === 'grid') {
      E[o + 20] = PP('scale'); E[o + 21] = PP('jitter'); E[o + 22] = PP('density'); E[o + 23] = PP('lit');
      E[o + 24] = PP('links'); E[o + 25] = b.place.p.lattice;
      E[o + 26] = clamp01(this.resp('g', Math.max(sig.melodic(), 0.4 * F.stem[1])) * 1.8);
      E[o + 18] = this.step(s, bi, 'lit', 1);
      E[o + 26] = Math.max(0.3, E[o + 26]);
      E[o + 27] = PP('twinkle');
    } else if (b.place.kind === 'ring') {
      E[o + 20] = b.place.p.n; E[o + 21] = PP('radius');
    } else if (b.place.kind === 'mirror') {
      E[o + 20] = b.place.p.axis;
    }

    // Deformation: slots 7-11.
    this.packDeform(s, b, bi, o + 28, sdt);
    const ops = b.deform.ops ?? [];
    this.packOps(s, ops, bi, sdt);

    // Emission: slot 12 (cover amount, tip, ops count, body visibility).
    E[o + 48] = b.emit.kind === 'cover' ? PE('amt') : 0;
    E[o + 49] = b.emit.kind === 'cover' || b.emit.kind === 'trail' ? PE('tip') : 0;
    E[o + 50] = ops.length;
    E[o + 51] = b.emit.kind === 'sparks' || b.emit.kind === 'slime' || b.emit.kind === 'flock' || b.emit.kind === 'ecosystem' ? PE('body') : 1;

    // Fuse: slot 13 (mode, blend radius, t, inside), 14-15 the fused shape.
    if (b.fuse) {
      const f = b.fuse;
      const PF = (k: string) => s.P('fu', bi, f.p, k, FUSE_SCHEMA);
      const bars = F.bars / Math.max(1, f.p.rate);
      const drv = [PF('t'), 0.5 - 0.5 * Math.cos(TAU * bars), F.stem[1] * F.gate[1], F.melody, F.loud, Math.min(1, F.surge * 0.8)][f.p.drive] ?? PF('t');
      const tt = mm('ft', PF('t'));
      set('ft', approach(tt, PF('t') + (drv - PF('t')) * PF('depth'), f.p.drive >= 2 ? 3 : 20, sdt));
      const bass = set('fbs', approach(mm('fbs'), F.stem[1] * F.gate[1], 4, sdt));
      E[o + 52] = f.p.mode;
      E[o + 53] = PF('k') * (0.6 + 0.9 * bass);
      E[o + 54] = clamp01(mm('ft'));
      E[o + 55] = f.p.inside;
      this.packShape(s, f.shape, bi, o + 56, sdt, true, copies);
    }

    // Emission side effects.
    if (b.emit.kind === 'dye') this.dye(s, b, bi, copies, PE('force'));
    if (b.emit.kind === 'sparks' && s.sparks === bi) this.sparkSpawn(s, b, copies);
    if (cls === 'curve' && !b.fuse) this.queueCurve(s, b, bi, copies);
    if (cls === 'flame' && b.shape.xforms) {
      const c0 = copies[0];
      const spin = b.motion.kind === 'spin' ? b.motion.p.rate * (b.motion.p.alt > 0.5 && F.barIndex % 2 === 1 ? -1 : 1) : 0;
      s.flameClock = { spin: this.clk.spin, bars: this.clk.bars % 48 };
      const xs = b.shape.xforms;
      if (F.dropStart) set('mt', mm('mt') > 0.5 ? 0 : 1);
      set('m', mm('m') + (mm('mt') - mm('m')) * (1 - Math.exp(-sdt * 0.8)));
      s.flameMorph = mm('m');
      s.flameHue = hue;
      const pulse = b.motion.kind === 'pulse' ? 1 + PM('amp') * F.beatPulse * F.gate[0] : 1;
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
        count: b.shape.p.count,
        iters: 4,
        rounds: b.shape.p.rounds,
        zoom: PS('zoom') * c0.s * pulse,
        offset: [c0.x, c0.y],
        camSpin: spin,
        gain,
        flow: PS('flow'),
        breathe: PS('breathe'),
      };
    }
  }

  private bodyHue = 0;

  /** Each copy's hue offset from the colour mapping (the placement's own offsets are the instrument mapping). */
  private mapHues(s: Slot, b: BodyGene, bi: number, copies: Copy[], sdt: number): void {
    const F = this.sig.F;
    const cm = b.color;
    const P = (k: string) => s.P('cm', bi, cm.p, k, MAPPING_SCHEMAS[cm.kind]);
    switch (cm.kind) {
      case 'fixed': case 'height': case 'age':
        for (const c of copies) c.hue = 0;
        break;
      case 'instrument': {
        const a = P('amount');
        for (const c of copies) c.hue *= a;
        break;
      }
      case 'pitch': {
        // The strongest pitch classes, one per copy (a grid picks its own per cell in the shader).
        const ch = this.sig.chroma;
        const order = Array.from({ length: 12 }, (_, i) => i).sort((x, y) => ch[y] - ch[x]);
        copies.forEach((c, i) => (c.hue = -1 - order[i % 12]));
        break;
      }
      case 'melody': {
        const a = P('amount');
        const mel = (s.mem[`b${bi}.hm`] = approach(s.mem[`b${bi}.hm`] ?? F.melody, F.melody, 4, sdt));
        for (const c of copies) c.hue = mel * a + c.hue * 0.15;
        break;
      }
      case 'speed': {
        const a = P('amount');
        copies.forEach((c, i) => {
          const k = `b${bi}.hs${i}`;
          const px = s.mem[k + 'x'] ?? c.x, py = s.mem[k + 'y'] ?? c.y;
          const v = Math.min(3, Math.hypot(c.x - px, c.y - py) / Math.max(sdt, 1e-3));
          s.mem[k + 'x'] = c.x;
          s.mem[k + 'y'] = c.y;
          const sm = (s.mem[k] = approach(s.mem[k] ?? v, v, 3, sdt));
          c.hue = Math.min(1, sm * 1.5) * a;
        });
        break;
      }
    }
  }

  private grooveOff: GrooveOffset = { dx: 0, dy: 0, da: 0, s: 1 };
  /** Timbre surface uniforms (two vec4 per body) and their smoothing state, per slot. */
  private tmbState = new WeakMap<Slot, { u: Float32Array; m: Float32Array }>();
  private tmbLook: TimbreLook = { sheen: 0, glass: 0, grain: 0, velvet: 0, edge: 0 };
  private tmbZero = new Float32Array(24);
  /** A slot's timbre surface uniforms (zeros when the genome has no timbre gene). */
  private tmbUniforms(s: Slot): Float32Array {
    return this.tmbState.get(s)?.u ?? this.tmbZero;
  }
  private clk: BodyClock = { mul: 1, s: 1, lock: true, bars: 0, spin: 0, barPhase: 0, beatPhase: 0 };
  private respBody = { s: null as Slot | null, bi: 0, b: null as BodyGene | null };

  /** The body's clock: exactly the song's at the neutral feel (bar unit, grid-locked). */
  private bodyClock(s: Slot, b: BodyGene, bi: number, sdt: number): BodyClock {
    const F = this.sig.F;
    const fp = b.feel.p;
    const sc = fp.div / 4;
    const mul = Math.min(2, Math.max(0.25, 1 / sc));
    const lock = fp.lock > 0.5;
    this.respBody = { s, bi, b };
    const m = s.mem;
    const k = (x: string) => `b${bi}.c${x}`;
    if (lock && mul === 1) return { mul, s: sc, lock, bars: F.bars, spin: F.spin, barPhase: F.barPhase, beatPhase: F.beatPhase };
    let dBars: number;
    if (lock) {
      const prev = m[k('pb')] ?? F.bars;
      dBars = F.bars - prev;
      if (dBars < -24) dBars += 48;
      if (dBars < 0 || dBars > 1) dBars = (sdt * F.bpm) / 240;
      m[k('pb')] = F.bars;
    } else dBars = ((sdt * F.bpm) / 240) * (1 + 0.04 * Math.sin(this.sig.clock * 0.13 + bi));
    const bars = (m[k('tb')] = ((m[k('tb')] ?? F.bars * mul) + dBars * mul) % 4096);
    const spin = (m[k('sp')] = ((m[k('sp')] ?? F.spin * mul) + this.sig.spinStep * mul) % (TAU * 1024));
    return { mul, s: sc, lock, bars, spin, barPhase: bars % 1, beatPhase: (bars * 4) % 1 };
  }

  /**
   * An event on the body's clock every `beats` beats (scaled by its feel): on the beat grid when
   * locked, at irregular intervals with the same average spacing when free.
   */
  private every(s: Slot, bi: number, key: string, beats: number): boolean {
    const F = this.sig.F;
    const m = s.mem;
    const k = `b${bi}.e${key}`;
    const span = Math.max(0.25, beats * this.clk.s);
    if (this.clk.lock) {
      const idx = Math.floor(F.beats / span + 1e-6);
      const prev = m[k];
      m[k] = idx;
      return prev !== undefined && idx !== prev;
    }
    const t = (m[k + 't'] = (m[k + 't'] ?? 0) + (F.dt * F.bpm) / 60);
    if (m[k + 'n'] === undefined) m[k + 'n'] = span * (0.6 + 0.8 * h11(bi * 7.1 + key.length));
    if (t < m[k + 'n']) return false;
    const c = (m[k + 'c'] = (m[k + 'c'] ?? 0) + 1);
    m[k + 't'] = 0;
    m[k + 'n'] = span * (0.6 + 0.8 * h11(c * 3.7 + bi));
    return true;
  }

  /** Index of the current clock step (grid refreshes, held levels). */
  private step(s: Slot, bi: number, key: string, beats: number): number {
    const m = s.mem;
    const k = `b${bi}.i${key}`;
    if (this.every(s, bi, key, beats)) m[k] = (m[k] ?? 0) + 1;
    return m[k] ?? 0;
  }

  /**
   * A music level through the body's response curve: threshold, sensitivity, held on the clock grid
   * (step feel), then attack / release smoothing. Neutral feel: the level unchanged.
   */
  private resp(key: string, raw: number): number {
    const { s, bi, b } = this.respBody;
    if (!s || !b) return raw;
    const fp = b.feel.p;
    let x = raw;
    if (fp.thr > 0) x = Math.max(0, x - fp.thr) / (1 - fp.thr);
    x *= fp.sens;
    const m = s.mem;
    const k = `b${bi}.r${key}`;
    if (b.feel.kind === 'step') {
      if (this.every(s, bi, 'q' + key, 1) || m[k + 'h'] === undefined) m[k + 'h'] = x;
      x = m[k + 'h'];
    }
    if (fp.atk > 0.006 || fp.rel > 0.006) {
      const e = m[k] ?? x;
      x = m[k] = approach(e, x, 1 / (x > e ? fp.atk : fp.rel), this.sig.F.dt);
    }
    return x;
  }

  /** Copy positions from the placement gene. */
  private placeCopies(s: Slot, b: BodyGene, bi: number, sdt: number): Copy[] {
    const F = this.sig.F;
    const m = s.mem;
    const key = (k: string) => `b${bi}.${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);
    const set = (k: string, v: number) => (m[key(k)] = v);
    const A = F.aspect * 0.5;
    const PP = (k: string) => s.P('pl', bi, b.place.p, k, PLACE_SCHEMAS[b.place.kind]);
    const p = b.place.p;
    const out: Copy[] = [];
    const add = (x: number, y: number, level = 1, hue = 0, sc = 1, a = 0): Copy => {
      const c: Copy = { x, y, a, s: sc, level, hue };
      out.push(c);
      return c;
    };
    const clk = this.clk;
    switch (b.place.kind) {
      case 'point':
        add(PP('x'), PP('y'));
        break;
      case 'orbit': {
        const n = p.count;
        const follow = PP('follow');
        const R = PP('radius');
        const ocx = PP('x') + follow * A * 0.64 * Math.sin((TAU * clk.bars) / 8);
        const ocy = PP('y') + follow * 0.63 * Math.sin((TAU * clk.bars) / 6 + 1);
        const lvl = this.resp('o', Math.max(F.stem[3], F.stem[2]) * 0.8 + 0.2 * F.loud);
        for (let i = 0; i < n; i++) {
          const ang = clk.spin * p.rate + (i / n) * TAU;
          add(ocx + R * Math.cos(ang), ocy + R * Math.sin(ang), lvl, i * 0.25 + 0.1, 1, ang);
        }
        break;
      }
      case 'row': {
        const n = p.count;
        const wv = PP('wander');
        for (let i = 0; i < n; i++) {
          const rx = (i - (n - 1) / 2) * (1.2 / Math.max(1, n)) * A + 0.05 * Math.sin(F.phase * 0.4 + i * 2.1);
          const k4 = i % 4;
          const lvl = this.resp('c' + i, k4 === 0 ? Math.max(F.onset[0], F.stem[0] * 0.5) : F.stem[k4]);
          add(rx + wv * Math.sin(F.phase * 0.23 * clk.mul + i * 1.9), PP('y'), lvl, i * 0.25 + 0.1);
        }
        break;
      }
      case 'stations': {
        // Sources at stations; with inst they follow their instruments (drums jump every bar or
        // every hit, bass swings out, vocals follow the melody, other roams).
        const n = p.count;
        const wv = PP('wander');
        const inst = PP('inst');
        const xs = PP('xs');
        const jump = p.jump === 1;
        const swap = p.swap === 1 && ((F.barIndex % 2) + 2) % 2 === 1 ? 0.333 : 0;
        const stations = [[-0.55 * A, -0.2], [0, -0.28], [0, 0.18], [0.55 * A, -0.05], [-0.3 * A, 0.25], [0.3 * A, 0.25]];
        if (inst > 0 && (jump ? F.hit > 0 : this.every(s, bi, 'jump', 4))) {
          const c = set('jn', mm('jn') + 1);
          set('dtx', (h11(c * 1.7) - 0.5) * 1.2 * A);
          set('dty', (h11(c * 2.9) - 0.5) * (jump ? 0.7 : 0.6));
        }
        for (let i = 0; i < n; i++) {
          const k4 = i % 4;
          const st = stations[i % 6];
          let sx = st[0] + wv * Math.sin(F.phase * 0.23 + i * 1.9);
          let sy = st[1] + wv * 0.7 * Math.sin(F.phase * 0.31 + i * 2.7);
          const lvlOld = k4 === 0 ? Math.max(F.onset[0], F.stem[0] * 0.5) : F.stem[k4];
          const lvlInst = k4 === 0 ? (jump ? F.hitPulse : F.onset[0]) : F.stem[k4];
          if (inst > 0) {
            const mir = i >= 4 ? -1 : 1;
            let tx: number, ty: number;
            if (k4 === 0) [tx, ty] = [mm('dtx', -0.55 * A), mm('dty', -0.2)];
            else if (k4 === 1) [tx, ty] = [Math.cos(F.phase * 0.2) * (0.1 + 0.45 * F.stem[1]) * A, -0.28 + 0.2 * F.stem[1]];
            else if (k4 === 2) [tx, ty] = [0.25 * A * Math.sin(F.phase * 0.15), (F.melody - 0.5) * 0.7];
            else [tx, ty] = [0.55 * A * Math.cos(F.phase * 0.11 + 1), 0.25 * Math.sin(F.phase * 0.13)];
            tx *= xs * mir;
            const rate = k4 === 0 ? (jump ? 30 : 3) : 1.5;
            const ix = set('x' + i, approach(mm('x' + i, tx), tx, rate, sdt));
            const iy = set('y' + i, approach(mm('y' + i, ty), ty, rate, sdt));
            sx += (ix - sx) * inst;
            sy += (iy - sy) * inst;
          }
          add(sx, sy, this.resp('c' + i, lvlOld + (lvlInst - lvlOld) * inst), i * 0.25 + 0.1 + swap);
        }
        break;
      }
      case 'float': {
        // Slow Lissajous paths; each copy's size follows its instrument (bass, bass, drums, other...).
        const n = p.count;
        const src = [1, 1, 0, 2, 3, 3];
        const spread = PP('spread');
        const speed = PP('speed');
        const ph = set('ph', (mm('ph') + sdt * F.speed * clk.mul) % 4096);
        for (let i = 0; i < n; i++) {
          const t = ph * (speed * 0.9 + 0.03 * i);
          const lv = this.resp('c' + i, src[i] === 0 ? F.onset[0] * 0.6 + F.stem[0] * 0.4 : F.stem[src[i]]);
          const tr = 0.75 + 1.25 * lv + (0.015 / 0.06) * F.beatPulse * F.gate[0];
          const rr = set('r' + i, approach(mm('r' + i, tr), tr, 8, sdt));
          add(spread * A * Math.sin(t + i * 1.7) * (0.6 + 0.4 * Math.sin(t * 0.37 + i)), 0.26 * (spread / 0.55) * Math.sin(t * 1.31 + i * 2.3), 1, 0, rr);
        }
        break;
      }
      case 'outline': {
        const n = p.count;
        const R = PP('radius');
        const cx = PP('x');
        const cy = PP('y');
        for (let i = 0; i < n; i++) {
          const u = (clk.spin * p.rate) / TAU + i / n;
          const a = u * TAU;
          let x: number, y: number;
          if (p.path === 0) [x, y] = [R * Math.cos(a), R * Math.sin(a)];
          else if (p.path === 1) {
            // Around a square: corners at the diagonals.
            const t = ((u % 1) + 1) % 1 * 4;
            const side = Math.floor(t);
            const f = t - side;
            const pts = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
            const [ax, ay] = pts[side];
            const [bx, by] = pts[(side + 1) % 4];
            x = (ax + (bx - ax) * f) * R * 0.75;
            y = (ay + (by - ay) * f) * R * 0.75;
          } else [x, y] = [R * 1.3 * Math.sin(a), R * 0.6 * Math.sin(2 * a)];
          add(cx + x, cy + y, this.resp('c' + i, this.voice(i % 4)), i * 0.2, 1, a);
        }
        break;
      }
      case 'walker':
        this.walk(s, b, bi, out);
        break;
      case 'grid':
        add(0, 0, 1, 0, 1, clk.spin * p.lock);
        break;
      case 'ring': {
        add(PP('x'), PP('y'));
        // Four voices for the ring's copies (drums, bass, vocals, other), cycling around it.
        for (let k = 1; k < 4; k++) add(PP('x'), PP('y'), 1, k * 0.25 + 0.1);
        out[0].hue = 0.1;
        for (let k = 0; k < 4; k++) out[k].level = 0.35 + 0.9 * this.resp('c' + k, this.voice(k));
        break;
      }
      case 'mirror':
        add(PP('x'), PP('y'));
        break;
    }
    if (!out.length) add(0, 0);
    return out;
  }

  /**
   * Walker heads roaming the screen: head 0 steers with the melody, head 1 with the bass. Head 0
   * turns every `every` beats, head 1 every 2 x `every`; turn-on-hits motion adds sharp turns on
   * drum hits / bass onsets. Turns are free or square, the path curves gently between them,
   * bounces off (or wraps around) the edges and moves `step` per beat.
   */
  private walk(s: Slot, b: BodyGene, bi: number, out: Copy[]): void {
    const F = this.sig.F;
    const P = (k: string) => s.P('pl', bi, b.place.p, k, PLACE_SCHEMAS.walker);
    const m = s.mem;
    const mm = (k: string, init = 0) => m[`b${bi}.w${k}`] ?? (m[`b${bi}.w${k}`] = init);
    const set = (k: string, v: number) => (m[`b${bi}.w${k}`] = v);
    const A = F.aspect * 0.5;
    const M = 0.07;
    const p = b.place.p;
    let db = F.beats - mm('lb', F.beats);
    if (db < 0) db += 256;
    if (db > 2) db = F.dt * 2;
    set('lb', F.beats);
    db *= this.clk.mul;
    const hits = b.motion.kind === 'hits' ? s.P('mo', bi, b.motion.p, 'amt', MOTION_SCHEMAS.hits) : 0;
    const hitRise = F.hit > 0.7 && mm('hp') <= 0.7;
    set('hp', F.hit);
    const bassRise = F.onset[1] > 0.6 && mm('bp') <= 0.6;
    set('bp', F.onset[1]);
    const step = P('step') * (0.7 + 0.5 * F.act);
    const turn = P('turn');
    const curve = P('curve');
    const every = p.every;
    const heads: [string, boolean, boolean, number, number, number, number, number][] = [
      ['m', this.every(s, bi, 'wm', every), hitRise, -0.3 * A, 0.15, 0.3, step * 1.18, F.melody],
      ['b', this.every(s, bi, 'wb', every * 2), bassRise, 0.3 * A, -0.2, 2.6, step * 0.82, F.stem[1]],
    ];
    for (let h = 0; h < p.heads; h++) {
      const [k, turnNow, heavyRaw, sx, sy, sa, perBeat, steer] = heads[h];
      const heavy = heavyRaw && hits > 0;
      const x = mm(k + 'x', sx);
      const y = mm(k + 'y', sy);
      let ang = mm(k + 'a', sa);
      if (turnNow || heavy) {
        const c = set(k + 'n', mm(k + 'n') + 1);
        const side = h11(c * 7.3 + h * 50) < 0.5 ? -1 : 1;
        if (p.square === 1) ang += side * (heavy ? Math.PI : Math.PI / 2);
        else ang += side * turn * (heavy ? (1.3 + 1.2 * h11(c * 3.7 + 1 + h * 59)) * hits : 0.45 + 0.9 * h11(c * 3.7 + 1 + h * 59));
      }
      if (p.square !== 1) ang += (steer - 0.5) * 0.8 * curve * F.dt;
      const d = db * perBeat;
      let nx = x + Math.cos(ang) * d;
      let ny = y + Math.sin(ang) * d;
      let px = x, py = y;
      if (p.wrap === 1) {
        if (nx < -A) { nx += 2 * A; px = nx; py = ny; }
        if (nx > A) { nx -= 2 * A; px = nx; py = ny; }
        if (ny < -0.5) { ny += 1; px = nx; py = ny; }
        if (ny > 0.5) { ny -= 1; px = nx; py = ny; }
      } else {
        if (nx < -A + M || nx > A - M) {
          ang = Math.PI - ang;
          nx = Math.min(A - M, Math.max(-A + M, nx));
        }
        if (ny < -0.5 + M || ny > 0.5 - M) {
          ang = -ang;
          ny = Math.min(0.5 - M, Math.max(-0.5 + M, ny));
        }
      }
      set(k + 'a', ang % (TAU * 64));
      set(k + 'x', nx);
      set(k + 'y', ny);
      const bass = this.resp('wb', F.stem[1]);
      const loud = this.resp('wl', F.loud);
      const level = h === 0 ? 0.25 + this.resp('wm', this.sig.melodic()) * 0.55 : 0.25 + 0.6 * bass * (0.3 + 0.7 * F.act);
      const hue = h === 0 ? F.melody * 0.7 + 0.2 * F.beatPulse : 0.5 + 0.4 * bass + 0.15 * F.barPulse;
      // Dots keep the snake's widths (melody thin, bass thick); larger shapes breathe more gently.
      const dot = b.shape.kind === 'dot';
      const sc = h === 0 ? (dot ? 0.6 + 0.8 * loud : 0.8 + 0.4 * loud) : dot ? 1.1 + 2 * bass : 0.85 + 0.5 * bass;
      out.push({ x: nx, y: ny, a: ang, s: sc, level, hue, px, py });
    }
    // The bass head is drawn first so the melody head paints over it.
    if (out.length === 2) out.reverse();
  }

  /** Motion genes move each copy (phase offset per copy). */
  private applyMotion(s: Slot, b: BodyGene, bi: number, copies: Copy[], sdt: number): void {
    const F = this.sig.F;
    const mo = b.motion;
    const P = (k: string) => s.P('mo', bi, mo.p, k, MOTION_SCHEMAS[mo.kind]);
    const m = s.mem;
    const key = (k: string) => `b${bi}.m${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);
    const set = (k: string, v: number) => (m[key(k)] = v);
    const A = F.aspect * 0.5;
    const solid = b.shape.kind === 'solid';
    const walker = b.place.kind === 'walker';
    copies.forEach((c, i) => {
      const ph = i * 1.3;
      switch (mo.kind) {
        case 'spin': {
          // Solids and flames turn their own way (3D yaw / the flame camera).
          if (solid || b.shape.kind === 'flame' || walker) break;
          const sign = mo.p.alt > 0.5 && F.barIndex % 2 === 1 ? -1 : 1;
          c.a += this.clk.spin * mo.p.rate * sign;
          break;
        }
        case 'sway': {
          const w = Math.sin((TAU * this.clk.bars) / mo.p.period + ph);
          c.x += P('amp') * w;
          c.a += P('tilt') * 0.35 * w;
          break;
        }
        case 'bob': {
          const amp = P('amp');
          const dp = set('dp', approach(mm('dp'), F.gate[0], 1, sdt / Math.max(1, copies.length)));
          c.x += 0.035 * amp * Math.sin(TAU * this.clk.barPhase + ph);
          c.y += 0.03 * Math.sin(F.phase * 0.01) + 0.008 * amp * Math.sin(TAU * this.clk.beatPhase + ph) * dp;
          break;
        }
        case 'drift': {
          if (i === 0) {
            set('dx', (mm('dx') + P('vx') * sdt * F.speed * this.clk.mul) % 64);
            set('dy', (mm('dy') + P('vy') * sdt * F.speed * this.clk.mul) % 64);
          }
          c.x += mm('dx');
          c.y += mm('dy');
          if (b.place.kind !== 'grid') {
            // Wrap around the screen.
            c.x = ((((c.x + A) % (2 * A)) + 2 * A) % (2 * A)) - A;
            c.y = ((((c.y + 0.5) % 1) + 1) % 1) - 0.5;
            if (c.px !== undefined) [c.px, c.py] = [c.x, c.y];
          }
          break;
        }
        case 'circle': {
          const a = (TAU * this.clk.bars) / mo.p.period + ph;
          c.x += P('radius') * Math.cos(a);
          c.y += P('radius') * Math.sin(a);
          break;
        }
        case 'hits': {
          if (walker) break;
          // A jolt that stays: each drum hit turns the copy.
          if (i === 0) {
            const rise = F.hit > 0.5 && mm('hp') <= 0.5;
            set('hp', F.hit);
            if (rise) {
              const n = set('hn', mm('hn') + 1);
              set('ha', mm('ha') + P('amt') * (0.5 + 0.5 * h11(n * 3.1)) * (h11(n * 7.7) < 0.5 ? -1 : 1));
            }
            set('hs', approach(mm('hs'), mm('ha'), 14, sdt));
          }
          c.a += mm('hs');
          break;
        }
        case 'pulse':
          c.s *= 1 + P('amp') * F.beatPulse * F.gate[0];
          break;
      }
    });
  }

  /**
   * A shape's parameter slots (two vec4 from `o`) plus its frame state. Returns the shape's
   * characteristic size in scene units (for arms, textures and tips).
   */
  private packShape(s: Slot, sh: ShapeGene, bi: number, o: number, sdt: number, fused: boolean, copies: Copy[]): number {
    const F = this.sig.F;
    const sig = this.sig;
    const E = s.bd;
    const group = fused ? 'fs' : 'sh';
    const P = (k: string) => s.P(group, bi, sh.p, k, SHAPE_SCHEMAS[sh.kind]);
    const m = s.mem;
    const key = (k: string) => `b${bi}.${group}.${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);
    const set = (k: string, v: number) => (m[key(k)] = v);
    const b = s.genome.bodies[bi];
    const beat = F.beatPulse * F.gate[0];
    switch (sh.kind) {
      case 'dot': {
        const r = P('r') * (1 + 0.08 * this.resp('bass', F.stem[1]));
        E[o] = r;
        return Math.max(r, 0.004);
      }
      case 'polygon':
        E[o] = sh.p.n; E[o + 1] = P('r'); E[o + 2] = P('round');
        return P('r');
      case 'star':
        E[o] = sh.p.n; E[o + 1] = P('r'); E[o + 2] = P('inner');
        return P('r');
      case 'segment':
        E[o] = P('len'); E[o + 1] = P('w');
        return P('len') * 0.5;
      case 'solid': {
        // Segments in the body's local frame; spin motion turns the solid in 3D.
        const solid = sh.p.solid === 5 ? ((F.sectionIndex % 4) + 4) % 4 : sh.p.solid;
        const scale = P('size') + 0.07 * this.resp('bass', F.stem[1]) + 0.02 * beat;
        const mo = b.motion;
        const ang = mo.kind === 'spin' ? this.clk.spin * mo.p.rate * (mo.p.alt > 0.5 && F.barIndex % 2 === 1 ? -1 : 1) : 0;
        let n = 0;
        if (solid === 4) n = polygonSegs(s, 0, sh.p.sides, scale * 0.5, ang, 1);
        else {
          const tilt = P('tilt') + 0.25 * Math.sin(F.phase * 0.07);
          n = projectSolid(SOLIDS[solid], s, 0, ang, tilt, scale, 1);
          const inner = clamp01((F.act - 0.45) / 0.25) * P('inner');
          if (inner > 0.01) n = projectSolid(SOLIDS[(solid + 2) % 4], s, n, -ang * 0.5, -tilt, scale * 0.45, inner * 0.7);
        }
        s.segN = n;
        E[o] = scale;
        return scale * 0.5;
      }
      case 'bars':
        E[o] = sh.p.mode; E[o + 1] = sh.p.bins; E[o + 2] = P('radius'); E[o + 3] = P('len');
        E[o + 4] = P('fill'); E[o + 5] = 0.025 * F.stem[1];
        return sh.p.mode === 1 || sh.p.mode === 2 ? P('radius') + P('len') * 0.5 : 0.3;
      case 'curve': {
        const u = s.wv;
        const w0 = bi * 16;
        const ph = set('ph', (mm('ph') + sdt * F.speed * 0.07) % 4096);
        u[w0] = sh.p.form; u[w0 + 1] = P('amp'); u[w0 + 2] = 0; u[w0 + 3] = 0;
        u[w0 + 4] = P('radius'); u[w0 + 5] = P('turns'); u[w0 + 6] = sh.p.ra + 0.004 * Math.sin(F.phase * 0.05); u[w0 + 7] = sh.p.rb * (F.minor ? 0.75 : 1) - 0.005;
        u[w0 + 8] = ph; u[w0 + 9] = 0; u[w0 + 10] = this.bodyHue; u[w0 + 11] = b.color.p.detail;
        u[w0 + 12] = u[w0 + 13] = u[w0 + 14] = u[w0 + 15] = 0;
        if (sh.p.form === 5) {
          const L = WAVE_RATIOS.length;
          const stepN = Math.floor(Math.max(0, F.barIndex) / sh.p.rb);
          const [ra, rb] = WAVE_RATIOS[(((F.keyTonic + stepN * 5 + sh.p.ra - 1) % L) + L) % L];
          const fx = set('fx', approach(mm('fx', ra), ra, 1.2, sdt));
          const fy = set('fy', approach(mm('fy', rb), rb, 1.2, sdt));
          const newBar = F.barIndex >= 0 && F.barIndex !== mm('bi', F.barIndex);
          set('bi', F.barIndex);
          const swing = set('sw', newBar ? 1 : mm('sw') * Math.exp(-sdt * 1.2));
          const det = 0.012 * (F.melody - 0.5) + 0.004 * Math.sin(F.phase * 0.05);
          u[w0 + 4] = fx; u[w0 + 5] = fx * 2 + 0.01 + det; u[w0 + 6] = fy; u[w0 + 7] = fy * (F.minor ? 1.5 : 2) - 0.01;
          u[w0 + 8] = P('radius') * 0.72 * (0.75 + 0.2 * F.loud + 0.3 * swing);
          u[w0 + 9] = 0;
          u[w0 + 11] = P('amp') * 1.5 + 0.35 * sig.melodic() + 0.35 * F.onset[0];
          u[w0 + 12] = F.spin * 0.25;
          u[w0 + 13] = F.phase * 0.07 + F.beats * 0.25;
          u[w0 + 14] = 1.3 + F.phase * 0.05 - F.beats * 0.18;
          u[w0 + 15] = 0.5 - F.phase * 0.04 + F.beats * 0.11;
        }
        E[o] = P('radius');
        return sh.p.form === 0 ? 0.3 : P('radius');
      }
      case 'superscope': {
        // AVS superscope: 3D tumble on the body's bar clock, pushed by the bass and the beat.
        const ph = set('ph', (mm('ph') + sdt * F.speed * 0.07) % 4096);
        packSuperscope(s.wv, bi * 16, sh.p, P, this.clk.spin, ph, this.bodyHue, b.color.p.detail, beat, F.stem[1]);
        E[o] = P('size');
        return P('size');
      }
      case 'aurora': {
        const t = set('t', (mm('t') + sdt * F.speed * 0.15) % 512);
        // A standing glow (0.15) keeps the curtains visible through instrumental stretches with no
        // vocals or melody, which otherwise left the whole screen black.
        const lvl = 0.15 + Math.max(F.stem[2], F.stem[3] * 0.7 * (1 - F.gate[2])) + 0.08 * F.loud;
        if (fused) {
          E[o + 1] = P('fall'); E[o + 2] = P('rays'); E[o + 3] = P('wav');
          E[o + 4] = t; E[o + 5] = lvl;
        } else {
          const f = bi * BODY_VEC4 * 4 + 64;
          E[f] = s.P('ma', bi, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
          E[f + 1] = this.bodyHue;
          E[f + 2] = lvl;
          E[f + 3] = t;
          E[f + 4] = 0; E[f + 5] = P('fall'); E[f + 6] = P('rays'); E[f + 7] = P('wav');
        }
        return 0.2;
      }
      case 'plasma': case 'terrain': case 'edge': case 'beams': case 'scene': case 'cells': case 'cymatics': case 'landscape': case 'tonnetz': case 'notes':
        this.packField(s, b, bi, sdt, copies);
        return 0.2;
      case 'flame':
        return 0.28 * (P('zoom') / 0.22);
    }
    return 0.1;
  }

  /** Full-screen chunk shapes: their four field slots (EA..ED = slots 16-19), as the old emitters packed them. */
  private packField(s: Slot, b: BodyGene, bi: number, sdt: number, copies: Copy[]): void {
    const F = this.sig.F;
    const E = s.bd;
    const o = bi * BODY_VEC4 * 4 + 64;
    const sh = b.shape;
    const P = (k: string) => s.P('sh', bi, sh.p, k, SHAPE_SCHEMAS[sh.kind]);
    const m = s.mem;
    const key = (k: string) => `b${bi}.f.${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);
    const gain = s.P('ma', bi, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
    E[o] = gain;
    E[o + 1] = this.bodyHue;
    switch (sh.kind) {
      case 'plasma': {
        const tempo = P('tempo');
        m[key('t')] = (mm('t') + sdt * F.speed * P('speed') * 0.66) % 256;
        const lb = mm('lb', F.beats);
        let db = F.beats - lb;
        if (db < 0) db += 256;
        if (db > 1) db = 0;
        m[key('lb')] = F.beats;
        const dBurst = sdt * (0.15 + 1.6 * F.beatPulse * F.gate[0]) * F.speed;
        const dTempo = db * 0.25 + sdt * 0.04 * F.speed;
        m[key('l')] = (mm('l') + dBurst + (dTempo - dBurst) * tempo) % 256;
        const bs = (m[key('bs')] = approach(mm('bs'), F.stem[1] * F.gate[1], 1.5, sdt));
        m[key('b')] = (mm('b') + sdt * bs * 0.25) % 256;
        m[key('mel')] = approach(mm('mel', 0.5), F.melody, 1.2, sdt);
        E[o + 2] = m[key('t')]; E[o + 3] = m[key('l')];
        E[o + 4] = P('scale'); E[o + 5] = P('warp'); E[o + 6] = P('bands'); E[o + 7] = P('lines');
        E[o + 8] = m[key('l')] * 0.1 + (m[key('b')] - m[key('l')] * 0.1) * tempo;
        E[o + 9] = F.stem[1] + (bs - F.stem[1]) * tempo;
        E[o + 10] = bs * tempo;
        E[o + 11] = m[key('mel')] * 0.3 * P('melHue');
        E[o + 12] = 1 + (0.9 * F.beatPulse - 0.45) * P('pulse');
        break;
      }
      case 'edge': {
        const side = sh.p.side;
        const perp = side === 0 || side === 3 ? Math.abs(s.shift[0]) : Math.abs(s.shift[1]);
        E[o + 2] = perp + 2 / this.h;
        E[o + 4] = sh.p.mode; E[o + 5] = side; E[o + 6] = P('base'); E[o + 7] = P('height');
        E[o + 8] = P('density');
        if (sh.p.mode === 0) {
          const bb = F.beatIndex >= 0 ? F.beatIndex : Math.floor(F.time * 2);
          let start = mm('start', bb);
          let wide = mm('wide', 1);
          if (bb >= start + wide || bb < start) {
            start = m[key('start')] = bb;
            wide = m[key('wide')] = h11(bb * 1.7) < 0.3 ? 2 : 1;
            const lv = 0.6 * F.loud + 0.4 * Math.max(F.stem[1], F.stem[3]);
            m[key('h')] = (0.04 + P('height') * Math.pow(h11(bb * 3.3), 1.6) + 0.12 * lv) * (0.6 + 0.4 * F.act);
            m[key('cols')] = 3 + Math.floor(h11(bb * 5.1) * 3) * wide;
            m[key('lit')] = (0.08 + 0.3 * Math.max(F.stem[3], F.stem[2]) + 0.15 * F.act) * (0.5 + P('density'));
          }
          const frac = (bb - start + (F.beatIndex >= 0 ? F.beatPhase : (F.time * 2) % 1)) / wide;
          E[o + 9] = mm('cols', 4);
          E[o + 12] = mm('h', 0.1); E[o + 13] = start; E[o + 14] = mm('lit', 0.3); E[o + 15] = frac;
        } else if (sh.p.mode === 1) {
          const y = (F.melody - 0.5) * 0.62 + P('base') * 0.3;
          const prev = mm('y', y);
          const yy = approach(prev, y, 6, sdt);
          m[key('y')] = yy;
          E[o] = gain * this.sig.melodic() * 1.2;
          E[o + 10] = yy; E[o + 11] = prev;
        } else if (sh.p.mode === 3) {
          const tgt = clamp01(0.6 * F.loud + 0.4 * F.stem[1] + 0.15 * Math.sin(F.phase * 0.4));
          m[key('th')] = approach(mm('th', tgt), tgt, 3, sdt);
          E[o + 10] = m[key('th')];
        }
        break;
      }
      case 'terrain': {
        const prev = mm('lb', F.beats);
        let d = F.beats - prev;
        if (d < 0) d += 256;
        if (d > 2) d = sdt * 2;
        m[key('lb')] = F.beats;
        m[key('sc')] = (mm('sc') + d * P('speed') * (0.5 + 0.5 * F.act)) % 1;
        E[o + 2] = m[key('sc')];
        E[o + 4] = 0; E[o + 5] = P('density'); E[o + 6] = P('peaks');
        E[o + 8] = P('terrain'); E[o + 9] = P('flash');
        break;
      }
      case 'cells': {
        const cf = { speed: F.speed, beats: F.beats, beatPulse: F.beatPulse, bass: F.stem[1], loud: F.loud };
        packCells(E, o, P, sh.p, cf, m, key, (k, raw) => this.resp(k, raw), sdt);
        break;
      }
      case 'beams': {
        const bf = { bars: this.clk.bars, loud: F.loud, melodic: this.sig.melodic(), drop: F.drop, speed: F.speed };
        packBeams(E, o, o - 56, P, sh.p, bf, m, key, copies[0]?.y ?? 0, (k, raw) => this.resp(k, raw), sdt);
        break;
      }
      case 'scene':
        packScene(s.scn, { F, sdt, P, raw: sh.p, mem: m, key: key('') });
        break;
      case 'landscape':
        if (this.land) packLandscape(s.lsu, { F, sdt, P, raw: sh.p, mem: m, key: key(''), world: this.land.world, now: this.land.now, keyHue: F.keyHue });
        break;
      case 'cymatics': {
        const cf = { chroma: this.sig.chroma, spec: this.sig.spec, keyTonic: F.keyTonic, minor: F.minor, sectionIndex: F.sectionIndex, bass: F.stem[1] * F.gate[1], loud: F.loud, bpm: F.bpm };
        packCymatics(E, o, o - 56, P, sh.p, cf, m, key, this.every(s, bi, 'cym', 4 * sh.p.hold), (k, raw) => this.resp(k, raw), sdt);
        break;
      }
      case 'tonnetz': {
        const tf = { chord: F.chord, tonnetzX: F.tonnetzX, tonnetzY: F.tonnetzY, keyTonic: F.keyTonic, tension: F.tension, chordPulse: F.chordPulse, resolve: F.resolve, loud: F.loud };
        packTonnetz(E, o, o - 56, o - 52, P, sh.p, tf, m, key, sdt);
        break;
      }
      case 'notes':
        packNotes(E, o, o - 56, o - 52, P, sh.p, F.notes, m, key, sdt);
        break;
    }
    void copies;
  }

  /** Deformation slots 7-11 (arms: count, reach, curl now, width; angles; lengths). */
  private packDeform(s: Slot, b: BodyGene, bi: number, o: number, sdt: number): void {
    const F = this.sig.F;
    const E = s.bd;
    const d = b.deform;
    const P = (k: string) => s.P('de', bi, d.p, k, DEFORM_SCHEMAS[d.kind]);
    const m = s.mem;
    const key = (k: string) => `b${bi}.d${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);
    switch (d.kind) {
      case 'arms': {
        // Each arm reaches and retracts with its instrument plus a breath of its own, sways every
        // 2 bars out of step with the others; the set turns once per `turn` bars.
        const n = d.p.count;
        const g = (k: number) => F.stem[k] * F.gate[k];
        const raw = [F.onset[0] * F.gate[0] * 0.6 + g(0) * 0.4, g(1), Math.max(g(2), 0.7 * this.sig.melodic()), g(3), F.loud];
        const drivers = raw.map((x, k) => this.resp('a' + k, x));
        const bars = this.clk.bars;
        const spin = (TAU * bars) / d.p.turn;
        const sway = P('sway');
        for (let k = 0; k < n; k++) {
          const ang = spin + (k * TAU) / n + 0.45 * sway * Math.sin((TAU * bars) / 2 + k * 1.3);
          const breath = 0.5 + 0.5 * Math.sin((TAU * bars) / (2 + (k % 5) * 0.5) + k * 2.1);
          const target = 0.2 + 0.8 * drivers[k % 5] + 0.45 * breath;
          const len = (m[key('len' + k)] = approach(mm('len' + k, target), target, 2.5, sdt));
          E[o + 4 + (k < 4 ? k : 4 + k - 4)] = ang;
          E[o + 12 + (k < 4 ? k : 4 + k - 4)] = len;
        }
        E[o] = n;
        E[o + 1] = P('reach');
        E[o + 2] = P('curl') * 0.5 * Math.sin((TAU * bars) / 3);
        E[o + 3] = P('width');
        break;
      }
      case 'wobble':
        E[o] = d.p.lobes;
        E[o + 1] = P('amp') * (0.6 + 0.8 * this.resp('bass', F.stem[1] * F.gate[1]));
        E[o + 2] = -this.clk.spin * d.p.rate * d.p.lobes;
        break;
      case 'noise': {
        const ph = (m[key('ph')] = (mm('ph') + sdt * P('speed') * F.speed) % 4096);
        E[o] = P('amp') * (0.6 + 0.8 * this.resp('loud', F.loud));
        E[o + 1] = P('scale');
        E[o + 2] = ph;
        break;
      }
      case 'twist':
        E[o] = P('amt') * (0.6 + 0.6 * this.resp('loud', F.loud));
        break;
    }
  }

  /**
   * A body's deform ops: the chain's per-frame rates become absolute shaping amounts,
   * breathing with the music (loudness, beat, bass). Folds and variations keep their meaning.
   */
  private packOps(s: Slot, ops: OpGene[], bi: number, sdt: number): void {
    const F = this.sig.F;
    const A = F.aspect * 0.5;
    const beat = F.beatPulse * F.gate[0];
    const mod = 0.55 + 0.6 * F.loud + 0.3 * beat;
    ops.forEach((op, i) => {
      const P = (k: string) => s.P('dr', bi * 3 + i, op.p, k, OP_SCHEMAS[op.op]);
      const a = s.drA;
      const j = (bi * 3 + i) * 4;
      a[j] = a[j + 1] = a[j + 2] = a[j + 3] = 0;
      s.drB[j] = drawOpId(op.op);
      const key = (k: string) => `dr${bi}.${i}.${k}`;
      const center = (w: number) => {
        a[j] = (op.p.cx ?? 0) + w * A * 0.64 * Math.sin((TAU * F.bars) / 8);
        a[j + 1] = (op.p.cy ?? 0) + w * 0.63 * Math.sin((TAU * F.bars) / 6 + 1);
      };
      switch (op.op) {
        case 'swirl':
          center(P('wander'));
          a[j + 2] = P('amt') * 60 * op.w * mod;
          a[j + 3] = P('k');
          break;
        case 'twist':
          center(0);
          a[j + 2] = P('amt') * 60 * op.w * mod;
          break;
        case 'ripple': {
          const ph = (s.mem[key('ph')] = ((s.mem[key('ph')] ?? 0) + sdt * P('speed') * F.speed * TAU) % 4096);
          a[j] = P('amp') * 20 * op.w * (0.6 + 0.8 * F.stem[1]);
          a[j + 1] = P('freq');
          a[j + 2] = ph;
          a[j + 3] = op.p.radial;
          break;
        }
        case 'noise': {
          const ph = (s.mem[key('ph')] = ((s.mem[key('ph')] ?? 0) + sdt * P('speed') * F.speed) % 4096);
          a[j] = P('amp') * 25 * op.w * mod;
          a[j + 1] = P('scale');
          a[j + 2] = ph;
          break;
        }
        case 'rotate': {
          center(P('wander'));
          const sign = op.p.alt > 0.5 && F.barIndex % 2 === 1 ? -1 : 1;
          const ang = (s.mem[key('ang')] = ((s.mem[key('ang')] ?? 0) - (op.p.lock * this.sig.spinStep + P('rate') * op.w * sdt * 60 * F.speed) * sign) % (TAU * 64));
          a[j + 2] = ang;
          break;
        }
        case 'zoom':
          center(P('wander'));
          a[j + 2] = 1 + P('rate') * 12 * op.w * (0.3 + 1.2 * beat + 0.5 * F.stem[1]);
          a[j + 3] = op.p.radial;
          break;
        case 'mirror':
          a[j] = op.p.axis;
          break;
        case 'kaleido':
          a[j] = op.p.n;
          a[j + 1] = F.spin * op.p.lock;
          break;
        default: // flame variations
          a[j] = Math.min(1, op.w * (0.7 + 0.3 * mod));
          a[j + 1] = P('s');
      }
    });
  }

  /** Dye emission: copies push the fluid (instrument pushes for stations, beat pulses otherwise). */
  private dye(s: Slot, b: BodyGene, bi: number, copies: Copy[], force: number): void {
    const F = this.sig.F;
    const fl = this.fluid && s.genome.carrier.kind === 'fluid' ? this.fluid : null;
    if (!fl || force <= 0) return;
    const m = s.mem;
    const key = (k: string) => `b${bi}.y${k}`;
    const mm = (k: string, init = 0) => m[key(k)] ?? (m[key(k)] = init);
    const inst = b.place.kind === 'stations' ? s.P('pl', bi, b.place.p, 'inst', PLACE_SCHEMAS.stations) : 0;
    const newBeat = F.beatIndex >= 0 && F.beatIndex !== mm('bt', F.beatIndex);
    m[key('bt')] = F.beatIndex;
    const hits = [F.hit > 0.5 && mm('h0') <= 0.5, F.onset[1] > 0.5 && mm('h1') <= 0.5, F.onset[2] > 0.5 && mm('h2') <= 0.5, F.onset[3] > 0.5 && mm('h3') <= 0.5];
    m[key('h0')] = F.hit;
    m[key('h1')] = F.onset[1];
    m[key('h2')] = F.onset[2];
    m[key('h3')] = F.onset[3];
    const tf = Math.min(1.6, Math.max(0.5, F.bpm / 120));
    const sdt = F.dt;
    if (b.place.kind === 'grid') {
      // A grid drops dye at a new star-like spot on every beat.
      if (newBeat) {
        const c = F.beatIndex * 1.37 + bi;
        const x = (h11(c) - 0.5) * F.aspect * 0.9;
        const y = h11(c * 2.1) - 0.5;
        const a = h11(c * 3.3) * TAU;
        const Fm = 300 * (0.3 + F.loud) * force * (0.4 + 0.6 * F.act);
        fl.splat(x / F.aspect + 0.5, y + 0.5, Math.cos(a) * Fm, Math.sin(a) * Fm, 0.002, 0);
      }
      return;
    }
    copies.forEach((c, i) => {
      const k4 = i % 4;
      const strength = c.level;
      const lvlInst = k4 === 0 ? F.onset[0] : F.stem[k4];
      const Fo = strength > 0.04 && (k4 > 0 || F.hit > 0) ? (k4 === 0 ? 1400 * F.hit : 420 * strength) : 0;
      let Fi = 0;
      if (inst > 0) {
        if (hits[k4]) m[key('a' + i)] = mm('a' + i, i * (TAU / 4)) + (0.35 + 0.5 * h11(F.beatIndex * 3.1 + i)) * (i % 2 ? -1 : 1);
        if (k4 === 0) Fi = hits[0] ? 650 * F.hit * tf : 0;
        else if (lvlInst > 0.04) Fi = 45 * lvlInst * tf * Math.min(3, sdt * 60) + (newBeat ? 260 * lvlInst * tf : 0);
      }
      const Fm = (Fo + (Fi - Fo) * inst) * (0.4 + 0.6 * F.act) * force;
      const a = inst > 0.5 ? mm('a' + i, i * (TAU / 4)) : F.spin * 0.5 + i * (TAU / 4) + c.a;
      if (Fm > 0) fl.splat(c.x / F.aspect + 0.5, c.y + 0.5, Math.cos(a) * Fm, Math.sin(a) * Fm, 0.002, 0);
    });
  }

  /** Where the sparks are born: the placement's copies (rotating emitters), its centre, ring or row. */
  private sparkSpawn(s: Slot, b: BodyGene, copies: Copy[]): void {
    const F = this.sig.F;
    const sp = s.spawn;
    sp.count = 3;
    sp.radius = 0.3;
    sp.angle = TAU * F.barPhase;
    const k = b.place.kind;
    const c0 = copies[0];
    if (b.fuse) {
      // Born on the fused shape's rim.
      sp.mode = 2;
      sp.count = 6;
      sp.radius = Math.max(0.05, s.bd[s.genome.bodies.indexOf(b) * BODY_VEC4 * 4 + 19]);
      sp.angle = F.spin * 0.25;
      return;
    }
    if (b.shape.kind === 'curve' && b.shape.p.form === 0) sp.mode = 1;
    else if (k === 'grid') sp.mode = 0;
    else if (k === 'ring') sp.mode = 3;
    else if (k === 'row' && b.place.p.y < -0.35) sp.mode = 5;
    else if (copies.length === 1 && Math.hypot(c0.x, c0.y) < 0.03) sp.mode = 4;
    else if (k === 'orbit' && Math.hypot(b.place.p.x, b.place.p.y) < 0.03) {
      sp.mode = 2;
      sp.count = copies.length;
      sp.radius = Math.hypot(c0.x, c0.y);
      sp.angle = Math.atan2(c0.y, c0.x);
    } else {
      // One copy per frame in turn: the sparks spread over all of them.
      const c = copies[this.frame % copies.length];
      sp.mode = 2;
      sp.count = 1;
      sp.radius = Math.hypot(c.x, c.y);
      sp.angle = Math.atan2(c.y, c.x);
    }
  }

  /** Curve bodies: one geometry draw per copy (ring and mirror folds become copies too). */
  private queueCurve(s: Slot, b: BodyGene, bi: number, copies: Copy[]): void {
    const F = this.sig.F;
    const gain = s.P('ma', bi, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
    const mk = b.material.kind;
    const list: number[][] = [];
    const c0 = copies[0];
    if (b.place.kind === 'ring') {
      const n = b.place.p.n;
      const rad = s.P('pl', bi, b.place.p, 'radius', PLACE_SCHEMAS.ring);
      for (let i = 0; i < n; i++) {
        const a = c0.a + (i / n) * TAU;
        list.push([c0.x + rad * Math.cos(a), c0.y + rad * Math.sin(a), a - Math.PI / 2, c0.s, 1, 1]);
      }
    } else if (b.place.kind === 'mirror') {
      const ax = b.place.p.axis;
      list.push([c0.x, c0.y, c0.a, c0.s, 1, 1]);
      if (ax === 0 || ax === 2) list.push([-c0.x, c0.y, -c0.a, c0.s, -1, 1]);
      if (ax === 1 || ax === 2) list.push([c0.x, -c0.y, -c0.a, c0.s, 1, -1]);
      if (ax === 2) list.push([-c0.x, -c0.y, c0.a, c0.s, -1, -1]);
    } else for (const c of copies) list.push([c.x, c.y, c.a, c.s * (copies.length > 1 ? 0.35 + 0.65 * Math.min(1, c.level + 0.3) : 1), 1, 1]);
    const d = b.deform;
    const o = bi * BODY_VEC4 * 4;
    const dk = { none: 0, arms: 1, wobble: 2, noise: 3, twist: 4 }[d.kind];
    s.curves.push({
      body: bi,
      top: b.emit.kind === 'none',
      bright: gain * 0.5 * (0.3 + 0.9 * Math.max(F.loud, 0.6 * this.sig.melodic())) * (b.emit.kind === 'sparks' ? b.emit.p.body : 1),
      thick: mk === 'line' ? b.material.p.width : mk === 'glow' ? 3 + b.material.p.width * 60 : 1.5,
      n: b.shape.kind === 'superscope' ? b.shape.p.n : b.shape.p.form === 3 || b.shape.p.form === 5 ? 1024 : 512,
      soft: mk === 'glow' ? 1 : 0,
      dash: mk === 'dots' ? 3 + b.material.p.spacing * 400 : 0,
      copies: list,
      deform: dk,
      dp: [s.bd[o + 28], s.bd[o + 29], s.bd[o + 30], s.bd[o + 31]],
      blend: b.material.p.blend ?? 0,
    });
  }

  private updateParticles(s: Slot, sdt: number, fluid: boolean): void {
    const eng = this.eng;
    if (!this.particles) this.particles = new Particles(eng.gl, eng.fs);
    const bi = s.sparks;
    const b = s.genome.bodies[bi];
    const sch = EMIT_SCHEMAS.sparks;
    const P = (k: string) => s.P('em', bi, b.emit.p, k, sch);
    const n = Math.min(b.emit.p.count, this.opts.particleCap);
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
    // Particles follow the chain's rotation too, so a vortex carries its sparks.
    let rot = 0;
    for (const o of s.genome.chain) if (o.op === 'rotate') rot += o.p.lock * 4 * (this.sig.spinStep / Math.max(sdt, 1e-3)) * 0.25;
    pu.rotFlow = -rot;
    pu.converge = F.build * 1.5;
    pu.drag = P('drag');
    pu.lifeRate = P('life');
    pu.speed = P('speed') * F.speed;
    // Surge: speed and zoom flow follow the beat envelope.
    const surge = 1 + (0.35 + F.surge - 1) * P('surge');
    pu.speed *= surge;
    pu.zoomFlow *= surge;
    pu.spawnFrom = pu.spawnTo = s.spawn.mode;
    pu.spawnMix = 0;
    pu.emitAngle = s.spawn.angle;
    pu.emitCount = s.spawn.count;
    pu.emitRadius = s.spawn.radius;
    pu.burst = F.hit > 0 ? 0.02 * F.act : 0;
    pu.burstSeed = Math.random() * 1000;
    pu.burstSpeed = 0.3 + F.stem[0] + F.drop;
    pu.liftX = 0;
    pu.liftY = P('lift') * F.speed;
    pu.spread = P('spread');
    this.particles.update(pu);
  }

  private drawParticles(s: Slot, weight: number, fb: boolean): void {
    if (!this.particles || s.sparks < 0) return;
    const F = this.sig.F;
    const b = s.genome.bodies[s.sparks];
    const sch = EMIT_SCHEMAS.sparks;
    const size = Math.max(1, s.P('em', s.sparks, b.emit.p, 'size', sch) * (this.h / 1080));
    const alive = 0.35 + 0.65 * F.act;
    const gain = s.P('ma', s.sparks, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
    const bright = gain * 1.4 * weight * (fb ? 0.5 : 1) * (0.7 + 0.6 * F.beatPulse * F.gate[0]);
    this.particles.draw(size, bright, alive, s.cols);
  }

  private drawCurve(s: Slot, c: CurveDraw, weight: number, sdt: number, fb: boolean): void {
    const eng = this.eng;
    const gl = eng.gl;
    const p = eng.pWave.use();
    this.setCommon(p, s, sdt, fb);
    // At least a pixel wide; below 1080p the energy is scaled down so small renders match.
    const want = c.thick * (this.h / 1080);
    const thick = Math.max(1, want);
    const o = c.body * BODY_VEC4 * 4;
    const bd = s.bd;
    p.f1('uN', c.n)
      .f1('uThick', thick)
      .f1('uBright', c.bright * weight * (fb ? 1 : 3) * Math.sqrt(Math.min(1, want / thick)) / Math.sqrt(Math.max(1, c.copies.length)))
      .f4v('uW', s.wv.subarray(c.body * 16, c.body * 16 + 16))
      .i1('uDk', c.deform)
      .f4('uDp', c.dp[0], c.dp[1], c.dp[2], c.dp[3])
      .f4('uArmA0', bd[o + 32], bd[o + 33], bd[o + 34], bd[o + 35])
      .f4('uArmA1', bd[o + 36], bd[o + 37], bd[o + 38], bd[o + 39])
      .f4('uArmL0', bd[o + 40], bd[o + 41], bd[o + 42], bd[o + 43])
      .f4('uArmL1', bd[o + 44], bd[o + 45], bd[o + 46], bd[o + 47])
      .i1('uDrBase', c.body * 3)
      .i1('uDrN', Math.round(bd[o + 50]));
    gl.bindVertexArray(eng.lineVao);
    const lines = c.blend ? setCurveBlend(gl, c.blend) : false;
    p.f1('uLines', lines ? 1 : 0);
    for (const cp of c.copies) {
      p.f4('uT', cp[0], cp[1], cp[2], cp[3]).f2('uFlip', cp[4], cp[5]).f1('uDash', c.dash).f1('uSoft', c.soft);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, c.n * 2);
    }
    if (c.blend) resetCurveBlend(gl);
  }

  /**
   * Where a body's agents are born or gather: its copies (uv x, uv y, radius in screen heights) into
   * `out`; fold placements and full-screen chunks cover the screen. Returns the count (1-6).
   */
  private agentHomes(s: Slot, bi: number, out: Float32Array): number {
    const b = s.genome.bodies[bi];
    const aspect = this.sig.F.aspect;
    const o = bi * BODY_VEC4 * 4;
    const R = Math.max(0.01, s.bd[o + 19]);
    const whole = SHAPE_CLASS[b.shape.kind] === 'field' || b.place.kind === 'grid';
    const n = whole ? 1 : Math.max(1, Math.min(COPY_SLOTS, Math.round(s.bd[o + 16])));
    for (let i = 0; i < n; i++) {
      const j = (bi * COPY_SLOTS + i) * 4;
      out[i * 3] = whole ? 0.5 : s.cp[j] / aspect + 0.5;
      out[i * 3 + 1] = whole ? 0.5 : s.cp[j + 1] + 0.5;
      out[i * 3 + 2] = whole ? 0.75 : b.place.kind === 'ring' ? b.place.p.radius + R : R * Math.max(0.3, s.cp[j + 3]);
    }
    return n;
  }

  /** The flock's overlay: the birds drawn again straight onto the composite (emit over > 0), sized in stage pixels. */
  private drawFlockOverlay(s: Slot): void {
    if (!this.flock) return;
    const b = s.genome.bodies[s.flock];
    const PE = (k: string) => s.P('em', s.flock, b.emit.p, k, EMIT_SCHEMAS.flock);
    const over = PE('over');
    if (!(over > 0.001)) return;
    const gain = s.P('ma', s.flock, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
    this.flock.draw(flockOverlaySize(PE('osize'), this.h), gain * FLOCK_GAIN * FLOCK_OVERLAY_GAIN * over * s.weight, PE('speed') * this.sig.F.speed, s.cols);
  }

  /** Boids: splat into the neighbourhood grid, then align, cohere, separate, home and move (genes/boidsGpu.ts). */
  private updateFlock(s: Slot, sdt: number): void {
    if (!this.flock) this.flock = new Boids(this.eng.gl, this.eng.fs, this.eng.hdr);
    if (this.flockOwner !== s) this.flock.reseed();
    this.flockOwner = s;
    const bi = s.flock;
    const b = s.genome.bodies[bi];
    const P = (k: string) => s.P('em', bi, b.emit.p, k, EMIT_SCHEMAS.flock);
    const F = this.sig.F;
    const homes = this.flockHomes;
    const n = this.agentHomes(s, bi, homes);
    const count = Math.max(1024, Math.round(b.emit.p.count * Math.min(1, (this.w * this.h) / (1280 * 720))));
    this.flock.step({
      dt: sdt, time: this.sig.clock, aspect: F.aspect, count, speed: P('speed') * F.speed, radius: P('radius'),
      align: P('align'), cohere: P('cohere'), separate: P('separate'), wander: P('wander'), home: P('home'),
      copies: homes, nCopies: n, burst: F.dropStart ? b.emit.p.onDrop : 0,
    });
  }
  private readonly flockHomes = new Float32Array(COPY_SLOTS * 3);

  /** Physarum: sense, move, deposit, diffuse and decay (genes/physarumGpu.ts). */
  private updateSlime(s: Slot, sdt: number): void {
    if (!this.slime) {
      this.slime = new Physarum(this.eng.gl, this.eng.fs, this.eng.hdr);
      this.slime.resize(this.w, this.h);
    }
    if (this.slimeOwner !== s) this.slime.reseed();
    this.slimeOwner = s;
    const bi = s.slime;
    const b = s.genome.bodies[bi];
    const sch = EMIT_SCHEMAS.slime;
    const P = (k: string) => s.P('em', bi, b.emit.p, k, sch);
    const F = this.sig.F;
    const pl = this.slimeCopies;
    const n = this.agentHomes(s, bi, pl);
    const gain = s.P('ma', bi, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
    this.slime.step({
      dt: sdt, time: this.sig.clock, aspect: F.aspect, count: b.emit.p.count,
      sa: P('sa'), sd: P('sd'), turn: P('steer'), step: P('step'), deposit: P('deposit'), decay: P('decay'), diffuse: P('diffuse'),
      feed: P('feed'), fb: s.fb.read.t, birth: P('birth'), copies: pl, nCopies: n,
      gain: gain * SLIME_GAIN, scale: slimeDisplayScale(b.emit.p), cols: s.cols,
      burst: F.dropStart ? b.emit.p.onDrop : 0,
    });
  }

  /** Stem ecosystem: populations from the mix, then births, hunting, grazing, blooming (genes/ecosystemGpu.ts). */
  private updateEco(s: Slot, sdt: number): void {
    if (!this.eco) {
      this.eco = new Ecosystem(this.eng.gl, this.eng.fs, this.eng.hdr);
      this.eco.resize(this.w, this.h);
    }
    if (this.ecoOwner !== s) this.eco.reseed();
    this.ecoOwner = s;
    const bi = s.eco;
    const b = s.genome.bodies[bi];
    const sch = EMIT_SCHEMAS.ecosystem;
    const p: Record<string, number> = {};
    for (const k of Object.keys(sch)) p[k] = s.P('em', bi, b.emit.p, k, sch);
    const F = this.sig.F;
    const pres = [0, 1, 2, 3].map((i) => F.gate[i] * (0.7 + 0.3 * Math.min(1, F.stem[i] * 2)));
    this.eco.step({
      dt: sdt, time: this.sig.clock, aspect: F.aspect, count: b.emit.p.count, cuts: ecoCuts(p), pres, env: F.stem,
      strike: F.onset[0] + 0.4 * F.beatPulse * F.gate[0], drop: F.drop, p,
    });
  }

  private feedbackPass(s: Slot, sdt: number, partOwner: boolean, flameOwner: boolean, slimeOwner = false, flockOwner = false, ecoOwner = false): void {
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
      .f1('uBlur', cp.blur + 0.08 * (this.lyricNudges.get(s)?.blur ?? 0))
      .f1('uDecaySub', (eng.hq ? 0.0015 : 1.5 / 255) * cp.floor * sdt * 60)
      .f1('uFlowAmt', s.P('car', 0, cp, 'famt', CARRIER_SCHEMA) * sdt * 60 * this.sig.F.speed)
      .f1('uFlowScale', cp.fscale);
    // Sharpen: reaction-diffusion growth, seeded by grain noise that follows the high end of the mix.
    const F = this.sig.F;
    const sharpen = g.carrier.kind === 'none' ? 0 : s.P('car', 0, cp, 'sharpen', CARRIER_SCHEMA);
    p.f1('uSharpen', sharpen)
      .f1('uGrain', s.P('car', 0, cp, 'grain', CARRIER_SCHEMA))
      .f1('uSharpNoise', sharpen * 0.2 * Math.min(1, Math.max(0, F.stem[3] * 1.4 + F.onset[0] * 0.4 - 0.45)));
    // Border: a frame in a palette colour that walks the three slots every 16 bars, swelling with the bass.
    const border = g.carrier.kind === 'none' ? 0 : s.P('car', 0, cp, 'border', CARRIER_SCHEMA);
    if (border > 0.001) {
      const t = ((F.bars / 16) % 1) * 3;
      const k = Math.floor(t);
      const f = t - k;
      const cc = s.cols;
      const a = (k % 3) * 3;
      const b = ((k + 1) % 3) * 3;
      // The bass swell is smoothed (about 0.15 s): the border floods the whole frame under a strong
      // inward flow, and a raw sixteenth-note bass line made that flood strobe.
      const bass = (s.mem['bord.b'] = approach(s.mem['bord.b'] ?? F.stem[1], F.stem[1], 11, sdt));
      const lvl = 0.35 + 0.5 * bass;
      p.f1('uBorder', Math.min(1, border))
        .f1('uBorderW', 0.004 + 0.01 * bass * border)
        .f3('uBorderCol', (cc[a] + (cc[b] - cc[a]) * f) * lvl, (cc[a + 1] + (cc[b + 1] - cc[a + 1]) * f) * lvl, (cc[a + 2] + (cc[b + 2] - cc[a + 2]) * f) * lvl);
    } else p.f1('uBorder', 0);
    const water = g.carrier.kind === 'none' ? 0 : s.P('car', 0, cp, 'water', CARRIER_SCHEMA);
    const wt = this.water?.tex;
    p.tex('uWater', wt ?? eng.black).f1('uWaterAmt', wt && water > 0.001 ? water : 0).f2('uWaterTexel', this.water ? 1 / this.water.w : 0, this.water ? 1 / this.water.h : 0);
    eng.fs.draw();

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (const c of s.curves) if (!c.top) this.drawCurve(s, c, 1, sdt, true);
    if (partOwner && s.sparks >= 0 && g.bodies[s.sparks].emit.p.top < 0.5) this.drawParticles(s, 1, true);
    if (slimeOwner && this.slime && s.slime >= 0) {
      const b = g.bodies[s.slime];
      const gain = s.P('ma', s.slime, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
      this.slime.draw(gain * SLIME_GAIN, slimeDisplayScale(b.emit.p), s.cols);
    }
    if (flockOwner && this.flock && s.flock >= 0) {
      const b = g.bodies[s.flock];
      const gain = s.P('ma', s.flock, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
      const size = Math.max(1, s.P('em', s.flock, b.emit.p, 'size', EMIT_SCHEMAS.flock) * (this.h / 1080));
      this.flock.draw(size, gain * FLOCK_GAIN * (0.7 + 0.5 * this.sig.F.loud), s.P('em', s.flock, b.emit.p, 'speed', EMIT_SCHEMAS.flock) * this.sig.F.speed, s.cols);
    }
    if (ecoOwner && this.eco && s.eco >= 0) {
      const b = g.bodies[s.eco];
      const PE = (k: string) => s.P('em', s.eco, b.emit.p, k, EMIT_SCHEMAS.ecosystem);
      const gain = s.P('ma', s.eco, b.material.p, 'gain', MATERIAL_SCHEMAS[b.material.kind]);
      const F = this.sig.F;
      const fill = Math.min(1, Math.max(0.03, (1 - s.decay) * 8));
      this.eco.draw({
        gain, fieldScale: ecoFieldScale(b.emit.p), field: PE('field'), size: PE('size'), trail: PE('trail'), glyph: b.emit.p.glyph, hues: b.emit.p.hues,
        bright: gain * fill * Math.min(1.2, Math.sqrt(16384 / Math.max(1, this.eco.count))), strike: F.onset[0], cols: s.cols,
      });
      s.fb.write.bind();
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
    // Lyrics gene: the sung line added into the feedback, so the chain carries and smears it. Scaled
    // by the fade rate so the trail's steady brightness stays about `smear` whatever the half-life.
    const ly = g.lyrics?.p;
    if (ly && ly.show > 0 && ly.smear > 0 && g.carrier.kind !== 'none' && this.caption?.has && this.caption.alpha > 0.002) {
      s.fb.write.bind();
      const c = s.cols;
      this.caption.draw(ly.smear * this.caption.alpha * Math.max(0.06, 1 - s.decay) * 0.6, 0.45 + 0.55 * c[0], 0.45 + 0.55 * c[1], 0.45 + 0.55 * c[2], this.cam);
    }
    gl.disable(gl.BLEND);
    s.fb.swap();
  }

  /** Ray-marches the slot's scene body into its reduced-resolution target. */
  private scenePass(s: Slot, sdt: number): void {
    const gl = this.eng.gl;
    const b = s.genome.bodies.find((x) => x.shape.kind === 'scene');
    if (!b || !s.progs.scene) return;
    const res = b.shape.p.res;
    const w = Math.max(2, Math.round(this.w * res));
    const h = Math.max(2, Math.round(this.h * res));
    if (!s.sceneT || s.sceneT.w !== w || s.sceneT.h !== h) {
      s.sceneT?.dispose();
      s.sceneT = new Target(gl, w, h, [this.eng.hdr], gl.LINEAR);
    }
    s.sceneT.bind();
    gl.disable(gl.BLEND);
    const p = s.progs.scene.use();
    this.setCommon(p, s, sdt, true);
    p.f2('uRes', w, h).f4v('uScn', s.scn);
    this.eng.fs.draw();
  }

  /** Ray-marches the slot's landscape body (the song's world map) into its reduced-resolution target. */
  private landPass(s: Slot, sdt: number): void {
    const gl = this.eng.gl;
    const b = s.genome.bodies.find((x) => x.shape.kind === 'landscape');
    if (!b || !s.progs.land || !this.land) return;
    const res = b.shape.p.res;
    const w = Math.max(2, Math.round(this.w * res));
    const h = Math.max(2, Math.round(this.h * res));
    if (!s.landT || s.landT.w !== w || s.landT.h !== h) {
      s.landT?.dispose();
      s.landT = new Target(gl, w, h, [this.eng.hdr], gl.LINEAR);
    }
    const tex = this.land.texture();
    s.landT.bind();
    gl.disable(gl.BLEND);
    const p = s.progs.land.use();
    this.setCommon(p, s, sdt, true);
    p.f2('uRes', w, h).f4v('uLs', s.lsu).tex('uWorld', tex);
    this.eng.fs.draw();
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
      .f4v('uBd', s.bd)
      .f4v('uTmb', this.tmbUniforms(s))
      .f4v('uCp', s.cp)
      .f4v('uCq', s.cq)
      .f4v('uWv', s.wv)
      .f4v('uDrA', s.drA)
      .f4v('uDrB', s.drB)
      .f4v('uSeg', s.seg)
      .f4v('uSegZ', s.segZ)
      .i1('uSegN', s.segN)
      .f4v('uChroma4', this.sig.chroma)
      .f1('uBarPulse', F.barPulse)
      .tex('uWave', this.sig.waveTex)
      .tex('uSpec', this.sig.specTex)
      .tex('uNote', this.sig.noteTex);
    if (s.sceneT && p !== s.progs.scene) p.tex('uScene', s.sceneT.tex[0]);
    if (s.landT && p !== s.progs.land) p.tex('uLand', s.landT.tex[0]);
  }

  /** Lyrics gene: eases the slot's nudges toward what the words ask for and composes them onto its pose. */
  private lyricize(s: Slot, state: MusicState, sdt: number, q: ChoreoPose): void {
    const g = s.genome.lyrics;
    if (!g) {
      this.lyricNudges.delete(s);
      return;
    }
    let n = this.lyricNudges.get(s);
    if (!n) this.lyricNudges.set(s, (n = { ...NEUTRAL_NUDGE }));
    const F = this.sig.F;
    const words = { tags: state.lyricTags, valence: num(state.lyricValence, 0.5), arousal: num(state.lyricArousal, 0.5), presence: num(state.lyricPresence, 0), pulse: 0 };
    lyricTarget(g, words, paletteHue(s.genome.palette.p, F.keyHue) + q.hue, F.time, this.lyricTgt);
    easeNudge(n, this.lyricTgt, sdt, g.p.lag);
    q.zoom *= n.zoom * lineKick(g, num(state.lyricPulse, 0));
    q.ty -= n.lift; // + lift moves the picture up (the camera looks lower)
    q.roll += n.roll;
    q.hue += n.hue;
    q.sat *= n.sat;
    q.exposure *= n.exposure;
  }

  /** Lyrics gene, chain group: swells the warp ops and carrier settings the genome already has. */
  private lyricDeltas(s: Slot, n: LyricNudge): void {
    const add = (key: string, d: number) => {
      if (Math.abs(d) > 1e-4) s.delta.set(key, (s.delta.get(key) ?? 0) + d);
    };
    s.genome.chain.forEach((o, i) => {
      if (o.op === 'ripple') add(`op${i}.amp`, 0.8 * n.ripple);
      else if (o.op === 'noise') add(`op${i}.amp`, 0.8 * n.noise);
      else if (o.op === 'swirl') add(`op${i}.amt`, (o.p.amt < 0 ? -0.6 : 0.6) * n.swirl);
      else if (o.op === 'twist') add(`op${i}.amt`, (o.p.amt < 0 ? -0.4 : 0.4) * n.swirl);
    });
    if (s.genome.carrier.kind !== 'none' && s.genome.carrier.p.water > 0.001) add('car0.water', 0.6 * n.water);
  }

  /** Harmony gene: runs the slot's motor and composes its pose onto the choreography's. */
  private harmonize(s: Slot, state: MusicState, sdt: number, q: ChoreoPose): void {
    let o = this.harmOut.get(s);
    if (!o) this.harmOut.set(s, (o = { ...IDLE_HARMONY }));
    const h = s.genome.harmony;
    if (!h) {
      Object.assign(o, IDLE_HARMONY);
      return;
    }
    let m = this.motors.get(s);
    if (!m) this.motors.set(s, (m = new HarmonyMotor()));
    const F = this.sig.F;
    const I = this.harmIn;
    I.tension = F.tension;
    I.resolve = F.resolve;
    I.chordPulse = F.chordPulse;
    I.modPulse = F.modPulse;
    I.tonnetzX = num(state.tonnetzX, 0.5);
    I.tonnetzY = num(state.tonnetzY, 0.2887);
    I.keyWalk = num(state.keyWalk, 0);
    m.update(h, I, sdt, o);
    q.zoom *= o.zoom;
    q.roll += o.roll;
    q.hue += o.hue;
    q.sat *= o.sat;
    q.exposure *= o.exposure;
  }

  /** The final pass warp: the slots' warp amounts blended by weight, the heaviest slot's style. */
  private harmonyWarp(slots: Slot[], state: MusicState): void {
    const w = this.harmWarp;
    w[0] = w[1] = w[2] = w[3] = 0;
    let top = 0;
    let wsum = 0;
    for (const s of slots) wsum += s.weight;
    for (const s of slots) {
      const o = this.harmOut.get(s);
      if (!o || !s.genome.harmony) continue;
      w[0] += (o.warp * s.weight) / Math.max(wsum, 1e-3);
      if (s.weight > top) {
        top = s.weight;
        w[1] = o.phase;
        w[2] = s.genome.harmony.p.style;
      }
    }
    // Each chord breaks the folds its own way: the loosening seed follows the chord.
    this.harmSeed = (num(state.chord, -1) + 1) * 1.618;
  }

  /** A fold op's loosening (OB.w) and seed (OB.z) from the slot's harmony motor. */
  private loosen(s: Slot, b: Float32Array, j: number, i: number, stage: string): void {
    const o = s.genome.harmony ? this.harmOut.get(s) : undefined;
    if (!o || o.brk === 0) return;
    b[j + 2] = this.harmSeed + i * 0.73;
    b[j + 3] = o.brk * (stage === 'warp' ? 0.35 : 1);
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
    const exposure = pp.exposure * (0.78 + 0.12 * F.beatPulse * F.act + 0.2 * F.drop + 0.15 * F.build + this.flash * 0.25) * this.pose.exposure;
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
      .f4('uCamM', this.cam[0], this.cam[1], this.cam[2], this.cam[3])
      .f2('uCamT', this.cam[4], this.cam[5])
      .f4('uHarm', this.harmWarp[0], this.harmWarp[1], this.harmWarp[2], this.harmWarp[3])
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
    this.caption?.dispose();
    this.slots = [];
    this.scene?.dispose();
    this.out?.dispose();
    this.bloom.dispose();
    this.avgLum.dispose();
    this.fluid?.dispose();
    this.water?.dispose();
    this.particles?.dispose();
    this.slime?.dispose();
    this.flock?.dispose();
    this.eco?.dispose();
    this.land?.dispose();
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

/** The parameter set a reaction target names (see schemaFor). */
export function paramsFor(g: Genome, group: GeneGroup, i: number): Params | null {
  switch (group) {
    case 'op': return g.chain[i]?.p ?? null;
    case 'car': return g.carrier.p;
    case 'col': return g.tone.p;
    case 'pal': return g.palette.p;
    case 'dr': return g.bodies[Math.floor(i / MAX_DRAW)]?.deform.ops?.[i % MAX_DRAW]?.p ?? null;
  }
  const b = g.bodies[i];
  if (!b) return null;
  switch (group) {
    case 'sh': return b.shape.p;
    case 'pl': return b.place.p;
    case 'mo': return b.motion.p;
    case 'de': return b.deform.p;
    case 'ma': return b.material.p;
    case 'em': return b.emit.p;
    case 'fe': return b.feel.p;
    case 'cm': return b.color.p;
    case 'fu': return b.fuse?.p ?? null;
    case 'fs': return b.fuse?.shape.p ?? null;
  }
  return null;
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
  private pending: { g: Genome; secs: number; home?: Genome; drift?: boolean } | null = null;
  /** The drift performance layer (genes/drift.ts): which genome to render for the home genome. */
  readonly drift = new DriftDriver();
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
    return this.pending?.home ?? this.pending?.g ?? this.to?.home ?? null;
  }

  /** The loaded song's sections, for the drift's plan (null: none, e.g. live input). */
  setSong(sections: readonly Section[] | null): void {
    this.drift.setSong(sections);
  }

  /**
   * Live edit of the genome on screen. A copy is taken. Same structure as what is showing (or already
   * compiling): its parameters are swapped in place, no recompile, no crossfade ('inplace'). A new
   * structure goes through the program cache and crossfades in over `secs` once linked ('compile');
   * if it fails to compile the current picture stays (see failed()).
   */
  edit(g: Genome, secs = 0.3): 'inplace' | 'compile' {
    const copy = cloneGenome(g);
    const key = structuralKey(copy);
    if (this.pending?.drift) this.pending = null;
    if (this.pending && structuralKey(this.pending.g) === key) {
      this.pending.g = copy;
      this.pending.home = copy;
      return 'compile';
    }
    if (this.to && structuralKey(this.to.home) === key) {
      // Same structure as the home genome: swap it in place (a drifting one re-plans next frame).
      this.pending = null;
      this.to.home = copy;
      if (this.to.key === key) this.to.retarget(copy);
      return 'inplace';
    }
    this.show(copy, secs);
    return 'compile';
  }

  /** True while a genome is waiting for its programs. */
  get compiling(): boolean {
    return !!this.pending;
  }

  /**
   * Live reaction readouts of the genome on screen: per reaction its source signal, its response
   * after the curve, and the driven parameter's value this frame (reactions applied).
   */
  reactionMeters(): { src: number; resp: number; value: number }[] {
    const s = this.to;
    if (!s) return [];
    const g = s.genome;
    return g.reactions.slice(0, MAX_REACTIONS).map((r, j) => {
      const schema = schemaFor(g, r.g, r.i);
      const p = paramsFor(g, r.g, r.i);
      const value = schema && p && r.k in schema ? s.P(r.g, r.i, p, r.k, schema) : NaN;
      return { src: s.meters[j * 2], resp: s.meters[j * 2 + 1], value };
    });
  }

  playing(): Genome | null {
    return this.to?.home ?? null;
  }

  failed(g: Genome): string | null {
    return this.cache.failed(g);
  }

  setSongComplexity(cx: number): void {
    this.main.sig.songCx = cx;
  }

  /** The analysed song, for landscape bodies' world map (null: none, e.g. live input). */
  setSongWorld(r: AnalysisResult | null): void {
    (this.main.land ??= new LandWorld(this.gl)).setSong(r);
  }

  /** The lyric caption for genomes that smear it into their feedback (null: no line). */
  setCaption(source: TexImageSource | null, version: number, alpha: number): void {
    if (!source && !this.main.caption) return;
    (this.main.caption ??= new CaptionLayer(this.gl, this.fs)).set(source, version, alpha);
  }

  render(state: MusicState): void {
    const t0 = performance.now();
    if (this.lastNow) this.stats.frameMs += (t0 - this.lastNow - this.stats.frameMs) * 0.05;
    this.lastNow = t0;
    this.pollTimer();
    const q = this.beginTimer();
    const dt = state.dt > 0 ? Math.min(state.dt, 0.1) : 1 / 60;
    this.adapt(dt);

    this.driveDrift(state);
    if (this.pending) {
      const pg = this.cache.get(this.pending.g, !this.to);
      if (pg) this.doSwitch(this.pending.g, pg, this.pending.secs, this.pending.home, this.pending.drift);
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

  /**
   * Drift performance layer: renders the home genome's planned path. A genome of the slot's structure
   * is swapped in place (uniforms); another structure compiles (ahead of the boundary when it can)
   * and crossfades in. User switches waiting to compile take precedence.
   */
  private driveDrift(state: MusicState): void {
    const s = this.to;
    if (!s || (this.pending && !this.pending.drift)) return;
    if (!s.home.drift && s.genome === s.home) return;
    const f = this.drift.frame(s.home, state);
    if (f.upcoming) this.cache.request(f.upcoming);
    if (f.key === s.key) {
      this.pending = null;
      if (s.genome !== f.genome) s.retarget(f.genome, f.key);
      return;
    }
    if (this.pending?.drift && structuralKey(this.pending.g) === f.key) {
      this.pending.g = f.genome;
      return;
    }
    this.pending = { g: f.genome, secs: f.xfade, home: s.home, drift: true };
    this.cache.request(f.genome);
  }

  private doSwitch(g: Genome, progs: GenomePrograms, secs: number, home: Genome = g, drift = false): void {
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
    s.home = home;
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
    if (!drift) this.onSwitched?.(g);
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

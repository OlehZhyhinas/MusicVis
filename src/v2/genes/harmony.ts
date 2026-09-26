// Harmony: a genome-wide gene that makes the picture follow the song's chord progression, read
// from the harmony map (src/analysis/harmony.ts). Same chord, same shape: every chord has a fixed
// deformation derived from where it sits relative to the key (its root's scale degree and whether
// it is major or minor, placed on the Tonnetz). The home chord is the clean picture, with the
// symmetric folds exact; chords further from home on the lattice loosen the fold ops (mirror,
// tile, polar, kaleido) and lean the whole frame more, each chord in its own fixed way. On a chord
// change the picture glides to the new chord's shape in about a third of a second and then holds
// it still for as long as the chord lasts, so a repeating progression repeats its shapes and a
// cadence home is a smooth glide back into the clean picture. Chord changes also walk the palette
// along the lattice and give a small lift of light; modulations swing the hue and roll the world.
//
// Pure except the per-slot HarmonyMotor (glide state). Imports only types from genome.ts so
// genome.ts can import it without a cycle.

import type { ParamSpec, Params, Schema } from '../genome';
import { registerGenomeGene } from '../geneRegistry';
import { tonnetzXY } from '../../analysis/harmony';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * brk: how far the folds loosen on a chord far from home (0 = folds stay exact; nearer chords
 * loosen less, the home chord not at all); warp: strength of the lopsided whole-frame warp on a
 * far chord; style: its shape (0 lean, 1 off-centre swirl, 2 buckle, a wavy asymmetric bend);
 * snap: the lift of light when the progression arrives home; settle: how long the glide to a new
 * chord's shape takes (0.15 = ~0.27 s up to 1.2 = ~0.4 s);
 * walk: palette hue shift per Tonnetz step of the chord from the tonic (chord changes move the
 * colours along the lattice); kick: a small lift of light on every chord change;
 * modHue: hue turn per fifth the key travels on a modulation; modTurn: camera roll per fifth
 * (turns); calm: the home chord mutes the colour and far chords saturate it (0 off).
 */
export const HARMONY_SCHEMA: Schema = {
  brk: P(0, 1, 0.6),
  warp: P(0, 1, 0.35),
  style: C([0, 1, 2], 0),
  snap: P(0, 1, 0.5),
  settle: C([0.15, 0.3, 0.6, 1.2], 0.3),
  walk: P(0, 0.25, 0.06),
  kick: P(0, 1, 0.2),
  modHue: P(0, 0.25, 0.08),
  modTurn: P(-0.02, 0.02, 0.006),
  calm: P(0, 1, 0.3),
};

registerGenomeGene({
  key: 'harmony',
  title: 'Harmony',
  schemas: HARMONY_SCHEMA,
  optional: true,
  glossary: 'same chord, same shape: every chord of the song gets its own fixed deformation from where it sits in the key, the home chord is the clean symmetric picture, and on each chord change the picture glides to that chord\'s shape and holds it still while the chord lasts, so a repeating progression repeats its shapes (brk = how far mirror/tile/polar/kaleido folds slide out of register on chords far from home, warp = a lopsided whole-frame warp on far chords, style 0 lean, 1 off-centre swirl, 2 buckle; settle = glide time, 0.15 about a quarter second up to 1.2 about 0.4 s); snap = lift of light when the progression comes home, kick = small lift of light on each chord change; walk = palette hue shift as chords move away from home; modulations (key changes) swing the hue (modHue per fifth) and roll the world (modTurn, turns per fifth); calm = the home chord mutes the colour, far chords saturate it',
  order: 2,
});

export interface HarmonyGene {
  p: Params;
}

/** Estimated GPU cost: a few ALU ops per folded pixel and one warp in the final pass. */
export const HARMONY_COST_MS = 0.04;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object';

function clampSpec(v: unknown, s: ParamSpec): number {
  let x = typeof v === 'number' && Number.isFinite(v) ? v : s.def;
  if (s.choices) {
    let best = s.choices[0];
    for (const c of s.choices) if (Math.abs(c - x) < Math.abs(best - x)) best = c;
    return best;
  }
  x = clamp(x, s.min, s.max);
  return s.int ? Math.round(x) : x;
}

/** A valid harmony gene from anything (missing or broken values take the defaults). */
export function repairHarmony(raw: unknown): HarmonyGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(HARMONY_SCHEMA)) p[k] = clampSpec(src[k], HARMONY_SCHEMA[k]);
  return { p };
}

/** Rule violations of a harmony gene (empty when valid). */
export function validateHarmony(h: HarmonyGene): string[] {
  const errs: string[] = [];
  if (!isObj(h) || !isObj(h.p)) return ['harmony params'];
  for (const k of Object.keys(HARMONY_SCHEMA)) {
    const s = HARMONY_SCHEMA[k];
    const v = h.p[k];
    const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min - 1e-9 && v <= s.max + 1e-9);
    if (!ok) errs.push(`harmony.${k}`);
  }
  for (const k of Object.keys(h.p)) if (!(k in HARMONY_SCHEMA)) errs.push(`harmony.${k} unknown`);
  return errs;
}

// ------------------------------------------------------------ breeding

type Rng = () => number;

function gauss(rng: Rng): number {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function randomValue(s: ParamSpec, rng: Rng): number {
  if (s.choices) return s.choices[Math.floor(rng() * s.choices.length)];
  return clampSpec(s.min + (s.max - s.min) * rng(), s);
}

/** A new random harmony gene (most settings near their defaults, a few pushed further). */
export function randomHarmony(rng: Rng): HarmonyGene {
  const p: Params = {};
  for (const k of Object.keys(HARMONY_SCHEMA)) {
    const s = HARMONY_SCHEMA[k];
    p[k] = rng() < 0.6 ? randomValue(s, rng) : s.def;
  }
  return { p };
}

/** Nudges some settings in place. */
export function jitterHarmony(h: HarmonyGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(HARMONY_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.35) continue;
    touched = true;
    const s = HARMONY_SCHEMA[k];
    h.p[k] = s.choices ? (rng() < 0.5 ? randomValue(s, rng) : h.p[k]) : clampSpec(h.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    h.p[k] = randomValue(HARMONY_SCHEMA[k], rng);
  }
}

/** Probability a child inherits the gene when only one parent has it. */
export const HARMONY_CARRY = 0.5;

/**
 * Crossover. Draws from the rng only when a parent has the gene, so children of parents without
 * it come out exactly as before the gene existed.
 */
export function crossHarmony(d: HarmonyGene | undefined, r: HarmonyGene | undefined, rng: Rng): HarmonyGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) return rng() < HARMONY_CARRY ? repairHarmony((d ?? r)!) : undefined;
  const p: Params = {};
  for (const k of Object.keys(HARMONY_SCHEMA)) {
    const s = HARMONY_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    const t = rng();
    p[k] = s.choices ? (t < 0.5 ? x : y) : x + (y - x) * t;
  }
  return repairHarmony({ p });
}

// --------------------------------------------------------------- motor

/** What the harmony map says this frame (from MusicState via the engine's Frame). */
export interface HarmonyInputs {
  /** Current chord: 0..11 major triad on that root, 12..23 minor, -1 unknown (holds the shape). */
  chord: number;
  /** Key tonic pitch class and mode (the home chord is the tonic triad of this mode). */
  keyTonic: number;
  minor: boolean;
  /** Signed fifths travelled by modulations. */
  keyWalk: number;
  /** Tempo, for the dwell a new chord must hold before the shape follows (half a beat). */
  bpm: number;
}

/** The harmony gene's effect this frame. */
export interface HarmonyOut {
  /** Fold loosening 0..1 (the chord's fixed shape; 0 on the home chord). */
  brk: number;
  /** The fold ops' loosening seed (fixed per chord relative to the key). */
  seed: number;
  /** Whole-frame lopsided warp amount 0..1. */
  warp: number;
  /** Warp pattern phase (radians, fixed per chord: its direction from home on the Tonnetz). */
  phase: number;
  /** Pose contributions (composed onto the choreography pose). */
  zoom: number;
  roll: number;
  hue: number;
  sat: number;
  exposure: number;
}

const TAU = Math.PI * 2;
const XY: [number, number] = [0, 0];
const mod12 = (x: number) => ((x % 12) + 12) % 12;
const smooth = (x: number) => x * x * (3 - 2 * x);

/** A chord's fixed shape relative to a key (pure: same chord in the same key, same shape). */
export interface ChordShape {
  /** Shape id: scale degree * 2 + minor (0 = the tonic major triad). */
  id: number;
  /** True on the home chord (the tonic triad of the key's mode). */
  home: boolean;
  /** Deformation amount 0..1, growing with the triad's Tonnetz distance from the home triad. */
  amt: number;
  /** Warp phase: the chord's direction from home on the lattice (radians). */
  phase: number;
  /** Fold seed. */
  seed: number;
  /** Tonnetz offset from the home triad (x along fifths, y up the thirds), for the palette walk. */
  dx: number;
  dy: number;
}

/** The fixed shape of a chord (0..23) in a key. */
export function chordShape(chord: number, tonic: number, minor: boolean): ChordShape {
  const deg = mod12((chord % 12) - tonic);
  const isMin = chord >= 12;
  const id = deg * 2 + (isMin ? 1 : 0);
  const home = deg === 0 && isMin === minor;
  tonnetzXY(chord, tonic, XY);
  const x = XY[0];
  const y = XY[1];
  tonnetzXY(tonic + (minor ? 12 : 0), tonic, XY);
  const dx = x - XY[0];
  const dy = y - XY[1];
  const dist = Math.hypot(dx, dy);
  return {
    id,
    home,
    amt: home ? 0 : clamp(0.3 + 0.35 * dist, 0, 1),
    phase: home ? 0 : Math.atan2(dy, dx),
    seed: (id + 1) * 1.618,
    dx: home ? 0 : dx,
    dy: home ? 0 : dy,
  };
}

/** Glide time to a new chord's shape: ~0.27 s (settle 0.15) up to 0.4 s (settle 1.2). */
export const glideTime = (settle: number): number => 0.25 + 0.125 * clamp(settle, 0, 1.2);

/** An eased (smoothstep) move from one value to another over a fixed time, still at both ends. */
class Glide {
  v = 0;
  private from = 0;
  private to = 0;
  private t = 1;
  private T = 1;

  set(to: number, T: number): void {
    if (to === this.to) return;
    this.from = this.v;
    this.to = to;
    this.t = 0;
    this.T = Math.max(1e-3, T);
  }

  jump(v: number): void {
    this.v = this.from = this.to = v;
    this.t = this.T;
  }

  get target(): number {
    return this.to;
  }

  get moving(): boolean {
    return this.t < this.T;
  }

  step(dt: number): number {
    if (this.t < this.T) {
      this.t = Math.min(this.T, this.t + dt);
      this.v = this.from + (this.to - this.from) * smooth(this.t / this.T);
    }
    return this.v;
  }
}

/**
 * Per-slot state. The picture takes each chord's fixed shape. Once a new chord has held for half a
 * beat (so a flickering live reading never flicks the shape), the warp amount, warp phase and
 * palette walk glide to it over glideTime(settle) and then hold still for the whole chord. The fold
 * seed cannot glide (tile and kaleido hash it), so the folds ease out to the exact fold over the
 * first half of the glide, switch seed there where nothing shows, and ease into the new chord's
 * shape over the second half. Unknown chords hold the current shape.
 */
export class HarmonyMotor {
  private amt = new Glide();
  private ph = new Glide();
  private walkX = new Glide();
  private walkY = new Glide();
  private fold = new Glide();
  private seed = 0;
  private wantSeed = 0;
  private wantFold = 0;
  private foldT = 0.3;
  private shape = -1; // accepted shape id (-1: none yet, the clean picture)
  private chord = -1; // accepted chord
  private key = -1; // accepted key (tonic + 12 * minor)
  private cand = -1; // candidate chord + 24 * key, waiting out the dwell
  private candFor = 0;
  private kick = 0;
  private kickEnv = 0;
  private snap = 0;
  private snapEnv = 0;
  private modHue = 0;
  private roll = 0;

  reset(): void {
    for (const g of [this.amt, this.ph, this.walkX, this.walkY, this.fold]) g.jump(0);
    this.seed = this.wantSeed = this.wantFold = 0;
    this.shape = this.chord = this.key = this.cand = -1;
    this.candFor = this.kick = this.kickEnv = this.snap = this.snapEnv = this.modHue = this.roll = 0;
  }

  /** The chord whose shape is showing (-1 before the first). */
  get heldChord(): number {
    return this.chord;
  }

  private accept(p: Params, c: number, tonic: number, minor: boolean): void {
    const s = chordShape(c, tonic, minor);
    const T = glideTime(p.settle);
    const wasHome = this.shape < 0 || this.amt.target === 0;
    if (c !== this.chord && this.chord >= 0) this.kick = 1;
    if (s.home && !wasHome) this.snap = 1;
    this.chord = c;
    this.key = tonic + (minor ? 12 : 0);
    this.shape = s.id;
    // Warp: from the clean picture the phase takes its new value at once (nothing shows); otherwise
    // it glides straight to the chord's own value (no wrapping, so a chord's phase is always the
    // same number: the swirl style reads it at a non-2pi period).
    if (this.amt.v < 1e-4) this.ph.jump(s.phase);
    else if (!s.home) this.ph.set(s.phase, T);
    this.amt.set(s.amt, T);
    this.walkX.set(s.dx, T);
    this.walkY.set(s.dy, T);
    // Folds: out to the exact fold, switch seed, into the new shape.
    this.wantSeed = s.seed;
    this.wantFold = s.amt;
    this.foldT = T;
    if (this.fold.v < 1e-4 && !this.fold.moving) {
      this.seed = s.seed;
      this.fold.set(s.amt, T);
    } else if (s.home || s.seed === this.seed) this.fold.set(s.amt, T);
    else this.fold.set(0, T * 0.5);
  }

  update(h: HarmonyGene | undefined, inp: HarmonyInputs, dt: number, out: HarmonyOut): HarmonyOut {
    out.brk = out.warp = out.seed = out.phase = 0;
    out.zoom = out.sat = out.exposure = 1;
    out.roll = out.hue = 0;
    if (!h) return out;
    const p = h.p;
    dt = clamp(Number.isFinite(dt) ? dt : 0, 0, 0.1);
    const c = Number.isFinite(inp.chord) ? Math.round(inp.chord) : -1;
    const tonic = mod12(Math.round(Number.isFinite(inp.keyTonic) ? inp.keyTonic : 0));
    const minor = !!inp.minor;
    const key = tonic + (minor ? 12 : 0);
    // A new chord (or the same chord heard in a new key) must hold for half a beat before the
    // shape follows; unknown chords (-1) keep whatever shape is showing.
    if (c >= 0 && c < 24) {
      const k = c + 24 * key;
      if (c === this.chord && key === this.key) this.cand = -1;
      else if (k !== this.cand) {
        this.cand = k;
        this.candFor = 0;
      } else this.candFor += dt;
      const bpm = Number.isFinite(inp.bpm) && inp.bpm > 30 ? inp.bpm : 120;
      const dwell = clamp(30 / bpm, 0.12, 0.35);
      if (this.cand >= 0 && (this.shape < 0 || this.candFor + 1e-9 >= dwell)) {
        this.accept(p, c, tonic, minor);
        this.cand = -1;
      }
    }
    this.amt.step(dt);
    this.ph.step(dt);
    this.walkX.step(dt);
    this.walkY.step(dt);
    this.fold.step(dt);
    // The folds reached the exact picture on their way out: take the new seed and ease in.
    if (!this.fold.moving && this.fold.v === 0 && this.seed !== this.wantSeed) {
      this.seed = this.wantSeed;
      if (this.wantFold > 0) this.fold.set(this.wantFold, this.foldT * 0.5);
    }
    // Light accents only (no geometry): a small lift on each chord change, a larger one arriving
    // home. Eased attacks, so the light never jumps in one frame.
    this.kick *= Math.exp(-dt / 0.35);
    this.kickEnv += (this.kick - this.kickEnv) * (1 - Math.exp(-dt / 0.05));
    this.snap *= Math.exp(-dt / Math.max(0.3, p.settle));
    this.snapEnv += (this.snap - this.snapEnv) * (1 - Math.exp(-dt / 0.06));
    // Modulations: the hue and roll swing by modHue / modTurn per fifth travelled, eased over ~1.5 s.
    const walk = Number.isFinite(inp.keyWalk) ? inp.keyWalk : 0;
    const e = 1 - Math.exp(-dt / 1.5);
    this.roll += (clamp(p.modTurn * walk, -0.03, 0.03) * TAU - this.roll) * e;
    this.modHue += (p.modHue * walk - this.modHue) * e;

    out.brk = p.brk * this.fold.v;
    out.seed = this.seed;
    out.warp = p.warp * this.amt.v;
    out.phase = this.ph.v;
    out.exposure = 1 + p.kick * 0.1 * this.kickEnv + p.snap * 0.3 * this.snapEnv;
    out.sat = clamp(1 + p.calm * (this.amt.v - 0.3) * 0.9, 0.5, 1.5);
    // The palette walk is relative to the home triad, so home is the palette as bred.
    out.hue = p.walk * (this.walkX.v + 0.6 * this.walkY.v) + this.modHue;
    out.roll = this.roll;
    return out;
  }
}

export const IDLE_HARMONY: Readonly<HarmonyOut> = { brk: 0, seed: 0, warp: 0, phase: 0, zoom: 1, roll: 0, hue: 0, sat: 1, exposure: 1 };

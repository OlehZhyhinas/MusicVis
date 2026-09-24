// Harmony: a genome-wide gene that makes the picture follow the song's chord progression, read
// from the harmony map (src/analysis/harmony.ts). Consonance is order: symmetric folds hold, the
// frame is still. Rising harmonic tension breaks the symmetry: the fold ops (mirror, tile, polar,
// kaleido) loosen so their halves and segments slide out of register, and a lopsided warp grows
// over the whole frame. A resolution to the tonic snaps everything back into order with a springy
// re-alignment and a flash of light. Chord changes walk the palette along the Tonnetz (a fifth up
// is one hue step, a third another), and modulations turn the whole world's hue and orientation.
//
// Pure except the per-slot HarmonyMotor (spring state). Imports only types from genome.ts so
// genome.ts can import it without a cycle.

import type { ParamSpec, Params, Schema } from '../genome';
import { registerGenomeGene } from '../geneRegistry';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * brk: how far full tension loosens the fold ops (0 = folds stay locked);
 * warp: strength of the lopsided whole-frame warp at full tension; style: its shape (0 lean,
 * 1 off-centre swirl, 2 buckle, a wavy asymmetric bend);
 * snap: the resolution moment (a springy re-alignment, a zoom kick and a flash); settle: seconds
 * the snap-back takes (short = a click into place, long = a wobbling settle);
 * walk: palette hue shift per Tonnetz step of the chord from the tonic (chord changes move the
 * colours along the lattice); kick: a small zoom pulse on every chord change;
 * modHue: hue turn per fifth the key travels on a modulation; modTurn: camera roll per fifth
 * (turns); calm: consonance mutes and tension saturates (0 off).
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
  glossary: 'follows the chord progression on a tonal lattice: consonant harmony keeps the picture symmetric and calm, rising harmonic tension breaks the symmetry (brk = how far mirror/tile/polar/kaleido folds slide out of register, warp = a lopsided whole-frame warp, style 0 lean, 1 off-centre swirl, 2 buckle), and a resolution to the home chord snaps everything back into order (snap = strength of the re-alignment kick and flash, settle = seconds to click back, short = a click, long = a wobble); walk = palette hue shift as chords move away from home, kick = zoom pulse on each chord change; modulations (key changes) turn the hue (modHue per fifth) and roll the world (modTurn, turns per fifth); calm = consonance mutes the colour, tension saturates it',
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
  tension: number; // 0..1, smoothed
  resolve: number; // resolve pulse (strength, decaying)
  chordPulse: number;
  modPulse: number;
  tonnetzX: number;
  tonnetzY: number;
  keyWalk: number; // signed fifths travelled by modulations
}

/** The harmony gene's effect this frame. */
export interface HarmonyOut {
  /** Fold loosening, about -0.3..1 (negative only briefly: the overshoot of the snap-back). */
  brk: number;
  /** Whole-frame lopsided warp amount 0..1. */
  warp: number;
  /** Warp pattern phase (radians, advances slowly). */
  phase: number;
  /** Pose contributions (composed onto the choreography pose). */
  zoom: number;
  roll: number;
  hue: number;
  sat: number;
  exposure: number;
}

const TAU = Math.PI * 2;
/** Loosening at a tension: nothing while consonant, full by strong tension. */
export const tensionShape = (t: number): number => {
  const x = clamp((t - 0.12) / 0.6, 0, 1);
  return x * x * (3 - 2 * x);
};

/**
 * Per-slot spring state. Normally the fold loosening follows the tension slowly (tension creeps
 * in); a resolution switches it to a stiff, underdamped spring for a moment, so the picture snaps
 * back into register with a small overshoot, and fires the snap envelope (zoom kick and flash).
 */
export class HarmonyMotor {
  brk = 0;
  vel = 0;
  phase = 0;
  snapEnv = 0;
  snapFor = 0;
  roll = 0;
  private lastResolve = 0;

  reset(): void {
    this.brk = this.vel = this.snapEnv = this.snapFor = this.roll = this.lastResolve = 0;
  }

  update(h: HarmonyGene | undefined, inp: HarmonyInputs, dt: number, out: HarmonyOut): HarmonyOut {
    out.brk = out.warp = 0;
    out.zoom = out.sat = out.exposure = 1;
    out.roll = out.hue = 0;
    out.phase = this.phase;
    if (!h) return out;
    const p = h.p;
    dt = clamp(Number.isFinite(dt) ? dt : 0, 0, 0.1);
    const t = clamp(inp.tension, 0, 1);
    const target = tensionShape(t);
    // A resolution: the pulse jumps up.
    if (inp.resolve > this.lastResolve + 0.15) {
      this.snapEnv = Math.max(this.snapEnv, clamp(inp.resolve, 0, 1));
      this.snapFor = p.settle * 2;
    }
    this.lastResolve = inp.resolve;
    if (this.snapFor > 0) {
      // Underdamped spring toward order (0), period ~ settle.
      const w = TAU / Math.max(0.1, p.settle);
      const zeta = 0.35;
      const acc = -w * w * this.brk - 2 * zeta * w * this.vel;
      // Integrate in small steps for stability at large dt.
      const n = Math.max(1, Math.ceil(dt / 0.01));
      const h2 = dt / n;
      for (let i = 0; i < n; i++) {
        const a = i === 0 ? acc : -w * w * this.brk - 2 * zeta * w * this.vel;
        this.vel += a * h2;
        this.brk += this.vel * h2;
      }
      this.snapFor -= dt;
      if (this.snapFor <= 0) this.vel = 0;
    } else {
      // Tension creeps in (~0.8 s), eases out a little faster.
      const tau = target > this.brk ? 0.8 : 0.5;
      this.brk += (target - this.brk) * (1 - Math.exp(-dt / tau));
      this.vel = 0;
    }
    this.snapEnv *= Math.exp(-dt / Math.max(0.1, p.settle));
    this.phase = (this.phase + dt * (0.3 + 0.9 * t)) % 4096;
    // Modulations: the world turns by modTurn per fifth travelled (eased over ~1.5 s).
    const rollTarget = clamp(p.modTurn * inp.keyWalk, -0.03, 0.03) * TAU;
    this.roll += (rollTarget - this.roll) * (1 - Math.exp(-dt / 1.5));

    out.brk = p.brk * this.brk;
    out.warp = p.warp * Math.max(0, this.brk);
    out.phase = this.phase;
    out.zoom = 1 + p.snap * 0.08 * this.snapEnv + p.kick * 0.03 * clamp(inp.chordPulse, 0, 1);
    out.exposure = 1 + p.snap * 0.45 * this.snapEnv;
    out.sat = clamp(1 + p.calm * (t - 0.3) * 0.9, 0.5, 1.5);
    // Relative to the tonic triad's centre, so home is the palette as bred.
    out.hue = p.walk * (inp.tonnetzX - 0.5 + 0.6 * (inp.tonnetzY - 0.2887)) + p.modHue * inp.keyWalk;
    out.roll = this.roll;
    return out;
  }
}

export const IDLE_HARMONY: Readonly<HarmonyOut> = { brk: 0, warp: 0, phase: 0, zoom: 1, roll: 0, hue: 0, sat: 1, exposure: 1 };

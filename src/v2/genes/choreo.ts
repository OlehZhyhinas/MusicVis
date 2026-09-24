// Choreography: a genome-wide gene that composes the picture over the song's timeline instead of
// only reacting frame by frame. Songs are analysed before playback, so the renderer knows how far
// away the next drop is; the gene uses that look-ahead to build anticipation over the last bars
// before a drop and to release it on the drop itself.
//
// Everything here is pure (no GL, no module state): the pose is a function of the gene and the
// music's timeline position, so seeking lands on the right pose and the tests can check it.
// This file imports only types from genome.ts so genome.ts can import it without a cycle.

import type { ParamSpec, Params, Schema } from '../genome';
import type { MusicState } from '../../types';
import { registerGenomeGene } from '../geneRegistry';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * Anticipation of a known drop:
 * lead: bars of build-up before the drop; curve: how late the build-up bites (1 linear, 4 a late rush);
 * push: camera push-in at the peak (fraction of the frame); roll: camera roll at the peak (turns);
 * drain: desaturation at the peak; dim: exposure dip at the peak.
 * Release on the drop:
 * punch: the slam on the drop (a zoom kick, a saturation and exposure flash); relax: bars it takes to settle.
 */
export const CHOREO_SCHEMA: Schema = {
  lead: C([2, 4, 8, 16], 8),
  curve: P(0.5, 4, 2),
  push: P(0, 0.3, 0.12),
  roll: P(-0.02, 0.02, 0),
  drain: P(0, 1, 0.5),
  dim: P(0, 0.6, 0.25),
  punch: P(0, 1, 0.6),
  relax: C([0.5, 1, 2, 4], 1),
};

registerGenomeGene({
  key: 'choreo',
  title: 'Choreography',
  schemas: CHOREO_SCHEMA,
  optional: true,
  glossary: 'composes the picture over the song from its known future: over the last lead bars before each drop the camera pushes in (push) and leans (roll, turns), colour drains (drain) and light dims (dim), rising late when curve is high; on the drop it snaps back with a slam of zoom, colour and light (punch) that settles over relax bars',
});

export interface ChoreoGene {
  p: Params;
}

/** Estimated GPU cost: a few uniforms and one 2x2 transform in the final pass. */
export const CHOREO_COST_MS = 0.02;

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

/** A valid choreography gene from anything (missing or broken values take the defaults). */
export function repairChoreo(raw: unknown): ChoreoGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(CHOREO_SCHEMA)) p[k] = clampSpec(src[k], CHOREO_SCHEMA[k]);
  return { p };
}

/** Rule violations of a choreography gene (empty when valid). */
export function validateChoreo(c: ChoreoGene): string[] {
  const errs: string[] = [];
  if (!isObj(c) || !isObj(c.p)) return ['choreo params'];
  for (const k of Object.keys(CHOREO_SCHEMA)) {
    const s = CHOREO_SCHEMA[k];
    const v = c.p[k];
    const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min - 1e-9 && v <= s.max + 1e-9);
    if (!ok) errs.push(`choreo.${k}`);
  }
  for (const k of Object.keys(c.p)) if (!(k in CHOREO_SCHEMA)) errs.push(`choreo.${k} unknown`);
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

/** A new random choreography (most settings near their defaults, a few pushed further). */
export function randomChoreo(rng: Rng): ChoreoGene {
  const p: Params = {};
  for (const k of Object.keys(CHOREO_SCHEMA)) {
    const s = CHOREO_SCHEMA[k];
    p[k] = rng() < 0.6 ? randomValue(s, rng) : s.def;
  }
  return { p };
}

/** Nudges some settings of a choreography in place. */
export function jitterChoreo(c: ChoreoGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(CHOREO_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.35) continue;
    touched = true;
    const s = CHOREO_SCHEMA[k];
    c.p[k] = s.choices ? (rng() < 0.5 ? randomValue(s, rng) : c.p[k]) : clampSpec(c.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    c.p[k] = randomValue(CHOREO_SCHEMA[k], rng);
  }
}

/** Probability a child inherits the choreography when only one parent has one. */
export const CHOREO_CARRY = 0.5;

/**
 * Crossover of the choreography. Draws from the rng only when a parent has one, so children of
 * parents without choreography come out exactly as before the gene existed.
 */
export function crossChoreo(d: ChoreoGene | undefined, r: ChoreoGene | undefined, rng: Rng): ChoreoGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) {
    const only = (d ?? r)!;
    return rng() < CHOREO_CARRY ? repairChoreo(only) : undefined;
  }
  const p: Params = {};
  for (const k of Object.keys(CHOREO_SCHEMA)) {
    const s = CHOREO_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    const t = rng();
    p[k] = s.choices ? (t < 0.5 ? x : y) : x + (y - x) * t;
  }
  return repairChoreo({ p });
}

// --------------------------------------------------------------- pose

/** The timeline position the choreography needs (look-ahead from the offline analysis). */
export interface ChoreoCue {
  /** Seconds until the next drop starts (Infinity when none is coming or unknown). */
  timeToDrop: number;
  /** Seconds since the last drop started (Infinity when none yet or unknown). */
  sinceDrop: number;
  /** Seconds per bar. */
  barSeconds: number;
}

/** What the choreography does to the picture this frame (identity: zoom 1, roll 0, sat 1, exposure 1). */
export interface ChoreoPose {
  zoom: number; // >= 1, camera push-in
  roll: number; // radians
  tx: number; // camera pan, fraction of the frame
  ty: number;
  sat: number; // saturation multiplier
  exposure: number; // exposure multiplier
}

export const IDENTITY_POSE: Readonly<ChoreoPose> = { zoom: 1, roll: 0, tx: 0, ty: 0, sat: 1, exposure: 1 };

/** The cue a MusicState carries (sampled songs only; live input has no look-ahead). */
export function cueOf(state: MusicState): ChoreoCue {
  const bpm = state.bpm > 0 && Number.isFinite(state.bpm) ? state.bpm : 120;
  const bar = state.barSeconds && state.barSeconds > 0 ? state.barSeconds : 240 / bpm;
  const ttd = state.timeToDrop;
  const sd = state.sinceDrop;
  return {
    timeToDrop: typeof ttd === 'number' && ttd >= 0 ? ttd : Infinity,
    sinceDrop: typeof sd === 'number' && sd >= 0 ? sd : Infinity,
    barSeconds: bar,
  };
}

/** Anticipation ramp 0..1 over the last `lead` bars before the drop. */
export function buildRamp(c: ChoreoGene, cue: ChoreoCue): number {
  const bars = cue.timeToDrop / cue.barSeconds;
  if (!(bars < c.p.lead)) return 0;
  return Math.pow(clamp(1 - bars / c.p.lead, 0, 1), c.p.curve);
}

/** Release envelope 1..0 over the `relax` bars after the drop. */
export function releaseEnv(c: ChoreoGene, cue: ChoreoCue): number {
  const bars = cue.sinceDrop / cue.barSeconds;
  if (!(bars < c.p.relax)) return 0;
  const x = 1 - bars / c.p.relax;
  return x * x;
}

/** The pose of a choreography at a timeline position (identity without a gene). */
export function choreoPose(c: ChoreoGene | undefined, cue: ChoreoCue, out: ChoreoPose = { ...IDENTITY_POSE }): ChoreoPose {
  Object.assign(out, IDENTITY_POSE);
  if (!c) return out;
  const p = c.p;
  // Tension: the camera creeps in and rolls, colour drains and the light dims toward the drop.
  const ramp = buildRamp(c, cue);
  // Release: on the drop the tension snaps back, and the punch slams in and settles over `relax` bars.
  const env = releaseEnv(c, cue);
  out.zoom = 1 + p.push * ramp + PUNCH_ZOOM * p.punch * env;
  out.roll = p.roll * TAU * ramp;
  out.sat = (1 - 0.85 * p.drain * ramp) * (1 + PUNCH_SAT * p.punch * env);
  out.exposure = (1 - p.dim * ramp) * (1 + PUNCH_EXPOSURE * p.punch * env);
  return out;
}

const TAU = Math.PI * 2;
const PUNCH_ZOOM = 0.15;
const PUNCH_SAT = 0.35;
const PUNCH_EXPOSURE = 0.4;

/** Weighted blend of poses (slots crossfading); weights need not sum to 1. */
export function blendPoses(poses: readonly ChoreoPose[], weights: readonly number[], out: ChoreoPose = { ...IDENTITY_POSE }): ChoreoPose {
  let wsum = 0;
  for (const w of weights) wsum += w;
  Object.assign(out, IDENTITY_POSE);
  if (wsum <= 1e-6) return out;
  out.zoom = out.roll = out.tx = out.ty = out.sat = out.exposure = 0;
  poses.forEach((q, i) => {
    const w = weights[i] / wsum;
    out.zoom += q.zoom * w;
    out.roll += q.roll * w;
    out.tx += q.tx * w;
    out.ty += q.ty * w;
    out.sat += q.sat * w;
    out.exposure += q.exposure * w;
  });
  return out;
}

/**
 * The 2D camera of a pose as the final pass uses it: screen uv -> scene uv is
 * uv' = 0.5 + M (uv - 0.5) + t. Written as M minus the identity (out[0..3], row-major) and t
 * (out[4..5]), so zero uniforms are the identity. The zoom is raised as far as a roll needs to keep
 * the frame covered, and the pan is clamped so no screen corner samples outside the scene.
 */
export function cameraUniforms(pose: ChoreoPose, aspect: number, out: Float32Array = new Float32Array(6)): Float32Array {
  const a = aspect > 0 && Number.isFinite(aspect) ? aspect : 1;
  const c = Math.cos(pose.roll);
  const sn = Math.sin(pose.roll);
  const cover = Math.abs(c) + Math.abs(sn) * Math.max(a, 1 / a);
  const s = Math.max(1, pose.zoom, sn === 0 ? 1 : cover * (1 + 1e-6));
  const m00 = c / s, m01 = -sn / a / s, m10 = (a * sn) / s, m11 = c / s;
  const ex = 0.5 * (Math.abs(m00) + Math.abs(m01));
  const ey = 0.5 * (Math.abs(m10) + Math.abs(m11));
  const lx = Math.max(0, 0.5 - ex - 1e-6);
  const ly = Math.max(0, 0.5 - ey - 1e-6);
  out[0] = m00 - 1;
  out[1] = m01;
  out[2] = m10;
  out[3] = m11 - 1;
  out[4] = clamp(pose.tx, -lx, lx);
  out[5] = clamp(pose.ty, -ly, ly);
  return out;
}

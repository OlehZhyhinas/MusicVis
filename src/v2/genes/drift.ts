// Drift: a genome-wide gene that lets the preset travel through gene space as the song unfolds.
// Instead of one fixed look, every section plays a small mutation of the section before it; a
// section type that comes back (the second chorus) returns toward the genome it had the first time,
// the breakdown wanders furthest, drops may jump, and the outro heads home.
//
// The whole path is planned offline from the song's analysis (genes/driftPath.ts), deterministic
// from the song, the genome and the drift seed, so seeking always lands on the same picture. The
// genome that carries the gene is the "home" genome: it is what is saved, voted and edited; the
// drift is a performance layer on top of it.
//
// This file imports only types from genome.ts so genome.ts can import it without a cycle.

import type { ParamSpec, Params, Schema } from '../genome';
import { registerGenomeGene } from '../geneRegistry';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * step: mutation strength per section (how far each section moves from the one before);
 * kinds: what may change beyond continuous parameters: 0 parameters only, 1 kinds too but only on
 *   drops (a jump), 2 kinds on any section boundary;
 * what: which gene groups drift: 0 all, 1 form (bodies, chain), 2 colour (palette, tone), 3 motion
 *   (chain, carrier, body motion and deform);
 * ret: how strongly a returning section type goes back to the genome it had before (1 = exactly);
 * morph: bars a section takes to morph from the previous section's genome (0 = a cut);
 * bound: the farthest the path may wander from the home genome (0..1 gene distance);
 * seed: which path (another seed, another journey through the same song).
 */
export const DRIFT_SCHEMA: Schema = {
  step: P(0.05, 1, 0.35),
  kinds: C([0, 1, 2], 0),
  what: C([0, 1, 2, 3], 0),
  ret: P(0, 1, 0.75),
  morph: C([0, 1, 2, 4, 8], 2),
  bound: P(0.02, 0.5, 0.15),
  seed: P(0, 1, 0.5),
};

registerGenomeGene({
  key: 'drift',
  title: 'Drift',
  schemas: DRIFT_SCHEMA,
  optional: true,
  order: 10,
  glossary: 'the preset travels through gene space over the song: each section is a small mutation of the last (step), morphing over morph bars; kinds 0 params only, 1 shapes may change on drops, 2 on any section; what 0 all, 1 form, 2 colour, 3 motion; a returning section type goes back to its earlier look (ret); bound=max distance from the saved preset; seed=which journey',
});

export interface DriftGene {
  p: Params;
}

/** Estimated GPU cost of the gene itself (per-frame parameter interpolation is CPU side). */
export const DRIFT_COST_MS = 0.02;
/**
 * Cost headroom a path genome may take over the home genome (fraction). The planner rejects any
 * section genome costing more than driftCost(home), which is therefore the worst case over the path.
 */
export const DRIFT_HEADROOM = 0.15;
/** At least this much headroom (ms), so cheap presets can still drift a little dearer. */
export const DRIFT_HEADROOM_MS = 0.3;

/** The most a path genome of a home genome costing `homeMs` may cost (planner limit, charged cost). */
export function driftCost(homeMs: number): number {
  return homeMs + Math.max(homeMs * DRIFT_HEADROOM, DRIFT_HEADROOM_MS);
}

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

/** A valid drift gene from anything (missing or broken values take the defaults). */
export function repairDrift(raw: unknown): DriftGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(DRIFT_SCHEMA)) p[k] = clampSpec(src[k], DRIFT_SCHEMA[k]);
  return { p };
}

/** Rule violations of a drift gene (empty when valid). */
export function validateDrift(d: DriftGene): string[] {
  const errs: string[] = [];
  if (!isObj(d) || !isObj(d.p)) return ['drift params'];
  for (const k of Object.keys(DRIFT_SCHEMA)) {
    const s = DRIFT_SCHEMA[k];
    const v = d.p[k];
    const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min - 1e-9 && v <= s.max + 1e-9);
    if (!ok) errs.push(`drift.${k}`);
  }
  for (const k of Object.keys(d.p)) if (!(k in DRIFT_SCHEMA)) errs.push(`drift.${k} unknown`);
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

/** A new random drift (gentle: parameters only most of the time, a modest step). */
export function randomDrift(rng: Rng): DriftGene {
  const p: Params = {};
  for (const k of Object.keys(DRIFT_SCHEMA)) p[k] = rng() < 0.5 ? randomValue(DRIFT_SCHEMA[k], rng) : DRIFT_SCHEMA[k].def;
  if (rng() < 0.7) p.kinds = 0;
  return repairDrift({ p });
}

/** Nudges some settings of a drift in place (the seed changes the whole journey, so it moves rarely). */
export function jitterDrift(d: DriftGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(DRIFT_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.35) continue;
    touched = true;
    const s = DRIFT_SCHEMA[k];
    d.p[k] = s.choices ? (rng() < 0.5 ? randomValue(s, rng) : d.p[k]) : clampSpec(d.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    d.p[k] = randomValue(DRIFT_SCHEMA[k], rng);
  }
}

/** Probability a child inherits the drift when only one parent has one. */
export const DRIFT_CARRY = 0.5;

/**
 * Crossover of the drift. Draws from the rng only when a parent has one, so children of parents
 * without a drift come out exactly as before the gene existed.
 */
export function crossDrift(d: DriftGene | undefined, r: DriftGene | undefined, rng: Rng): DriftGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) return rng() < DRIFT_CARRY ? repairDrift(d ?? r) : undefined;
  const p: Params = {};
  for (const k of Object.keys(DRIFT_SCHEMA)) {
    const s = DRIFT_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    const t = rng();
    p[k] = s.choices || k === 'seed' ? (t < 0.5 ? x : y) : x + (y - x) * t;
  }
  return repairDrift({ p });
}

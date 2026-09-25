// Accents: small built-in gestures every preset gets by default, so the song's riff reads on
// screen even when a preset's own reactions miss it.
//
//  * hook: on each repeat of the song's hook (src/analysis/hooks.ts) the camera makes the same small
//    gesture on every note of the motif, a nudge whose direction follows the note's place in the
//    motif, plus a lift on the repeat's first note. Every repeat moves the same way, so the riff rhymes.
//
// Everything is a camera / colour pose composed onto the choreography's (see choreo.ts): no extra
// passes and no flashes. A genome without the gene gets the defaults; the gene tunes each part and a
// part set to 0 is off.
//
// Pure (no GL, no module state besides the harness switch): the pose is a function of the gene,
// the genome and the music's position. Imports only types from genome.ts.

import type { Genome, ParamSpec, Params, Schema } from '../genome';
import { registerGenomeGene } from '../geneRegistry';
import type { ChoreoCue, ChoreoPose } from './choreo';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });

/** hook: strength of the hook gesture; shot: which set of gesture directions (a seed). */
export const ACCENT_SCHEMA: Schema = {
  hook: P(0, 1, 0.5),
  shot: P(0, 1, 0.5),
};

registerGenomeGene({
  key: 'accent',
  title: 'Accents',
  schemas: ACCENT_SCHEMA,
  optional: true,
  glossary: 'built-in accents every preset has by default (absent gene = these defaults; add the gene to tune them, set a part to 0 to turn it off): hook = on every repeat of the song\'s riff or sung hook the camera nudges on each note of the motif the same way every time, so the riff rhymes visually; shot = which set of nudge directions',
});

export interface AccentGene {
  p: Params;
}

/** Estimated GPU cost: nothing beyond the choreography's camera uniforms. */
export const ACCENT_COST_MS = 0;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object';

function clampSpec(v: unknown, s: ParamSpec): number {
  const x = typeof v === 'number' && Number.isFinite(v) ? v : s.def;
  return clamp(x, s.min, s.max);
}

/** A valid accent gene from anything (missing or broken values take the defaults). */
export function repairAccent(raw: unknown): AccentGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(ACCENT_SCHEMA)) p[k] = clampSpec(src[k], ACCENT_SCHEMA[k]);
  return { p };
}

/** Rule violations of an accent gene (empty when valid). */
export function validateAccent(c: AccentGene): string[] {
  if (!isObj(c) || !isObj(c.p)) return ['accent params'];
  const errs: string[] = [];
  for (const k of Object.keys(ACCENT_SCHEMA)) {
    const s = ACCENT_SCHEMA[k];
    const v = c.p[k];
    if (!(typeof v === 'number' && Number.isFinite(v) && v >= s.min - 1e-9 && v <= s.max + 1e-9)) errs.push(`accent.${k}`);
  }
  for (const k of Object.keys(c.p)) if (!(k in ACCENT_SCHEMA)) errs.push(`accent.${k} unknown`);
  return errs;
}

// ------------------------------------------------------------ breeding

type Rng = () => number;

function gauss(rng: Rng): number {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/** A new random accent gene (most parts near their defaults, a few pushed or turned off). */
export function randomAccent(rng: Rng): AccentGene {
  const p: Params = {};
  for (const k of Object.keys(ACCENT_SCHEMA)) {
    const s = ACCENT_SCHEMA[k];
    const r = rng();
    p[k] = r < 0.15 && k !== 'shot' ? 0 : r < 0.6 ? clamp(s.min + (s.max - s.min) * rng(), s.min, s.max) : s.def;
  }
  return { p };
}

/** Nudges some settings of an accent gene in place. */
export function jitterAccent(c: AccentGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(ACCENT_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.35) continue;
    touched = true;
    const s = ACCENT_SCHEMA[k];
    c.p[k] = clamp(c.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s.min, s.max);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    const s = ACCENT_SCHEMA[k];
    c.p[k] = clamp(s.min + (s.max - s.min) * rng(), s.min, s.max);
  }
}

/** Probability a child inherits the accent gene when only one parent has one (else it gets the defaults). */
export const ACCENT_CARRY = 0.5;

/**
 * Crossover of the accent gene. Draws from the rng only when a parent has one, so children of
 * parents without it come out exactly as before the gene existed.
 */
export function crossAccent(d: AccentGene | undefined, r: AccentGene | undefined, rng: Rng): AccentGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) return rng() < ACCENT_CARRY ? repairAccent(d ?? r) : undefined;
  const p: Params = {};
  for (const k of Object.keys(ACCENT_SCHEMA)) {
    const s = ACCENT_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    p[k] = x + (y - x) * rng();
  }
  return repairAccent({ p });
}

// --------------------------------------------------------------- plan

/** The accent parts a genome plays, resolved from its gene (or the defaults) and its other genes. */
export interface AccentPlan {
  hook: number;
  shot: number;
}

const OFF: AccentPlan = { hook: 0, shot: 0.5 };
const HOOK_SIGNALS = new Set(['hook', 'hookphase', 'hookon']);

let override: boolean | readonly string[] | null = null;

/**
 * Harness switch for before / after comparisons: false turns every accent off, a list keeps only
 * those parts (e.g. ['hook']), null restores the genomes' own.
 */
export function setAccentOverride(on: boolean | readonly string[] | null): void {
  override = on === true ? null : on;
}

/** What a genome plays. Without an accent gene: the defaults, minus the hook gesture for one that reacts to the hook signals itself. */
export function accentPlan(g: Genome, out: AccentPlan = { ...OFF }): AccentPlan {
  if (override === false) return Object.assign(out, OFF);
  const a = (g as Genome & { accent?: AccentGene }).accent;
  const src = a?.p;
  const d = (k: string) => (src && typeof src[k] === 'number' ? src[k] : ACCENT_SCHEMA[k].def);
  out.hook = d('hook');
  out.shot = src ? d('shot') : (((g.palette?.p?.hue ?? 0.5) % 1) + 1) % 1;
  if (!a && g.reactions.some((r) => HOOK_SIGNALS.has(r.src))) out.hook = 0;
  if (Array.isArray(override)) for (const k of Object.keys(out) as (keyof AccentPlan)[]) if (k !== 'shot' && !override.includes(k)) out[k] = 0;
  return out;
}

// --------------------------------------------------------------- pose

/** The music an accent pose needs this frame. */
export interface AccentInput {
  cue: ChoreoCue;
  /** Hook signals (MusicState hook fields). */
  hookOn: number;
  hookPulse: number;
  hookNotePulse: number;
  hookNote: number;
  hookId: number;
}

/** Hook gesture at full strength: zoom lift on a repeat's first note, zoom and nudge per note (fraction of the frame), roll per note (turns), held push while the hook plays. */
const HOOK_START_ZOOM = 0.08;
const HOOK_NOTE_ZOOM = 0.05;
const HOOK_NOTE_PAN = 0.03;
const HOOK_NOTE_ROLL = 0.006;
const HOOK_HOLD_ZOOM = 0.04;
const HOOK_START_HUE = 0.05;
const TAU = Math.PI * 2;
/** The nudge direction of a motif note (the same for that note in every repeat). */
function noteAngle(plan: AccentPlan, note: number, hook: number): number {
  return TAU * (plan.shot + 0.38197 * note + 0.25 * hook);
}

/** Composes the accents onto a pose (the choreography's, usually) in place. */
export function applyAccents(plan: AccentPlan, m: AccentInput, q: ChoreoPose): ChoreoPose {
  let zoom = 1;
  let tx = 0;
  let ty = 0;
  let roll = 0;
  let hue = 0;
  if (plan.hook > 0 && (m.hookOn > 0 || m.hookPulse > 0 || m.hookNotePulse > 0)) {
    const h = plan.hook;
    const np = m.hookNotePulse;
    zoom *= 1 + h * (HOOK_HOLD_ZOOM * m.hookOn + HOOK_START_ZOOM * m.hookPulse + HOOK_NOTE_ZOOM * np);
    if (m.hookNote >= 0 && np > 0) {
      const a = noteAngle(plan, m.hookNote, Math.max(0, m.hookId));
      tx += Math.cos(a) * HOOK_NOTE_PAN * h * np;
      ty += Math.sin(a) * HOOK_NOTE_PAN * h * np;
      roll += (m.hookNote % 2 === 0 ? 1 : -1) * HOOK_NOTE_ROLL * TAU * h * np;
    }
    hue += HOOK_START_HUE * h * m.hookPulse;
  }
  q.zoom *= zoom;
  q.tx += tx;
  q.ty += ty;
  q.roll += roll;
  q.hue += hue;
  return q;
}

// Accents: small built-in gestures every preset gets by default, so the song's riff, its section
// changes and its drum hits read on screen even when a preset's own reactions miss them.
//
//  * hook: on each repeat of the song's hook (src/analysis/hooks.ts) the camera makes the same small
//    gesture on every note of the motif, a nudge whose direction follows the note's place in the
//    motif, plus a lift on the repeat's first note. Every repeat moves the same way, so the riff rhymes.
//  * section: every section type gets its own light, colour and hue step (calm sections dimmer and
//    paler, choruses and drops fuller), reached in a fraction of a second at the boundary, so a chorus
//    looks like the other choruses and not like the verse; an optional framing per section type (a
//    push, a pan, a lean; off by default: a push-in magnifies fine flickering texture); drop: a zoom,
//    colour and light punch on each drop that settles over a bar.
//  * kick: a small zoom and exposure punch on drum hits, for presets without a hit or drums reaction.
//
// Everything is a camera / colour pose composed onto the choreography's (see choreo.ts): no extra
// passes and no full-frame flashes (the kick's exposure punch stays under 10 % at the default, far
// below the WCAG flash threshold; the drop's light lift is one swell per drop). A genome without the
// gene gets the defaults; the gene tunes each part and a part set to 0 is off. Parts the genome's choreography already does (a scene framing, a scene hue,
// a drop punch) are left to the choreography so nothing is applied twice.
//
// Pure (no GL, no module state besides the harness switch): the pose is a function of the gene,
// the genome and the music's position. Imports only types from genome.ts.

import type { Genome, ParamSpec, Params, Schema } from '../genome';
import { registerGenomeGene } from '../geneRegistry';
import { IDENTITY_POSE, sceneFraming, type ChoreoCue, type ChoreoGene, type ChoreoPose } from './choreo';
import type { SectionLabel } from '../../types';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });

/**
 * hook: strength of the hook gesture; section: the light and colour step per section type; hue: the
 * hue step per section type; frame: how far each section type's framing departs from the plain view
 * (off by default); drop: the punch on drops; kick: the punch on drum hits; shot: which set of framings and gesture directions (a seed).
 */
export const ACCENT_SCHEMA: Schema = {
  hook: P(0, 1, 0.5),
  section: P(0, 1, 0.5),
  hue: P(0, 1, 0.5),
  frame: P(0, 1, 0),
  drop: P(0, 1, 0.5),
  kick: P(0, 1, 0.5),
  shot: P(0, 1, 0.5),
};

registerGenomeGene({
  key: 'accent',
  title: 'Accents',
  schemas: ACCENT_SCHEMA,
  optional: true,
  glossary: 'built-in accents every preset has by default (absent gene = these defaults; add the gene to tune them, set a part to 0 to turn it off): hook = on every repeat of the song\'s riff or sung hook the camera nudges on each note of the motif the same way every time, so the riff rhymes visually; section = each section type gets its own light and colour (calm sections dimmer and paler, choruses and drops fuller) so choruses match each other and differ from verses; hue = hue step per section type; frame = each section type also gets its own framing (push, pan, lean; 0 by default); drop = zoom, colour and light punch on drops; kick = small zoom and brightness punch on drum hits (applied by default only to presets with no hit/drums reaction); shot = which set of framings and nudge directions. Parts the choreography gene already does (its frame, scene, punch) are left to it',
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
  section: number;
  hue: number;
  frame: number;
  drop: number;
  kick: number;
  shot: number;
}

const OFF: AccentPlan = { hook: 0, section: 0, hue: 0, frame: 0, drop: 0, kick: 0, shot: 0.5 };
const HIT_SIGNALS = new Set(['hit', 'drums']);
const HOOK_SIGNALS = new Set(['hook', 'hookphase', 'hookon']);

let override: boolean | readonly string[] | null = null;

/**
 * Harness switch for before / after comparisons: false turns every accent off, a list keeps only
 * those parts (e.g. ['hook']), null restores the genomes' own.
 */
export function setAccentOverride(on: boolean | readonly string[] | null): void {
  override = on === true ? null : on;
}

/** Whether a genome already answers drum hits itself (a hit or drums reaction, or a body moving on hits). */
export function hasHitResponse(g: Genome): boolean {
  return g.reactions.some((r) => HIT_SIGNALS.has(r.src)) || g.bodies.some((b) => b.motion.kind === 'hits');
}

/**
 * What a genome plays. The gene's values (or the defaults), minus the kick for a preset that
 * already answers drum hits and the hook gesture for one that reacts to the hook signals itself. Always: the section framing, hue and drop punch are left to a
 * choreography that does them.
 */
export function accentPlan(g: Genome, out: AccentPlan = { ...OFF }): AccentPlan {
  if (override === false) return Object.assign(out, OFF);
  const a = (g as Genome & { accent?: AccentGene }).accent;
  const src = a?.p;
  const d = (k: string): number => (src && typeof src[k] === 'number' ? src[k] : ACCENT_SCHEMA[k].def);
  out.hook = d('hook');
  out.section = d('section');
  out.hue = d('hue');
  out.frame = d('frame');
  out.drop = d('drop');
  out.kick = d('kick');
  out.shot = src ? d('shot') : (((g.palette?.p?.hue ?? 0.5) % 1) + 1) % 1;
  // Presets that answer drum hits (or the hook signals) themselves keep only their own response,
  // with or without a gene (a seed's gene may predate a part and carry its default).
  if (hasHitResponse(g)) out.kick = 0;
  if (g.reactions.some((r) => HOOK_SIGNALS.has(r.src))) out.hook = 0;
  const c: ChoreoGene | undefined = g.choreo;
  if (c) {
    if (c.p.frame > 0) out.frame = 0;
    if (c.p.scene > 0) out.hue = 0;
    if (c.p.punch > 0) out.drop = 0;
  }
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
  /** Drum hit envelope (the engine's hit pulse, 0..~1.4). */
  hit: number;
}

/** Seconds the camera takes to move into a new section's framing. */
export const SECTION_GLIDE_S = 0.35;
/** Hook gesture at full strength: zoom lift on a repeat's first note, zoom and nudge per note (fraction of the frame), roll per note (turns), held push while the hook plays. */
const HOOK_START_ZOOM = 0.08;
const HOOK_NOTE_ZOOM = 0.05;
const HOOK_NOTE_PAN = 0.03;
const HOOK_NOTE_ROLL = 0.006;
const HOOK_HOLD_ZOOM = 0.04;
const HOOK_START_HUE = 0.05;
/**
 * Section framing at full strength goes through choreo's sceneFraming with frame = SECTION_FRAME *
 * frame. Off by default: a push-in magnifies fine flickering texture (rain, sparks), which tipped a
 * busy preset over the flash limit in the harness, so the default look change is light and colour.
 */
const SECTION_FRAME = 0.6;
/**
 * Light and colour per section type at full strength (section = 1): exposure and saturation
 * multipliers. Calm sections are dimmer and paler, loud ones fuller; never brighter than the preset's
 * own exposure, so a section change cannot add flashes.
 */
const SECTION_TONE: Record<SectionLabel, [number, number]> = {
  intro: [0.7, 0.7], verse: [0.85, 0.85], build: [0.92, 1], chorus: [1, 1.25], drop: [1, 1.35], breakdown: [0.7, 0.65], outro: [0.7, 0.7],
};
/** Hue step per section type at full strength, as choreo's scene amount. */
const SECTION_SCENE = 0.35;
/** Drop punch at full strength (zoom kick, saturation and exposure lift), settling over DROP_RELAX_BARS. */
const DROP_ZOOM = 0.16;
const DROP_SAT = 0.3;
const DROP_EXPOSURE = 0.3;
const DROP_RELAX_BARS = 1;
/** Kick at full strength: zoom and exposure lift per unit hit pulse. */
const KICK_ZOOM = 0.035;
const KICK_EXPOSURE = 0.15;

const TAU = Math.PI * 2;
const FRAME_A: ChoreoPose = { ...IDENTITY_POSE };
const FRAME_B: ChoreoPose = { ...IDENTITY_POSE };
const FAKE: ChoreoGene = { p: { frame: 0, scene: 0, shot: 0.5 } };

/** A section type's look: framing, hue, light and colour. */
function sectionLook(plan: AccentPlan, label: SectionLabel, out: ChoreoPose): ChoreoPose {
  FAKE.p.frame = SECTION_FRAME * plan.frame;
  FAKE.p.scene = SECTION_SCENE * plan.hue;
  FAKE.p.shot = plan.shot;
  sceneFraming(FAKE, label, out);
  const [e, sa] = SECTION_TONE[label] ?? [1, 1];
  const k = Math.min(1, plan.section);
  out.exposure = 1 + (e - 1) * k;
  out.sat = 1 + (sa - 1) * k;
  return out;
}

/** The look of the current section (gliding in from the previous one's over SECTION_GLIDE_S). */
function sectionFrame(plan: AccentPlan, cue: ChoreoCue, out: ChoreoPose): ChoreoPose {
  sectionLook(plan, cue.label, out);
  const x = clamp(cue.sinceSection / SECTION_GLIDE_S, 0, 1);
  const g = x * x * (3 - 2 * x);
  if (g < 1) {
    const q = cue.prevLabel ? sectionLook(plan, cue.prevLabel, FRAME_B) : IDENTITY_POSE;
    out.zoom = q.zoom + (out.zoom - q.zoom) * g;
    out.roll = q.roll + (out.roll - q.roll) * g;
    out.tx = q.tx + (out.tx - q.tx) * g;
    out.ty = q.ty + (out.ty - q.ty) * g;
    out.hue = q.hue + (out.hue - q.hue) * g;
    out.exposure = q.exposure + (out.exposure - q.exposure) * g;
    out.sat = q.sat + (out.sat - q.sat) * g;
  }
  return out;
}

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
  let sat = 1;
  let exposure = 1;
  if (plan.section > 0 || plan.hue > 0 || plan.frame > 0) {
    const f = sectionFrame(plan, m.cue, FRAME_A);
    zoom *= f.zoom;
    tx += f.tx;
    ty += f.ty;
    roll += f.roll;
    hue += f.hue;
    exposure *= f.exposure;
    sat *= f.sat;
  }
  if (plan.drop > 0 && m.cue.sinceDrop < DROP_RELAX_BARS * m.cue.barSeconds) {
    const x = 1 - m.cue.sinceDrop / (DROP_RELAX_BARS * m.cue.barSeconds);
    const env = x * x;
    zoom *= 1 + DROP_ZOOM * plan.drop * env;
    sat *= 1 + DROP_SAT * plan.drop * env;
    exposure *= 1 + DROP_EXPOSURE * plan.drop * env;
  }
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
  if (plan.kick > 0 && m.hit > 0) {
    const k = plan.kick * Math.min(1, m.hit);
    zoom *= 1 + KICK_ZOOM * k;
    exposure *= 1 + KICK_EXPOSURE * k;
  }
  q.zoom *= zoom;
  q.tx += tx;
  q.ty += ty;
  q.roll += roll;
  q.hue += hue;
  q.sat *= sat;
  q.exposure *= exposure;
  return q;
}

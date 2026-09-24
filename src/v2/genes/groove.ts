// Groove: a genome-wide gene that gives motion the music's timing feel (src/analysis/groove.ts
// measures it: swing, push, humanity, syncopation).
//
//   swung music     the body clocks are warped so each beat's second half starts late, exactly
//                   where the music's off-beat lands: spins, sways, orbits and pulses lope; a sway
//                   across every two beats gives the lilt, an off-beat pulse lands on the swung "and"
//   push / pull     a laid-back backbeat drags the motion behind the grid, a pushed one leads it
//   quantized       (low humanity) motion moves in exact ticks: it holds, then snaps on each tick
//                   (hard locks, a small click of scale on every tick: crystalline)
//   human timing    organic jitter: each onset nudges the copies a little, differently every time,
//                   so the looseness is locked to the real hits rather than random wobble
//   syncopation     onsets on weak slots kick the copies (accents off the grid)
//
// Everything is CPU side (the body clocks and copy transforms), pure and seek-safe apart from the
// short onset envelopes. This file imports only types from genome.ts so genome.ts can import it.

import type { ParamSpec, Params, Schema } from '../genome';
import type { GrooveStats } from '../../types';
import { registerGenomeGene } from '../geneRegistry';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * swing: how much of the measured swing the motion takes (1 = lands exactly as late as the music,
 *   up to 1.5 = exaggerated); sub: the swung subdivision (8 = 8th pairs, 16 = 16th pairs);
 * sway: a sideways lilt across every two beats (scene units), fuller the more the music swings;
 * off: a size pulse on the (swung) off-beat;
 * lean: how far the backbeat's push / pull drags the motion (1 = 0.12 beat at full push);
 * crisp: how hard quantized music ticks (motion holds, then snaps each tick; scaled by 1 - humanity);
 * tick: ticks per beat;
 * jitter: organic nudges on each onset, scaled by the music's humanity;
 * accent: kicks on syncopated onsets, scaled by the music's syncopation.
 */
export const GROOVE_SCHEMA: Schema = {
  swing: P(0, 1.5, 1),
  sub: C([8, 16], 8),
  sway: P(0, 0.08, 0.02),
  off: P(0, 1, 0.3),
  lean: P(0, 1, 0.5),
  crisp: P(0, 1, 0.5),
  tick: C([1, 2, 4], 2),
  jitter: P(0, 1, 0.4),
  accent: P(0, 1, 0.3),
};

registerGenomeGene({
  key: 'groove',
  title: 'Groove',
  schemas: GROOVE_SCHEMA,
  optional: true,
  order: 12,
  glossary: 'gives motion the music\'s timing feel: swung music makes spins, sways and pulses land late on the off-beats (swing=how much of the measured swing, sub 8/16=which pairs), sway=sideways lilt over two beats, off=pulse on the swung off-beat; lean=laid-back/pushed backbeat drags/leads the motion; quantized music ticks crisply (crisp=hold-then-snap strength, tick=ticks per beat); human timing adds organic nudges on each hit (jitter); accent=kicks on syncopated hits',
});

export interface GrooveGene {
  p: Params;
}

/** Estimated cost: CPU only (clock warp and copy offsets), a token amount for the uniforms it moves. */
export const GROOVE_COST_MS = 0.01;

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

/** A valid groove gene from anything (missing or broken values take the defaults). */
export function repairGroove(raw: unknown): GrooveGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(GROOVE_SCHEMA)) p[k] = clampSpec(src[k], GROOVE_SCHEMA[k]);
  return { p };
}

/** Rule violations of a groove gene (empty when valid). */
export function validateGroove(g: GrooveGene): string[] {
  if (!isObj(g) || !isObj(g.p)) return ['groove params'];
  const errs: string[] = [];
  for (const k of Object.keys(GROOVE_SCHEMA)) {
    const s = GROOVE_SCHEMA[k];
    const v = g.p[k];
    const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min - 1e-9 && v <= s.max + 1e-9);
    if (!ok) errs.push(`groove.${k}`);
  }
  for (const k of Object.keys(g.p)) if (!(k in GROOVE_SCHEMA)) errs.push(`groove.${k} unknown`);
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

/** A new random groove (most settings near their defaults, a few pushed further). */
export function randomGroove(rng: Rng): GrooveGene {
  const p: Params = {};
  for (const k of Object.keys(GROOVE_SCHEMA)) {
    const s = GROOVE_SCHEMA[k];
    p[k] = rng() < 0.6 ? randomValue(s, rng) : s.def;
  }
  return { p };
}

/** Nudges some settings of a groove in place. */
export function jitterGroove(g: GrooveGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(GROOVE_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.35) continue;
    touched = true;
    const s = GROOVE_SCHEMA[k];
    g.p[k] = s.choices ? (rng() < 0.5 ? randomValue(s, rng) : g.p[k]) : clampSpec(g.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    g.p[k] = randomValue(GROOVE_SCHEMA[k], rng);
  }
}

/** Probability a child inherits the groove when only one parent has one. */
export const GROOVE_CARRY = 0.5;

/**
 * Crossover of the groove. Draws from the rng only when a parent has one, so children of parents
 * without a groove come out exactly as before the gene existed.
 */
export function crossGroove(d: GrooveGene | undefined, r: GrooveGene | undefined, rng: Rng): GrooveGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) return rng() < GROOVE_CARRY ? repairGroove(d ?? r) : undefined;
  const p: Params = {};
  for (const k of Object.keys(GROOVE_SCHEMA)) {
    const s = GROOVE_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    const t = rng();
    p[k] = s.choices ? (t < 0.5 ? x : y) : x + (y - x) * t;
  }
  return repairGroove({ p });
}

// --------------------------------------------------------------- feel

/** The off-beat's position within its pair for a swing amount (0.5 straight .. 2/3 triplet .. 0.75). */
export function offFrac(swing: number): number {
  return 0.5 + clamp(swing, 0, 1.5) * (1 / 6);
}

/** Swing warp of a position within one pair (0..1): the pair's midpoint is reached at time f. */
function swingPair(x: number, f: number): number {
  return x < f ? (0.5 * x) / f : 0.5 + (0.5 * (x - f)) / (1 - f);
}

/** Hold-then-snap easing of 0..1 (p = 1 linear; large p nearly a step at 0.5). */
function snap(u: number, p: number): number {
  if (p <= 1.0001) return u;
  const a = Math.pow(u, p);
  const b = Math.pow(1 - u, p);
  return a / (a + b);
}

/**
 * The motion's beat position for a song beat position: swung (the off-beat reached late), leaned
 * (push / pull), then ticked (held and snapped on the tick grid when the music is quantized).
 * Monotone and continuous: integer beats stay put apart from the lean.
 */
export function grooveBeat(g: GrooveGene, st: GrooveStats, beat: number): number {
  const p = g.p;
  const f = offFrac(p.swing * clamp(st.swing, 0, 1));
  const n = Math.floor(beat);
  const x = beat - n;
  let m: number;
  if (p.sub === 16) {
    const h = x < 0.5 ? 0 : 0.5;
    m = h + 0.5 * swingPair((x - h) * 2, f);
  } else m = swingPair(x, f);
  let b = n + m - p.lean * clamp(st.push, -1, 1) * 0.12;
  const q = p.crisp * (1 - clamp(st.humanity, 0, 1));
  if (q > 0.001) {
    const y = b * p.tick;
    const k = Math.floor(y);
    b = (k + snap(y - k, 1 + 9 * q)) / p.tick;
  }
  return b;
}

/** A body clock (bars / spin / phases) as the groove warps it. */
export interface GrooveClock {
  bars: number;
  spin: number;
  barPhase: number;
  beatPhase: number;
}

/** Warps a grid-locked body clock in place (spin moves with the bars it gains or loses). */
export function grooveClock(g: GrooveGene, st: GrooveStats, clk: GrooveClock): GrooveClock {
  const beat = clk.bars * 4;
  const w = grooveBeat(g, st, beat);
  const dBars = (w - beat) / 4;
  clk.bars += dBars;
  clk.spin += Math.PI * 2 * dBars;
  clk.barPhase = clk.bars - Math.floor(clk.bars);
  clk.beatPhase = clk.bars * 4 - Math.floor(clk.bars * 4);
  return clk;
}

/** What the groove adds to every copy of a body this frame. */
export interface GrooveOffset {
  dx: number;
  dy: number;
  da: number;
  /** Scale multiplier (1 = none). */
  s: number;
}

/** Inputs for the copy offsets (the song clock, the drum onsets, frame time). */
export interface GrooveCue {
  /** Song beat position (beat index + phase). */
  beat: number;
  /** Seconds per beat. */
  period: number;
  dt: number;
  /** Drum onset strength now, 0..1. */
  onset: number;
}

function hash11(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

/**
 * The copy offsets. mem / key hold the onset envelopes (a per-body record, as the engine's slot
 * memory). Pure apart from those.
 */
export function grooveOffset(g: GrooveGene, st: GrooveStats, cue: GrooveCue, mem: Record<string, number>, key: string, out: GrooveOffset): GrooveOffset {
  const p = g.p;
  const swing = clamp(st.swing, 0, 1);
  const human = clamp(st.humanity, 0, 1);
  const warped = grooveBeat(g, st, cue.beat);
  // Sway: right on one beat, left on the next, fuller in swung music; tilts with the swing.
  const sw = p.sway * (0.35 + 0.65 * swing) * Math.sin(Math.PI * warped);
  out.dx = sw;
  out.dy = 0;
  out.da = sw * 3;
  out.s = 1;
  // Off-beat pulse on the swung "and".
  const f = offFrac(p.swing * swing);
  const x = cue.beat - Math.floor(cue.beat);
  const half = p.sub === 16 ? 0.5 : 1;
  const xp = (x % half) / half;
  if (xp >= f) out.s += 0.18 * p.off * Math.exp((-(xp - f) * half * cue.period) / 0.09);
  // Crisp ticks: a click of scale on each tick when the music is quantized.
  const q = p.crisp * (1 - human);
  if (q > 0.001) {
    const u = cue.beat * p.tick - Math.floor(cue.beat * p.tick);
    out.s += 0.05 * q * Math.exp((-u * cue.period) / p.tick / 0.03);
  }
  // Onsets: a new nudge on each rising edge (jitter), a kick when it lands on a weak slot (accent).
  const k = (s: string) => `${key}${s}`;
  const prev = mem[k('o')] ?? 0;
  mem[k('o')] = cue.onset;
  const decay = Math.exp(-cue.dt / 0.22);
  let jx = (mem[k('jx')] ?? 0) * decay;
  let jy = (mem[k('jy')] ?? 0) * decay;
  let ja = (mem[k('ja')] ?? 0) * decay;
  let ac = (mem[k('ac')] ?? 0) * Math.exp(-cue.dt / 0.18);
  if (cue.onset > 0.45 && prev <= 0.45) {
    const n = (mem[k('n')] = (mem[k('n')] ?? 0) + 1);
    const h = p.jitter * human;
    jx = 0.035 * h * hash11(n * 1.37);
    jy = 0.035 * h * hash11(n * 2.71 + 5);
    ja = 0.25 * h * hash11(n * 3.93 + 9);
    const weak = (x > 0.17 && x < 0.4) || (x > 0.6 && x < 0.88) || (p.sub === 8 && x > 0.4 && x <= 0.6);
    if (weak) ac = Math.max(ac, cue.onset);
  }
  mem[k('jx')] = jx;
  mem[k('jy')] = jy;
  mem[k('ja')] = ja;
  mem[k('ac')] = ac;
  // Smoothed follow of the nudges (a quick glide, not a jump).
  const fk = 1 - Math.exp(-cue.dt / 0.03);
  const sx = (mem[k('sx')] = (mem[k('sx')] ?? 0) + (jx - (mem[k('sx')] ?? 0)) * fk);
  const sy = (mem[k('sy')] = (mem[k('sy')] ?? 0) + (jy - (mem[k('sy')] ?? 0)) * fk);
  const sa = (mem[k('sa')] = (mem[k('sa')] ?? 0) + (ja - (mem[k('sa')] ?? 0)) * fk);
  out.dx += sx;
  out.dy += sy;
  out.da += sa;
  out.s *= 1 + 0.3 * p.accent * clamp(st.synco * 1.5, 0, 1) * ac;
  return out;
}

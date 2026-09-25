// Lyrics: a genome-wide gene that lets what the words are about steer the picture. The lyric sampler
// (src/lyrics) reads each line's imagery (fire, water, night...) and mood (valence, arousal); this
// gene turns that into bounded, temporary nudges on top of the saved genome, never written into it:
//
//   palette  the hue leans toward the imagery's colour (fire warm, water and cold blue, love pink,
//            nature green, night deep blue, space violet, gold amber)
//   tone     exposure and saturation (night and dark dim it, light and gold brighten it, a happy
//            line is more colourful, a sad one paler)
//   motion   the camera and the pace (rise and sky pull back and lift, fall sinks and pushes in,
//            speed and intense lines run faster, dream and calm lines slow down, storms shake)
//   chain    the warp ops the genome already has (water swells ripples and water, storm and speed
//            stir noise, dream swirls and blurs)
//
// Without lyrics (or between sung passages) every nudge eases back to nothing, so a preset with the
// gene looks exactly like the one without it. It can also show the line being sung (see
// src/lyrics/overlay.ts). This file is pure and imports only types from genome.ts.

import type { ParamSpec, Params, Schema } from '../genome';
import type { LyricTag } from '../../lyrics/lexicon';
import { registerGenomeGene } from '../geneRegistry';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * strength: how far the words steer (0 = display only);
 * pal / tone / motion / chain: which groups the words may steer (1 on, 0 off);
 * lag: seconds the nudges take to follow the words;
 * kick: a small camera kick on each new line;
 * show: display the line being sung (0 off, 1 subtle caption, 2 karaoke with the next line);
 * smear: how much of the caption is drawn into the feedback, so the visual carries and smears it.
 */
export const LYRICS_SCHEMA: Schema = {
  strength: P(0, 1, 0.6),
  pal: C([0, 1], 1),
  tone: C([0, 1], 1),
  motion: C([0, 1], 1),
  chain: C([0, 1], 0),
  lag: P(0.3, 6, 1.5),
  kick: P(0, 1, 0.3),
  show: C([0, 1, 2], 1),
  smear: P(0, 1, 0),
};

registerGenomeGene({
  key: 'lyrics',
  title: 'Lyrics',
  schemas: LYRICS_SCHEMA,
  optional: true,
  order: 14,
  glossary: 'lets what the sung words are about steer the picture (lyrics looked up online): imagery words pull the palette hue (fire warm, water/cold blue, love pink, nature green, night deep blue, space violet, gold amber), tone (night/dark dim, light/gold brighten, happy lines more colourful, sad paler), motion (rise/sky pull back and lift, fall sinks, speed/intense lines faster, dream/calm slower, storms shake) and existing warp ops (water ripples, storm noise, dream swirl). strength=how far; pal/tone/motion/chain=1 lets it steer that group; lag=seconds to follow; kick=camera kick per line; show 0 off/1 subtle caption/2 karaoke; smear=caption drawn into the feedback so it smears. Temporary: never changes the saved preset; no effect without lyrics.',
});

export interface LyricsGene {
  p: Params;
}

/** Estimated cost: CPU nudges only; the smeared caption adds one small textured draw. */
export function lyricsCost(p: Params): number {
  return 0.01 + (p.show > 0 && p.smear > 0 ? 0.05 : 0);
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

/** A valid lyrics gene from anything (missing or broken values take the defaults). */
export function repairLyrics(raw: unknown): LyricsGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(LYRICS_SCHEMA)) p[k] = clampSpec(src[k], LYRICS_SCHEMA[k]);
  return { p };
}

/** Rule violations of a lyrics gene (empty when valid). */
export function validateLyrics(g: LyricsGene): string[] {
  if (!isObj(g) || !isObj(g.p)) return ['lyrics params'];
  const errs: string[] = [];
  for (const k of Object.keys(LYRICS_SCHEMA)) {
    const s = LYRICS_SCHEMA[k];
    const v = g.p[k];
    const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min - 1e-9 && v <= s.max + 1e-9);
    if (!ok) errs.push(`lyrics.${k}`);
  }
  for (const k of Object.keys(g.p)) if (!(k in LYRICS_SCHEMA)) errs.push(`lyrics.${k} unknown`);
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

/** A new random lyrics gene (most settings near their defaults; the caption rarely smears). */
export function randomLyrics(rng: Rng): LyricsGene {
  const p: Params = {};
  for (const k of Object.keys(LYRICS_SCHEMA)) {
    const s = LYRICS_SCHEMA[k];
    p[k] = rng() < 0.5 ? randomValue(s, rng) : s.def;
  }
  if (rng() < 0.6) p.smear = 0;
  // At least one group steers, or the gene would be a caption only.
  if (!p.pal && !p.tone && !p.motion && !p.chain) p.pal = 1;
  return { p };
}

/** Nudges some settings of a lyrics gene in place. */
export function jitterLyrics(g: LyricsGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(LYRICS_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.3) continue;
    touched = true;
    const s = LYRICS_SCHEMA[k];
    g.p[k] = s.choices ? (rng() < 0.5 ? randomValue(s, rng) : g.p[k]) : clampSpec(g.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    g.p[k] = randomValue(LYRICS_SCHEMA[k], rng);
  }
}

/** Probability a child inherits the gene when only one parent has it. */
export const LYRICS_CARRY = 0.5;

/**
 * Crossover of the lyrics gene. Draws from the rng only when a parent has one, so children of
 * parents without it come out exactly as before the gene existed.
 */
export function crossLyrics(d: LyricsGene | undefined, r: LyricsGene | undefined, rng: Rng): LyricsGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) return rng() < LYRICS_CARRY ? repairLyrics(d ?? r) : undefined;
  const p: Params = {};
  for (const k of Object.keys(LYRICS_SCHEMA)) {
    const s = LYRICS_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    const t = rng();
    p[k] = s.choices ? (t < 0.5 ? x : y) : x + (y - x) * t;
  }
  return repairLyrics({ p });
}

// --------------------------------------------------------------- nudges

/** What the words do to one slot this frame. Neutral: all multipliers 1, all offsets 0. */
export interface LyricNudge {
  /** Palette hue shift, turns. */
  hue: number;
  /** Saturation and exposure multipliers. */
  sat: number;
  exposure: number;
  /** Camera push-in multiplier (>= 1), lift (fraction of the frame, + up) and roll (radians). */
  zoom: number;
  lift: number;
  roll: number;
  /** Pace multiplier for the motion (chain op rates, body clocks). */
  speed: number;
  /** 0..1 swell of existing warp ops and carrier settings. */
  ripple: number;
  swirl: number;
  noise: number;
  water: number;
  blur: number;
}

export const NEUTRAL_NUDGE: Readonly<LyricNudge> = { hue: 0, sat: 1, exposure: 1, zoom: 1, lift: 0, roll: 0, speed: 1, ripple: 0, swirl: 0, noise: 0, water: 0, blur: 0 };

/** The words' state the nudges read (MusicState lyric fields; see src/lyrics/sampler.ts). */
export interface LyricInput {
  tags?: ArrayLike<number>;
  valence: number;
  arousal: number;
  presence: number;
  pulse: number;
}

type Effect = { hue?: number; exp?: number; sat?: number; lift?: number; zin?: number; zout?: number; speed?: number; roll?: number; ripple?: number; swirl?: number; noise?: number; water?: number; blur?: number };

/** What each tag does (in LYRIC_TAGS order): colour it leans to, light, camera, pace and warp. */
export const TAG_EFFECTS: Record<LyricTag, Effect> = {
  fire: { hue: 0.03, exp: 0.08, sat: 0.1, lift: 0.6, speed: 0.15 },
  water: { hue: 0.55, lift: -0.2, ripple: 1, water: 1, swirl: 0.3 },
  sky: { hue: 0.57, exp: 0.05, lift: 0.5, zout: 0.4 },
  night: { hue: 0.68, exp: -0.25, sat: -0.1, speed: -0.15 },
  love: { hue: 0.94, sat: 0.15, zin: 0.4, speed: -0.1 },
  fall: { exp: -0.08, lift: -0.8, zin: 0.3 },
  rise: { exp: 0.06, lift: 0.8, zout: 0.8 },
  speed: { speed: 0.6, zin: 0.2, noise: 0.3 },
  cold: { hue: 0.53, sat: -0.3, exp: 0.03, speed: -0.2 },
  gold: { hue: 0.12, sat: 0.15, exp: 0.1 },
  dark: { hue: 0.72, exp: -0.3, sat: -0.2 },
  light: { hue: 0.14, exp: 0.2, sat: -0.05 },
  dream: { hue: 0.8, sat: -0.05, speed: -0.25, swirl: 0.6, blur: 1 },
  city: { hue: 0.86, sat: 0.2, speed: 0.2 },
  nature: { hue: 0.3, sat: 0.1, speed: -0.1 },
  space: { hue: 0.74, exp: -0.1, zout: 0.5, speed: -0.1, roll: 0.3 },
  storm: { hue: 0.64, exp: -0.1, speed: 0.4, noise: 1, roll: 1 },
};
const EFFECTS = Object.values(TAG_EFFECTS);

/** Signed shortest turn from a to b (-0.5..0.5). */
const turn = (a: number, b: number) => {
  const d = (((b - a) % 1) + 1.5) % 1;
  return d - 0.5;
};

/**
 * The nudge the words ask for right now (before easing). `baseHue` is the slot's palette hue
 * (turns) so the hue shift leans toward the imagery's colour from wherever the palette is; `time`
 * drives the storm shake and the slow space roll.
 */
export function lyricTarget(g: LyricsGene | undefined, w: LyricInput, baseHue: number, time: number, out: LyricNudge = { ...NEUTRAL_NUDGE }): LyricNudge {
  Object.assign(out, NEUTRAL_NUDGE);
  if (!g || !w.tags) return out;
  const p = g.p;
  const a = clamp(p.strength, 0, 1) * clamp(w.presence, 0, 1);
  if (a <= 1e-4) return out;
  let hx = 0, hy = 0, hw = 0, exp = 0, sat = 0, lift = 0, zin = 0, zout = 0, speed = 0, roll = 0;
  let ripple = 0, swirl = 0, noise = 0, water = 0, blur = 0;
  let total = 0;
  for (let i = 0; i < EFFECTS.length; i++) {
    const t = clamp(w.tags[i] ?? 0, 0, 1);
    if (t <= 0) continue;
    total += t;
    const e = EFFECTS[i];
    if (e.hue !== undefined) {
      hx += t * Math.cos(2 * Math.PI * e.hue);
      hy += t * Math.sin(2 * Math.PI * e.hue);
      hw += t;
    }
    exp += t * (e.exp ?? 0);
    sat += t * (e.sat ?? 0);
    lift += t * (e.lift ?? 0);
    zin += t * (e.zin ?? 0);
    zout += t * (e.zout ?? 0);
    speed += t * (e.speed ?? 0);
    roll += t * (e.roll ?? 0);
    ripple += t * (e.ripple ?? 0);
    swirl += t * (e.swirl ?? 0);
    noise += t * (e.noise ?? 0);
    water += t * (e.water ?? 0);
    blur += t * (e.blur ?? 0);
  }
  // Many tags at once share the effect instead of stacking without bound.
  const norm = total > 1 ? 1 / Math.sqrt(total) : 1;
  const v = clamp(w.valence, 0, 1) - 0.5;
  const ar = clamp(w.arousal, 0, 1) - 0.5;
  if (p.pal && hw > 0) {
    const target = Math.atan2(hy, hx) / (2 * Math.PI);
    const coherence = Math.hypot(hx, hy) / hw; // 1: all tags agree on a colour
    out.hue = turn(baseHue, target) * a * 0.85 * clamp(hw, 0, 1) * coherence;
  }
  if (p.tone) {
    out.exposure = clamp(1 + a * (exp * norm + 0.15 * v), 0.7, 1.25);
    out.sat = clamp(1 + a * (sat * norm + 0.4 * v), 0.65, 1.3);
  }
  if (p.motion) {
    out.speed = clamp(1 + a * (speed * norm + 0.8 * ar), 0.6, 1.6);
    out.zoom = clamp(1 + a * (0.06 + 0.1 * zin * norm - 0.06 * zout * norm), 1, 1.2);
    out.lift = clamp(a * lift * norm * 0.035, -0.035, 0.035);
    const shake = Math.sin(time * 7.3) * 0.6 + Math.sin(time * 11.1) * 0.4;
    out.roll = clamp(a * norm * roll * (0.012 * shake + 0.02 * Math.sin(time * 0.21)), -0.03, 0.03);
  }
  if (p.chain) {
    const k = a * norm;
    out.ripple = clamp(ripple * k, 0, 1);
    out.swirl = clamp(swirl * k, 0, 1);
    out.noise = clamp(noise * k, 0, 1);
    out.water = clamp(water * k, 0, 1);
    out.blur = clamp(blur * k, 0, 1);
  }
  return out;
}

/** The camera kick on a new line (zoom multiplier; applied after easing so it stays a kick). */
export function lineKick(g: LyricsGene | undefined, pulse: number): number {
  if (!g || !g.p.motion) return 1;
  return 1 + 0.05 * g.p.kick * g.p.strength * clamp(pulse, 0, 1);
}

/** Eases a nudge toward its target over `lag` seconds (in place). */
export function easeNudge(cur: LyricNudge, target: LyricNudge, dt: number, lag: number): LyricNudge {
  const k = 1 - Math.exp(-Math.max(0, dt) / Math.max(0.05, lag));
  for (const key of Object.keys(NEUTRAL_NUDGE) as (keyof LyricNudge)[]) cur[key] += (target[key] - cur[key]) * k;
  return cur;
}

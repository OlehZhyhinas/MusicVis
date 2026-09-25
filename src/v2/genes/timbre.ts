// Timbre as material: a genome-wide gene that turns what the sound is made of (src/analysis/timbre.ts:
// brightness, noisiness, roughness, attack) into what the bodies are made of.
//
//   bright sound        metallic sheen: a specular highlight and palette reflections on each body
//   pure, tonal sound   glass: the body's inside turns transparent, a bright Fresnel rim, a refracted
//                       inner edge
//   noisy / rough sound grain: a gritty, sparkling surface texture and ragged edges
//   breathy sound       velvet: noisy but soft (no sharp attack): a soft bloom halo and a dimmed,
//                       plush interior
//   sharp attacks       edges flash crisp on each pluck or hit (edge)
//   rough / noisy mix   the carried picture is embossed as a lit surface (emboss, via the tone's
//                       relief), glossier and more metallic the brighter the sound
//
// A source picks whose timbre drives it: the mix or one stem. Distance-field bodies get the surface
// (a wrapper around the body's material in the shader); every species gets the emboss. The music
// values are smoothed on the CPU and sent as two vec4 per body. This file imports only types from
// genome.ts so genome.ts can import it without a cycle.

import type { ParamSpec, Params, Schema } from '../genome';
import type { StemName, TimbreStats } from '../../types';
import { registerGenomeGene } from '../geneRegistry';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * src: whose timbre (0 mix, 1 drums, 2 bass, 3 vocals, 4 other);
 * sheen: metallic highlight from brightness; glass: transparency and rim from tonal purity;
 * grain: grit from noisiness and roughness (scale = grain size, grains per body); velvet: soft bloom
 * from breathiness; edge: crisp edge flashes on sharp attacks; emboss: the carried picture lit as a
 * surface from roughness (glossy and metallic with brightness).
 */
export const TIMBRE_SCHEMA: Schema = {
  src: C([0, 1, 2, 3, 4], 0),
  sheen: P(0, 1, 0.6),
  glass: P(0, 1, 0.5),
  grain: P(0, 1, 0.5),
  scale: P(4, 60, 18),
  velvet: P(0, 1, 0.4),
  edge: P(0, 1, 0.3),
  emboss: P(0, 1, 0),
};

export const TIMBRE_SOURCES: readonly ('mix' | StemName)[] = ['mix', 'drums', 'bass', 'vocals', 'other'];

registerGenomeGene({
  key: 'timbre',
  title: 'Timbre',
  schemas: TIMBRE_SCHEMA,
  optional: true,
  order: 13,
  glossary: 'turns the sound\'s timbre into what the bodies are made of: src=whose timbre (0 mix,1 drums,2 bass,3 vocals,4 other); bright sound gives metallic sheen (sheen), pure tonal sound turns bodies to glass with a bright rim (glass), noisy/rough sound adds grit (grain, scale=grain size), breathy sound a soft velvet bloom (velvet), sharp attacks flash the edges (edge); emboss=the whole picture lit as a surface from roughness',
});

export interface TimbreGene {
  p: Params;
}

/** GPU cost per distance-field evaluation of the surface (a few ALU and one hash), and of the emboss. */
export const TIMBRE_EVAL_COST = 0.06;
export const TIMBRE_EMBOSS_COST = 0.12;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const clamp01 = (v: number) => clamp(v, 0, 1);
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

export function repairTimbre(raw: unknown): TimbreGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(TIMBRE_SCHEMA)) p[k] = clampSpec(src[k], TIMBRE_SCHEMA[k]);
  return { p };
}

export function validateTimbre(g: TimbreGene): string[] {
  if (!isObj(g) || !isObj(g.p)) return ['timbre params'];
  const errs: string[] = [];
  for (const k of Object.keys(TIMBRE_SCHEMA)) {
    const s = TIMBRE_SCHEMA[k];
    const v = g.p[k];
    const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min - 1e-9 && v <= s.max + 1e-9);
    if (!ok) errs.push(`timbre.${k}`);
  }
  for (const k of Object.keys(g.p)) if (!(k in TIMBRE_SCHEMA)) errs.push(`timbre.${k} unknown`);
  return errs;
}

/**
 * Cost of the gene on a genome: `sdfEvals` = per distance-field body, the material evaluations it
 * pays for (the surface wraps each), `relief` = whether the tone already lights the picture.
 */
export function timbreCost(g: TimbreGene, sdfEvals: number[], relief: boolean): number {
  let ms = 0;
  for (const n of sdfEvals) ms += TIMBRE_EVAL_COST * n;
  if (g.p.emboss > 0.001 && !relief) ms += TIMBRE_EMBOSS_COST;
  return ms;
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

export function randomTimbre(rng: Rng): TimbreGene {
  const p: Params = {};
  for (const k of Object.keys(TIMBRE_SCHEMA)) {
    const s = TIMBRE_SCHEMA[k];
    p[k] = rng() < 0.6 ? randomValue(s, rng) : s.def;
  }
  return { p };
}

export function jitterTimbre(g: TimbreGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(TIMBRE_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.35) continue;
    touched = true;
    const s = TIMBRE_SCHEMA[k];
    g.p[k] = s.choices ? (rng() < 0.5 ? randomValue(s, rng) : g.p[k]) : clampSpec(g.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    g.p[k] = randomValue(TIMBRE_SCHEMA[k], rng);
  }
}

export const TIMBRE_CARRY = 0.5;

/** Crossover; draws from the rng only when a parent has the gene. */
export function crossTimbre(d: TimbreGene | undefined, r: TimbreGene | undefined, rng: Rng): TimbreGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) return rng() < TIMBRE_CARRY ? repairTimbre(d ?? r) : undefined;
  const p: Params = {};
  for (const k of Object.keys(TIMBRE_SCHEMA)) {
    const s = TIMBRE_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    const t = rng();
    p[k] = s.choices ? (t < 0.5 ? x : y) : x + (y - x) * t;
  }
  return repairTimbre({ p });
}

// --------------------------------------------------------------- look

/** What the surface does this frame, before smoothing: sheen, glass, grain, velvet, edge. */
export interface TimbreLook {
  sheen: number;
  glass: number;
  grain: number;
  velvet: number;
  edge: number;
}

/** The timbre a gene reads from the music (neutral when not analysed). */
export function timbreSource(g: TimbreGene, all: Record<'mix' | StemName, TimbreStats> | undefined): TimbreStats {
  const key = TIMBRE_SOURCES[g.p.src] ?? 'mix';
  return all?.[key] ?? { bright: 0, noise: 0, rough: 0, attack: 0 };
}

/** The surface amounts for a timbre (each gene param scaled by the matching quality of the sound). */
export function timbreLook(g: TimbreGene, t: TimbreStats, out: TimbreLook = { sheen: 0, glass: 0, grain: 0, velvet: 0, edge: 0 }): TimbreLook {
  const p = g.p;
  const noisy = clamp01(Math.max(t.noise, t.rough * 0.9));
  out.sheen = p.sheen * clamp01((t.bright - 0.25) / 0.5);
  out.glass = p.glass * clamp01(1 - 1.4 * t.noise) * clamp01(1 - 0.8 * t.rough);
  out.grain = p.grain * noisy;
  out.velvet = p.velvet * clamp01(t.noise * 1.3) * clamp01(1 - t.attack * 1.2);
  out.edge = p.edge * clamp01(t.attack);
  return out;
}

/**
 * The two vec4 of a body's surface uniforms, smoothed in `state` (5 numbers per body, kept by the
 * caller): (sheen, glass, grain, velvet), (grain scale, edge flash, grain phase, -).
 */
export function timbreUniforms(g: TimbreGene, look: TimbreLook, state: Float32Array, bi: number, time: number, dt: number, out: Float32Array): void {
  const o = bi * 8;
  const s = bi * 5;
  const k = 1 - Math.exp(-dt / 0.25);
  const kEdge = 1 - Math.exp(-dt / (look.edge > state[s + 4] ? 0.02 : 0.2));
  state[s] += (look.sheen - state[s]) * k;
  state[s + 1] += (look.glass - state[s + 1]) * k;
  state[s + 2] += (look.grain - state[s + 2]) * k;
  state[s + 3] += (look.velvet - state[s + 3]) * k;
  state[s + 4] += (look.edge - state[s + 4]) * kEdge;
  out[o] = state[s];
  out[o + 1] = state[s + 1];
  out[o + 2] = state[s + 2];
  out[o + 3] = state[s + 3];
  out[o + 4] = g.p.scale;
  out[o + 5] = state[s + 4];
  out[o + 6] = time;
  out[o + 7] = 0;
}

/** The tone parameters the gene adjusts (relief, gloss, metal), from a tone getter. */
export function timbreTone(g: TimbreGene, t: TimbreStats, get: (k: string) => number, k: string): number {
  const v = get(k);
  const e = g.p.emboss;
  if (e <= 0.001) return v;
  const rough = clamp01(Math.max(t.rough, t.noise * 0.8));
  switch (k) {
    case 'relief': return clamp01(v + e * (0.25 + 0.6 * rough));
    case 'gloss': return clamp01(v + e * 0.6 * clamp01(t.bright));
    case 'metal': return clamp01(v + e * 0.5 * clamp01((t.bright - 0.3) / 0.5) * (1 - t.noise));
    case 'bump': return Math.min(3, v * (0.8 + 1.2 * rough));
  }
  return v;
}

// --------------------------------------------------------------- shader

/** Uniform declaration plus the surface function (two vec4 per body). */
export function timbreGlsl(nb: number): string {
  return /* glsl */ `
uniform vec4 uTmb[${nb * 2}];
vec4 timbreSurface(vec4 m, vec3 s, vec2 q, float R, inout vec3 ex, vec4 A, vec4 B) {
  if (A.x + A.y + A.z + A.w + B.y < 0.002) return m;
  R = max(R, 1e-3);
  // A dome over the body: flat outside, rising from the edge to the centre, lit by its slope.
  vec2 gd = vec2(dFdx(s.x), dFdy(s.x));
  float gl = length(gd);
  gd = gl > 1e-7 ? gd / gl : vec2(0.0);
  float depth = clamp(-s.x / R, 0.0, 1.0);
  float slope = sqrt(1.0 - depth) * 3.0;
  vec3 n = normalize(vec3(gd * slope, 1.0));
  vec3 col = m.rgb;
  float lv = luma(col);
  // Grain: grit that sparkles at the grain size, ragged near the edge.
  if (A.z > 0.001) {
    vec2 gq = q / R * B.x;
    float gr = hash12(floor(gq) + floor(B.z * 12.0) * 0.37);
    float sp = step(1.0 - 0.08 * A.z, hash12(floor(gq * 1.7) + 3.1 + floor(B.z * 20.0)));
    col *= 1.0 - A.z * 0.65 * gr;
    col += sp * A.z * (0.3 + lv) * 0.6;
    m.a *= 1.0 - A.z * 0.8 * step(depth, 0.18) * gr;
    n = normalize(n + vec3((gr - 0.5) * A.z * 0.8, (hash12(floor(gq) + 9.7) - 0.5) * A.z * 0.8, 0.0));
  }
  // Sheen: palette reflections along the normal and a white specular highlight.
  if (A.x > 0.001) {
    vec3 L = normalize(vec3(-0.45, 0.55, 0.7));
    float spec = pow(max(dot(reflect(-L, n), vec3(0.0, 0.0, 1.0)), 0.0), 28.0);
    vec3 env = pal(0.45 + 0.35 * n.x + 0.25 * n.y) * (0.35 + 1.1 * sqrt(max(lv, 0.0)));
    col = mix(col, env, A.x * 0.55) + vec3(spec) * A.x * (0.25 + lv) * 1.4;
  }
  // Glass: the inside goes clear, the rim catches the light, a refracted inner edge.
  if (A.y > 0.001) {
    float fres = pow(1.0 - n.z, 1.5);
    float rim = glow(s.x + 0.12 * R, 0.08 * R);
    m.a *= mix(1.0, 0.18 + 0.82 * clamp(fres * 1.6, 0.0, 1.0), A.y);
    col += (col + pal(0.6 + 0.3 * n.x) * 0.3) * (fres * 0.9 + rim * 0.7) * A.y;
  }
  // Velvet: a plush, dimmer interior and a soft bloom halo outside.
  if (A.w > 0.001) {
    col *= 1.0 - 0.35 * A.w * depth;
    ex += col * exp(-max(s.x, 0.0) / (0.5 * R)) * step(0.0, s.x) * A.w * 0.35;
  }
  // Edge: a crisp line on the outline on each sharp attack.
  if (B.y > 0.001) ex += (col * 1.5 + vec3(0.25)) * glow(s.x, max(px() * 1.5, 0.004 * R)) * B.y;
  return vec4(col, m.a);
}
`;
}

/**
 * Wraps a body's material (MAT_i in the slotted body code) with the timbre surface: the original is
 * renamed MATB_i and a MAT_i that calls it and applies timbreSurface is added before the body.
 * Bodies without a material function (curves, fields, flames) are returned unchanged.
 */
export function timbreWrap(code: string, bi: number): string {
  const def = `vec4 MAT_${bi}(`;
  const at = code.indexOf(`vec3 body_${bi}(`);
  if (!code.includes(def) || at < 0) return code;
  const wrapper = `vec4 MAT_${bi}(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex) {
  vec4 m = MATB_${bi}(s, q, Q, R, p, ex);
  return timbreSurface(m, s, q, R, ex, uTmb[${bi * 2}], uTmb[${bi * 2 + 1}]);
}
`;
  const renamed = code.replace(def, `vec4 MATB_${bi}(`);
  const at2 = renamed.indexOf(`vec3 body_${bi}(`);
  return renamed.slice(0, at2) + wrapper + renamed.slice(at2);
}

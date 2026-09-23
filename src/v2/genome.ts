// V2 genome: one shared vocabulary so any two presets can swap parts.
//
//   chain      ordered space transforms (vec2 -> vec2). Motion ops run in the
//              feedback warp; fold ops (mirror, tile, polar, kaleido) and the
//              flame variations can run in the warp ('warp' stage) or at
//              display time ('view' stage, applied before sampling).
//   emitters   1-3 distinct light sources (fragment fields, line geometry,
//              GPU particles, a fractal-flame IFS).
//   carrier    how existing light moves: feedback warp, fluid, flow field, none.
//   color      palette around the key hue plus post settings.
//   reactions  music signal -> parameter, with a gain.
//
// Every numeric parameter has a spec (range, integer, discrete choices) so
// mutation and crossover always stay in range; repair() enforces all of it.

import { FLAME_VARIATIONS, type FlameVar } from './variations';

export { FLAME_VARIATIONS };
export type { FlameVar };

export interface ParamSpec {
  min: number;
  max: number;
  def: number;
  int?: boolean;
  log?: boolean;
  choices?: number[];
}
export type Schema = Record<string, ParamSpec>;
export type Params = Record<string, number>;

const P = (min: number, max: number, def: number, extra: Partial<ParamSpec> = {}): ParamSpec => ({ min, max, def, ...extra });
const I = (min: number, max: number, def: number): ParamSpec => ({ min, max, def, int: true });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });
const LOCKS = [-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25];

// ------------------------------------------------------------------ chain

export const MOTION_OPS = ['zoom', 'rotate', 'translate', 'swirl', 'twist', 'ripple', 'noise'] as const;
export const FOLD_OPS = ['mirror', 'tile', 'polar', 'kaleido'] as const;
export const VAR_OPS = FLAME_VARIATIONS.map((v) => `v_${v}`) as `v_${FlameVar}`[];
export type OpKind = (typeof MOTION_OPS)[number] | (typeof FOLD_OPS)[number] | `v_${FlameVar}`;
export const OP_KINDS: OpKind[] = [...MOTION_OPS, ...FOLD_OPS, ...VAR_OPS];
export type Stage = 'warp' | 'view';

const VAR_SCHEMA: Schema = { s: P(0.4, 3, 1.2) };

export const OP_SCHEMAS: Record<string, Schema> = {
  // rate > 0: content streams outward (flying in); < 0: inward.
  zoom: { rate: P(-0.02, 0.04, 0.008), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), radial: P(0, 1, 0), wander: P(0, 0.35, 0) },
  // lock: turns per bar (bar-locked spin); rate: free rotation, rad per frame.
  rotate: { lock: C(LOCKS, 0.0625), rate: P(-0.01, 0.01, 0), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), alt: C([0, 1], 0) },
  // content velocity, p units per second (applied as whole-pixel shifts)
  translate: { vx: P(-0.5, 0.5, 0), vy: P(-0.5, 0.5, 0) },
  swirl: { amt: P(-0.03, 0.03, 0.01), k: P(1, 12, 6), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), wander: P(0, 0.35, 0) },
  twist: { amt: P(-0.012, 0.012, 0.003), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0) },
  ripple: { amp: P(0, 0.004, 0.0008), freq: P(2, 40, 8), speed: P(0.2, 4, 0.7), radial: C([0, 1], 0) },
  noise: { amp: P(0, 0.003, 0.0012), scale: P(0.8, 5, 2), speed: P(0.05, 1, 0.3) },
  mirror: { axis: C([0, 1, 2], 0) },
  tile: { n: P(1.2, 4, 2) },
  polar: { scale: P(0.4, 1.6, 1), lock: C(LOCKS, 0) },
  kaleido: { n: I(3, 12, 6), lock: C(LOCKS, 0.0625) },
};
for (const v of VAR_OPS) OP_SCHEMAS[v] = VAR_SCHEMA;

export function isFold(op: OpKind): boolean {
  return (FOLD_OPS as readonly string[]).includes(op);
}
export function isVarOp(op: OpKind): boolean {
  return op.startsWith('v_');
}
/** Ops that may live in either stage; motion ops are warp-only. */
export function stageFree(op: OpKind): boolean {
  return isFold(op) || isVarOp(op);
}

export interface OpGene {
  op: OpKind;
  stage: Stage;
  /** Strength 0..1 (blend amount for variations, rate multiplier for motion, unused by folds). */
  w: number;
  p: Params;
}

// --------------------------------------------------------------- emitters

export const EMITTER_KINDS = [
  'wave', 'spectrum', 'particles', 'stars', 'ink', 'wire', 'plasma', 'aurora', 'blobs', 'flame', 'edge', 'tiles', 'horizon', 'orb',
] as const;
export type EmitterKind = (typeof EMITTER_KINDS)[number];
export type Layer = 'fb' | 'top';

const COMMON_EMIT: Schema = { gain: P(0.05, 3, 1), hue: P(0, 1, 0) };

export const EMITTER_SCHEMAS: Record<EmitterKind, Schema> = {
  // shape: 0 line, 1 circle, 2 spiral, 3 lissajous (harmonograph)
  wave: { ...COMMON_EMIT, shape: C([0, 1, 2, 3], 0), amp: P(0.05, 0.5, 0.24), x: P(-0.3, 0.3, 0), y: P(-0.35, 0.35, 0), radius: P(0.08, 0.45, 0.27), turns: P(1, 6, 3), ra: I(1, 6, 2), rb: I(1, 6, 3), thick: P(0.8, 3, 1.5) },
  // mode: 0 bars on a baseline, 1 upper arc, 2 full ring, 3 mirrored bars
  spectrum: { ...COMMON_EMIT, mode: C([0, 1, 2, 3], 1), x: P(-0.3, 0.3, 0), y: P(-0.45, 0.2, -0.42), bins: I(12, 64, 40), radius: P(0.08, 0.35, 0.2), len: P(0.1, 0.5, 0.3), fill: P(0.3, 0.9, 0.6) },
  // spawn: 0 anywhere, 1 waveform, 2 rotating emitters, 3 ring, 4 centre, 5 bottom edge
  particles: { ...COMMON_EMIT, spawn: C([0, 1, 2, 3, 4, 5], 4), count: P(1024, 65536, 16384, { log: true, int: true }), size: P(1.5, 6, 2.6), speed: P(0.05, 1, 0.3), curl: P(0, 0.5, 0.05), zoomFlow: P(-0.5, 1.5, 0.5), lift: P(-0.2, 0.2, 0), drag: P(1, 5, 2.5), life: P(0.1, 0.8, 0.3), spread: P(0.01, 0.2, 0.05) },
  stars: { ...COMMON_EMIT, density: P(0.2, 0.8, 0.5), scale: P(3, 12, 6.5), links: P(0, 1, 0.8), twinkle: P(0, 1, 0.3), drift: P(0, 0.1, 0.03) },
  // orbit 0: blobs sit at stem stations; 1: they circle the centre, bar locked
  // row 1: blobs line up along the bottom edge (smoke sources)
  ink: { ...COMMON_EMIT, count: I(1, 6, 4), orbit: P(0, 1, 0), row: P(0, 1, 0), radius: P(0.1, 0.45, 0.3), size: P(0.008, 0.05, 0.02), wander: P(0, 0.2, 0.1), force: P(0, 1.5, 1) },
  // solid: 0 tetra, 1 cube, 2 octa, 3 icosa, 4 polygon, 5 follow sections
  wire: { ...COMMON_EMIT, solid: C([0, 1, 2, 3, 4, 5], 5), sides: I(3, 8, 6), scale: P(0.06, 0.35, 0.2), tilt: P(0, 1, 0.45), lock: C(LOCKS, 0.25), inner: P(0, 1, 0.6), thick: P(0.8, 2.5, 1.3) },
  plasma: { ...COMMON_EMIT, scale: P(0.8, 3, 1.7), warp: P(0.5, 3, 2), bands: P(3, 12, 8), lines: P(0, 1, 0.8), speed: P(0.05, 0.4, 0.15) },
  aurora: { ...COMMON_EMIT, y: P(-0.3, 0.25, 0), fall: P(2, 10, 5), rays: P(8, 40, 24), wav: P(0.5, 3, 1.4) },
  blobs: { ...COMMON_EMIT, count: I(3, 6, 6), size: P(0.03, 0.12, 0.06), spread: P(0.3, 0.7, 0.55), speed: P(0.05, 0.3, 0.12), chrome: P(0, 1, 1) },
  // flow: vars <-> alt morph cycles per 8 bars (0 = only drops morph); breathe: camera zoom per unit bass
  flame: { ...COMMON_EMIT, count: C([65536, 131072, 262144, 524288], 262144), zoom: P(0.1, 0.45, 0.22), camSpin: C(LOCKS, 0.0625), rounds: I(1, 2, 2), ox: P(-0.2, 0.2, 0), oy: P(-0.2, 0.2, 0), flow: P(0, 1, 0), breathe: P(0, 0.4, 0) },
  // mode: 0 skyline, 1 melody ribbon, 2 spectral rain, 3 terrain ridge. side: 0 right, 1 top, 2 bottom, 3 left
  edge: { ...COMMON_EMIT, mode: C([0, 1, 2, 3], 0), side: C([0, 1, 2, 3], 0), base: P(-0.3, 0.3, -0.16), height: P(0.1, 0.5, 0.3), density: P(0.2, 1, 0.6) },
  // shape: 0 hex, 1 square, 2 triangle
  tiles: { ...COMMON_EMIT, shape: C([0, 1, 2], 0), scale: P(3, 10, 5), lock: C(LOCKS, 0.0625), lit: P(0.2, 0.8, 0.5), edges: P(0, 1, 0.6) },
  horizon: { ...COMMON_EMIT, y: P(-0.3, 0.1, -0.06), speed: P(0.2, 1, 0.5), density: P(0.5, 3, 1.5), peaks: P(0, 1, 1) },
  orb: { ...COMMON_EMIT, x: P(-0.5, 0.5, 0), y: P(-0.3, 0.35, 0.1), radius: P(0.04, 0.25, 0.1), halo: P(0, 1, 0.5), stripes: P(0, 1, 0), craters: P(0, 1, 0) },
};

export interface FlameXformGene {
  /** [a, b, c, d, e, f]: x' = a x + b y + e, y' = c x + d y + f */
  aff: number[];
  weight: number; // 0.05..1
  color: number; // 0..1
  vars: Partial<Record<FlameVar, number>>; // each 0..1
  /** Variation mix the transform morphs toward (flow cycles / drops); omitted = no morph. */
  alt?: Partial<Record<FlameVar, number>>;
  spin: number; // turns per bar, from LOCKS
  bass: number; // 0..0.3 scale per unit bass
  drift: [number, number]; // 0..0.3, bar-locked drift of the translation
  pulse: number; // 0..0.3 scale kick per beat pulse
}
export const MAX_XFORMS = 4;
export const AFF_RANGE = [-1.2, 1.2];
export const XFORM_SPIN = LOCKS;

export interface EmitterGene {
  kind: EmitterKind;
  layer: Layer;
  p: Params;
  xforms?: FlameXformGene[]; // flame only
}

// ---------------------------------------------------------------- carrier

export const CARRIER_KINDS = ['warp', 'fluid', 'flow', 'none'] as const;
export type CarrierKind = (typeof CARRIER_KINDS)[number];
export const CARRIER_SCHEMA: Schema = {
  halfLife: P(0.04, 25, 0.5, { log: true }), // seconds for the feedback to fade to half
  floor: P(0, 2, 1), // black-level subtraction
  blur: P(0, 0.4, 0),
  amount: P(0.4, 1.5, 1), // fluid advection amount
  vort: P(5, 40, 28),
  fnoise: P(0, 0.6, 0.35),
  fscale: P(1, 4, 2),
  famt: P(0.0004, 0.003, 0.0012),
};
export interface CarrierGene {
  kind: CarrierKind;
  p: Params;
}

// ----------------------------------------------------------------- colour

export const SCHEMES = ['analogous', 'complementary', 'triad', 'split', 'mono'] as const;
export type Scheme = (typeof SCHEMES)[number];
export const COLOR_SCHEMA: Schema = {
  hue: P(0, 1, 0.5),
  sat: P(0.2, 1, 0.9),
  exposure: P(0.6, 1.4, 1),
  contrast: P(0, 0.08, 0.03),
  bloom: P(0.4, 1.5, 1),
  adapt: P(0.1, 0.6, 0.3),
  vignette: P(0, 0.7, 0.45),
  ca: P(0, 0.008, 0.0015),
  reflect: C([0, 1], 0),
  reflectY: P(-0.45, 0, -0.16),
  tonemap: C([0, 1], 0), // 1 = flame log-density
};
export interface ColorGene {
  scheme: Scheme;
  p: Params;
}

// -------------------------------------------------------------- reactions

export const SIGNALS = ['drums', 'bass', 'vocals', 'other', 'hit', 'beat', 'bar', 'complexity', 'drop', 'loud', 'melody', 'build'] as const;
export type Signal = (typeof SIGNALS)[number];
export type GeneGroup = 'op' | 'em' | 'car' | 'col';
export interface ReactionGene {
  src: Signal;
  g: GeneGroup;
  i: number; // index into chain / emitters (0 for car / col)
  k: string; // parameter name
  gain: number; // -1..1, fraction of the parameter's half range per unit signal
}
export const MAX_REACTIONS = 6;
/** Parameters reactions may not touch (structural switches). */
const NO_REACT = new Set(['mode', 'shape', 'spawn', 'solid', 'side', 'axis', 'count', 'reflect', 'tonemap', 'alt', 'radial', 'rounds', 'lock', 'camSpin', 'sides', 'ra', 'rb', 'n', 'bins', 'halfLife']);

// ----------------------------------------------------------------- genome

export interface Genome {
  v: 1;
  chain: OpGene[];
  emitters: EmitterGene[];
  carrier: CarrierGene;
  color: ColorGene;
  reactions: ReactionGene[];
  energy: [number, number]; // complexity range the preset suits
}

export const MAX_CHAIN = 6;
export const MAX_EMITTERS = 3;

export type Species =
  | 'terrain' | 'rain' | 'vortex' | 'ink' | 'scope' | 'plasma' | 'mirror'
  | 'spectrum' | 'stars' | 'wire' | 'chrome' | 'aurora' | 'flame';
export const SPECIES: Species[] = ['terrain', 'rain', 'vortex', 'ink', 'scope', 'plasma', 'mirror', 'spectrum', 'stars', 'wire', 'chrome', 'aurora', 'flame'];
export const SPECIES_LABEL: Record<Species, string> = {
  terrain: 'terrain/skyline', rain: 'rain/smoke', vortex: 'vortex/tunnel', ink: 'ink/fluid', scope: 'oscilloscope',
  plasma: 'plasma/field', mirror: 'mirror/tiled', spectrum: 'spectrum', stars: 'starfield', wire: 'wireframe',
  chrome: 'chrome/blobs', aurora: 'aurora', flame: 'fractal flame',
};
export type Energy = 'calm' | 'energetic';

// ---------------------------------------------------------------- helpers

export function clampParam(v: number, s: ParamSpec): number {
  if (!Number.isFinite(v)) v = s.def;
  if (s.choices) {
    let best = s.choices[0];
    for (const c of s.choices) if (Math.abs(c - v) < Math.abs(best - v)) best = c;
    return best;
  }
  v = Math.min(s.max, Math.max(s.min, v));
  return s.int ? Math.round(v) : v;
}

export function inRange(v: number, s: ParamSpec): boolean {
  if (typeof v !== 'number' || !Number.isFinite(v)) return false;
  if (s.choices) return s.choices.includes(v);
  if (v < s.min - 1e-9 || v > s.max + 1e-9) return false;
  return !s.int || Number.isInteger(v);
}

function repairParams(p: unknown, schema: Schema): Params {
  const src = (p && typeof p === 'object' ? p : {}) as Record<string, unknown>;
  const out: Params = {};
  for (const k of Object.keys(schema)) {
    const v = src[k];
    out[k] = clampParam(typeof v === 'number' ? v : schema[k].def, schema[k]);
  }
  return out;
}

export function defaultParams(schema: Schema): Params {
  const out: Params = {};
  for (const k of Object.keys(schema)) out[k] = schema[k].def;
  return out;
}

export function schemaFor(g: Genome, group: GeneGroup, i: number): Schema | null {
  if (group === 'op') return g.chain[i] ? OP_SCHEMAS[g.chain[i].op] : null;
  if (group === 'em') return g.emitters[i] ? EMITTER_SCHEMAS[g.emitters[i].kind] : null;
  if (group === 'car') return CARRIER_SCHEMA;
  return COLOR_SCHEMA;
}

export function reactable(schema: Schema): string[] {
  return Object.keys(schema).filter((k) => !NO_REACT.has(k) && !schema[k].choices);
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function repairXform(x: Partial<FlameXformGene> | undefined): FlameXformGene {
  const src = x ?? {};
  const aff = Array.isArray(src.aff) ? src.aff.slice(0, 6) : [];
  while (aff.length < 6) aff.push(aff.length === 0 || aff.length === 3 ? 0.5 : 0);
  const vars: Partial<Record<FlameVar, number>> = {};
  let any = false;
  for (const v of FLAME_VARIATIONS) {
    const w = num(src.vars?.[v], 0);
    if (w > 0.001) {
      vars[v] = clamp(w, 0, 1);
      any = true;
    }
  }
  if (!any) vars.linear = 1;
  let alt: Partial<Record<FlameVar, number>> | undefined;
  if (src.alt && typeof src.alt === 'object') {
    for (const v of FLAME_VARIATIONS) {
      const w = num(src.alt[v], 0);
      if (w > 0.001) (alt ??= {})[v] = clamp(w, 0, 1);
    }
  }
  const out: FlameXformGene = {
    aff: aff.map((a) => clamp(num(a, 0), AFF_RANGE[0], AFF_RANGE[1])),
    weight: clamp(num(src.weight, 0.5), 0.05, 1),
    color: clamp(num(src.color, 0.5), 0, 1),
    vars,
    spin: clampParam(num(src.spin, 0), C(LOCKS, 0)),
    bass: clamp(num(src.bass, 0), 0, 0.3),
    drift: [clamp(num(src.drift?.[0], 0), 0, 0.3), clamp(num(src.drift?.[1], 0), 0, 0.3)],
    pulse: clamp(num(src.pulse, 0), 0, 0.3),
  };
  if (alt) out.alt = alt;
  return out;
}

function repairOp(o: Partial<OpGene> | undefined): OpGene | null {
  if (!o || !OP_KINDS.includes(o.op as OpKind)) return null;
  const op = o.op as OpKind;
  const stage: Stage = stageFree(op) ? (o.stage === 'view' ? 'view' : 'warp') : 'warp';
  return { op, stage, w: clamp(num(o.w, 1), 0, 1), p: repairParams(o.p, OP_SCHEMAS[op]) };
}

function repairEmitter(e: Partial<EmitterGene> | undefined): EmitterGene | null {
  if (!e || !EMITTER_KINDS.includes(e.kind as EmitterKind)) return null;
  const kind = e.kind as EmitterKind;
  // Flames and particles need accumulation; field emitters may sit on either layer.
  const layer: Layer = kind === 'flame' ? 'fb' : e.layer === 'top' ? 'top' : 'fb';
  const out: EmitterGene = { kind, layer, p: repairParams(e.p, EMITTER_SCHEMAS[kind]) };
  if (kind === 'flame') {
    const xs = (Array.isArray(e.xforms) ? e.xforms : []).slice(0, MAX_XFORMS).map(repairXform);
    if (!xs.length) xs.push(repairXform({ vars: { linear: 0.5, spherical: 0.5 } }), repairXform({ aff: [0.5, 0, 0, 0.5, 0.5, 0], vars: { sinusoidal: 1 } }));
    out.xforms = xs;
  }
  return out;
}

/**
 * Returns a valid genome: every parameter clamped into its spec, unknown
 * fields dropped, caps enforced, dangling reactions retargeted or removed.
 * Idempotent: repair(repair(g)) deep-equals repair(g).
 */
export function repair(input: unknown): Genome {
  const g = (input && typeof input === 'object' ? input : {}) as Partial<Genome>;
  const chain = (Array.isArray(g.chain) ? g.chain : []).map(repairOp).filter((o): o is OpGene => !!o).slice(0, MAX_CHAIN);
  const seen = new Set<EmitterKind>();
  const emitters: EmitterGene[] = [];
  for (const e of Array.isArray(g.emitters) ? g.emitters : []) {
    const r = repairEmitter(e);
    if (!r || seen.has(r.kind)) continue;
    seen.add(r.kind);
    emitters.push(r);
    if (emitters.length >= MAX_EMITTERS) break;
  }
  if (!emitters.length) emitters.push({ kind: 'wave', layer: 'fb', p: defaultParams(EMITTER_SCHEMAS.wave) });
  const ck = CARRIER_KINDS.includes(g.carrier?.kind as CarrierKind) ? (g.carrier!.kind as CarrierKind) : 'warp';
  const carrier: CarrierGene = { kind: ck, p: repairParams(g.carrier?.p, CARRIER_SCHEMA) };
  const scheme = SCHEMES.includes(g.color?.scheme as Scheme) ? (g.color!.scheme as Scheme) : 'analogous';
  const color: ColorGene = { scheme, p: repairParams(g.color?.p, COLOR_SCHEMA) };

  let lo = clamp(num(g.energy?.[0], 0.2), 0, 1);
  let hi = clamp(num(g.energy?.[1], 0.7), 0, 1);
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi - lo < 0.15) {
    const c = clamp((lo + hi) / 2, 0.075, 0.925);
    lo = c - 0.075;
    hi = c + 0.075;
  }
  const out: Genome = { v: 1, chain, emitters, carrier, color, reactions: [], energy: [round4(lo), round4(hi)] };

  for (const r of Array.isArray(g.reactions) ? g.reactions : []) {
    if (out.reactions.length >= MAX_REACTIONS) break;
    if (!r || !SIGNALS.includes(r.src as Signal)) continue;
    const group = (['op', 'em', 'car', 'col'] as GeneGroup[]).includes(r.g) ? r.g : null;
    if (!group) continue;
    const len = group === 'op' ? chain.length : group === 'em' ? emitters.length : 1;
    if (!len) continue;
    const i = group === 'op' || group === 'em' ? ((Math.floor(num(r.i, 0)) % len) + len) % len : 0;
    const keys = reactable(schemaFor(out, group, i)!);
    if (!keys.length) continue;
    const k = keys.includes(r.k) ? r.k : keys.includes('gain') ? 'gain' : keys[0];
    const gain = clamp(num(r.gain, 0.3), -1, 1);
    if (out.reactions.some((q) => q.g === group && q.i === i && q.k === k && q.src === r.src)) continue;
    out.reactions.push({ src: r.src as Signal, g: group, i, k, gain });
  }
  return out;
}

function round4(x: number): number {
  return Math.round(x * 1e4) / 1e4;
}

/** Lists every rule a genome breaks (empty when valid). Does not modify it. */
export function validate(g: Genome): string[] {
  const errs: string[] = [];
  const chk = (p: Params, s: Schema, where: string) => {
    for (const k of Object.keys(s)) if (!inRange(p[k], s[k])) errs.push(`${where}.${k}=${p[k]} out of range`);
    for (const k of Object.keys(p)) if (!(k in s)) errs.push(`${where}.${k} unknown`);
  };
  if (g.v !== 1) errs.push('version');
  if (!Array.isArray(g.chain) || g.chain.length > MAX_CHAIN) errs.push('chain length');
  g.chain?.forEach((o, i) => {
    if (!OP_KINDS.includes(o.op)) errs.push(`chain[${i}] op ${o.op}`);
    else chk(o.p, OP_SCHEMAS[o.op], `chain[${i}]`);
    if (!(o.w >= 0 && o.w <= 1)) errs.push(`chain[${i}].w`);
    if (o.stage === 'view' && !stageFree(o.op)) errs.push(`chain[${i}] stage`);
  });
  if (!Array.isArray(g.emitters) || g.emitters.length < 1 || g.emitters.length > MAX_EMITTERS) errs.push('emitter count');
  const kinds = new Set<string>();
  g.emitters?.forEach((e, i) => {
    if (!EMITTER_KINDS.includes(e.kind)) errs.push(`emitters[${i}] kind`);
    else chk(e.p, EMITTER_SCHEMAS[e.kind], `emitters[${i}]`);
    if (kinds.has(e.kind)) errs.push(`emitters[${i}] duplicate kind`);
    kinds.add(e.kind);
    if (e.kind === 'flame') {
      if (!e.xforms || e.xforms.length < 1 || e.xforms.length > MAX_XFORMS) errs.push('flame xforms count');
      e.xforms?.forEach((x, j) => {
        if (x.aff.length !== 6 || x.aff.some((a) => !(a >= AFF_RANGE[0] && a <= AFF_RANGE[1]))) errs.push(`xform[${j}].aff`);
        if (!(x.weight >= 0.05 && x.weight <= 1)) errs.push(`xform[${j}].weight`);
        if (!(x.color >= 0 && x.color <= 1)) errs.push(`xform[${j}].color`);
        const vs = Object.entries(x.vars);
        if (!vs.length || vs.some(([k, w]) => !FLAME_VARIATIONS.includes(k as FlameVar) || !(w! >= 0 && w! <= 1))) errs.push(`xform[${j}].vars`);
        if (!LOCKS.includes(x.spin)) errs.push(`xform[${j}].spin`);
        if (!(x.bass >= 0 && x.bass <= 0.3)) errs.push(`xform[${j}].bass`);
        if (!(x.pulse >= 0 && x.pulse <= 0.3)) errs.push(`xform[${j}].pulse`);
        if (!Array.isArray(x.drift) || x.drift.length !== 2 || x.drift.some((d) => !(d >= 0 && d <= 0.3))) errs.push(`xform[${j}].drift`);
        if (x.alt && Object.entries(x.alt).some(([k, w]) => !FLAME_VARIATIONS.includes(k as FlameVar) || !(w! >= 0 && w! <= 1))) errs.push(`xform[${j}].alt`);
      });
      if (e.layer !== 'fb') errs.push('flame layer');
    } else if (e.xforms) errs.push(`emitters[${i}] stray xforms`);
  });
  if (!CARRIER_KINDS.includes(g.carrier?.kind)) errs.push('carrier kind');
  else chk(g.carrier.p, CARRIER_SCHEMA, 'carrier');
  if (!SCHEMES.includes(g.color?.scheme)) errs.push('scheme');
  else chk(g.color.p, COLOR_SCHEMA, 'color');
  if (!(g.energy?.[0] >= 0 && g.energy[1] <= 1 && g.energy[0] < g.energy[1])) errs.push('energy');
  if (g.reactions?.length > MAX_REACTIONS) errs.push('reaction count');
  g.reactions?.forEach((r, i) => {
    const s = schemaFor(g, r.g, r.i);
    if (!s || !reactable(s).includes(r.k)) errs.push(`reactions[${i}] target ${r.g}${r.i}.${r.k}`);
    if (!SIGNALS.includes(r.src)) errs.push(`reactions[${i}] src`);
    if (!(r.gain >= -1 && r.gain <= 1)) errs.push(`reactions[${i}] gain`);
  });
  return errs;
}

// ------------------------------------------------------------- identity

export function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Everything that changes the compiled shaders (numeric params are uniforms). */
export function structuralKey(g: Genome): string {
  const ops = g.chain.map((o) => `${o.op}${o.stage === 'view' ? '@v' : ''}`).join(',');
  const em = g.emitters.map((e) => `${e.kind}@${e.layer}`).sort().join(',');
  return `${ops}|${em}|${g.carrier.kind}|r${g.color.p.reflect}t${g.color.p.tonemap}`;
}

export function genomeHash(g: Genome): number {
  return fnv1a(JSON.stringify(g, (_k, v) => (typeof v === 'number' ? Math.round(v * 1000) / 1000 : v)));
}

export function cloneGenome(g: Genome): Genome {
  return JSON.parse(JSON.stringify(g)) as Genome;
}

// -------------------------------------------------------- classification

export function speciesScores(g: Genome): Record<Species, number> {
  const s = Object.fromEntries(SPECIES.map((k) => [k, 0])) as Record<Species, number>;
  for (const e of g.emitters) {
    const p = e.p;
    const w = 0.6 + 0.4 * Math.min(1, p.gain);
    switch (e.kind) {
      case 'flame': s.flame += 3.2 * w; break;
      case 'wave': s.scope += 2.4 * w; break;
      case 'spectrum': s.spectrum += 2.6 * w; break;
      case 'particles':
        s.stars += 1.8 * w;
        if (p.spawn === 5 && p.lift > 0.03) s.rain += 1.6;
        if (p.spawn === 4 && p.zoomFlow > 0.5) s.stars += 1;
        break;
      case 'stars': s.stars += 2.6 * w; break;
      case 'ink': (p.orbit > 0.5 ? (s.vortex += 0.8) : (s.ink += 1.4)); s.ink += 0.4; break;
      case 'wire': s.wire += 2.6 * w; break;
      case 'plasma': s.plasma += 2.6 * w; break;
      case 'aurora': s.aurora += 2.6 * w; break;
      case 'blobs': s.chrome += 2.6 * w; break;
      case 'edge': (p.mode === 2 ? (s.rain += 2.6) : (s.terrain += 2.6)); if (p.side === 1 || p.side === 2) s.rain += 0.6; break;
      case 'tiles': s.mirror += 2.4 * w; break;
      case 'horizon': s.terrain += 2.6 * w; break;
      case 'orb': s.terrain += 1.2 * w; break;
    }
  }
  if (g.carrier.kind === 'fluid') s.ink += 2.6;
  if (g.carrier.kind === 'flow') s.rain += 0.4;
  for (const o of g.chain) {
    const p = o.p;
    switch (o.op) {
      case 'zoom': s.vortex += Math.min(2.2, Math.abs(p.rate) * o.w * 110); break;
      case 'rotate': s.vortex += Math.min(1.2, (Math.abs(p.lock) * 4 + Math.abs(p.rate) * 60) * o.w); break;
      case 'swirl': s.vortex += Math.min(1.8, Math.abs(p.amt) * o.w * 120); break;
      case 'twist': s.vortex += Math.min(1, Math.abs(p.amt) * o.w * 150); break;
      case 'translate':
        if (Math.abs(p.vy) > Math.abs(p.vx) && Math.abs(p.vy) > 0.08) s.rain += 1.2;
        else if (Math.abs(p.vx) > 0.08) s.terrain += 1;
        break;
      case 'noise': s.ink += 0.3; s.rain += 0.2; break;
      case 'mirror': s.mirror += 1.6; break;
      case 'tile': s.mirror += 1.8; break;
      case 'kaleido': s.mirror += 2.2; break;
      case 'polar': s.vortex += 1; break;
      default: s.flame += 0.5 * o.w; // variations
    }
  }
  return s;
}

export interface Classification {
  primary: Species;
  secondary: Species | null;
  label: string; // "vortex × flame" style
}

export function classify(g: Genome): Classification {
  const s = speciesScores(g);
  const ranked = SPECIES.slice().sort((a, b) => s[b] - s[a]);
  const primary = ranked[0];
  const second = ranked[1];
  const secondary = s[second] >= 1.5 && s[second] >= 0.6 * s[primary] ? second : null;
  const short = (k: Species) => SPECIES_LABEL[k].split('/')[0];
  return { primary, secondary, label: secondary ? `${short(primary)} × ${short(secondary)}` : SPECIES_LABEL[primary] };
}

export function energyOf(g: Genome): Energy {
  return (g.energy[0] + g.energy[1]) / 2 >= 0.5 ? 'energetic' : 'calm';
}

// -------------------------------------------------------------- cost

/** Estimated GPU milliseconds per frame at 2560x1440 on Apple Silicon (rough model). */
export function estimateCost(g: Genome): number {
  let ms = 1.3; // feedback + composite + bloom + exposure + final
  for (const o of g.chain) ms += o.op === 'noise' ? 0.35 : isVarOp(o.op) ? 0.12 : 0.05;
  if (g.carrier.kind === 'fluid') ms += 1.1;
  if (g.carrier.kind === 'flow') ms += 0.35;
  if (g.carrier.p.blur > 0) ms += 0.2;
  const field: Partial<Record<EmitterKind, number>> = {
    spectrum: 0.25, stars: 1.1, ink: 0.25, wire: 1.5, plasma: 1.8, aurora: 0.9, blobs: 0.5, edge: 0.2, tiles: 0.5, horizon: 0.4, orb: 0.3, wave: 0.15,
  };
  for (const e of g.emitters) {
    if (e.kind === 'particles') ms += 0.25 + (e.p.count / 65536) * 0.6;
    else if (e.kind === 'flame') ms += (e.p.count / 262144) * e.p.rounds * 1.3;
    else ms += field[e.kind] ?? 0.3;
  }
  return ms;
}
export const COST_BUDGET_MS = 8;

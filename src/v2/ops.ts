// Genetic operators: random genes, crossover, mutation. Every operator returns
// a repaired (valid, in-range) genome.

import {
  BASIC_KINDS, CARRIER_KINDS, CARRIER_SCHEMA, COLOR_SCHEMA, COST_BUDGET_MS, DRAW_OPS, EMITTER_SCHEMAS, FLAME_VARIATIONS,
  FOLD_OPS, GEOMETRY_KINDS, MAX_CHAIN, MAX_CHILD_EMITTERS, MAX_DRAW, MAX_FLAT, MAX_REACTIONS, MAX_XFORMS, MOTION_OPS,
  OP_KINDS, OP_SCHEMAS, SCHEMES, SIGNALS, XFORM_DRIFT, XFORM_SPIN,
  cloneGenome, defaultParams, estimateCost, flatEmitters, isDrawOp, isSdfKind, isVarOp, reactable, repair, repairXform,
  schemaFor, stageFree,
  type EmitterGene, type EmitterKind, type FlameVar, type FlameXformGene, type GeneGroup, type Genome, type OpGene,
  type OpKind, type ParamSpec, type Params, type ReactionGene, type Schema,
} from './genome';

export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = <T>(rng: Rng, xs: readonly T[]): T => xs[Math.floor(rng() * xs.length) % xs.length];
const randInt = (rng: Rng, lo: number, hi: number) => lo + Math.floor(rng() * (hi - lo + 1));
function gauss(rng: Rng): number {
  const u = Math.max(1e-9, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ------------------------------------------------------------ params

function randomValue(s: ParamSpec, rng: Rng): number {
  if (s.choices) return pick(rng, s.choices);
  let v: number;
  if (s.log && s.min > 0) v = s.min * Math.pow(s.max / s.min, rng());
  else v = s.min + (s.max - s.min) * rng();
  return s.int ? Math.round(v) : v;
}

function jitterValue(v: number, s: ParamSpec, rng: Rng, amt: number): number {
  if (s.choices) {
    const i = Math.max(0, s.choices.indexOf(v));
    const j = Math.min(s.choices.length - 1, Math.max(0, i + (rng() < 0.5 ? -1 : 1)));
    return s.choices[j];
  }
  if (s.int) return Math.round(Math.min(s.max, Math.max(s.min, v + (rng() < 0.5 ? -1 : 1) * Math.max(1, Math.round((s.max - s.min) * 0.15 * amt)))));
  let out: number;
  if (s.log && s.min > 0) out = Math.exp(Math.log(Math.max(v, s.min)) + gauss(rng) * 0.35 * amt * Math.log(s.max / s.min) * 0.3);
  else out = v + gauss(rng) * (s.max - s.min) * 0.12 * amt;
  return Math.min(s.max, Math.max(s.min, out));
}

function jitterParams(p: Params, schema: Schema, rng: Rng, amt: number, frac = 0.4): void {
  const keys = Object.keys(schema);
  let touched = false;
  for (const k of keys) {
    if (rng() < frac) {
      p[k] = jitterValue(p[k] ?? schema[k].def, schema[k], rng, amt);
      touched = true;
    }
  }
  if (!touched && keys.length) {
    const k = pick(rng, keys);
    p[k] = jitterValue(p[k] ?? schema[k].def, schema[k], rng, amt);
  }
}

/** Blend two param sets: each key picked from either parent or interpolated. */
function mixParams(a: Params, b: Params, schema: Schema, rng: Rng): Params {
  const out: Params = {};
  for (const k of Object.keys(schema)) {
    const s = schema[k];
    const x = a[k] ?? s.def;
    const y = b[k] ?? s.def;
    const r = rng();
    if (s.choices || r < 0.5) out[k] = r < 0.25 || (s.choices && r < 0.5) ? x : y;
    else {
      const t = rng();
      out[k] = s.int ? Math.round(x + (y - x) * t) : x + (y - x) * t;
    }
  }
  return out;
}

function randomParams(schema: Schema, rng: Rng, spread = 0.6): Params {
  const out: Params = {};
  for (const k of Object.keys(schema)) {
    const s = schema[k];
    out[k] = rng() < spread ? randomValue(s, rng) : s.def;
  }
  return out;
}

// -------------------------------------------------------------- genes

const MOTION_WEIGHT = 3; // motion ops are the backbone; bias random picks toward them

export function randomOpKind(rng: Rng): OpKind {
  const r = rng() * (MOTION_OPS.length * MOTION_WEIGHT + FOLD_OPS.length + 12);
  if (r < MOTION_OPS.length * MOTION_WEIGHT) return MOTION_OPS[Math.floor(r / MOTION_WEIGHT)];
  if (r < MOTION_OPS.length * MOTION_WEIGHT + FOLD_OPS.length) return FOLD_OPS[Math.floor(r - MOTION_OPS.length * MOTION_WEIGHT)];
  return `v_${pick(rng, FLAME_VARIATIONS)}` as OpKind;
}

export function randomOp(rng: Rng, kind?: OpKind): OpGene {
  const op = kind ?? randomOpKind(rng);
  const p = randomParams(OP_SCHEMAS[op], rng, 0.5);
  const stage = stageFree(op) ? (isVarOp(op) ? (rng() < 0.6 ? 'warp' : 'view') : rng() < 0.75 ? 'view' : 'warp') : 'warp';
  const w = isVarOp(op) ? (stage === 'view' ? 0.2 + 0.6 * rng() : 0.1 + 0.5 * rng()) : 0.5 + 0.5 * rng();
  return { op, stage, w, p };
}

export function randomXform(rng: Rng): FlameXformGene {
  const ang = rng() * Math.PI * 2;
  const sc = 0.3 + 0.55 * rng();
  const sk = (rng() - 0.5) * 0.4;
  const vars: Partial<Record<FlameVar, number>> = {};
  const n = randInt(rng, 1, 2);
  for (let i = 0; i < n; i++) vars[pick(rng, FLAME_VARIATIONS)] = 0.3 + 0.7 * rng();
  const alt: Partial<Record<FlameVar, number>> | undefined = rng() < 0.5 ? { [pick(rng, FLAME_VARIATIONS)]: 0.4 + 0.6 * rng() } : undefined;
  return repairXform({
    aff: [Math.cos(ang) * sc, -Math.sin(ang) * sc + sk, Math.sin(ang) * sc, Math.cos(ang) * sc, (rng() - 0.5) * 1.2, (rng() - 0.5) * 1.2],
    weight: 0.2 + 0.8 * rng(),
    color: rng(),
    vars,
    alt,
    spin: rng() < 0.6 ? 0 : pick(rng, XFORM_SPIN),
    bass: rng() < 0.5 ? 0 : 0.25 * rng(),
    drift: rng() < 0.5 ? [0, 0] : [0.2 * rng(), 0.2 * rng()],
    pulse: rng() < 0.5 ? 0 : 0.2 * rng(),
  });
}

export function randomEmitter(rng: Rng, kind?: EmitterKind): EmitterGene {
  const k = kind ?? pick(rng, BASIC_KINDS);
  const schema = EMITTER_SCHEMAS[k];
  const p = randomParams(schema, rng, 0.45);
  p.gain = 0.6 + 0.8 * rng();
  if (k === 'particles') p.count = Math.min(p.count, 32768);
  if (k === 'flame') p.count = pick(rng, [65536, 131072, 262144]);
  const layer = k === 'flame' || (k === 'snake' && rng() < 0.9) ? 'fb' : (k === 'plasma' || k === 'blobs' || k === 'horizon' || k === 'orb') && rng() < 0.6 ? 'top' : rng() < 0.15 ? 'top' : 'fb';
  const e: EmitterGene = { kind: k, layer, p };
  if (k === 'flame') e.xforms = Array.from({ length: randInt(rng, 2, 3) }, () => randomXform(rng));
  return e;
}

function randomReaction(g: Genome, rng: Rng): ReactionGene | null {
  const groups: GeneGroup[] = [];
  if (g.chain.length) groups.push('op', 'op');
  if (g.draw?.length) groups.push('dr');
  groups.push('em', 'em', 'em', 'col');
  const group = pick(rng, groups);
  const len = group === 'op' ? g.chain.length : group === 'em' ? flatEmitters(g).length : group === 'dr' ? g.draw!.length : 1;
  const i = randInt(rng, 0, len - 1);
  const keys = reactable(schemaFor(g, group, i)!);
  if (!keys.length) return null;
  return { src: pick(rng, SIGNALS), g: group, i, k: pick(rng, keys), gain: (rng() < 0.75 ? 1 : -1) * (0.15 + 0.6 * rng()) };
}

export function randomGenome(rng: Rng): Genome {
  const chain = Array.from({ length: randInt(rng, 1, 4) }, () => randomOp(rng));
  const emitters: EmitterGene[] = [];
  const n = randInt(rng, 1, 2);
  while (emitters.length < n) {
    const e = randomEmitter(rng);
    if (!emitters.some((x) => x.kind === e.kind)) emitters.push(e);
  }
  const g: Genome = {
    v: 2, chain, emitters,
    carrier: { kind: pick(rng, ['warp', 'warp', 'fluid', 'flow', 'none'] as const), p: randomParams(CARRIER_SCHEMA, rng, 0.5) },
    color: { scheme: pick(rng, SCHEMES), p: randomParams(COLOR_SCHEMA, rng, 0.4) },
    reactions: [],
    energy: [rng() * 0.5, 0.5 + rng() * 0.5],
  };
  const nr = randInt(rng, 1, 3);
  for (let i = 0; i < nr; i++) {
    const r = randomReaction(g, rng);
    if (r) g.reactions.push(r);
  }
  return repair(g);
}

// --------------------------------------------------- structure transfer
// Flame xforms and chain ops share the variation vocabulary, and a chain's
// rotate / zoom / translate ops are affine maps, so structure can cross over
// in both directions.

/** A chain op re-expressed as a flame transform. */
export function opToXform(o: OpGene, rng: Rng): FlameXformGene {
  const x = randomXform(rng);
  if (isVarOp(o.op)) {
    const v = o.op.slice(2) as FlameVar;
    x.vars = { [v]: 0.5 + 0.5 * o.w, linear: 0.3 * rng() };
  } else if (o.op === 'rotate' || o.op === 'swirl' || o.op === 'twist' || o.op === 'kaleido' || o.op === 'polar') {
    const a = (o.op === 'rotate' ? (o.p.lock || 0.0625) : 0.1) * Math.PI * 2 * (rng() < 0.5 ? 1 : -1) + rng() * 0.4;
    const s = 0.55 + 0.3 * rng();
    x.aff = [Math.cos(a) * s, -Math.sin(a) * s, Math.sin(a) * s, Math.cos(a) * s, x.aff[4], x.aff[5]];
    x.vars = o.op === 'rotate' ? { linear: 0.7, swirl: 0.3 } : o.op === 'polar' ? { polar: 0.8 } : { swirl: 0.7, linear: 0.3 };
    x.spin = o.op === 'rotate' ? o.p.lock : x.spin;
  } else if (o.op === 'zoom' || o.op === 'translate') {
    const s = 0.4 + 0.3 * rng();
    x.aff = [s, 0, 0, s, (o.p.cx ?? o.p.vx ?? 0) + (rng() - 0.5) * 0.8, (o.p.cy ?? o.p.vy ?? 0) + (rng() - 0.5) * 0.8];
    x.vars = { linear: 0.6, spherical: 0.4 };
  } else if (o.op === 'mirror' || o.op === 'tile') {
    const s = 0.5;
    x.aff = [-s, 0, 0, s, 0.3, 0];
    x.vars = { linear: 1 };
  } else {
    x.vars = { sinusoidal: 0.6, [pick(rng, FLAME_VARIATIONS)]: 0.4 };
  }
  return repairXform(x);
}

/** A flame transform re-expressed as a chain op (its dominant variation). */
export function xformToOp(x: FlameXformGene, rng: Rng): OpGene {
  let best: FlameVar = 'linear';
  let bw = -1;
  for (const [k, w] of Object.entries(x.vars)) {
    if (k !== 'linear' && (w ?? 0) > bw) {
      best = k as FlameVar;
      bw = w ?? 0;
    }
  }
  if (bw < 0) {
    // purely linear: its rotation becomes a rotate op
    const o = randomOp(rng, 'rotate');
    o.p.lock = x.spin || 0.0625;
    return o;
  }
  const o = randomOp(rng, `v_${best}` as OpKind);
  o.w = Math.min(1, 0.15 + 0.4 * (bw ?? 0.5));
  return o;
}

// ------------------------------------------------------------ crossover
// A child is ONE picture: the dominant parent's drawing (its emitter set,
// normally one emitter) shaped by the recessive parent. The recessive parent's
// motion ops bend the drawing itself (the draw-space chain), and its colour,
// carrier, folds and reactions are picked per gene. On top of that:
//   morph   both parents have an emitter of the same kind: one emitter with
//           interpolated params (flames: paired transforms, blended variations);
//   merged  different kinds that can be written as distance fields fuse into
//           one merge emitter (smooth union, music-driven morph, or one shape
//           as the region the other lights up in);
//   layered a rare mutation adds a second, separate emitter (at most two).

export type CrossTag = 'fused' | 'morph' | 'merged' | 'layered';
export interface CrossResult {
  genome: Genome;
  tag: CrossTag;
}
/** Chance that a crossover child gets a second emitter as a separate layer. */
export const LAYER_CHANCE = 0.08;

/** Interpolation weight per gene, peaked at 0.5 (0.1..0.9). */
const midT = (rng: Rng) => 0.5 + (rng() + rng() - 1) * 0.4;

/** Same-kind morph of two param sets: numbers interpolate, switches come from either parent. */
export function morphParams(a: Params, b: Params, schema: Schema, rng: Rng): Params {
  const out: Params = {};
  for (const k of Object.keys(schema)) {
    const s = schema[k];
    const x = a[k] ?? s.def;
    const y = b[k] ?? s.def;
    if (s.choices) {
      out[k] = rng() < 0.5 ? x : y;
      continue;
    }
    const t = midT(rng);
    let v = s.log && s.min > 0 ? Math.exp(Math.log(Math.max(x, s.min)) * (1 - t) + Math.log(Math.max(y, s.min)) * t) : x + (y - x) * t;
    if (s.int) v = Math.round(v);
    out[k] = Math.min(s.max, Math.max(s.min, v));
  }
  return out;
}

function lerpVars(x: Partial<Record<FlameVar, number>> | undefined, y: Partial<Record<FlameVar, number>> | undefined, t: number): Partial<Record<FlameVar, number>> {
  const out: Partial<Record<FlameVar, number>> = {};
  for (const v of FLAME_VARIATIONS) {
    const w = (x?.[v] ?? 0) * (1 - t) + (y?.[v] ?? 0) * t;
    if (w > 0.02) out[v] = Math.min(1, w);
  }
  return out;
}

/** Flame morph: transforms paired by weight rank; unpaired ones join half the time. */
export function morphXforms(xa: FlameXformGene[], xb: FlameXformGene[], rng: Rng): FlameXformGene[] {
  const A = xa.slice().sort((p, q) => q.weight - p.weight);
  const B = xb.slice().sort((p, q) => q.weight - p.weight);
  const out: FlameXformGene[] = [];
  for (let i = 0; i < Math.max(A.length, B.length); i++) {
    const x = A[i];
    const y = B[i];
    if (x && y) {
      const t = midT(rng);
      const L = (p: number, q: number) => p + (q - p) * t;
      const alt = x.alt || y.alt ? lerpVars(x.alt ?? x.vars, y.alt ?? y.vars, t) : undefined;
      const xf: FlameXformGene = {
        aff: x.aff.map((v, k) => L(v, y.aff[k])),
        weight: L(x.weight, y.weight),
        color: L(x.color, y.color),
        vars: lerpVars(x.vars, y.vars, t),
        spin: rng() < 0.5 ? x.spin : y.spin,
        bass: L(x.bass, y.bass),
        drift: [L(x.drift[0], y.drift[0]), L(x.drift[1], y.drift[1])],
        pulse: L(x.pulse, y.pulse),
      };
      if (alt && Object.keys(alt).length) xf.alt = alt;
      out.push(xf);
    } else if (rng() < 0.5) out.push((x ?? y)!);
  }
  if (!out.length) out.push(A[0] ?? B[0] ?? randomXform(rng));
  return out.slice(0, MAX_XFORMS).map(repairXform);
}

/** Two emitters of the same kind become one. */
export function morphEmitter(a: EmitterGene, b: EmitterGene, rng: Rng): EmitterGene {
  const e: EmitterGene = { kind: a.kind, layer: rng() < 0.5 ? a.layer : b.layer, p: morphParams(a.p, b.p, EMITTER_SCHEMAS[a.kind], rng) };
  if (a.kind === 'flame') e.xforms = morphXforms(a.xforms ?? [], b.xforms ?? [], rng);
  if (a.kind === 'merge' && a.parts && b.parts) {
    e.parts = a.parts.map((x, i) => {
      const y = b.parts!.find((q) => q.kind === x.kind);
      return y ? morphEmitter(x, y, rng) : cloneEmitter(b.parts![i] && rng() < 0.3 && !a.parts!.some((q) => q.kind === b.parts![i].kind) ? b.parts![i] : x);
    });
  }
  return e;
}

const cloneEmitter = (e: EmitterGene): EmitterGene => JSON.parse(JSON.stringify(e)) as EmitterGene;

/** Where a shape sits (its x/y params), or null when it has no position. */
function shapeCenter(e: EmitterGene): [number, number] | null {
  if (e.kind === 'orb' || e.kind === 'wave') return [e.p.x, e.p.y];
  if (e.kind === 'spectrum' && (e.p.mode === 1 || e.p.mode === 2)) return [e.p.x, e.p.y];
  return null;
}
function setCenter(e: EmitterGene, c: [number, number]): void {
  const put = (k: string, v: number) => {
    const s = EMITTER_SCHEMAS[e.kind][k];
    if (s) e.p[k] = Math.min(s.max, Math.max(s.min, v));
  };
  if (e.kind === 'orb' || e.kind === 'wave' || (e.kind === 'spectrum' && (e.p.mode === 1 || e.p.mode === 2))) {
    put('x', c[0]);
    put('y', c[1]);
  }
  if (e.kind === 'aurora') put('y', c[1]);
}

/** Rough radius of a shape, for fitting two shapes into one figure. */
function shapeSize(e: EmitterGene): number {
  switch (e.kind) {
    case 'orb': return e.p.radius * (1 + 0.8 * e.p.arms);
    case 'wire': return e.p.scale * 0.5;
    case 'wave': return e.p.shape === 0 ? 0.3 : e.p.radius;
    case 'spectrum': return e.p.radius + e.p.len * 0.5;
    case 'blobs': return e.p.size * 2 + 0.1;
    case 'ink': return e.p.radius;
    default: return 0.12;
  }
}

/**
 * Moves and sizes two shapes so they overlap as one figure: a shared centre
 * (the origin when one of them is drawn around it: wireframes, orbiting ink,
 * blobs), ink sources orbiting just outside the other shape, blobs drawn in,
 * a small moon enlarged, a ribbon or bar baseline running through / under it.
 */
function fitTogether(a: EmitterGene, b: EmitterGene, dom: EmitterGene): void {
  const originBound = (x: EmitterGene) => x.kind === 'wire' || x.kind === 'ink' || x.kind === 'blobs';
  const other = dom === a ? b : a;
  const anchor: [number, number] = originBound(a) || originBound(b) ? [0, 0] : shapeCenter(dom) ?? shapeCenter(other) ?? [0, 0];
  const put = (x: EmitterGene, k: string, v: number) => {
    const sp = EMITTER_SCHEMAS[x.kind][k];
    if (sp) x.p[k] = Math.min(sp.max, Math.max(sp.min, v));
  };
  for (const [x, y] of [[a, b], [b, a]] as const) {
    if (shapeCenter(x)) setCenter(x, anchor);
    if (x.kind === 'orb') put(x, 'radius', Math.max(x.p.radius, 0.08));
    if (x.kind === 'blobs') put(x, 'spread', Math.min(x.p.spread, 0.35));
    if (x.kind === 'ink') {
      // Sources circle the other shape (instrument moves kept, but on a short leash).
      put(x, 'orbit', Math.max(0.75, x.p.orbit));
      put(x, 'follow', 0);
      put(x, 'row', 0);
      put(x, 'xs', Math.min(x.p.xs, 0.3));
      put(x, 'radius', shapeSize(y) * 1.3 + 0.04);
    }
    if (x.kind === 'aurora') put(x, 'y', anchor[1] - 0.05);
    if (x.kind === 'spectrum' && (x.p.mode === 0 || x.p.mode === 3)) put(x, 'y', anchor[1] - shapeSize(y) * 0.8);
  }
}

/** Can these two different kinds fuse into one merge emitter? */
export function canMerge(a: EmitterGene, b: EmitterGene): boolean {
  if (a.kind === b.kind || a.kind === 'merge' || b.kind === 'merge') return false;
  return isSdfKind(a.kind) || isSdfKind(b.kind);
}

/**
 * One merge emitter from two different kinds. Both distance fields: smooth
 * union, music-driven morph or mask; one field: it becomes the region the
 * other kind lights up in. The shapes are moved onto a shared centre so the
 * result is one figure, not two pictures side by side.
 */
export function makeMerge(e: EmitterGene, f: EmitterGene, rng: Rng, mode?: number): EmitterGene | null {
  if (!canMerge(e, f)) return null;
  let a = cloneEmitter(e);
  let b = cloneEmitter(f);
  const both = isSdfKind(a.kind) && isSdfKind(b.kind);
  let m: number;
  if (mode !== undefined) m = both ? mode : 2;
  else if (both) {
    const r = rng();
    m = r < 0.5 ? 0 : r < 0.82 ? 1 : 2;
  } else m = 2;
  if (m === 2 && !isSdfKind(a.kind)) [a, b] = [b, a];
  a.layer = 'fb';
  b.layer = 'fb';
  delete (a as Partial<EmitterGene>).parts;
  delete (b as Partial<EmitterGene>).parts;
  fitTogether(a, b, e.kind === a.kind ? a : b);
  // Particles are born around the origin and flames centre on it: the region shape moves there.
  if (m === 2 && GEOMETRY_KINDS.includes(b.kind)) setCenter(a, [0, 0]);
  if (m !== 2 || !GEOMETRY_KINDS.includes(b.kind)) {
    for (const x of [a, b]) if (x.kind === 'wave' && (x.p.shape === 0 || x.p.shape === 3 || x.p.shape === 5)) {
      // A full-width line or an open curve reads as a separate stroke: close it into a ring or spiral.
      if (x.p.shape !== 0 || rng() < 0.6) x.p.shape = rng() < 0.7 ? 1 : 2;
    }
  }
  const solid = (k: EmitterKind) => k === 'orb' || k === 'blobs' || k === 'spectrum' || k === 'ink';
  const p = defaultParams(EMITTER_SCHEMAS.merge);
  p.mode = m;
  p.gain = Math.min(3, Math.max(0.05, ((e.p.gain ?? 1) + (f.p.gain ?? 1)) / 2));
  p.hue = e.p.hue ?? 0;
  p.k = 0.05 + 0.12 * rng();
  p.t = 0.3 + 0.4 * rng();
  p.drive = m === 1 ? pick(rng, [1, 1, 2, 3, 4, 5]) : pick(rng, [0, 1, 2, 4]);
  p.depth = m === 1 ? 0.6 + 0.4 * rng() : 0.3 + 0.5 * rng();
  p.rate = pick(rng, [4, 8, 8, 16]);
  p.line = 0.5 + 0.5 * rng();
  p.fill = solid(a.kind) || (m !== 2 && solid(b.kind)) ? 0.45 + 0.45 * rng() : 0.15 + 0.3 * rng();
  p.width = 0.8 + 1.0 * rng();
  // Lines have no inside: the consumer lights up along them.
  p.inside = solid(a.kind) ? 1 : 0;
  return { kind: 'merge', layer: GEOMETRY_KINDS.includes(b.kind) ? 'fb' : e.layer, p, parts: [a, b] };
}

/** Emitters whose shaders apply the draw-space chain (flames and particles are V1 geometry). */
const drawReceptive = (e: EmitterGene) => e.kind !== 'flame' && e.kind !== 'particles';

/** A chain op re-expressed as a draw-space op (motion rates keep their params). */
function toDrawOp(o: OpGene): OpGene | null {
  if (!isDrawOp(o.op)) return null;
  return { op: o.op, stage: 'warp', w: o.w, p: { ...o.p } };
}

/**
 * Adds a second emitter as its own layer (the other parent's, or a random
 * kind), keeping kinds unique and the child cap of two. Returns false when
 * there is no room.
 */
export function addLayer(g: Genome, rng: Rng, from?: EmitterGene[], strict = false): boolean {
  if (g.emitters.length >= MAX_CHILD_EMITTERS) return false;
  const flat = flatEmitters(g);
  if (flat.length >= MAX_FLAT) return false;
  const used = new Set(flat.map((e) => e.kind));
  const cands = (from ?? []).filter((e) => e.kind !== 'merge' && !used.has(e.kind));
  if (strict && !cands.length) return false;
  const e = cands.length ? cloneEmitter(pick(rng, cands)) : randomEmitter(rng, pick(rng, BASIC_KINDS.filter((k) => !used.has(k))));
  g.emitters.push(e);
  return true;
}

/** Remapped reactions, 'op' targets looked up in the chain and then in the draw chain (or the reverse). */
function remapReactions(parent: Genome, child: Genome, opsToDraw = false): ReactionGene[] {
  const out: ReactionGene[] = [];
  const pFlat = flatEmitters(parent);
  const cFlat = flatEmitters(child);
  const draw = child.draw ?? [];
  for (const r of parent.reactions) {
    if (r.g === 'op' || r.g === 'dr') {
      const op = (r.g === 'op' ? parent.chain[r.i] : parent.draw?.[r.i])?.op;
      const inChain = child.chain.findIndex((o) => o.op === op);
      const inDraw = draw.findIndex((o) => o.op === op);
      const preferDraw = opsToDraw || r.g === 'dr';
      if (preferDraw && inDraw >= 0) out.push({ ...r, g: 'dr', i: inDraw });
      else if (inChain >= 0) out.push({ ...r, g: 'op', i: inChain });
      else if (inDraw >= 0) out.push({ ...r, g: 'dr', i: inDraw });
    } else if (r.g === 'em') {
      const kind = pFlat[r.i]?.kind;
      const j = cFlat.findIndex((e) => e.kind === kind);
      if (j >= 0) out.push({ ...r, i: j });
    } else out.push({ ...r });
  }
  return out;
}

/** Plain crossover (the genome only). */
export function crossover(aIn: Genome, bIn: Genome, rng: Rng, bias = 0): Genome {
  return crossoverTagged(aIn, bIn, rng, bias).genome;
}

/**
 * Crossover with the outcome tag. bias > 0 makes `a` the likelier dominant
 * parent (pass the fitness difference); the dominant is still random.
 */
export function crossoverTagged(aIn: Genome, bIn: Genome, rng: Rng, bias = 0): CrossResult {
  const a = cloneGenome(aIn);
  const b = cloneGenome(bIn);
  const pa = 0.5 + 0.3 * Math.max(-1, Math.min(1, bias));
  const [D, R] = rng() < pa ? [a, b] : [b, a];

  // Body: the dominant parent's drawing. Two emitters only if it already had two.
  let body = D.emitters.map(cloneEmitter);
  if (body.length > 1) body = rng() < 0.5 ? [body[0], pick(rng, body.slice(1))] : [body[0]];
  const rPlain = flatEmitters(R).filter((e) => e.kind !== 'merge');
  let tag: CrossTag = 'fused';

  // Same-kind morph.
  body = body.map((e) => {
    if (e.kind === 'merge') {
      const rm = R.emitters.find((x) => x.kind === 'merge');
      let hit = false;
      const parts = e.parts!.map((x) => {
        const y = rPlain.find((q) => q.kind === x.kind);
        if (y && rng() < 0.9) {
          hit = true;
          return morphEmitter(x, y, rng);
        }
        return x;
      });
      if (rm || hit) tag = 'morph';
      return { ...e, p: rm ? morphParams(e.p, rm.p, EMITTER_SCHEMAS.merge, rng) : e.p, parts };
    }
    const y = rPlain.find((q) => q.kind === e.kind);
    if (y && rng() < 0.9) {
      tag = 'morph';
      return morphEmitter(e, y, rng);
    }
    return e;
  });

  // Cross-kind fusion through distance fields.
  const rDrawable = [...(R.draw ?? []), ...R.chain.filter((o) => isDrawOp(o.op))];
  const plainBody = body.length === 1 && body[0].kind !== 'merge' ? body[0] : null;
  const preMerge = body;
  if (tag === 'fused' && plainBody) {
    const cands = rPlain.filter((f) => canMerge(plainBody, f));
    if (cands.length && rng() < (rDrawable.length ? 0.45 : 0.9)) {
      const m = makeMerge(plainBody, pick(rng, cands), rng);
      if (m) {
        body = [m];
        tag = 'merged';
      }
    }
  }

  const build = (emitters: EmitterGene[]): Genome => {
    const chain = D.chain.map((o) => ({ ...o, p: { ...o.p } }));
    const draw: OpGene[] = (D.draw ?? []).map((o) => ({ ...o, p: { ...o.p } }));
    const receptive = emitters.some(drawReceptive);
    const hasFlame = flatEmitters({ emitters }).find((e) => e.kind === 'flame');
    // The recessive parent's motion bends the drawing (draw chain), or, for
    // flames and particles, enters the flame transforms / the carrier.
    const shaping = rDrawable.slice().sort(() => rng() - 0.5).slice(0, 1 + (rng() < 0.45 ? 1 : 0));
    const before = chain.length + draw.length;
    for (const o of shaping) {
      const d = toDrawOp(o);
      if (d && receptive && draw.length < MAX_DRAW && !draw.some((x) => x.op === d.op)) draw.push(d);
      else if (hasFlame && rng() < 0.7) {
        const xs = hasFlame.xforms!;
        if (xs.length < MAX_XFORMS && rng() < 0.6) xs.push(opToXform(o, rng));
        else if (isVarOp(o.op)) pick(rng, xs).vars[o.op.slice(2) as FlameVar] = 0.3 + 0.5 * o.w;
        else xs[randInt(rng, 0, xs.length - 1)] = opToXform(o, rng);
      } else if (!chain.some((x) => x.op === o.op)) chain.push({ ...o, p: { ...o.p } });
    }
    // A recessive flame shapes through one of its transforms as a variation.
    const rFlame = rPlain.find((e) => e.kind === 'flame');
    if (rFlame && !hasFlame && rng() < 0.6) {
      const o = xformToOp(pick(rng, rFlame.xforms!), rng);
      if (receptive && draw.length < MAX_DRAW && isDrawOp(o.op)) draw.push({ ...o, stage: 'warp', w: Math.min(1, 0.25 + 0.5 * rng()) });
      else chain.splice(randInt(rng, 0, chain.length), 0, o);
    }
    // Composite folds and, sometimes, the recessive carrier motion; nearly
    // always when nothing else of the recessive parent's structure got in.
    const weak = tag === 'fused' && chain.length + draw.length === before;
    for (const o of R.chain) {
      if (o.stage === 'view' && !chain.some((x) => x.op === o.op) && rng() < (weak ? 0.9 : 0.45)) chain.push({ ...o, p: { ...o.p } });
    }
    const motion = R.chain.filter((o) => o.stage === 'warp' && !chain.some((x) => x.op === o.op));
    if (motion.length && rng() < (weak ? 0.6 : 0.25)) chain.push({ ...pick(rng, motion) });
    while (chain.length > MAX_CHAIN) chain.splice(randInt(rng, D.chain.length ? Math.min(D.chain.length, chain.length - 1) : 0, chain.length - 1), 1);

    // Carrier: per gene. Accumulating bodies keep a feedback carrier.
    const src = rng() < 0.6 ? D : R;
    let carrier = { kind: src.carrier.kind, p: D.carrier.kind === R.carrier.kind ? mixParams(D.carrier.p, R.carrier.p, CARRIER_SCHEMA, rng) : { ...src.carrier.p } };
    if (carrier.kind === 'none' && D.carrier.kind !== 'none') carrier = { kind: D.carrier.kind, p: { ...D.carrier.p } };
    if (src === R && D.carrier.kind !== R.carrier.kind && rng() < 0.5) carrier.p.halfLife = D.carrier.p.halfLife;
    const mg = emitters.find((e) => e.kind === 'merge');
    if (mg && GEOMETRY_KINDS.includes(mg.parts![1].kind) && carrier.kind === 'none') carrier = { kind: 'warp', p: { ...carrier.p, halfLife: Math.max(0.3, carrier.p.halfLife) } };

    const color = { scheme: rng() < 0.5 ? D.color.scheme : R.color.scheme, p: mixParams(D.color.p, R.color.p, COLOR_SCHEMA, rng) };
    const flameParent = [D, R].find((x) => flatEmitters(x).some((e) => e.kind === 'flame'));
    if (hasFlame && flameParent && rng() < 0.75) color.p.tonemap = flameParent.color.p.tonemap;
    else if (!hasFlame) color.p.tonemap = D.color.p.tonemap;

    const child: Genome = { v: 2, chain, emitters, carrier, color, reactions: [], energy: [0, 1] };
    if (draw.length) child.draw = draw.slice(0, MAX_DRAW);
    const t = rng();
    child.energy = [D.energy[0] + (R.energy[0] - D.energy[0]) * t, D.energy[1] + (R.energy[1] - D.energy[1]) * t];
    const seen = new Set<string>();
    const reactions = [...remapReactions(D, child), ...remapReactions(R, child, true)].sort(() => rng() - 0.5).filter((r) => {
      const k = `${r.g}${r.i}.${r.k}.${r.src}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    if (mg && rng() < 0.6) {
      // The fused shape breathes with the music: the blend radius or the morph follows a stem.
      const i = flatEmitters(child).indexOf(mg);
      reactions.unshift({ src: pick(rng, ['bass', 'drums', 'loud', 'surge'] as const), g: 'em', i, k: mg.p.mode === 1 ? 't' : 'k', gain: 0.3 + 0.4 * rng() });
    }
    child.reactions = reactions.slice(0, MAX_REACTIONS);
    return repair(child);
  };

  let child = build(body);
  if (tag === 'merged' && (estimateCost(child) > COST_BUDGET_MS * 0.95 || !child.emitters.some((e) => e.kind === 'merge'))) {
    // Over the GPU budget: keep the dominant drawing, shaped only.
    tag = 'fused';
    child = build(preMerge);
  }
  // Rare: the other parent's drawing as a separate second layer.
  if (rng() < LAYER_CHANCE && addLayer(child, rng, rPlain, true)) {
    const layered = repair(child);
    if (estimateCost(layered) <= COST_BUDGET_MS * 0.95 && layered.emitters.length > 1) return { genome: layered, tag: 'layered' };
    child = build(tag === 'merged' ? body : preMerge);
  }
  return { genome: child, tag };
}

// ------------------------------------------------------------- mutation

type Mutator = (g: Genome, rng: Rng, amt: number) => boolean;

const MUTATORS: [number, string, Mutator][] = [
  [5, 'jitter-op', (g, rng, amt) => {
    if (!g.chain.length) return false;
    const o = pick(rng, g.chain);
    jitterParams(o.p, OP_SCHEMAS[o.op], rng, amt);
    if (rng() < 0.3) o.w = Math.min(1, Math.max(0, o.w + gauss(rng) * 0.15 * amt));
    return true;
  }],
  [5, 'jitter-emitter', (g, rng, amt) => {
    const e = pick(rng, flatEmitters(g));
    jitterParams(e.p, EMITTER_SCHEMAS[e.kind], rng, amt);
    return true;
  }],
  [2, 'jitter-carrier', (g, rng, amt) => (jitterParams(g.carrier.p, CARRIER_SCHEMA, rng, amt, 0.3), true)],
  [3, 'jitter-color', (g, rng, amt) => {
    jitterParams(g.color.p, COLOR_SCHEMA, rng, amt, 0.3);
    if (rng() < 0.3) g.color.p.hue = (g.color.p.hue + gauss(rng) * 0.15 + 1) % 1;
    return true;
  }],
  [2, 'insert-op', (g, rng) => {
    if (g.chain.length >= MAX_CHAIN) return false;
    g.chain.splice(randInt(rng, 0, g.chain.length), 0, randomOp(rng));
    return true;
  }],
  [1.5, 'delete-op', (g, rng) => {
    if (!g.chain.length) return false;
    g.chain.splice(randInt(rng, 0, g.chain.length - 1), 1);
    return true;
  }],
  [1, 'reorder-op', (g, rng) => {
    if (g.chain.length < 2) return false;
    const i = randInt(rng, 0, g.chain.length - 1);
    let j = randInt(rng, 0, g.chain.length - 2);
    if (j >= i) j++;
    [g.chain[i], g.chain[j]] = [g.chain[j], g.chain[i]];
    return true;
  }],
  [1.5, 'replace-op', (g, rng) => {
    if (!g.chain.length) return false;
    g.chain[randInt(rng, 0, g.chain.length - 1)] = randomOp(rng);
    return true;
  }],
  [0.8, 'toggle-stage', (g, rng) => {
    const free = g.chain.filter((o) => stageFree(o.op));
    if (!free.length) return false;
    const o = pick(rng, free);
    o.stage = o.stage === 'warp' ? 'view' : 'warp';
    return true;
  }],
  [1.5, 'swap-emitter', (g, rng) => {
    const i = randInt(rng, 0, g.emitters.length - 1);
    const used = flatEmitters(g).map((e) => e.kind);
    const e = randomEmitter(rng, pick(rng, BASIC_KINDS.filter((k) => !used.includes(k))));
    e.p.gain = g.emitters[i].p.gain;
    g.emitters[i] = e;
    return true;
  }],
  // Layering is rare: a second, separate emitter, never more than two.
  [0.6, 'add-layer', (g, rng) => addLayer(g, rng)],
  [1, 'remove-emitter', (g, rng) => {
    if (g.emitters.length <= 1) return false;
    g.emitters.splice(randInt(rng, 0, g.emitters.length - 1), 1);
    return true;
  }],
  [0.6, 'flip-layer', (g, rng) => {
    const e = pick(rng, g.emitters.filter((x) => x.kind !== 'flame'));
    if (!e) return false;
    e.layer = e.layer === 'fb' ? 'top' : 'fb';
    return true;
  }],
  [1, 'change-carrier', (g, rng) => {
    const kinds = CARRIER_KINDS.filter((k) => k !== g.carrier.kind);
    g.carrier.kind = pick(rng, kinds);
    return true;
  }],
  [0.6, 'change-scheme', (g, rng) => ((g.color.scheme = pick(rng, SCHEMES)), true)],
  [2, 'flame-variation', (g, rng) => {
    const f = flatEmitters(g).find((e) => e.kind === 'flame');
    if (f) {
      const x = pick(rng, f.xforms!);
      const keys = Object.keys(x.vars) as FlameVar[];
      const drop = pick(rng, keys);
      const w = x.vars[drop] ?? 0.5;
      if (keys.length > 1 && rng() < 0.6) delete x.vars[drop];
      x.vars[pick(rng, FLAME_VARIATIONS)] = Math.min(1, w * (0.6 + 0.8 * rng()));
      return true;
    }
    const vo = g.chain.filter((o) => isVarOp(o.op));
    if (!vo.length) return false;
    const o = pick(rng, vo);
    o.op = `v_${pick(rng, FLAME_VARIATIONS)}` as OpKind;
    return true;
  }],
  [1.5, 'flame-affine', (g, rng, amt) => {
    const f = flatEmitters(g).find((e) => e.kind === 'flame');
    if (!f) return false;
    const x = pick(rng, f.xforms!);
    x.aff = x.aff.map((v) => v + gauss(rng) * 0.12 * amt);
    if (rng() < 0.3) x.weight = Math.min(1, Math.max(0.05, x.weight + gauss(rng) * 0.2));
    if (rng() < 0.3) x.color = (x.color + gauss(rng) * 0.2 + 1) % 1;
    return true;
  }],
  [1.2, 'flame-motion', (g, rng) => {
    // The music-coupled motion of a flame: morph target, drift, beat kick, bass breathing.
    const f = flatEmitters(g).find((e) => e.kind === 'flame');
    if (!f) return false;
    const x = pick(rng, f.xforms!);
    const r = rng();
    if (r < 0.3) x.alt = rng() < 0.2 ? undefined : { [pick(rng, FLAME_VARIATIONS)]: 0.4 + 0.6 * rng() };
    else if (r < 0.5) x.drift = [Math.min(XFORM_DRIFT, Math.max(0, x.drift[0] + gauss(rng) * 0.08)), Math.min(XFORM_DRIFT, Math.max(0, x.drift[1] + gauss(rng) * 0.08))];
    else if (r < 0.7) x.pulse = Math.min(0.3, Math.max(0, x.pulse + gauss(rng) * 0.08));
    else if (r < 0.85) f.p.flow = Math.min(2, Math.max(0, f.p.flow + gauss(rng) * 0.25));
    else f.p.breathe = Math.min(0.4, Math.max(0, f.p.breathe + gauss(rng) * 0.1));
    return true;
  }],
  [0.8, 'flame-xform-count', (g, rng) => {
    const f = flatEmitters(g).find((e) => e.kind === 'flame');
    if (!f) return false;
    const xs = f.xforms!;
    if (xs.length < MAX_XFORMS && (xs.length <= 1 || rng() < 0.5)) {
      // New transform, sometimes borrowed from the chain.
      xs.push(g.chain.length && rng() < 0.4 ? opToXform(pick(rng, g.chain), rng) : randomXform(rng));
    } else if (xs.length > 1) xs.splice(randInt(rng, 0, xs.length - 1), 1);
    return true;
  }],
  [0.6, 'xform-to-chain', (g, rng) => {
    const f = flatEmitters(g).find((e) => e.kind === 'flame');
    if (!f || g.chain.length >= MAX_CHAIN) return false;
    g.chain.splice(randInt(rng, 0, g.chain.length), 0, xformToOp(pick(rng, f.xforms!), rng));
    return true;
  }],
  // Draw-space shaping: bend the drawing itself.
  [2, 'jitter-draw', (g, rng, amt) => {
    if (!g.draw?.length) return false;
    const o = pick(rng, g.draw);
    jitterParams(o.p, OP_SCHEMAS[o.op], rng, amt);
    if (rng() < 0.4) o.w = Math.min(1, Math.max(0, o.w + gauss(rng) * 0.2 * amt));
    return true;
  }],
  [0.7, 'insert-draw', (g, rng) => {
    if ((g.draw?.length ?? 0) >= MAX_DRAW || !flatEmitters(g).some((e) => e.kind !== 'flame' && e.kind !== 'particles')) return false;
    // Half the time the drawing borrows one of the carrier's own motions.
    const own = g.chain.filter((o) => isDrawOp(o.op));
    const src = own.length && rng() < 0.5 ? { ...pick(rng, own) } : randomOp(rng, pick(rng, DRAW_OPS));
    (g.draw ??= []).push({ op: src.op, stage: 'warp', w: src.w, p: { ...src.p } });
    return true;
  }],
  [0.6, 'delete-draw', (g, rng) => {
    if (!g.draw?.length) return false;
    g.draw.splice(randInt(rng, 0, g.draw.length - 1), 1);
    if (!g.draw.length) delete g.draw;
    return true;
  }],
  // Merges stay breedable: blend mode, music driver, one side swapped.
  [1.2, 'merge-mode', (g, rng) => {
    const m = g.emitters.find((e) => e.kind === 'merge');
    if (!m) return false;
    const [a, b] = m.parts!;
    const modes = isSdfKind(a.kind) && isSdfKind(b.kind) ? [0, 1, 2].filter((x) => x !== m.p.mode) : [2];
    if (m.p.mode === 2 && isSdfKind(b.kind) && rng() < 0.3) m.parts = [b, a];
    else if (modes.length) m.p.mode = pick(rng, modes);
    else return false;
    if (m.p.mode === 2) m.p.inside = rng() < 0.5 ? 0 : 1;
    return true;
  }],
  [1.2, 'merge-drive', (g, rng, amt) => {
    const m = g.emitters.find((e) => e.kind === 'merge');
    if (!m) return false;
    const r = rng();
    if (r < 0.4) m.p.drive = pick(rng, EMITTER_SCHEMAS.merge.drive.choices!);
    else if (r < 0.6) m.p.rate = pick(rng, EMITTER_SCHEMAS.merge.rate.choices!);
    else jitterParams(m.p, { depth: EMITTER_SCHEMAS.merge.depth, t: EMITTER_SCHEMAS.merge.t, k: EMITTER_SCHEMAS.merge.k }, rng, amt, 0.6);
    return true;
  }],
  [0.8, 'merge-side', (g, rng) => {
    const m = g.emitters.find((e) => e.kind === 'merge');
    if (!m) return false;
    const i = rng() < 0.5 ? 0 : 1;
    const used = flatEmitters(g).map((e) => e.kind);
    const needSdf = i === 0 || m.p.mode !== 2;
    const kinds = BASIC_KINDS.filter((k) => !used.includes(k) && (!needSdf || isSdfKind(k)) && (!GEOMETRY_KINDS.includes(k) || m.layer === 'fb'));
    if (!kinds.length) return false;
    const e = randomEmitter(rng, pick(rng, kinds));
    e.layer = 'fb';
    e.p.gain = m.parts![i].p.gain;
    m.parts![i] = e;
    return true;
  }],
  [1.5, 'reaction', (g, rng) => {
    const r = rng();
    if (r < 0.4 && g.reactions.length < MAX_REACTIONS) {
      const x = randomReaction(g, rng);
      if (!x) return false;
      g.reactions.push(x);
    } else if (r < 0.6 && g.reactions.length) g.reactions.splice(randInt(rng, 0, g.reactions.length - 1), 1);
    else if (g.reactions.length) {
      const x = pick(rng, g.reactions);
      if (rng() < 0.5) x.gain = Math.min(1, Math.max(-1, x.gain + gauss(rng) * 0.3));
      else x.src = pick(rng, SIGNALS);
    } else return false;
    return true;
  }],
  [0.8, 'energy', (g, rng) => {
    const d = gauss(rng) * 0.12;
    g.energy = [g.energy[0] + d, g.energy[1] + d + gauss(rng) * 0.05];
    return true;
  }],
];
export const MUTATION_NAMES = MUTATORS.map((m) => m[1]);
/** Mutations that change what a preset is, damped for light (post-crossover) mutation. */
const STRUCTURAL = new Set(['swap-emitter', 'remove-emitter', 'add-layer', 'change-carrier', 'replace-op', 'flip-layer', 'merge-side']);

const BODY = new Set(['swap-emitter', 'remove-emitter', 'add-layer', 'merge-side']);

/** 1-3 random mutations (more for larger `amt`). Returns a repaired copy. */
export function mutate(gIn: Genome, rng: Rng, amt = 1, log?: string[], keepBody = false): Genome {
  const g = cloneGenome(gIn);
  const n = Math.max(1, Math.min(4, Math.round(1 + rng() * 2 * amt)));
  // keepBody: polish a crossover child without replacing or adding what it draws.
  const weight = (x: (typeof MUTATORS)[number]) => (keepBody && BODY.has(x[1]) ? 0 : amt < 0.6 && STRUCTURAL.has(x[1]) ? x[0] * 0.25 : x[0]);
  const total = MUTATORS.reduce((s, m) => s + weight(m), 0);
  let done = 0;
  for (let tries = 0; done < n && tries < 30; tries++) {
    let r = rng() * total;
    let m = MUTATORS[0];
    for (const x of MUTATORS) {
      r -= weight(x);
      if (r <= 0) {
        m = x;
        break;
      }
    }
    if (m[2](g, rng, amt)) {
      done++;
      log?.push(m[1]);
    }
  }
  return repair(g);
}

/** Checks two genomes differ (used to reject no-op children). */
export function sameGenome(a: Genome, b: Genome): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export { OP_KINDS };

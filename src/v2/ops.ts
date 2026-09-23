// Genetic operators: random genes, crossover, mutation. Every operator returns
// a repaired (valid, in-range) genome.

import {
  CARRIER_KINDS, CARRIER_SCHEMA, COLOR_SCHEMA, COST_BUDGET_MS, DEFORM_KINDS, DRAW_OPS, EMIT_KINDS, FLAME_VARIATIONS,
  FOLD_OPS, FUSE_SCHEMA, LOCI, LOCUS_KINDS, MATERIAL_KINDS, MAX_CHAIN, MAX_CHILD_BODIES, MAX_DRAW, MAX_REACTIONS,
  MAX_XFORMS, MOTION_KINDS, FEEL_SCHEMAS, REACTION_SCHEMA, MOTION_OPS, OP_KINDS, OP_SCHEMAS, PLACE_KINDS, SCHEMES, SHAPE_CLASS, SHAPE_KINDS,
  SHAPE_SCHEMAS, SIGNALS, UNIQUE_SHAPES, XFORM_DRIFT, XFORM_SPIN, BODY_GROUPS,
  cloneBody, cloneGenome, countKey, defaultParams, estimateCost, isDrawOp, isFoldPlace, isVarOp, locusSchema, reactable,
  repair, repairBody, repairXform, schemaFor, sdfCapable, stageFree,
  type BodyGene, type FlameVar, type FlameXformGene, type Gene, type GeneGroup, type Genome, type Locus, type OpGene,
  type OpKind, type ParamSpec, type Params, type ReactionGene, type Schema, type ShapeGene, type ShapeKind,
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

function jitterParams(p: Params, schema: Schema, rng: Rng, amt: number, frac = 0.4): boolean {
  const keys = Object.keys(schema);
  if (!keys.length) return false;
  let touched = false;
  for (const k of keys) {
    if (rng() < frac) {
      p[k] = jitterValue(p[k] ?? schema[k].def, schema[k], rng, amt);
      touched = true;
    }
  }
  if (!touched) {
    const k = pick(rng, keys);
    p[k] = jitterValue(p[k] ?? schema[k].def, schema[k], rng, amt);
  }
  return true;
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

/** Random pick weights per locus kind (the favoured kinds read well on most shapes). */
const KIND_WEIGHTS: Partial<Record<Locus, Record<string, number>>> = {
  shape: { dot: 3, polygon: 1.5, star: 1.2, segment: 0.8, solid: 1.5, bars: 1, curve: 2, plasma: 0.8, aurora: 0.8, terrain: 0.5, edge: 0.6, flame: 1 },
  place: { point: 3, orbit: 2, walker: 1.5, stations: 1.5, row: 0.7, float: 1.2, outline: 1.2, grid: 1.2, ring: 1.5, mirror: 1 },
  motion: { none: 1.5, spin: 2, sway: 1.5, bob: 1.2, drift: 0.8, circle: 1, hits: 1, pulse: 1 },
  deform: { none: 3, arms: 1.2, wobble: 1.2, noise: 0.8, twist: 0.8 },
  material: { line: 2, fill: 1.5, glow: 2, dots: 0.8, textured: 0.8, chrome: 0.8 },
  emit: { none: 1.2, trail: 3, cover: 1, dye: 0.8, sparks: 0.8 },
};

function weightedKind(locus: Locus, rng: Rng, exclude: string[] = []): string {
  const kinds = LOCUS_KINDS[locus].filter((k) => !exclude.includes(k));
  const w = KIND_WEIGHTS[locus] ?? {};
  let total = 0;
  for (const k of kinds) total += w[k] ?? 1;
  let r = rng() * total;
  for (const k of kinds) {
    r -= w[k] ?? 1;
    if (r <= 0) return k;
  }
  return kinds[kinds.length - 1];
}

/** A random gene for one locus. */
export function randomGene(locus: Locus, rng: Rng, kind?: string): Gene {
  const k = kind ?? weightedKind(locus, rng);
  const p = randomParams(locusSchema(locus, k), rng, 0.45);
  if (locus === 'material') p.gain = 0.6 + 0.8 * rng();
  if (locus === 'emit' && k === 'sparks') p.count = Math.min(p.count, 32768);
  if (locus === 'shape' && k === 'flame') p.count = pick(rng, [65536, 131072, 262144]);
  const g: Gene = { kind: k, p };
  if (locus === 'shape' && k === 'flame') (g as ShapeGene).xforms = Array.from({ length: randInt(rng, 2, 3) }, () => randomXform(rng));
  return g;
}

/**
 * Makes a body's loci read well together after a locus changed hands: a dot
 * drawn with a solid material gets a visible size, shapes shrink to fit a grid
 * cell or a crowd of copies, costly shapes on a grid are evaluated once per cell.
 */
export function fitLoci(b: BodyGene): void {
  const sh = b.shape;
  const size = shapeSize(sh);
  const solidMat = b.material.kind !== 'glow';
  if (sh.kind === 'dot' && solidMat && sh.p.r < 0.012) sh.p.r = b.place.kind === 'grid' ? 0.3 / b.place.p.scale : 0.04;
  if (sh.kind === 'dot' && !solidMat && b.material.p.width < 0.004 && b.place.kind !== 'grid') b.material.p.width = 0.015;
  if (b.place.kind === 'grid') {
    const cell = 1 / b.place.p.scale;
    if (size > cell * 0.48) setShapeSize(sh, cell * 0.45);
    if (sh.kind === 'solid' || sh.kind === 'bars' || sh.kind === 'curve') b.place.p.jitter = Math.min(b.place.p.jitter, 0.2);
    if (sh.kind === 'dot' && b.material.kind === 'glow') b.material.p.width = Math.min(b.material.p.width, cell * 0.12);
  } else if (!isFoldPlace(b.place.kind)) {
    const ck = countKey(b.place.kind);
    const n = ck ? b.place.p[ck] : 1;
    if (n > 2 && size > 0.2) setShapeSize(sh, 0.14);
    if (b.place.kind === 'walker' && size > 0.1) setShapeSize(sh, 0.07);
  }
  if (b.place.kind === 'ring' && size > b.place.p.radius * 0.9 && b.place.p.radius > 0.05) setShapeSize(sh, Math.max(0.03, b.place.p.radius * 0.6));
}

/** Rough radius of a shape (scene units). */
export function shapeSize(s: ShapeGene): number {
  switch (s.kind) {
    case 'dot': return s.p.r;
    case 'polygon': case 'star': return s.p.r;
    case 'segment': return s.p.len * 0.5;
    case 'solid': return s.p.size * 0.5;
    case 'bars': return s.p.mode === 1 || s.p.mode === 2 ? s.p.radius + s.p.len * 0.5 : 0.5;
    case 'curve': return s.p.form === 0 ? 0.6 : s.p.radius;
    default: return 0.5;
  }
}
function setShapeSize(s: ShapeGene, r: number): void {
  const put = (k: string, v: number) => {
    const sp = SHAPE_SCHEMAS[s.kind][k];
    if (sp) s.p[k] = Math.min(sp.max, Math.max(sp.min, v));
  };
  switch (s.kind) {
    case 'dot': put('r', r); break;
    case 'polygon': case 'star': put('r', r); break;
    case 'segment': put('len', r * 2); break;
    case 'solid': put('size', r * 2); break;
    case 'bars': if (s.p.mode === 1 || s.p.mode === 2) { put('radius', r * 0.6); put('len', r * 0.6); } break;
    case 'curve': if (s.p.form !== 0) put('radius', r); break;
  }
}

export function randomBody(rng: Rng, shapeKind?: ShapeKind): BodyGene {
  const shape = randomGene('shape', rng, shapeKind) as ShapeGene;
  const b: BodyGene = {
    shape,
    place: randomGene('place', rng) as BodyGene['place'],
    motion: randomGene('motion', rng) as BodyGene['motion'],
    deform: randomGene('deform', rng) as BodyGene['deform'],
    material: randomGene('material', rng) as BodyGene['material'],
    emit: randomGene('emit', rng) as BodyGene['emit'],
    feel: randomGene('feel', rng) as BodyGene['feel'],
  };
  fitLoci(b);
  return repairBody(b);
}

function randomReaction(g: Genome, rng: Rng): ReactionGene | null {
  const groups: GeneGroup[] = [];
  if (g.chain.length) groups.push('op', 'op');
  groups.push('ma', 'ma', 'sh', 'pl', 'de', 'col');
  const group = pick(rng, groups);
  const i = group === 'op' ? randInt(rng, 0, g.chain.length - 1) : group === 'col' ? 0 : randInt(rng, 0, g.bodies.length - 1);
  const keys = reactable(schemaFor(g, group, i) ?? {});
  if (!keys.length) return null;
  return { src: pick(rng, SIGNALS), g: group, i, k: pick(rng, keys), gain: (rng() < 0.75 ? 1 : -1) * (0.15 + 0.6 * rng()), ...randomCurve(rng) };
}

/** A reaction's response curve and clock: mostly smooth (musical), sometimes raw or quantised. */
export function randomCurve(rng: Rng): Pick<ReactionGene, 'atk' | 'rel' | 'thr' | 'q' | 'div'> {
  const r = rng();
  const atk = r < 0.25 ? 0.005 : 0.01 + 0.2 * rng() * rng();
  const rel = r < 0.25 ? 0.005 : 0.08 + 1.2 * rng() * rng();
  return { atk, rel, thr: rng() < 0.6 ? 0 : 0.3 * rng(), q: rng() < 0.15 ? 1 : 0, div: pick(rng, [1, 2, 4, 4, 8]) };
}

export function randomGenome(rng: Rng): Genome {
  const chain = Array.from({ length: randInt(rng, 1, 4) }, () => randomOp(rng));
  const bodies: BodyGene[] = [randomBody(rng)];
  if (rng() < 0.3) bodies.push(randomBody(rng));
  const g: Genome = {
    v: 4, chain, bodies,
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
// Flame xforms and chain ops share the variation vocabulary, and rotate / zoom /
// translate ops are affine maps, so structure can cross over in both directions.

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
    const o = randomOp(rng, 'rotate');
    o.p.lock = x.spin || 0.0625;
    return o;
  }
  const o = randomOp(rng, `v_${best}` as OpKind);
  o.w = Math.min(1, 0.15 + 0.4 * (bw ?? 0.5));
  return o;
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

/** Two genes of the same locus and kind become one with interpolated params. */
function morphGene<G extends Gene>(locus: Locus, a: G, b: G, rng: Rng): G {
  const out = { ...a, p: morphParams(a.p, b.p, locusSchema(locus, a.kind), rng) } as G;
  if (locus === 'shape' && a.kind === 'flame') (out as unknown as ShapeGene).xforms = morphXforms((a as unknown as ShapeGene).xforms ?? [], (b as unknown as ShapeGene).xforms ?? [], rng);
  return out;
}

const cloneGene = <G>(g: G): G => JSON.parse(JSON.stringify(g)) as G;

/** Would this body keep `locus` at the given kind (renderer rules)? */
function keeps(b: BodyGene, locus: Locus, kind: string): boolean {
  return repairBody(b)[locus].kind === kind;
}

// ------------------------------------------------------------ crossover
// A child is ONE picture built from ideas of both parents. Every body locus
// (shape, placement, motion, deformation, material, emission) is a separate
// gene with a homologous slot, so the dominant parent's shape can walk the
// other parent's path, wear its material, leave its kind of trail. On top:
//   morph   both parents have the same shape kind: one shape with interpolated
//           params (flames: paired transforms, blended variations);
//   merged  the other parent's shape fuses into this one through their distance
//           fields (smooth union, music-driven morph, or a region it lights up);
//   fused   the default: one parent's shape, the other parent's ideas shape it;
//   layered a rare mutation adds the other parent's body as a second layer.

export type CrossTag = 'fused' | 'morph' | 'merged' | 'layered';
export interface CrossResult {
  genome: Genome;
  tag: CrossTag;
}
/** Chance that a crossover child gets a second body as a separate layer. */
export const LAYER_CHANCE = 0.08;
/** Chance per locus that the recessive parent's gene is taken (when it fits the child's shape). */
export const LOCUS_SWAP = 0.45;

/** Can this shape fuse the other one in? */
export function canFuse(a: BodyGene, s: ShapeGene): boolean {
  if (a.fuse || s.kind === a.shape.kind || !sdfCapable(s)) return false;
  return !(UNIQUE_SHAPES.includes(s.kind) && s.kind === a.shape.kind);
}

/**
 * The other parent's shape fused into this body (smooth union, music-driven
 * morph, or the region this body lights up inside). Both shapes share the
 * body's placement, so they are always one figure.
 */
export function makeFuse(b: BodyGene, s: ShapeGene, rng: Rng, mode?: number): BodyGene | null {
  if (!canFuse(b, s)) return null;
  const out = cloneBody(b);
  const both = sdfCapable(out.shape);
  let m: number;
  if (mode !== undefined) m = both ? mode : 2;
  else if (both) {
    const r = rng();
    m = r < 0.5 ? 0 : r < 0.82 ? 1 : 2;
  } else m = 2;
  const shape = cloneGene(s);
  // Open curves read as separate strokes: close them into a ring or spiral.
  if (shape.kind === 'curve' && shape.p.form === 0 && rng() < 0.6) shape.p.form = rng() < 0.7 ? 1 : 2;
  if (out.shape.kind === 'curve' && (out.shape.p.form === 3 || out.shape.p.form === 5)) out.shape.p.form = 1;
  // Fit the two into one figure.
  const sa = shapeSize(out.shape);
  const sb = shapeSize(shape);
  if (sb > sa * 2.2 && sa < 0.3) setShapeSize(shape, sa * 1.6);
  if (sb < sa * 0.3) setShapeSize(shape, sa * 0.7);
  const p = defaultParams(FUSE_SCHEMA);
  p.mode = m;
  p.k = 0.05 + 0.12 * rng();
  p.t = 0.3 + 0.4 * rng();
  p.drive = m === 1 ? pick(rng, [1, 1, 2, 3, 4, 5]) : pick(rng, [0, 1, 2, 4]);
  p.depth = m === 1 ? 0.6 + 0.4 * rng() : 0.3 + 0.5 * rng();
  p.rate = pick(rng, [4, 8, 8, 16]);
  p.inside = shape.kind === 'curve' || shape.kind === 'segment' || shape.kind === 'solid' ? 0 : 1;
  out.fuse = { shape, p };
  if (out.material.kind === 'glow' && both && m !== 2) out.material = { kind: 'fill', p: { ...defaultParams(locusSchema('material', 'fill')), gain: Math.min(3, out.material.p.gain * 2), hue: out.material.p.hue, soft: 0.3 } };
  fitLoci(out);
  const r = repairBody(out);
  return r.fuse ? r : null;
}

/** Adds a second body as its own layer (the other parent's, or a random one). */
export function addLayer(g: Genome, rng: Rng, from?: BodyGene[], strict = false): boolean {
  if (g.bodies.length >= MAX_CHILD_BODIES) return false;
  const used = new Set(g.bodies.flatMap((b) => [b.shape.kind, ...(b.fuse ? [b.fuse.shape.kind] : [])]));
  const cands = (from ?? []).filter((b) => !(UNIQUE_SHAPES.includes(b.shape.kind) && used.has(b.shape.kind)) && !g.bodies.some((x) => JSON.stringify(x) === JSON.stringify(b)));
  if (strict && !cands.length) return false;
  const b = cands.length ? cloneBody(pick(rng, cands)) : randomBody(rng, pick(rng, SHAPE_KINDS.filter((k) => !(UNIQUE_SHAPES.includes(k) && used.has(k)))));
  g.bodies.push(b);
  return true;
}

/** Plain crossover (the genome only). */
export function crossover(aIn: Genome, bIn: Genome, rng: Rng, bias = 0): Genome {
  return crossoverTagged(aIn, bIn, rng, bias).genome;
}

/** A body's loci recombined with a homologous body of the other parent. */
function recombineBody(d: BodyGene, r: BodyGene, rng: Rng): { body: BodyGene; morph: boolean; took: Locus[] } {
  const body = cloneBody(d);
  let morph = false;
  const took: Locus[] = [];
  for (const locus of LOCI) {
    const dg = body[locus] as Gene;
    const rg = r[locus] as Gene;
    if (dg.kind === rg.kind) {
      // Same idea in both parents: blend (the shape nearly always, other loci half the time).
      if (rng() < (locus === 'shape' ? 0.9 : 0.5)) {
        (body as unknown as Record<string, Gene>)[locus] = morphGene(locus, dg, rg, rng);
        if (locus === 'shape') morph = true;
      } else if (rng() < 0.5) (body as unknown as Record<string, Gene>)[locus] = cloneGene(rg);
      continue;
    }
    if (locus === 'shape') continue; // the dominant parent's shape is the body
    if (rng() >= LOCUS_SWAP) continue;
    const trial = cloneBody(body);
    (trial as unknown as Record<string, Gene>)[locus] = cloneGene(rg);
    if (locus === 'deform' && d.deform.ops && !trial.deform.ops) trial.deform.ops = d.deform.ops.map((o) => ({ ...o, p: { ...o.p } }));
    if (!keeps(trial, locus, rg.kind)) continue;
    Object.assign(body, trial);
    took.push(locus);
  }
  // A moving parent should not give a frozen child: when the recombined body would sit still (a fixed
  // placement, no motion, no deformation), it keeps the dominant parent's placement and motion.
  if (isStill(body) && !isStill(d)) {
    body.place = cloneGene(d.place);
    body.motion = cloneGene(d.motion);
    for (const l of ['place', 'motion'] as const) {
      const i = took.indexOf(l);
      if (i >= 0) took.splice(i, 1);
    }
  }
  fitLoci(body);
  return { body: repairBody(body), morph, took };
}

/** A body that sits still apart from its own shape's animation. */
function isStill(b: BodyGene): boolean {
  const fixed = b.place.kind === 'point' || b.place.kind === 'mirror' || b.place.kind === 'ring' || b.place.kind === 'grid';
  const cls = SHAPE_CLASS[b.shape.kind];
  return fixed && b.motion.kind === 'none' && b.deform.kind === 'none' && (cls === 'sdf') && b.shape.kind !== 'solid' && b.shape.kind !== 'bars';
}

/** A chain op re-expressed as a deform op (motion rates keep their params). */
function toDrawOp(o: OpGene): OpGene | null {
  if (!isDrawOp(o.op)) return null;
  return { op: o.op, stage: 'warp', w: o.w, p: { ...o.p } };
}

/**
 * Remapped reactions: chain targets by op kind, body-locus targets onto the
 * child's body when that locus is still the same kind there.
 */
function remapReactions(parent: Genome, child: Genome, pBody: number, cBody: number): ReactionGene[] {
  const out: ReactionGene[] = [];
  const groupLocus = Object.fromEntries(Object.entries(BODY_GROUPS).map(([l, g]) => [g, l])) as Record<string, Locus>;
  for (const r of parent.reactions) {
    if (r.g === 'op') {
      const op = parent.chain[r.i]?.op;
      const j = child.chain.findIndex((o) => o.op === op);
      if (j >= 0) out.push({ ...r, i: j });
    } else if (r.g === 'car' || r.g === 'col') out.push({ ...r });
    else if (r.i === pBody) {
      const locus = groupLocus[r.g];
      const pb = parent.bodies[pBody];
      const cb = child.bodies[cBody];
      if (!pb || !cb) continue;
      if (locus && (pb[locus] as Gene).kind === (cb[locus] as Gene).kind) out.push({ ...r, i: cBody });
      else if ((r.g === 'fu' && cb.fuse) || (r.g === 'fs' && cb.fuse && pb.fuse?.shape.kind === cb.fuse.shape.kind)) out.push({ ...r, i: cBody });
    }
  }
  return out;
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

  // The dominant parent's main body, recombined with the recessive parent's homologous body.
  const di = D.bodies.length > 1 && rng() < 0.3 ? 1 : 0;
  const ri = Math.min(di, R.bodies.length - 1);
  const rb = rng() < 0.75 ? R.bodies[ri] : pick(rng, R.bodies);
  const rec = recombineBody(D.bodies[di], rb, rng);
  let body = rec.body;
  let tag: CrossTag = rec.morph ? 'morph' : 'fused';

  // Cross-kind fusion through distance fields.
  const plain = body;
  const rShapes = R.bodies.flatMap((x) => [x.shape, ...(x.fuse ? [x.fuse.shape] : [])]);
  if (tag === 'fused' && !body.fuse) {
    const cands = rShapes.filter((s) => canFuse(body, s));
    if (cands.length && rng() < (rec.took.length >= 2 ? 0.3 : 0.55)) {
      const m = makeFuse(body, pick(rng, cands), rng);
      if (m) {
        body = m;
        tag = 'merged';
      }
    }
  }

  const build = (main: BodyGene): Genome => {
    const bodies = [main];
    const chain = D.chain.map((o) => ({ ...o, p: { ...o.p } }));
    // The recessive parent's motion ops bend the drawing (deform ops), or enter the flame transforms.
    const rMotion = R.chain.filter((o) => isDrawOp(o.op));
    const shaping = rMotion.slice().sort(() => rng() - 0.5).slice(0, rng() < 0.5 ? 1 : 0);
    for (const o of shaping) {
      const d = toDrawOp(o);
      const ops = main.deform.ops ?? [];
      if (main.shape.kind === 'flame' && main.shape.xforms) {
        const xs = main.shape.xforms;
        if (xs.length < MAX_XFORMS && rng() < 0.6) xs.push(opToXform(o, rng));
        else if (isVarOp(o.op)) pick(rng, xs).vars[o.op.slice(2) as FlameVar] = 0.3 + 0.5 * o.w;
        else xs[randInt(rng, 0, xs.length - 1)] = opToXform(o, rng);
      } else if (d && ops.length < MAX_DRAW && !ops.some((x) => x.op === d.op)) main.deform.ops = [...ops, d];
    }
    // Folds of the displayed picture and, sometimes, the recessive carrier motion.
    const weak = tag === 'fused' && rec.took.length === 0;
    for (const o of R.chain) {
      if (o.stage === 'view' && !chain.some((x) => x.op === o.op) && rng() < (weak ? 0.8 : 0.35)) chain.push({ ...o, p: { ...o.p } });
    }
    const motion = R.chain.filter((o) => o.stage === 'warp' && !chain.some((x) => x.op === o.op));
    if (motion.length && rng() < (weak ? 0.5 : 0.2)) chain.push({ ...pick(rng, motion) });
    while (chain.length > MAX_CHAIN) chain.splice(randInt(rng, D.chain.length ? Math.min(D.chain.length, chain.length - 1) : 0, chain.length - 1), 1);

    // Carrier: per gene. Trails need a feedback carrier.
    const src = rng() < 0.6 ? D : R;
    let carrier = { kind: src.carrier.kind, p: D.carrier.kind === R.carrier.kind ? mixParams(D.carrier.p, R.carrier.p, CARRIER_SCHEMA, rng) : { ...src.carrier.p } };
    if (carrier.kind === 'none' && (D.carrier.kind !== 'none' || main.emit.kind !== 'none')) carrier = { kind: D.carrier.kind === 'none' ? R.carrier.kind : D.carrier.kind, p: { ...(D.carrier.kind === 'none' ? R.carrier.p : D.carrier.p) } };
    if (carrier.kind === 'none' && main.emit.kind !== 'none') carrier = { kind: 'warp', p: { ...carrier.p, halfLife: Math.max(0.3, carrier.p.halfLife) } };
    if (src === R && D.carrier.kind !== R.carrier.kind && rng() < 0.5) carrier.p.halfLife = D.carrier.p.halfLife;
    // Dye wants the fluid.
    if (main.emit.kind === 'dye' && carrier.kind !== 'fluid' && rng() < 0.7) carrier = { kind: 'fluid', p: { ...carrier.p, floor: Math.max(carrier.p.floor, 1), halfLife: Math.max(carrier.p.halfLife, 1) } };

    const color = { scheme: rng() < 0.5 ? D.color.scheme : R.color.scheme, p: mixParams(D.color.p, R.color.p, COLOR_SCHEMA, rng) };
    const hasFlame = main.shape.kind === 'flame';
    const flameParent = [D, R].find((x) => x.bodies.some((q) => q.shape.kind === 'flame'));
    if (hasFlame && flameParent && rng() < 0.75) color.p.tonemap = flameParent.color.p.tonemap;
    else if (!hasFlame) color.p.tonemap = D.color.p.tonemap;

    const child: Genome = { v: 4, chain, bodies, carrier, color, reactions: [], energy: [0, 1] };
    const t = rng();
    child.energy = [D.energy[0] + (R.energy[0] - D.energy[0]) * t, D.energy[1] + (R.energy[1] - D.energy[1]) * t];
    const seen = new Set<string>();
    const reactions = [...remapReactions(D, child, di, 0), ...remapReactions(R, child, R.bodies.indexOf(rb), 0)].sort(() => rng() - 0.5).filter((r) => {
      const k = `${r.g}${r.i}.${r.k}.${r.src}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    if (main.fuse && rng() < 0.6) {
      // The fused shape breathes with the music: the blend radius or the morph follows a stem.
      reactions.unshift({ src: pick(rng, ['bass', 'drums', 'loud', 'surge'] as const), g: 'fu', i: 0, k: main.fuse.p.mode === 1 ? 't' : 'k', gain: 0.3 + 0.4 * rng(), atk: 0.03, rel: 0.4, thr: 0, q: 0, div: 4 });
    }
    child.reactions = reactions.slice(0, MAX_REACTIONS);
    return repair(child);
  };

  let child = build(body);
  if (tag === 'merged' && (estimateCost(child) > COST_BUDGET_MS * 0.95 || !child.bodies[0].fuse)) {
    tag = rec.morph ? 'morph' : 'fused';
    child = build(plain);
  }
  // Rare: the other parent's body as a separate second layer.
  if (rng() < LAYER_CHANCE && addLayer(child, rng, R.bodies, true)) {
    const layered = repair(child);
    if (estimateCost(layered) <= COST_BUDGET_MS * 0.95 && layered.bodies.length > 1) return { genome: layered, tag: 'layered' };
    child = build(tag === 'merged' ? body : plain);
  }
  return { genome: child, tag };
}

// ------------------------------------------------------------- mutation

type Mutator = (g: Genome, rng: Rng, amt: number) => boolean;

/** Replace one locus of a random body with a new kind (keeping what still applies). */
function swapLocus(locus: Locus): Mutator {
  return (g, rng) => {
    const bi = randInt(rng, 0, g.bodies.length - 1);
    const b = g.bodies[bi];
    const cur = (b[locus] as Gene).kind;
    for (let tries = 0; tries < 6; tries++) {
      const gene = randomGene(locus, rng, weightedKind(locus, rng, [cur]));
      if (locus === 'material') gene.p.gain = b.material.p.gain;
      const trial = cloneBody(b);
      (trial as unknown as Record<string, Gene>)[locus] = gene;
      if (locus === 'shape' && UNIQUE_SHAPES.includes(gene.kind as ShapeKind) && g.bodies.some((x, j) => j !== bi && (x.shape.kind === gene.kind || x.fuse?.shape.kind === gene.kind))) continue;
      if (!keeps(trial, locus, gene.kind)) continue;
      fitLoci(trial);
      g.bodies[bi] = repairBody(trial);
      return true;
    }
    return false;
  };
}

const MUTATORS: [number, string, Mutator][] = [
  [5, 'jitter-op', (g, rng, amt) => {
    if (!g.chain.length) return false;
    const o = pick(rng, g.chain);
    jitterParams(o.p, OP_SCHEMAS[o.op], rng, amt);
    if (rng() < 0.3) o.w = Math.min(1, Math.max(0, o.w + gauss(rng) * 0.15 * amt));
    return true;
  }],
  [6, 'jitter-locus', (g, rng, amt) => {
    const b = pick(rng, g.bodies);
    const locus = pick(rng, LOCI);
    const gene = b[locus] as Gene;
    return jitterParams(gene.p, locusSchema(locus, gene.kind), rng, amt);
  }],
  [2, 'jitter-carrier', (g, rng, amt) => jitterParams(g.carrier.p, CARRIER_SCHEMA, rng, amt, 0.3)],
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
  // Sub-gene structural mutations: one idea replaced, the rest of the body kept.
  [1, 'swap-shape', swapLocus('shape')],
  [1.4, 'swap-place', swapLocus('place')],
  [1.4, 'swap-motion', swapLocus('motion')],
  [1.1, 'swap-deform', swapLocus('deform')],
  [1.2, 'swap-material', swapLocus('material')],
  [1, 'swap-emit', swapLocus('emit')],
  // Feel: how the body responds (flow / step), its clock, a reaction's source or curve.
  [0.8, 'swap-feel', swapLocus('feel')],
  [1.2, 'change-clock', (g, rng) => {
    const b = pick(rng, g.bodies);
    const divs = FEEL_SCHEMAS.flow.div.choices!;
    if (rng() < 0.25) b.feel.p.lock = b.feel.p.lock > 0.5 ? 0 : 1;
    else {
      const i = divs.indexOf(b.feel.p.div);
      b.feel.p.div = divs[Math.min(divs.length - 1, Math.max(0, i + (rng() < 0.5 ? -1 : 1)))];
    }
    return true;
  }],
  [1, 'rewire-reaction', (g, rng) => {
    if (!g.reactions.length) return false;
    const r = pick(rng, g.reactions);
    r.src = pick(rng, SIGNALS.filter((x) => x !== r.src));
    return true;
  }],
  [1, 'reaction-curve', (g, rng, amt) => {
    if (!g.reactions.length) return false;
    const r = pick(rng, g.reactions) as unknown as Params;
    jitterParams(r, { atk: REACTION_SCHEMA.atk, rel: REACTION_SCHEMA.rel, thr: REACTION_SCHEMA.thr, q: REACTION_SCHEMA.q, div: REACTION_SCHEMA.div }, rng, amt, 0.5);
    return true;
  }],
  // Layering is rare: a second, separate body, never more than two.
  [0.6, 'add-layer', (g, rng) => addLayer(g, rng)],
  [1, 'remove-body', (g, rng) => {
    if (g.bodies.length <= 1) return false;
    g.bodies.splice(randInt(rng, 0, g.bodies.length - 1), 1);
    return true;
  }],
  [1, 'change-carrier', (g, rng) => {
    const kinds = CARRIER_KINDS.filter((k) => k !== g.carrier.kind);
    g.carrier.kind = pick(rng, kinds);
    return true;
  }],
  [0.6, 'change-scheme', (g, rng) => ((g.color.scheme = pick(rng, SCHEMES)), true)],
  [2, 'flame-variation', (g, rng) => {
    const f = g.bodies.find((b) => b.shape.kind === 'flame')?.shape;
    if (f?.xforms) {
      const x = pick(rng, f.xforms);
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
    const f = g.bodies.find((b) => b.shape.kind === 'flame')?.shape;
    if (!f?.xforms) return false;
    const x = pick(rng, f.xforms);
    x.aff = x.aff.map((v) => v + gauss(rng) * 0.12 * amt);
    if (rng() < 0.3) x.weight = Math.min(1, Math.max(0.05, x.weight + gauss(rng) * 0.2));
    if (rng() < 0.3) x.color = (x.color + gauss(rng) * 0.2 + 1) % 1;
    return true;
  }],
  [1.2, 'flame-motion', (g, rng) => {
    const f = g.bodies.find((b) => b.shape.kind === 'flame')?.shape;
    if (!f?.xforms) return false;
    const x = pick(rng, f.xforms);
    const r = rng();
    if (r < 0.3) x.alt = rng() < 0.2 ? undefined : { [pick(rng, FLAME_VARIATIONS)]: 0.4 + 0.6 * rng() };
    else if (r < 0.5) x.drift = [Math.min(XFORM_DRIFT, Math.max(0, x.drift[0] + gauss(rng) * 0.08)), Math.min(XFORM_DRIFT, Math.max(0, x.drift[1] + gauss(rng) * 0.08))];
    else if (r < 0.7) x.pulse = Math.min(0.3, Math.max(0, x.pulse + gauss(rng) * 0.08));
    else if (r < 0.85) f.p.flow = Math.min(2, Math.max(0, f.p.flow + gauss(rng) * 0.25));
    else f.p.breathe = Math.min(0.4, Math.max(0, f.p.breathe + gauss(rng) * 0.1));
    return true;
  }],
  [0.8, 'flame-xform-count', (g, rng) => {
    const f = g.bodies.find((b) => b.shape.kind === 'flame')?.shape;
    if (!f?.xforms) return false;
    const xs = f.xforms;
    if (xs.length < MAX_XFORMS && (xs.length <= 1 || rng() < 0.5)) xs.push(g.chain.length && rng() < 0.4 ? opToXform(pick(rng, g.chain), rng) : randomXform(rng));
    else if (xs.length > 1) xs.splice(randInt(rng, 0, xs.length - 1), 1);
    return true;
  }],
  [0.6, 'xform-to-chain', (g, rng) => {
    const f = g.bodies.find((b) => b.shape.kind === 'flame')?.shape;
    if (!f?.xforms || g.chain.length >= MAX_CHAIN) return false;
    g.chain.splice(randInt(rng, 0, g.chain.length), 0, xformToOp(pick(rng, f.xforms), rng));
    return true;
  }],
  // Deform ops: bend the drawing itself.
  [1.5, 'jitter-deform-op', (g, rng, amt) => {
    const b = pick(rng, g.bodies);
    if (!b.deform.ops?.length) return false;
    const o = pick(rng, b.deform.ops);
    jitterParams(o.p, OP_SCHEMAS[o.op], rng, amt);
    if (rng() < 0.4) o.w = Math.min(1, Math.max(0, o.w + gauss(rng) * 0.2 * amt));
    return true;
  }],
  [0.7, 'insert-deform-op', (g, rng) => {
    const b = pick(rng, g.bodies);
    if ((b.deform.ops?.length ?? 0) >= MAX_DRAW || b.shape.kind === 'flame') return false;
    const own = g.chain.filter((o) => isDrawOp(o.op));
    const src = own.length && rng() < 0.5 ? { ...pick(rng, own) } : randomOp(rng, pick(rng, DRAW_OPS));
    (b.deform.ops ??= []).push({ op: src.op, stage: 'warp', w: src.w, p: { ...src.p } });
    return true;
  }],
  [0.6, 'delete-deform-op', (g, rng) => {
    const b = pick(rng, g.bodies);
    if (!b.deform.ops?.length) return false;
    b.deform.ops.splice(randInt(rng, 0, b.deform.ops.length - 1), 1);
    if (!b.deform.ops.length) delete b.deform.ops;
    return true;
  }],
  // Fused shapes stay breedable: blend mode, music driver, the other shape swapped.
  [1, 'fuse-mode', (g, rng) => {
    const b = g.bodies.find((x) => x.fuse);
    if (!b?.fuse) return false;
    const modes = sdfCapable(b.shape) ? [0, 1, 2].filter((x) => x !== b.fuse!.p.mode) : [];
    if (!modes.length) return false;
    b.fuse.p.mode = pick(rng, modes);
    if (b.fuse.p.mode === 2) b.fuse.p.inside = rng() < 0.5 ? 0 : 1;
    return true;
  }],
  [1, 'fuse-drive', (g, rng, amt) => {
    const b = g.bodies.find((x) => x.fuse);
    if (!b?.fuse) return false;
    const r = rng();
    if (r < 0.4) b.fuse.p.drive = pick(rng, FUSE_SCHEMA.drive.choices!);
    else if (r < 0.6) b.fuse.p.rate = pick(rng, FUSE_SCHEMA.rate.choices!);
    else jitterParams(b.fuse.p, { depth: FUSE_SCHEMA.depth, t: FUSE_SCHEMA.t, k: FUSE_SCHEMA.k }, rng, amt, 0.6);
    return true;
  }],
  [0.7, 'fuse-shape', (g, rng) => {
    const bi = randInt(rng, 0, g.bodies.length - 1);
    const b = g.bodies[bi];
    if (b.fuse && rng() < 0.3) {
      delete b.fuse;
      return true;
    }
    const used = g.bodies.flatMap((x) => [x.shape.kind, ...(x.fuse ? [x.fuse.shape.kind] : [])]);
    const kinds = SHAPE_KINDS.filter((k) => SHAPE_CLASS[k] === 'sdf' && k !== b.shape.kind && !(UNIQUE_SHAPES.includes(k) && used.includes(k)));
    const s = randomGene('shape', rng, pick(rng, kinds)) as ShapeGene;
    const base = cloneBody(b);
    delete base.fuse;
    const m = makeFuse(base, s, rng, b.fuse?.p.mode);
    if (!m) return false;
    g.bodies[bi] = m;
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
const STRUCTURAL = new Set(['swap-shape', 'remove-body', 'add-layer', 'change-carrier', 'replace-op', 'swap-emit', 'fuse-shape']);
/** Mutations that replace or add what a crossover child draws. */
const BODY = new Set(['swap-shape', 'remove-body', 'add-layer', 'fuse-shape']);

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

export { OP_KINDS, PLACE_KINDS, MOTION_KINDS, DEFORM_KINDS, MATERIAL_KINDS, EMIT_KINDS };

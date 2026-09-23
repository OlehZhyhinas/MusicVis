// Genetic operators: random genes, crossover, mutation. Every operator returns
// a repaired (valid, in-range) genome.

import {
  CARRIER_KINDS, CARRIER_SCHEMA, COLOR_SCHEMA, EMITTER_KINDS, EMITTER_SCHEMAS, FLAME_VARIATIONS, FOLD_OPS, MAX_CHAIN,
  MAX_EMITTERS, MAX_REACTIONS, MAX_XFORMS, MOTION_OPS, OP_KINDS, OP_SCHEMAS, SCHEMES, SIGNALS, XFORM_DRIFT, XFORM_SPIN,
  cloneGenome, isVarOp, reactable, repair, repairXform, schemaFor, stageFree,
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
  const k = kind ?? pick(rng, EMITTER_KINDS);
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
  groups.push('em', 'em', 'em', 'col');
  const group = pick(rng, groups);
  const len = group === 'op' ? g.chain.length : group === 'em' ? g.emitters.length : 1;
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
    v: 1, chain, emitters,
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

export function crossover(aIn: Genome, bIn: Genome, rng: Rng): Genome {
  const a = cloneGenome(aIn);
  const b = cloneGenome(bIn);
  const [p, q] = rng() < 0.5 ? [a, b] : [b, a]; // p = "primary" parent

  // Chain: splice at random cut points, or uniform pick per slot.
  let chain: OpGene[];
  if (rng() < 0.65 || !p.chain.length || !q.chain.length) {
    const cutP = randInt(rng, 0, p.chain.length);
    const cutQ = randInt(rng, 0, q.chain.length);
    chain = [...p.chain.slice(0, cutP), ...q.chain.slice(cutQ)];
  } else {
    const n = Math.max(p.chain.length, q.chain.length);
    chain = [];
    for (let i = 0; i < n; i++) {
      const x = p.chain[i];
      const y = q.chain[i];
      if (x && y && x.op === y.op) chain.push({ ...x, w: (x.w + y.w) / 2, p: mixParams(x.p, y.p, OP_SCHEMAS[x.op], rng) });
      else {
        const c = rng() < 0.5 ? x ?? y : y ?? x;
        if (c) chain.push(c);
      }
    }
  }
  if (!chain.length && (p.chain.length || q.chain.length)) chain.push(pick(rng, [...p.chain, ...q.chain]));
  while (chain.length > MAX_CHAIN) chain.splice(randInt(rng, 0, chain.length - 1), 1);

  // Emitters: union (same kind -> blended), then keep up to 3 with at least one from each parent.
  const byKind = new Map<EmitterKind, { e: EmitterGene; from: number }>();
  for (const [src, list] of [[0, p.emitters], [1, q.emitters]] as const) {
    for (const e of list) {
      const cur = byKind.get(e.kind);
      if (!cur) byKind.set(e.kind, { e, from: src });
      else {
        const merged: EmitterGene = { kind: e.kind, layer: rng() < 0.5 ? cur.e.layer : e.layer, p: mixParams(cur.e.p, e.p, EMITTER_SCHEMAS[e.kind], rng) };
        if (e.kind === 'flame') merged.xforms = crossXforms(cur.e.xforms ?? [], e.xforms ?? [], rng);
        byKind.set(e.kind, { e: merged, from: 2 });
      }
    }
  }
  const pool = [...byKind.values()];
  const want = Math.min(MAX_EMITTERS, pool.length, randInt(rng, 1, Math.min(MAX_EMITTERS, pool.length)) + (rng() < 0.5 ? 1 : 0));
  const chosen: EmitterGene[] = [];
  const fromP = pool.filter((x) => x.from !== 1);
  const fromQ = pool.filter((x) => x.from !== 0);
  if (fromP.length) chosen.push(pick(rng, fromP).e);
  if (fromQ.length && chosen.length < want && rng() < 0.85) {
    const rest = fromQ.filter((x) => !chosen.includes(x.e));
    if (rest.length) chosen.push(pick(rng, rest).e);
  }
  for (const x of pool.sort(() => rng() - 0.5)) {
    if (chosen.length >= Math.max(want, 1)) break;
    if (!chosen.includes(x.e)) chosen.push(x.e);
  }
  const emitters = chosen.slice(0, MAX_EMITTERS);

  // Flame structure transfer.
  const flame = emitters.find((e) => e.kind === 'flame');
  const flameParent = p.emitters.some((e) => e.kind === 'flame') ? p : q.emitters.some((e) => e.kind === 'flame') ? q : null;
  const other = flameParent === p ? q : p;
  if (flame && flameParent && other !== flameParent && rng() < 0.6) {
    // Another parent's chain op becomes a flame transform, or tints an existing one's variations.
    const src = other.chain.length ? pick(rng, other.chain) : null;
    if (src) {
      const xs = flame.xforms!;
      if (xs.length < MAX_XFORMS && rng() < 0.6) xs.push(opToXform(src, rng));
      else if (isVarOp(src.op)) {
        const x = pick(rng, xs);
        x.vars[src.op.slice(2) as FlameVar] = 0.3 + 0.5 * src.w;
      } else xs[randInt(rng, 0, xs.length - 1)] = opToXform(src, rng);
    }
  }
  if (flameParent && !flame && rng() < 0.5 && chain.length < MAX_CHAIN) {
    // The flame itself is dropped: one of its transforms lives on in the chain.
    const fx = flameParent.emitters.find((e) => e.kind === 'flame')!.xforms!;
    chain.splice(randInt(rng, 0, chain.length), 0, xformToOp(pick(rng, fx), rng));
  }

  // Carrier: from one parent, params blended when both share its kind.
  const carrierSrc = rng() < 0.5 ? p.carrier : q.carrier;
  const carrier = { kind: carrierSrc.kind, p: p.carrier.kind === q.carrier.kind ? mixParams(p.carrier.p, q.carrier.p, CARRIER_SCHEMA, rng) : { ...carrierSrc.p } };
  // Keep the carrier's persistence from the parent whose emitters dominate, most of the time.
  if (rng() < 0.5) carrier.p.halfLife = Math.sqrt(p.carrier.p.halfLife * q.carrier.p.halfLife);

  // Colour and reactions per gene from either parent.
  const color = { scheme: rng() < 0.5 ? p.color.scheme : q.color.scheme, p: mixParams(p.color.p, q.color.p, COLOR_SCHEMA, rng) };
  if (emitters.some((e) => e.kind === 'flame')) {
    const fp = flameParent!.color.p.tonemap;
    if (rng() < 0.75) color.p.tonemap = fp;
  }

  const child: Genome = { v: 1, chain, emitters, carrier, color, reactions: [], energy: [0, 1] };
  const t = rng();
  child.energy = [p.energy[0] + (q.energy[0] - p.energy[0]) * t, p.energy[1] + (q.energy[1] - p.energy[1]) * t];
  // Reactions: re-anchor each to the child's genes by name where possible.
  const reactions = [...remapReactions(p, child), ...remapReactions(q, child)].sort(() => rng() - 0.5);
  child.reactions = reactions.slice(0, randInt(rng, 1, MAX_REACTIONS));
  return repair(child);
}

function crossXforms(a: FlameXformGene[], b: FlameXformGene[], rng: Rng): FlameXformGene[] {
  const out: FlameXformGene[] = [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i];
    const y = b[i];
    if (x && y) {
      if (rng() < 0.5) out.push(rng() < 0.5 ? x : y);
      else {
        const t = rng();
        const vars: Partial<Record<FlameVar, number>> = {};
        for (const v of FLAME_VARIATIONS) {
          const w = (x.vars[v] ?? 0) * (1 - t) + (y.vars[v] ?? 0) * t;
          if (w > 0.02) vars[v] = w;
        }
        out.push({
          aff: x.aff.map((v, k) => v + (y.aff[k] - v) * t),
          weight: x.weight + (y.weight - x.weight) * t,
          color: rng() < 0.5 ? x.color : y.color,
          vars,
          alt: rng() < 0.5 ? x.alt : y.alt,
          spin: rng() < 0.5 ? x.spin : y.spin,
          bass: (x.bass + y.bass) / 2,
          drift: [(x.drift[0] + y.drift[0]) / 2, (x.drift[1] + y.drift[1]) / 2],
          pulse: rng() < 0.5 ? x.pulse : y.pulse,
        });
      }
    } else if (rng() < 0.6) out.push((x ?? y)!);
  }
  if (!out.length) out.push(a[0] ?? b[0] ?? randomXform(rng));
  return out.slice(0, MAX_XFORMS).map(repairXform);
}

function remapReactions(parent: Genome, child: Genome): ReactionGene[] {
  const out: ReactionGene[] = [];
  for (const r of parent.reactions) {
    if (r.g === 'op') {
      const op = parent.chain[r.i]?.op;
      const j = child.chain.findIndex((o) => o.op === op);
      if (j >= 0) out.push({ ...r, i: j });
    } else if (r.g === 'em') {
      const kind = parent.emitters[r.i]?.kind;
      const j = child.emitters.findIndex((e) => e.kind === kind);
      if (j >= 0) out.push({ ...r, i: j });
    } else out.push({ ...r });
  }
  return out;
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
    const e = pick(rng, g.emitters);
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
    const kinds = EMITTER_KINDS.filter((k) => !g.emitters.some((e) => e.kind === k));
    const e = randomEmitter(rng, pick(rng, kinds));
    e.p.gain = g.emitters[i].p.gain;
    g.emitters[i] = e;
    return true;
  }],
  [1, 'add-emitter', (g, rng) => {
    if (g.emitters.length >= MAX_EMITTERS) return false;
    const kinds = EMITTER_KINDS.filter((k) => !g.emitters.some((e) => e.kind === k));
    g.emitters.push(randomEmitter(rng, pick(rng, kinds)));
    return true;
  }],
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
    const f = g.emitters.find((e) => e.kind === 'flame');
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
    const f = g.emitters.find((e) => e.kind === 'flame');
    if (!f) return false;
    const x = pick(rng, f.xforms!);
    x.aff = x.aff.map((v) => v + gauss(rng) * 0.12 * amt);
    if (rng() < 0.3) x.weight = Math.min(1, Math.max(0.05, x.weight + gauss(rng) * 0.2));
    if (rng() < 0.3) x.color = (x.color + gauss(rng) * 0.2 + 1) % 1;
    return true;
  }],
  [1.2, 'flame-motion', (g, rng) => {
    // The music-coupled motion of a flame: morph target, drift, beat kick, bass breathing.
    const f = g.emitters.find((e) => e.kind === 'flame');
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
    const f = g.emitters.find((e) => e.kind === 'flame');
    if (!f) return false;
    const xs = f.xforms!;
    if (xs.length < MAX_XFORMS && (xs.length <= 1 || rng() < 0.5)) {
      // New transform, sometimes borrowed from the chain.
      xs.push(g.chain.length && rng() < 0.4 ? opToXform(pick(rng, g.chain), rng) : randomXform(rng));
    } else if (xs.length > 1) xs.splice(randInt(rng, 0, xs.length - 1), 1);
    return true;
  }],
  [0.6, 'xform-to-chain', (g, rng) => {
    const f = g.emitters.find((e) => e.kind === 'flame');
    if (!f || g.chain.length >= MAX_CHAIN) return false;
    g.chain.splice(randInt(rng, 0, g.chain.length), 0, xformToOp(pick(rng, f.xforms!), rng));
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
const STRUCTURAL = new Set(['swap-emitter', 'remove-emitter', 'add-emitter', 'change-carrier', 'replace-op', 'flip-layer']);

/** 1-3 random mutations (more for larger `amt`). Returns a repaired copy. */
export function mutate(gIn: Genome, rng: Rng, amt = 1, log?: string[]): Genome {
  const g = cloneGenome(gIn);
  const n = Math.max(1, Math.min(4, Math.round(1 + rng() * 2 * amt)));
  const weight = (x: (typeof MUTATORS)[number]) => (amt < 0.6 && STRUCTURAL.has(x[1]) ? x[0] * 0.25 : x[0]);
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

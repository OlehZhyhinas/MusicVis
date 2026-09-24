// The drift's journey through gene space (genes/drift.ts), planned offline from the song analysis.
//
// planDrift() walks the song's sections once and derives one genome per section: a small mutation
// (ops.mutate + repair) of the previous section's genome, limited to the gene groups and kinds the
// gene allows, pulled back toward the genome a returning section type had before, kept inside the
// wander bound around the home genome and screened for cost. The rng is seeded from the song's
// sections, the home genome and the drift seed, so the same song and preset always take the same
// path, and performedAt() is a pure function of the plan and the song time: seeking lands on the
// same picture as playing through.
//
// Between sections the continuous parameters morph from the previous section's genome over `morph`
// bars (uniforms, no recompile); structure (kinds, discrete switches) changes only at a boundary.

import {
  CARRIER_SCHEMA, FUSE_SCHEMA, LOCI, LOCUS_SCHEMAS, OP_SCHEMAS, PALETTE_SCHEMAS, SHAPE_SCHEMAS, TONE_SCHEMA,
  cloneGenome, estimateCost, fnv1a, genomeHash, repair, structuralKey, validate,
  type Genome, type Locus, type ParamSpec, type Params, type Schema,
} from '../genome';
import { mulberry32, mutate } from '../ops';
import type { Section, SectionLabel } from '../../types';
import { driftCost, type DriftGene } from './drift';

export type DriftGroup = 'form' | 'colour' | 'motion';

/** One gene of a genome as the drift sees it: where it is, its kind, params and schema. */
interface GeneRef {
  path: string;
  kind: string;
  obj: Record<string, unknown> & { p: Params };
  schema: Schema;
  groups: readonly DriftGroup[];
}

const LOCUS_GROUPS: Record<Locus, readonly DriftGroup[]> = {
  shape: ['form'], place: ['form', 'motion'], motion: ['motion'], deform: ['form', 'motion'], material: ['form'],
  emit: ['form'], feel: ['motion'], color: ['colour'],
};

/** Every parameterised gene of a genome, keyed by a stable path. */
function genesOf(g: Genome): GeneRef[] {
  const out: GeneRef[] = [];
  const add = (path: string, kind: string, obj: unknown, schema: Schema | undefined, groups: readonly DriftGroup[]) => {
    if (obj && schema) out.push({ path, kind, obj: obj as GeneRef['obj'], schema, groups });
  };
  g.chain.forEach((o, i) => add(`op${i}`, o.op, o, OP_SCHEMAS[o.op], ['form', 'motion']));
  add('car', g.carrier.kind, g.carrier, CARRIER_SCHEMA, ['motion']);
  add('pal', g.palette.kind, g.palette, PALETTE_SCHEMAS[g.palette.kind], ['colour']);
  add('tone', '', g.tone, TONE_SCHEMA, ['colour']);
  g.bodies.forEach((b, i) => {
    for (const l of LOCI) {
      const gene = b[l] as { kind: string; p: Params };
      add(`b${i}.${l}`, gene.kind, gene, LOCUS_SCHEMAS[l][gene.kind], LOCUS_GROUPS[l]);
    }
    if (b.fuse) {
      add(`b${i}.fuse`, '', b.fuse, FUSE_SCHEMA, ['form']);
      add(`b${i}.fshape`, b.fuse.shape.kind, b.fuse.shape, SHAPE_SCHEMAS[b.fuse.shape.kind], ['form']);
    }
    b.deform.ops?.forEach((o, j) => add(`b${i}.dop${j}`, o.op, o, OP_SCHEMAS[o.op], ['form', 'motion']));
  });
  return out;
}

const geneMap = (g: Genome) => new Map(genesOf(g).map((r) => [r.path, r]));

/** True for a parameter that morphs smoothly (not a choice list, not an integer count). */
export const continuous = (s: ParamSpec): boolean => !s.choices && !s.int;

/** Replaces a gene object's contents with a copy of another's (kind, params and extras). */
function assignGene(dst: Record<string, unknown>, src: Record<string, unknown>): void {
  for (const k of Object.keys(dst)) delete dst[k];
  Object.assign(dst, JSON.parse(JSON.stringify(src)));
}

/** Distance of one parameter, 0..1 (log scale where the schema is logarithmic). */
function paramDist(a: number, b: number, s: ParamSpec): number {
  if (s.choices || s.int) return a === b ? 0 : 1;
  if (s.log && s.min > 0) return Math.min(1, Math.abs(Math.log(Math.max(a, s.min) / Math.max(b, s.min))) / Math.log(s.max / s.min));
  return Math.min(1, Math.abs(a - b) / Math.max(s.max - s.min, 1e-9));
}

/**
 * Gene distance of two genomes, 0 (identical) .. 1: the mean over every gene either has of 1 for a
 * missing gene or another kind, else the mean parameter distance (a changed choice counts 1).
 */
export function driftDistance(a: Genome, b: Genome): number {
  const ga = geneMap(a);
  const gb = geneMap(b);
  const paths = new Set([...ga.keys(), ...gb.keys()]);
  let sum = 0;
  for (const path of paths) {
    const x = ga.get(path);
    const y = gb.get(path);
    if (!x || !y || x.kind !== y.kind) {
      sum += 1;
      continue;
    }
    const keys = Object.keys(x.schema);
    if (!keys.length) continue;
    let d = 0;
    for (const k of keys) d += paramDist(x.obj.p[k] ?? x.schema[k].def, y.obj.p[k] ?? y.schema[k].def, x.schema[k]);
    sum += d / keys.length;
  }
  return paths.size ? sum / paths.size : 0;
}

/**
 * Parameters of `a` moved toward `target` by t (structure from target when t >= 0.5, else from a):
 * continuous parameters of genes both share (same path, same kind) interpolate; everything else
 * comes from the structure source. Not repaired: interpolated values stay inside their specs.
 */
export function blendToward(a: Genome, target: Genome, t: number): Genome {
  const [src, other, w] = t >= 0.5 ? [target, a, 1 - t] : [a, target, t];
  const out = cloneGenome(src);
  if (w <= 0) return out;
  const og = geneMap(other);
  for (const r of genesOf(out)) {
    const o = og.get(r.path);
    if (!o || o.kind !== r.kind) continue;
    for (const k of Object.keys(r.schema)) {
      const s = r.schema[k];
      if (!continuous(s) || typeof o.obj.p[k] !== 'number') continue;
      r.obj.p[k] = r.obj.p[k] + (o.obj.p[k] - r.obj.p[k]) * w;
    }
  }
  return out;
}

/** The genome between two section genomes at morph progress x (structure of `to`). */
export function morphGenome(from: Genome, to: Genome, x: number): Genome {
  if (x >= 1) return to;
  const out = cloneGenome(to);
  const fg = geneMap(from);
  for (const r of genesOf(out)) {
    const f = fg.get(r.path);
    if (!f || f.kind !== r.kind) continue;
    for (const k of Object.keys(r.schema)) {
      if (!continuous(r.schema[k]) || typeof f.obj.p[k] !== 'number') continue;
      r.obj.p[k] = f.obj.p[k] + (r.obj.p[k] - f.obj.p[k]) * x;
    }
  }
  return out;
}

/** The home genome as the drift plays it: no drift gene of its own (so a path never recurses). */
export function stripDrift(g: Genome): Genome {
  const c = cloneGenome(g);
  delete c.drift;
  return c;
}

const GROUPS_OF: Record<number, readonly DriftGroup[]> = { 0: ['form', 'colour', 'motion'], 1: ['form'], 2: ['colour'], 3: ['motion'] };

/**
 * A mutated candidate cut back to what the drift allows: kinds and body/chain structure change only
 * when `allowKinds` (and the form group drifts); genes of groups that do not drift keep the previous
 * genome's kind and params; reactions, energy and genome-wide genes stay the home genome's.
 */
function limit(cand: Genome, prev: Genome, home: Genome, allowKinds: boolean, groups: readonly DriftGroup[]): Genome {
  const r = cloneGenome(allowKinds && groups.includes('form') ? cand : prev);
  const cg = geneMap(cand);
  const pg = geneMap(prev);
  const apply = (paramsOnly: boolean) => {
    for (const g of genesOf(r)) {
      const allowed = g.groups.some((x) => groups.includes(x));
      const from = allowed ? cg.get(g.path) : pg.get(g.path);
      if (!from) continue;
      if (from.kind !== g.kind) {
        if (allowed ? allowKinds && !paramsOnly : true) assignGene(g.obj, from.obj);
        continue;
      }
      for (const k of Object.keys(g.schema)) {
        if (paramsOnly && !continuous(g.schema[k])) continue;
        if (typeof from.obj.p[k] === 'number') g.obj.p[k] = from.obj.p[k];
      }
    }
  };
  apply(false);
  const keep = (x: Genome) => {
    const o = x as unknown as Record<string, unknown>;
    const h = home as unknown as Record<string, unknown>;
    x.reactions = JSON.parse(JSON.stringify(home.reactions));
    x.energy = [...home.energy] as [number, number];
    for (const k of Object.keys(o)) if (!['v', 'chain', 'bodies', 'carrier', 'palette', 'tone', 'reactions', 'energy'].includes(k)) delete o[k];
    for (const k of Object.keys(h)) if (!(k in o) && k !== 'drift') o[k] = JSON.parse(JSON.stringify(h[k]));
    delete o.drift;
  };
  keep(r);
  let out = repair(r);
  if (!allowKinds && structuralKey(out) !== structuralKey(prev)) {
    // A discrete switch changed the shaders: only the continuous parameters move this time.
    const q = cloneGenome(prev);
    const qg = genesOf(q);
    for (const g of qg) {
      if (!g.groups.some((x) => groups.includes(x))) continue;
      const from = cg.get(g.path);
      if (!from || from.kind !== g.kind) continue;
      for (const k of Object.keys(g.schema)) if (continuous(g.schema[k]) && typeof from.obj.p[k] === 'number') g.obj.p[k] = from.obj.p[k];
    }
    keep(q);
    out = repair(q);
    if (structuralKey(out) !== structuralKey(prev)) out = repair(prev);
  }
  return out;
}

/**
 * Besides the discrete mutations, every drifting gene's continuous parameters take a small gaussian
 * step (a share of the parameter's range growing with the step), so each section reads as a variation
 * of the one before rather than one changed setting. Repaired.
 */
function wander(g: Genome, rng: () => number, amt: number, groups: readonly DriftGroup[]): Genome {
  const out = cloneGenome(g);
  const gauss = () => {
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  };
  for (const r of genesOf(out)) {
    if (!r.groups.some((x) => groups.includes(x))) continue;
    for (const k of Object.keys(r.schema)) {
      const s = r.schema[k];
      if (!continuous(s) || rng() > 0.5) continue;
      const v = r.obj.p[k];
      // A dormant feature (a param resting at its minimum, e.g. blur, sharpen, relief 0) stays off.
      if (v <= s.min) continue;
      if (s.log && s.min > 0) r.obj.p[k] = Math.min(s.max, Math.max(s.min, Math.max(v, s.min) * Math.exp(gauss() * Math.log(s.max / s.min) * WANDER * amt)));
      else r.obj.p[k] = Math.min(s.max, Math.max(s.min, v + gauss() * (s.max - s.min) * WANDER * amt));
    }
  }
  return repair(out);
}
/** Share of a parameter's range one unit of step moves it (one standard deviation). */
const WANDER = 0.1;

/** The kinds of every body locus (what a jump must change). */
const bodyKinds = (g: Genome) => g.bodies.map((b) => LOCI.map((l) => (b[l] as { kind: string }).kind).join('/')).join('+');

/** How far a section type moves (times step), and how much of the wander bound it may use. */
const LABEL_STEP: Record<SectionLabel, number> = { intro: 0.5, verse: 1, build: 1, chorus: 1, drop: 1.8, breakdown: 2, outro: 0.6 };
const LABEL_BOUND: Record<SectionLabel, number> = { intro: 0.4, verse: 0.7, build: 0.7, chorus: 0.75, drop: 1, breakdown: 1, outro: 0.5 };
/** Candidates tried per section before it falls back to a safe genome. */
const TRIES = 8;
/** Mutation rounds per unit of step. */
const ROUNDS = 5;

export interface DriftStop {
  start: number;
  end: number;
  label: SectionLabel;
  /** The section's genome (repaired, no drift gene). */
  genome: Genome;
  /** structuralKey(genome): when it differs from the previous stop's, the boundary recompiles. */
  key: string;
  /** Distance from the home genome. */
  dist: number;
  /** A hard cut into this section (drops, or morph 0) rather than a morph. */
  cut: boolean;
  /** Index of the earlier stop this one returned toward (-1: none). */
  parent: number;
}

export interface DriftPlan {
  /** fnv1a over the song's sections: identifies the song the plan was made for. */
  song: number;
  /** genomeHash of the home genome the plan was made for (drift gene included). */
  home: number;
  morph: number;
  stops: DriftStop[];
  /** Worst estimated cost over the stops (ms). */
  worstCost: number;
}

/** Identifies a song by its section layout (what the plan depends on). */
export function songHash(sections: readonly Section[]): number {
  return fnv1a(sections.map((s) => `${s.start.toFixed(2)}-${s.end.toFixed(2)}${s.label}`).join('|'));
}

/**
 * The drift's path through the song: one genome per section. Deterministic in the home genome
 * (drift gene included) and the song's sections. Without a drift gene, a single home stop.
 */
export function planDrift(homeIn: Genome, sections: readonly Section[]): DriftPlan {
  const home = stripDrift(homeIn);
  const d: DriftGene | undefined = homeIn.drift;
  const song = songHash(sections);
  const hh = genomeHash(homeIn);
  const homeCost = estimateCost(home);
  const maxCost = driftCost(homeCost) + 1e-6;
  const secs = sections.length ? sections : [{ start: 0, end: Infinity, label: 'verse' as SectionLabel, energy: 0.5 }];
  const stops: DriftStop[] = [];
  const homeKey = structuralKey(home);
  const plan: DriftPlan = { song, home: hh, morph: d?.p.morph ?? 0, stops, worstCost: homeCost };
  if (!d) {
    stops.push({ start: secs[0].start, end: secs[secs.length - 1].end, label: secs[0].label, genome: home, key: homeKey, dist: 0, cut: false, parent: -1 });
    return plan;
  }
  const p = d.p;
  const groups = GROUPS_OF[p.what] ?? GROUPS_OF[0];
  const firstOf = new Map<SectionLabel, number>();
  const seed = fnv1a(`${song}|${genomeHash(home)}|${p.seed.toFixed(4)}|${p.step.toFixed(3)}|${p.kinds}|${p.what}`);
  let prev = home;
  secs.forEach((sec, i) => {
    const rng = mulberry32((seed ^ Math.imul(i + 1, 0x9e3779b9)) >>> 0);
    const parentIdx = firstOf.get(sec.label) ?? -1;
    const parent = parentIdx >= 0 ? stops[parentIdx].genome : null;
    const bound = p.bound * LABEL_BOUND[sec.label];
    const ok = (g: Genome) => driftDistance(g, home) <= bound + 1e-9 && estimateCost(g) <= maxCost && !validate(g).length;
    let genome: Genome | null = null;
    if (i === 0) genome = home; // the song opens on the saved preset
    const allowKinds = p.kinds === 2 || (p.kinds === 1 && sec.label === 'drop');
    const jump = p.kinds === 1 && sec.label === 'drop';
    for (let t = 0; !genome && t < TRIES; t++) {
      // A few rounds of mutation (more for a bigger step), each cut back to what the drift allows.
      const amt = p.step * LABEL_STEP[sec.label] * (1 - t / (TRIES * 1.5));
      let cand = prev;
      for (let k = 0, n = 1 + Math.round(amt * ROUNDS); k < n; k++) cand = limit(mutate(cand, rng, Math.min(1, amt)), prev, home, allowKinds, groups);
      cand = wander(cand, rng, amt, groups);
      if (driftDistance(cand, prev) < 1e-4) continue; // every change fell on something the drift keeps
      if (!allowKinds && structuralKey(cand) !== structuralKey(prev)) continue; // a threshold crossed: shaders would change
      // A drop with kinds allowed on drops only is a jump: insist on a new body kind while tries remain.
      if (jump && t < TRIES - 2 && bodyKinds(cand) === bodyKinds(prev)) continue;
      if (parent && p.ret > 0) cand = repair(blendToward(cand, parent, p.ret));
      if (sec.label === 'outro') cand = repair(blendToward(cand, home, Math.max(p.ret, 0.5)));
      if (ok(cand)) genome = cand;
    }
    if (!genome) {
      // Nothing fitted: the section type's earlier genome, else halfway back home from here.
      const back = parent ?? repair(blendToward(prev, home, 0.5));
      genome = ok(back) ? back : home;
    }
    const cost = estimateCost(genome);
    if (cost > plan.worstCost) plan.worstCost = cost;
    stops.push({
      start: sec.start, end: sec.end, label: sec.label, genome, key: structuralKey(genome), dist: driftDistance(genome, home),
      cut: p.morph === 0 || sec.label === 'drop', parent: parentIdx,
    });
    if (parentIdx < 0) firstOf.set(sec.label, i);
    prev = genome;
  });
  return plan;
}

/** Index of the stop playing at song time t (clamped to the first / last). */
export function stopAt(plan: DriftPlan, t: number): number {
  const s = plan.stops;
  let lo = 0;
  let hi = s.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (s[mid].start <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** 0..1 progress of the morph into stop i at song time t (1: arrived, always 1 for a cut). */
export function morphProgress(plan: DriftPlan, i: number, t: number, barSeconds: number): number {
  const st = plan.stops[i];
  if (i === 0 || st.cut || plan.morph <= 0) return 1;
  const x = Math.min(1, Math.max(0, (t - st.start) / Math.max(plan.morph * barSeconds, 1e-3)));
  return x * x * (3 - 2 * x);
}

/**
 * The genome the drift plays at song time t: the current section's genome, morphing in from the
 * previous section's over the first `morph` bars. Pure: the same plan and time give the same genome.
 * Returns the stop's own object once the morph has arrived (callers must not mutate it).
 */
export function performedAt(plan: DriftPlan, t: number, barSeconds: number): { genome: Genome; index: number; key: string } {
  const i = stopAt(plan, t);
  const st = plan.stops[i];
  const x = morphProgress(plan, i, t, barSeconds);
  if (x >= 1) return { genome: st.genome, index: i, key: st.key };
  const g = morphGenome(plan.stops[i - 1].genome, st.genome, x);
  return { genome: g, index: i, key: structuralKey(g) };
}

// Gene editor model: the controls a genome exposes (generated from the parameter
// schemas, so new genes get controls automatically) and the edits the editor
// makes (parameter changes, kind switches, chain / reaction list edits, silent
// alleles, save as a new member). Pure logic (no DOM, no GL) so it runs in the
// Node tests. Every structural edit returns a repaired, valid genome or a reason.

import {
  BODY_GROUPS, CARRIER_KINDS, MAX_BODIES, CARRIER_SCHEMA, DRAW_OPS, FLAME_VARIATIONS, FUSE_SCHEMA, LOCI, LOCUS_KINDS, MAX_CHAIN, MAX_DRAW,
  MAX_REACTIONS, MAX_XFORMS, OP_KINDS, OP_SCHEMAS, PALETTE_KINDS, PALETTE_SCHEMAS, REACTION_SCHEMA, SHAPE_CLASS, SHAPE_KINDS,
  SHAPE_SCHEMAS, SIGNALS, TONE_SCHEMA, UNIQUE_SHAPES, XFORM_DRIFT, XFORM_SPIN, AFF_RANGE,
  cloneBody, cloneGenome, defaultParams, isFold, isVarOp, locusSchema, reactable, repair, repairBody, repairShape, repairXform,
  schemaFor, sdfCapable, stageFree, validate,
  type BodyGene, type CarrierKind, type FlameVar, type FlameXformGene, type Gene, type GeneGroup, type Genome, type Locus,
  type OpGene, type OpKind, type PaletteKind, type ParamSpec, type Params, type ReactionGene, type Schema, type ShapeGene,
  type ShapeKind, type Signal, type Stage,
} from './genome';
import { fitLoci } from './ops';
import {
  createGenomeGene, genomeGene, genomeGeneSchema, genomeGenes, repairGenomeGenes,
  type GenomeGeneValue,
} from './geneRegistry';
import type { Member, Population } from './population';

// ------------------------------------------------------------ addressing

/** Where a set of editable parameters lives in a genome. */
export type Target =
  | { t: 'locus'; b: number; locus: Locus }
  | { t: 'fuse'; b: number }
  | { t: 'fuseShape'; b: number }
  | { t: 'drawOp'; b: number; j: number }
  | { t: 'xform'; b: number; j: number }
  | { t: 'op'; j: number }
  | { t: 'carrier' }
  | { t: 'palette' }
  | { t: 'tone' }
  | { t: 'reaction'; j: number }
  | { t: 'energy' }
  /** A registered genome-wide gene (geneRegistry.ts), by its genome key. */
  | { t: 'gene'; key: string };

export const targetId = (t: Target): string => {
  switch (t.t) {
    case 'locus': return `b${t.b}.${t.locus}`;
    case 'fuse': case 'fuseShape': return `b${t.b}.${t.t}`;
    case 'drawOp': case 'xform': return `b${t.b}.${t.t}${t.j}`;
    case 'op': case 'reaction': return `${t.t}${t.j}`;
    case 'gene': return t.key;
    default: return t.t;
  }
};

const P = (min: number, max: number, def: number, extra: Partial<ParamSpec> = {}): ParamSpec => ({ min, max, def, ...extra });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/** An op's strength (blend for variations, rate multiplier for motion ops). */
export const OP_W: ParamSpec = P(0, 1, 1);
/** A flame transform's settings as a flat schema (aff a..f, drift dx / dy, variation weights var.<name>). */
export const XFORM_SCHEMA: Schema = {
  weight: P(0.05, 1, 0.5), color: P(0, 1, 0.5), spin: C(XFORM_SPIN, 0), bass: P(0, 0.3, 0), pulse: P(0, 0.3, 0),
  a: P(AFF_RANGE[0], AFF_RANGE[1], 0.5), b: P(AFF_RANGE[0], AFF_RANGE[1], 0), c: P(AFF_RANGE[0], AFF_RANGE[1], 0),
  d: P(AFF_RANGE[0], AFF_RANGE[1], 0.5), e: P(AFF_RANGE[0], AFF_RANGE[1], 0), f: P(AFF_RANGE[0], AFF_RANGE[1], 0),
  dx: P(0, XFORM_DRIFT, 0), dy: P(0, XFORM_DRIFT, 0),
};
const VAR_SPEC: ParamSpec = P(0, 1, 0.5);
const AFF_KEYS = ['a', 'b', 'c', 'd', 'e', 'f'];
export const ENERGY_SCHEMA: Schema = { lo: P(0, 1, 0.2), hi: P(0, 1, 0.7) };
const ENERGY_GAP = 0.15;

/** The registered genome-wide gene stored under `key` (undefined when absent). */
export function geneValue(g: Genome, key: string): GenomeGeneValue | undefined {
  const v = (g as unknown as Record<string, unknown>)[key];
  return v && typeof v === 'object' ? (v as GenomeGeneValue) : undefined;
}

function bodyOf(g: Genome, b: number): BodyGene | null {
  return g.bodies[b] ?? null;
}

/** The schema of the parameters at a target (null when the target does not exist). */
export function schemaAt(g: Genome, t: Target): Schema | null {
  switch (t.t) {
    case 'locus': {
      const b = bodyOf(g, t.b);
      return b ? locusSchema(t.locus, (b[t.locus] as Gene).kind) : null;
    }
    case 'fuse': return bodyOf(g, t.b)?.fuse ? FUSE_SCHEMA : null;
    case 'fuseShape': {
      const f = bodyOf(g, t.b)?.fuse;
      return f ? SHAPE_SCHEMAS[f.shape.kind] : null;
    }
    case 'drawOp': {
      const o = bodyOf(g, t.b)?.deform.ops?.[t.j];
      return o ? { w: OP_W, ...OP_SCHEMAS[o.op] } : null;
    }
    case 'xform': {
      const x = bodyOf(g, t.b)?.shape.xforms?.[t.j];
      if (!x) return null;
      const s: Schema = { ...XFORM_SCHEMA };
      for (const v of Object.keys(x.vars)) s[`var.${v}`] = VAR_SPEC;
      return s;
    }
    case 'op': {
      const o = g.chain[t.j];
      return o ? { w: OP_W, ...OP_SCHEMAS[o.op] } : null;
    }
    case 'carrier': return CARRIER_SCHEMA;
    case 'palette': return PALETTE_SCHEMAS[g.palette.kind];
    case 'tone': return TONE_SCHEMA;
    case 'reaction': return g.reactions[t.j] ? REACTION_SCHEMA : null;
    case 'energy': return ENERGY_SCHEMA;
    case 'gene': {
      const spec = genomeGene(t.key);
      const v = geneValue(g, t.key);
      return spec && v ? genomeGeneSchema(spec, v.kind) : null;
    }
  }
}

function xformGet(x: FlameXformGene, k: string): number {
  const ai = AFF_KEYS.indexOf(k);
  if (ai >= 0) return x.aff[ai];
  if (k === 'dx') return x.drift[0];
  if (k === 'dy') return x.drift[1];
  if (k.startsWith('var.')) return x.vars[k.slice(4) as FlameVar] ?? 0;
  return (x as unknown as Record<string, number>)[k];
}
function xformSet(x: FlameXformGene, k: string, v: number): void {
  const ai = AFF_KEYS.indexOf(k);
  if (ai >= 0) x.aff[ai] = v;
  else if (k === 'dx') x.drift[0] = v;
  else if (k === 'dy') x.drift[1] = v;
  else if (k.startsWith('var.')) x.vars[k.slice(4) as FlameVar] = v;
  else (x as unknown as Record<string, number>)[k] = v;
}

/** Current value of one parameter at a target. */
export function getParam(g: Genome, t: Target, k: string): number {
  switch (t.t) {
    case 'locus': return (g.bodies[t.b][t.locus] as Gene).p[k];
    case 'fuse': return g.bodies[t.b].fuse!.p[k];
    case 'fuseShape': return g.bodies[t.b].fuse!.shape.p[k];
    case 'drawOp': {
      const o = g.bodies[t.b].deform.ops![t.j];
      return k === 'w' ? o.w : o.p[k];
    }
    case 'xform': return xformGet(g.bodies[t.b].shape.xforms![t.j], k);
    case 'op': return k === 'w' ? g.chain[t.j].w : g.chain[t.j].p[k];
    case 'carrier': return g.carrier.p[k];
    case 'palette': return g.palette.p[k];
    case 'tone': return g.tone.p[k];
    case 'reaction': return (g.reactions[t.j] as unknown as Params)[k];
    case 'energy': return k === 'lo' ? g.energy[0] : g.energy[1];
    case 'gene': return geneValue(g, t.key)!.p[k];
  }
}

/** A value snapped into its spec (range, integer, discrete choices). */
export function clampTo(v: number, s: ParamSpec): number {
  if (!Number.isFinite(v)) v = s.def;
  if (s.choices) {
    let best = s.choices[0];
    for (const c of s.choices) if (Math.abs(c - v) < Math.abs(best - v)) best = c;
    return best;
  }
  v = Math.min(s.max, Math.max(s.min, v));
  return s.int ? Math.round(v) : v;
}

/**
 * Sets one parameter in place (clamped into its spec) and returns the value applied. Energy keeps
 * lo < hi with the minimum gap repair() enforces.
 */
export function setParam(g: Genome, t: Target, k: string, raw: number): number {
  const s = schemaAt(g, t)?.[k];
  if (!s) return NaN;
  const v = clampTo(raw, s);
  switch (t.t) {
    case 'locus': (g.bodies[t.b][t.locus] as Gene).p[k] = v; break;
    case 'fuse': g.bodies[t.b].fuse!.p[k] = v; break;
    case 'fuseShape': g.bodies[t.b].fuse!.shape.p[k] = v; break;
    case 'drawOp': {
      const o = g.bodies[t.b].deform.ops![t.j];
      if (k === 'w') o.w = v;
      else o.p[k] = v;
      break;
    }
    case 'xform': xformSet(g.bodies[t.b].shape.xforms![t.j], k, v); break;
    case 'op':
      if (k === 'w') g.chain[t.j].w = v;
      else g.chain[t.j].p[k] = v;
      break;
    case 'carrier': g.carrier.p[k] = v; break;
    case 'palette': g.palette.p[k] = v; break;
    case 'tone': g.tone.p[k] = v; break;
    case 'reaction': (g.reactions[t.j] as unknown as Params)[k] = v; break;
    case 'gene': geneValue(g, t.key)!.p[k] = v; break;
    case 'energy': {
      let [lo, hi] = g.energy;
      if (k === 'lo') {
        lo = Math.min(v, 1 - ENERGY_GAP);
        hi = Math.max(hi, lo + ENERGY_GAP);
      } else {
        hi = Math.max(v, ENERGY_GAP);
        lo = Math.min(lo, hi - ENERGY_GAP);
      }
      g.energy = [Math.round(lo * 1e4) / 1e4, Math.round(hi * 1e4) / 1e4];
      return k === 'lo' ? g.energy[0] : g.energy[1];
    }
  }
  return v;
}

/**
 * Applies a parameter edit to a genome and keeps it valid: the value is set in place; when the edit
 * breaks a rule (a chunk shape given copies, say), the genome is repaired. Returns the genome to use
 * (the same object unless it had to be repaired) and whether the repair changed anything else.
 */
export function editParam(g: Genome, t: Target, k: string, v: number): { genome: Genome; repaired: boolean } {
  setParam(g, t, k, v);
  if (!validate(g).length) return { genome: g, repaired: false };
  return { genome: repairKeeping(g), repaired: true };
}

/** repair(), keeping the registered genome-wide genes (repaired against their schemas). */
export function repairKeeping(g: Genome): Genome {
  const out = repair(g);
  repairGenomeGenes(g as unknown as Record<string, unknown>, out as unknown as Record<string, unknown>);
  return out;
}

// --------------------------------------------------------------- sliders

export const SLIDER_STEPS = 1000;

/** Slider position (0..SLIDER_STEPS) of a value: linear, or logarithmic for log specs. */
export function toSlider(v: number, s: ParamSpec): number {
  let t: number;
  if (s.log && s.min > 0) t = Math.log(Math.max(v, s.min) / s.min) / Math.log(s.max / s.min);
  else t = s.max > s.min ? (v - s.min) / (s.max - s.min) : 0;
  return Math.round(Math.min(1, Math.max(0, t)) * SLIDER_STEPS);
}

/** Value at a slider position (clamped into the spec, integers rounded). */
export function fromSlider(pos: number, s: ParamSpec): number {
  const t = Math.min(1, Math.max(0, pos / SLIDER_STEPS));
  const v = s.log && s.min > 0 ? s.min * Math.pow(s.max / s.min, t) : s.min + (s.max - s.min) * t;
  return clampTo(v, s);
}

/** Display form of a value: integers plain, others with precision fitting the range. */
export function formatValue(v: number, s: ParamSpec): string {
  if (!Number.isFinite(v)) return '–';
  if (s.int || Number.isInteger(v) && s.choices) return String(Math.round(v));
  const span = s.log ? Math.min(Math.abs(v) || s.min, s.max) : s.max - s.min;
  const dp = span < 0.01 ? 5 : span < 0.1 ? 4 : span < 2 ? 3 : span < 50 ? 2 : 1;
  return v.toFixed(dp);
}

// -------------------------------------------------------------- controls

export type Widget = 'slider' | 'segmented' | 'select';

export interface ParamControl {
  key: string;
  label: string;
  spec: ParamSpec;
  value: number;
  widget: Widget;
  options?: { value: number; label: string }[];
}

/** Readable names for the terse parameter keys (the key itself is the fallback). */
const LABELS: Record<string, string> = {
  r: 'radius', n: 'count', w: 'strength', len: 'length', atk: 'attack', rel: 'release', thr: 'threshold', sens: 'sensitivity',
  div: 'clock', lock: 'grid lock', vx: 'velocity x', vy: 'velocity y', cx: 'centre x', cy: 'centre y', amt: 'amount',
  amp: 'amplitude', freq: 'frequency', k: 'blend', t: 'mix', q: 'quantise', s: 'scale', halfLife: 'half-life', fnoise: 'flow noise',
  fscale: 'flow scale', famt: 'flow amount', grain: 'grain size', vort: 'vorticity', ca: 'aberration', sat: 'saturation', reflectY: 'reflect y',
  s1: 'slot 2', s2: 'slot 3', ra: 'ratio a', rb: 'ratio b', melHue: 'melody hue', xs: 'roam', inst: 'instruments',
  dx: 'drift x', dy: 'drift y', lo: 'from', hi: 'to', zoomFlow: 'zoom flow', rounds: 'rounds', bins: 'bins',
};
export function labelFor(key: string): string {
  if (key.startsWith('var.')) return key.slice(4);
  return LABELS[key] ?? key;
}

const ON_OFF = ['off', 'on'];
/** Names for coded choices, by `<gene kind>.<key>` (then by key alone). */
const CHOICE_NAMES: Record<string, string[]> = {
  'solid.solid': ['tetra', 'cube', 'octa', 'icosa', 'polygon', 'by section'],
  'bars.mode': ['baseline', 'arc', 'ring', 'mirrored'],
  'curve.form': ['wave', 'circle', 'spiral', 'lissajous', 'arc', 'harmonograph'],
  'edge.mode': ['skyline', 'melody', 'rain', 'ridge'],
  'edge.side': ['right', 'top', 'bottom', 'left'],
  'textured.tex': ['craters', 'stripes', 'windows'],
  'grid.lattice': ['square', 'hex', 'triangle'],
  'outline.path': ['circle', 'polygon', 'figure 8'],
  'mirror.axis': ['vertical', 'horizontal', 'both'],
  'push.axis': ['x', 'y', 'radial'],
  'fuse.mode': ['union', 'morph', 'region'],
  'fuse.drive': ['none', 'sweep', 'bass', 'melody', 'loud', 'surge'],
  'fuse.inside': ['along', 'inside'],
  'tone.tonemap': ['filmic', 'flame'],
  'reaction.q': ['free', 'on clock'],
  'feel.lock': ['free', 'grid'],
};
function fraction(v: number): string {
  if (v === 0) return '0';
  const sign = v < 0 ? '−' : '';
  const a = Math.abs(v);
  if (a >= 1) return `${sign}${a}`;
  return `${sign}1/${Math.round(1 / a)}`;
}
function choiceLabel(ctx: string, key: string, v: number, s: ParamSpec): string {
  const named = CHOICE_NAMES[`${ctx}.${key}`];
  const idx = s.choices!.indexOf(v);
  if (named && idx >= 0 && s.choices!.every((c, i) => c === i)) return named[idx] ?? String(v);
  if (s.choices!.length === 2 && s.choices![0] === 0 && s.choices![1] === 1) return ON_OFF[v];
  if (key === 'div') return v < 1 ? '½ beat' : `${v} beat${v === 1 ? '' : 's'}`;
  if (s.choices!.some((c) => c !== 0 && Math.abs(c) < 1)) return fraction(v);
  return String(v);
}

/** Controls for every parameter of a schema (ctx: the gene kind, used for choice names). */
export function paramControls(schema: Schema, value: (k: string) => number, ctx = ''): ParamControl[] {
  return Object.keys(schema).map((key) => {
    const spec = schema[key];
    const c: ParamControl = { key, label: labelFor(key), spec, value: value(key), widget: 'slider' };
    if (spec.choices) {
      c.options = spec.choices.map((v) => ({ value: v, label: choiceLabel(ctx, key, v, spec) }));
      c.widget = spec.choices.length <= 4 ? 'segmented' : 'select';
    }
    return c;
  });
}

export interface KindControl {
  value: string;
  options: string[];
}

export interface ListItem {
  id: string;
  title: string;
  target: Target;
  kind?: KindControl;
  params: ParamControl[];
  /** Chain ops that may run in either stage. */
  stage?: Stage;
  canUp: boolean;
  canDown: boolean;
}

export interface SectionModel {
  id: string;
  title: string;
  /** Body index for body sections, -1 for genome-wide ones. */
  body: number;
  target?: Target;
  kind?: KindControl;
  params: ParamControl[];
  /** Ordered lists (chain ops, deform ops, flame transforms, reactions). */
  items?: ListItem[];
  list?: 'chain' | 'drawOps' | 'xforms' | 'reactions';
  /** Whether another item can be added. */
  canAdd?: boolean;
  /** Silent alleles of a body. */
  alleles?: { locus: Locus; kind: string; summary: string }[];
  /** A registered genome-wide gene: whether the genome has it and whether it may be removed / added. */
  gene?: { key: string; present: boolean; optional: boolean };
  open: boolean;
}

export const LOCUS_TITLE: Record<Locus, string> = {
  shape: 'Shape', place: 'Placement', motion: 'Motion', deform: 'Deformation', material: 'Material', emit: 'Emission',
  feel: 'Feel', color: 'Colour mapping',
};

/** Shapes a fused shape may be (distance-field shapes). */
export const FUSE_SHAPE_KINDS: ShapeKind[] = SHAPE_KINDS.filter((k) => SHAPE_CLASS[k] === 'sdf' || k === 'curve' || k === 'aurora');

function geneSummary(locus: Locus, gene: Gene): string {
  const s = locusSchema(locus, gene.kind);
  const parts = Object.keys(s).slice(0, 3).map((k) => `${labelFor(k)} ${formatValue(gene.p[k], s[k])}`);
  return parts.join(', ');
}

/** The editor's sections for a genome, in display order. */
export function buildModel(g: Genome): SectionModel[] {
  const out: SectionModel[] = [];
  g.bodies.forEach((b, bi) => {
    for (const locus of LOCI) {
      const gene = b[locus] as Gene;
      const target: Target = { t: 'locus', b: bi, locus };
      const sec: SectionModel = {
        id: targetId(target), title: LOCUS_TITLE[locus], body: bi, target,
        kind: { value: gene.kind, options: [...LOCUS_KINDS[locus]] },
        params: paramControls(locusSchema(locus, gene.kind), (k) => gene.p[k], gene.kind),
        open: locus === 'shape' || locus === 'feel',
      };
      if (locus === 'shape' && gene.kind === 'flame') {
        const xs = b.shape.xforms ?? [];
        sec.list = 'xforms';
        sec.canAdd = xs.length < MAX_XFORMS;
        sec.items = xs.map((x, j) => {
          const t: Target = { t: 'xform', b: bi, j };
          return {
            id: targetId(t), title: `Transform ${j + 1}`, target: t, params: paramControls(schemaAt(g, t)!, (k) => xformGet(x, k), 'xform'),
            canUp: false, canDown: false,
          };
        });
      }
      if (locus === 'deform') {
        const ops = b.deform.ops ?? [];
        sec.list = 'drawOps';
        sec.canAdd = ops.length < MAX_DRAW;
        sec.items = ops.map((o, j) => {
          const t: Target = { t: 'drawOp', b: bi, j };
          return {
            id: targetId(t), title: `Op ${j + 1}`, target: t, kind: { value: o.op, options: [...DRAW_OPS] },
            params: paramControls(schemaAt(g, t)!, (k) => (k === 'w' ? o.w : o.p[k]), o.op), canUp: j > 0, canDown: j < ops.length - 1,
          };
        });
      }
      out.push(sec);
      if (locus === 'shape' && b.fuse) {
        const f = b.fuse;
        const ft: Target = { t: 'fuse', b: bi };
        out.push({ id: targetId(ft), title: 'Fuse', body: bi, target: ft, params: paramControls(FUSE_SCHEMA, (k) => f.p[k], 'fuse'), open: false });
        const st: Target = { t: 'fuseShape', b: bi };
        out.push({
          id: targetId(st), title: 'Fused shape', body: bi, target: st, kind: { value: f.shape.kind, options: [...FUSE_SHAPE_KINDS] },
          params: paramControls(SHAPE_SCHEMAS[f.shape.kind], (k) => f.shape.p[k], f.shape.kind), open: false,
        });
      }
    }
    const alleles = (Object.keys(b.alt ?? {}) as Locus[]).map((locus) => ({ locus, kind: b.alt![locus]!.kind, summary: geneSummary(locus, b.alt![locus]!) }));
    if (alleles.length) out.push({ id: `b${bi}.alt`, title: 'Silent alleles', body: bi, params: [], alleles, open: false });
  });
  const pal: Target = { t: 'palette' };
  out.push({
    id: 'palette', title: 'Palette', body: -1, target: pal, kind: { value: g.palette.kind, options: [...PALETTE_KINDS] },
    params: paramControls(PALETTE_SCHEMAS[g.palette.kind], (k) => g.palette.p[k], 'palette'), open: false,
  });
  out.push({ id: 'tone', title: 'Tone', body: -1, target: { t: 'tone' }, params: paramControls(TONE_SCHEMA, (k) => g.tone.p[k], 'tone'), open: false });
  out.push({
    id: 'carrier', title: 'Carrier', body: -1, target: { t: 'carrier' }, kind: { value: g.carrier.kind, options: [...CARRIER_KINDS] },
    params: paramControls(CARRIER_SCHEMA, (k) => g.carrier.p[k], 'carrier'), open: false,
  });
  out.push({
    id: 'chain', title: `Space chain (${g.chain.length}/${MAX_CHAIN})`, body: -1, params: [], list: 'chain', canAdd: g.chain.length < MAX_CHAIN, open: false,
    items: g.chain.map((o, j) => {
      const t: Target = { t: 'op', j };
      return {
        id: targetId(t), title: `${j + 1}`, target: t, kind: { value: o.op, options: [...OP_KINDS] },
        params: paramControls(schemaAt(g, t)!, (k) => (k === 'w' ? o.w : o.p[k]), o.op),
        stage: stageFree(o.op) ? o.stage : undefined, canUp: j > 0, canDown: j < g.chain.length - 1,
      };
    }),
  });
  out.push({
    id: 'reactions', title: `Reactions (${g.reactions.length}/${MAX_REACTIONS})`, body: -1, params: [], list: 'reactions',
    canAdd: g.reactions.length < MAX_REACTIONS && freeTargets(g).length > 0, open: false,
    items: g.reactions.map((r, j) => {
      const t: Target = { t: 'reaction', j };
      return {
        id: targetId(t), title: `${r.src} → ${targetLabel(g, r)}`, target: t, kind: { value: r.src, options: [...SIGNALS] },
        params: paramControls(REACTION_SCHEMA, (k) => (r as unknown as Params)[k], 'reaction'), canUp: false, canDown: false,
      };
    }),
  });
  for (const spec of genomeGenes()) {
    const v = geneValue(g, spec.key);
    const t: Target = { t: 'gene', key: spec.key };
    const gene = { key: spec.key, present: !!v, optional: spec.optional };
    if (!v) {
      out.push({ id: spec.key, title: spec.title, body: -1, params: [], gene, open: false });
      continue;
    }
    out.push({
      id: spec.key, title: spec.title, body: -1, target: t, gene,
      kind: spec.kinds ? { value: v.kind ?? spec.kinds[0], options: [...spec.kinds] } : undefined,
      params: paramControls(genomeGeneSchema(spec, v.kind), (k) => v.p[k], v.kind ?? spec.key), open: false,
    });
  }
  out.push({ id: 'energy', title: 'Energy range', body: -1, target: { t: 'energy' }, params: paramControls(ENERGY_SCHEMA, (k) => (k === 'lo' ? g.energy[0] : g.energy[1])), open: false });
  return out;
}

// --------------------------------------------------------- structural edits

export interface EditResult {
  genome: Genome;
  ok: boolean;
  reason?: string;
}
const fail = (g: Genome, reason: string): EditResult => ({ genome: g, ok: false, reason });

/** Keys two schemas share with the same spec (a material's gain, a mapping's hue, the feel settings). */
function carryParams(old: Params, from: Schema, to: Schema): Params {
  const p = defaultParams(to);
  for (const k of Object.keys(to)) {
    const a = from[k];
    const b = to[k];
    if (a && a.min === b.min && a.max === b.max && !!a.int === !!b.int && JSON.stringify(a.choices) === JSON.stringify(b.choices) && typeof old[k] === 'number') p[k] = old[k];
  }
  return p;
}

/** Reactions whose target no longer exists (or no longer takes that key) are dropped. */
function dropStaleReactions(g: Genome): void {
  g.reactions = g.reactions.filter((r) => {
    const s = schemaFor(g, r.g, r.i);
    return !!s && reactable(s).includes(r.k);
  });
}

function finish(before: Genome, g: Genome, check: (out: Genome) => string | null): EditResult {
  dropStaleReactions(g);
  const out = repairKeeping(g);
  if (out.bodies.length !== g.bodies.length) return fail(before, 'the renderer cannot show that combination');
  const why = check(out);
  if (why) return fail(before, why);
  const errs = validate(out);
  if (errs.length) return fail(before, errs[0]);
  return { genome: out, ok: true };
}

/**
 * Switches the kind at a target (a body locus, the fused shape, a chain / deform op, the palette,
 * the carrier) to a default gene of the new kind (keeping settings both kinds share), repaired.
 * Fails (returning the input) when the renderer would not keep that kind in this body.
 */
export function switchKind(gIn: Genome, t: Target, kind: string): EditResult {
  const g = cloneGenome(gIn);
  switch (t.t) {
    case 'locus': {
      const b = g.bodies[t.b];
      if (!b) return fail(gIn, 'no such body');
      if (!LOCUS_KINDS[t.locus].includes(kind)) return fail(gIn, `unknown ${t.locus} kind "${kind}"`);
      const old = b[t.locus] as Gene;
      if (old.kind === kind) return { genome: gIn, ok: true };
      if (t.locus === 'shape' && UNIQUE_SHAPES.includes(kind as ShapeKind) && g.bodies.some((x, j) => j !== t.b && (x.shape.kind === kind || x.fuse?.shape.kind === kind))) {
        return fail(gIn, `only one ${kind} per preset`);
      }
      const p = carryParams(old.p, locusSchema(t.locus, old.kind), locusSchema(t.locus, kind));
      const gene: Gene = t.locus === 'shape' ? repairShape({ kind, p }) : { kind, p };
      const body = cloneBody(b);
      (body as unknown as Record<string, Gene>)[t.locus] = gene;
      if (body.alt?.[t.locus]?.kind === kind) delete body.alt[t.locus];
      if (body.alt && !Object.keys(body.alt).length) delete body.alt;
      if (t.locus === 'shape' || t.locus === 'place' || t.locus === 'material') fitLoci(body);
      g.bodies[t.b] = repairBody(body);
      return finish(gIn, g, (out) => ((out.bodies[t.b][t.locus] as Gene).kind === kind ? null : `${kind} does not work with this body's ${t.locus === 'shape' ? 'placement / material' : 'shape'}`));
    }
    case 'fuseShape': {
      const b = g.bodies[t.b];
      if (!b?.fuse) return fail(gIn, 'no fused shape');
      if (b.fuse.shape.kind === kind) return { genome: gIn, ok: true };
      if (UNIQUE_SHAPES.includes(kind as ShapeKind) && g.bodies.some((x) => x.shape.kind === kind || (x !== b && x.fuse?.shape.kind === kind))) return fail(gIn, `only one ${kind} per preset`);
      b.fuse.shape = repairShape({ kind, p: carryParams(b.fuse.shape.p, SHAPE_SCHEMAS[b.fuse.shape.kind], SHAPE_SCHEMAS[kind as ShapeKind] ?? {}) });
      if (!sdfCapable(b.fuse.shape)) return fail(gIn, `${kind} has no distance field to fuse`);
      return finish(gIn, g, (out) => (out.bodies[t.b].fuse?.shape.kind === kind ? null : `${kind} cannot fuse into this shape`));
    }
    case 'op': {
      const o = g.chain[t.j];
      if (!o || !OP_KINDS.includes(kind as OpKind)) return fail(gIn, 'unknown op');
      g.chain[t.j] = newOp(kind as OpKind, o);
      return finish(gIn, g, () => null);
    }
    case 'drawOp': {
      const o = g.bodies[t.b]?.deform.ops?.[t.j];
      if (!o || !DRAW_OPS.includes(kind as OpKind)) return fail(gIn, 'unknown deform op');
      g.bodies[t.b].deform.ops![t.j] = { ...newOp(kind as OpKind, o), stage: 'warp' };
      return finish(gIn, g, () => null);
    }
    case 'palette': {
      if (!PALETTE_KINDS.includes(kind as PaletteKind)) return fail(gIn, 'unknown palette');
      g.palette = { kind: kind as PaletteKind, p: carryParams(g.palette.p, PALETTE_SCHEMAS[g.palette.kind], PALETTE_SCHEMAS[kind as PaletteKind]) };
      if ('hue' in g.palette.p) g.palette.p.hue = gIn.palette.p.hue;
      return finish(gIn, g, () => null);
    }
    case 'carrier': {
      if (!CARRIER_KINDS.includes(kind as CarrierKind)) return fail(gIn, 'unknown carrier');
      g.carrier = { kind: kind as CarrierKind, p: { ...g.carrier.p } };
      return finish(gIn, g, () => null);
    }
    case 'reaction': return setReactionSource(gIn, t.j, kind as Signal);
    case 'gene': {
      const spec = genomeGene(t.key);
      const v = geneValue(g, t.key);
      if (!spec?.kinds || !v) return fail(gIn, `no ${t.key} gene to switch`);
      if (!spec.kinds.includes(kind)) return fail(gIn, `unknown ${t.key} kind "${kind}"`);
      if (v.kind === kind) return { genome: gIn, ok: true };
      (g as unknown as Record<string, unknown>)[t.key] = { kind, p: carryParams(v.p, genomeGeneSchema(spec, v.kind), genomeGeneSchema(spec, kind)) };
      return finish(gIn, g, () => null);
    }
    default: return fail(gIn, 'this section has no kind');
  }
}

/** A default op of a kind (keeping the previous op's strength and stage where they apply). */
export function newOp(kind: OpKind, prev?: OpGene): OpGene {
  const stage: Stage = stageFree(kind) ? (prev && stageFree(prev.op) ? prev.stage : isFold(kind) ? 'view' : 'warp') : 'warp';
  const w = prev ? prev.w : isVarOp(kind) ? 0.3 : 1;
  return { op: kind, stage, w, p: defaultParams(OP_SCHEMAS[kind]) };
}

// Chain lists: reaction indices follow their op.

function remapIndexed(g: Genome, group: GeneGroup, map: (i: number) => number | null): void {
  const out: ReactionGene[] = [];
  for (const r of g.reactions) {
    if (r.g !== group) {
      out.push(r);
      continue;
    }
    const i = map(r.i);
    if (i !== null) out.push({ ...r, i });
  }
  g.reactions = out;
}

export function addOp(gIn: Genome, kind: OpKind): EditResult {
  if (gIn.chain.length >= MAX_CHAIN) return fail(gIn, `the chain holds at most ${MAX_CHAIN} ops`);
  if (!OP_KINDS.includes(kind)) return fail(gIn, 'unknown op');
  const g = cloneGenome(gIn);
  g.chain.push(newOp(kind));
  return finish(gIn, g, (out) => (out.chain.length === g.chain.length ? null : 'could not add the op'));
}

export function removeOp(gIn: Genome, j: number): EditResult {
  if (!gIn.chain[j]) return fail(gIn, 'no such op');
  const g = cloneGenome(gIn);
  g.chain.splice(j, 1);
  remapIndexed(g, 'op', (i) => (i === j ? null : i > j ? i - 1 : i));
  return finish(gIn, g, () => null);
}

export function moveOp(gIn: Genome, j: number, dir: -1 | 1): EditResult {
  const k = j + dir;
  if (!gIn.chain[j] || !gIn.chain[k]) return fail(gIn, 'cannot move there');
  const g = cloneGenome(gIn);
  [g.chain[j], g.chain[k]] = [g.chain[k], g.chain[j]];
  remapIndexed(g, 'op', (i) => (i === j ? k : i === k ? j : i));
  return finish(gIn, g, () => null);
}

export function setStage(gIn: Genome, j: number, stage: Stage): EditResult {
  const o = gIn.chain[j];
  if (!o) return fail(gIn, 'no such op');
  if (stage === 'view' && !stageFree(o.op)) return fail(gIn, `${o.op} runs in the warp only`);
  const g = cloneGenome(gIn);
  g.chain[j].stage = stage;
  return finish(gIn, g, () => null);
}

// Deform (draw-space) ops of a body: reaction index body * MAX_DRAW + op.

export function addDrawOp(gIn: Genome, b: number, kind: OpKind): EditResult {
  const body = gIn.bodies[b];
  if (!body) return fail(gIn, 'no such body');
  if ((body.deform.ops?.length ?? 0) >= MAX_DRAW) return fail(gIn, `a body bends through at most ${MAX_DRAW} ops`);
  if (!DRAW_OPS.includes(kind)) return fail(gIn, `${kind} cannot bend a body`);
  const g = cloneGenome(gIn);
  (g.bodies[b].deform.ops ??= []).push({ ...newOp(kind), stage: 'warp' });
  return finish(gIn, g, (out) => ((out.bodies[b].deform.ops?.length ?? 0) === g.bodies[b].deform.ops!.length ? null : 'could not add the op'));
}

export function removeDrawOp(gIn: Genome, b: number, j: number): EditResult {
  if (!gIn.bodies[b]?.deform.ops?.[j]) return fail(gIn, 'no such op');
  const g = cloneGenome(gIn);
  const ops = g.bodies[b].deform.ops!;
  ops.splice(j, 1);
  if (!ops.length) delete g.bodies[b].deform.ops;
  const base = b * MAX_DRAW;
  remapIndexed(g, 'dr', (i) => (i < base || i >= base + MAX_DRAW ? i : i === base + j ? null : i > base + j ? i - 1 : i));
  return finish(gIn, g, () => null);
}

export function moveDrawOp(gIn: Genome, b: number, j: number, dir: -1 | 1): EditResult {
  const ops = gIn.bodies[b]?.deform.ops;
  const k = j + dir;
  if (!ops?.[j] || !ops[k]) return fail(gIn, 'cannot move there');
  const g = cloneGenome(gIn);
  const o = g.bodies[b].deform.ops!;
  [o[j], o[k]] = [o[k], o[j]];
  const base = b * MAX_DRAW;
  remapIndexed(g, 'dr', (i) => (i === base + j ? base + k : i === base + k ? base + j : i));
  return finish(gIn, g, () => null);
}

// Flame transforms.

export function addXform(gIn: Genome, b: number): EditResult {
  const xs = gIn.bodies[b]?.shape.xforms;
  if (!xs) return fail(gIn, 'not a flame');
  if (xs.length >= MAX_XFORMS) return fail(gIn, `a flame has at most ${MAX_XFORMS} transforms`);
  const g = cloneGenome(gIn);
  g.bodies[b].shape.xforms!.push(repairXform({ aff: [0.5, 0, 0, 0.5, 0.3, 0.2], weight: 0.5, color: Math.random(), vars: { sinusoidal: 1 } }));
  return finish(gIn, g, () => null);
}

export function removeXform(gIn: Genome, b: number, j: number): EditResult {
  const xs = gIn.bodies[b]?.shape.xforms;
  if (!xs?.[j]) return fail(gIn, 'no such transform');
  if (xs.length <= 1) return fail(gIn, 'a flame needs at least one transform');
  const g = cloneGenome(gIn);
  g.bodies[b].shape.xforms!.splice(j, 1);
  return finish(gIn, g, () => null);
}

export function addVariation(gIn: Genome, b: number, j: number, v: FlameVar): EditResult {
  const x = gIn.bodies[b]?.shape.xforms?.[j];
  if (!x || !FLAME_VARIATIONS.includes(v)) return fail(gIn, 'no such transform');
  const g = cloneGenome(gIn);
  g.bodies[b].shape.xforms![j].vars[v] = 0.5;
  return finish(gIn, g, () => null);
}

// Reactions.

export interface ReactTarget {
  g: GeneGroup;
  i: number;
  k: string;
  /** Where the parameter lives (the option group), e.g. "Body 1 · Shape (dot)". */
  group: string;
}
export const reactKey = (r: Pick<ReactionGene, 'g' | 'i' | 'k'>): string => `${r.g}|${r.i}|${r.k}`;

/** Every parameter a reaction may drive, grouped by locus. */
export function reactionTargets(g: Genome): ReactTarget[] {
  const out: ReactTarget[] = [];
  const add = (grp: GeneGroup, i: number, group: string) => {
    const s = schemaFor(g, grp, i);
    if (!s) return;
    for (const k of reactable(s)) out.push({ g: grp, i, k, group });
  };
  const nb = g.bodies.length > 1;
  g.bodies.forEach((b, bi) => {
    const pre = nb ? `Body ${bi + 1} · ` : '';
    for (const locus of LOCI) add(BODY_GROUPS[locus], bi, `${pre}${LOCUS_TITLE[locus]} (${(b[locus] as Gene).kind})`);
    if (b.fuse) {
      add('fu', bi, `${pre}Fuse`);
      add('fs', bi, `${pre}Fused shape (${b.fuse.shape.kind})`);
    }
    b.deform.ops?.forEach((o, j) => add('dr', bi * MAX_DRAW + j, `${pre}Deform op ${j + 1} (${o.op})`));
  });
  g.chain.forEach((o, j) => add('op', j, `Chain ${j + 1} (${o.op})`));
  add('pal', 0, `Palette (${g.palette.kind})`);
  add('col', 0, 'Tone');
  add('car', 0, `Carrier (${g.carrier.kind})`);
  return out;
}

/** Targets no reaction drives yet. */
export function freeTargets(g: Genome, except = -1): ReactTarget[] {
  const used = new Set(g.reactions.filter((_r, j) => j !== except).map(reactKey));
  return reactionTargets(g).filter((t) => !used.has(reactKey(t)));
}

export function targetLabel(g: Genome, r: Pick<ReactionGene, 'g' | 'i' | 'k'>): string {
  const t = reactionTargets(g).find((x) => reactKey(x) === reactKey(r));
  return t ? `${t.group.replace(/ \(.*\)$/, '')} · ${labelFor(r.k)}` : `${r.g}${r.i}.${r.k}`;
}

export function addReaction(gIn: Genome, src: Signal = 'bass'): EditResult {
  if (gIn.reactions.length >= MAX_REACTIONS) return fail(gIn, `at most ${MAX_REACTIONS} reactions`);
  const free = freeTargets(gIn);
  if (!free.length) return fail(gIn, 'every parameter is already driven');
  const pick = free.find((t) => t.g === 'ma' && t.k === 'gain') ?? free.find((t) => t.g === 'sh') ?? free[0];
  const g = cloneGenome(gIn);
  g.reactions.push({ src, g: pick.g, i: pick.i, k: pick.k, gain: 0.3, atk: 0.01, rel: 0.3, thr: 0, q: 0, div: 1 });
  return finish(gIn, g, (out) => (out.reactions.length === g.reactions.length ? null : 'could not add the reaction'));
}

export function removeReaction(gIn: Genome, j: number): EditResult {
  if (!gIn.reactions[j]) return fail(gIn, 'no such reaction');
  const g = cloneGenome(gIn);
  g.reactions.splice(j, 1);
  return finish(gIn, g, () => null);
}

export function setReactionSource(gIn: Genome, j: number, src: Signal): EditResult {
  if (!gIn.reactions[j] || !SIGNALS.includes(src)) return fail(gIn, 'unknown signal');
  const g = cloneGenome(gIn);
  g.reactions[j].src = src;
  return finish(gIn, g, (out) => (out.reactions.length === g.reactions.length ? null : 'could not rewire'));
}

/** Points a reaction at another parameter; a parameter is never driven twice. */
export function setReactionTarget(gIn: Genome, j: number, target: Pick<ReactionGene, 'g' | 'i' | 'k'>): EditResult {
  const r = gIn.reactions[j];
  if (!r) return fail(gIn, 'no such reaction');
  if (!freeTargets(gIn, j).some((t) => reactKey(t) === reactKey(target))) return fail(gIn, 'that parameter is already driven (or cannot react)');
  const g = cloneGenome(gIn);
  Object.assign(g.reactions[j], { g: target.g, i: target.i, k: target.k });
  return finish(gIn, g, (out) => (out.reactions[j] && reactKey(out.reactions[j]) === reactKey(target) ? null : 'could not rewire'));
}

// Silent alleles and fuses.

/** Expresses a body's silent allele; the expressed gene goes silent in its place. */
export function expressAllele(gIn: Genome, b: number, locus: Locus): EditResult {
  const silent = gIn.bodies[b]?.alt?.[locus];
  if (!silent) return fail(gIn, 'no silent allele there');
  const g = cloneGenome(gIn);
  const body = g.bodies[b];
  const shown = body[locus] as Gene;
  if (locus === 'shape' && UNIQUE_SHAPES.includes(silent.kind as ShapeKind) && g.bodies.some((x, j) => j !== b && (x.shape.kind === silent.kind || x.fuse?.shape.kind === silent.kind))) {
    return fail(gIn, `only one ${silent.kind} per preset`);
  }
  (body as unknown as Record<string, Gene>)[locus] = locus === 'shape' ? repairShape(silent) : silent;
  if (shown.kind === 'flame') delete body.alt![locus];
  else body.alt![locus] = { kind: shown.kind, p: { ...shown.p } };
  if (!Object.keys(body.alt!).length) delete body.alt;
  fitLoci(body);
  g.bodies[b] = repairBody(body);
  return finish(gIn, g, (out) => ((out.bodies[b][locus] as Gene).kind === silent.kind ? null : `${silent.kind} does not work with this body`));
}

export function removeFuse(gIn: Genome, b: number): EditResult {
  if (!gIn.bodies[b]?.fuse) return fail(gIn, 'no fused shape');
  const g = cloneGenome(gIn);
  delete g.bodies[b].fuse;
  return finish(gIn, g, () => null);
}

// Registered genome-wide genes.

/** Adds an optional genome-wide gene (its default kind and params). */
export function addGenomeGene(gIn: Genome, key: string, kind?: string): EditResult {
  const spec = genomeGene(key);
  if (!spec) return fail(gIn, `unknown gene "${key}"`);
  if (geneValue(gIn, key)) return { genome: gIn, ok: true };
  const g = cloneGenome(gIn);
  (g as unknown as Record<string, unknown>)[key] = createGenomeGene(spec, kind);
  return finish(gIn, g, (out) => (geneValue(out, key) ? null : `could not add ${spec.title.toLowerCase()}`));
}

/** Removes an optional genome-wide gene. */
export function removeGenomeGene(gIn: Genome, key: string): EditResult {
  const spec = genomeGene(key);
  if (!spec) return fail(gIn, `unknown gene "${key}"`);
  if (!spec.optional) return fail(gIn, `${spec.title} cannot be removed`);
  if (!geneValue(gIn, key)) return { genome: gIn, ok: true };
  const g = cloneGenome(gIn);
  delete (g as unknown as Record<string, unknown>)[key];
  return finish(gIn, g, () => null);
}

// Bodies.

export interface NewBody {
  shape: ShapeKind;
  place?: string;
  material?: string;
  emit?: string;
  motion?: string;
}

/** Adds a body of default genes (the kinds given, sensible defaults for the rest), up to MAX_BODIES. */
export function addBody(gIn: Genome, spec: NewBody): EditResult {
  if (gIn.bodies.length >= MAX_BODIES) return fail(gIn, `a preset has at most ${MAX_BODIES} bodies`);
  if (!SHAPE_KINDS.includes(spec.shape)) return fail(gIn, `unknown shape "${spec.shape}"`);
  if (UNIQUE_SHAPES.includes(spec.shape) && gIn.bodies.some((x) => x.shape.kind === spec.shape || x.fuse?.shape.kind === spec.shape)) {
    return fail(gIn, `only one ${spec.shape} per preset`);
  }
  const pick = (locus: Locus, kind: string | undefined, def: string): Gene => {
    const k = kind && LOCUS_KINDS[locus].includes(kind) ? kind : def;
    return { kind: k, p: defaultParams(locusSchema(locus, k)) };
  };
  for (const [locus, kind] of [['place', spec.place], ['material', spec.material], ['emit', spec.emit], ['motion', spec.motion]] as [Locus, string | undefined][]) {
    if (kind !== undefined && !LOCUS_KINDS[locus].includes(kind)) return fail(gIn, `unknown ${locus} kind "${kind}"`);
  }
  const cls = SHAPE_CLASS[spec.shape];
  const body = {
    shape: repairShape({ kind: spec.shape }),
    place: pick('place', spec.place, 'point'),
    motion: pick('motion', spec.motion, cls === 'sdf' ? 'spin' : 'none'),
    deform: pick('deform', undefined, 'none'),
    material: pick('material', spec.material, cls === 'curve' ? 'line' : 'glow'),
    emit: pick('emit', spec.emit, 'trail'),
    feel: pick('feel', undefined, 'flow'),
    color: pick('color', undefined, 'fixed'),
  } as BodyGene;
  fitLoci(body);
  const g = cloneGenome(gIn);
  g.bodies.push(repairBody(body));
  return finish(gIn, g, (out) => (out.bodies[out.bodies.length - 1].shape.kind === spec.shape ? null : `${spec.shape} does not work with that placement / material`));
}

/** Removes a body (a preset keeps at least one); reactions on it go, later bodies' reactions follow their body. */
export function removeBody(gIn: Genome, b: number): EditResult {
  if (!gIn.bodies[b]) return fail(gIn, 'no such body');
  if (gIn.bodies.length <= 1) return fail(gIn, 'a preset needs at least one body');
  const g = cloneGenome(gIn);
  g.bodies.splice(b, 1);
  const bodyGroups = new Set<GeneGroup>([...Object.values(BODY_GROUPS), 'fu', 'fs']);
  g.reactions = g.reactions.flatMap((r) => {
    if (r.g === 'dr') {
      const ob = Math.floor(r.i / MAX_DRAW);
      if (ob === b) return [];
      return [ob > b ? { ...r, i: r.i - MAX_DRAW } : r];
    }
    if (!bodyGroups.has(r.g)) return [r];
    if (r.i === b) return [];
    return [r.i > b ? { ...r, i: r.i - 1 } : r];
  });
  return finish(gIn, g, (out) => (out.bodies.length === g.bodies.length ? null : 'could not remove the body'));
}

/** Adds a reaction driving one specific parameter (never one that is already driven). */
export function addReactionTo(gIn: Genome, src: Signal, target: Pick<ReactionGene, 'g' | 'i' | 'k'>, gain = 0.3): EditResult {
  if (gIn.reactions.length >= MAX_REACTIONS) return fail(gIn, `at most ${MAX_REACTIONS} reactions`);
  if (!SIGNALS.includes(src)) return fail(gIn, `unknown signal "${src}"`);
  if (!reactionTargets(gIn).some((t) => reactKey(t) === reactKey(target))) return fail(gIn, 'that parameter cannot react');
  if (gIn.reactions.some((r) => reactKey(r) === reactKey(target))) return fail(gIn, 'that parameter is already driven by a reaction');
  const g = cloneGenome(gIn);
  g.reactions.push({ src, g: target.g, i: target.i, k: target.k, gain: clampTo(gain, REACTION_SCHEMA.gain), atk: 0.01, rel: 0.3, thr: 0, q: 0, div: 1 });
  return finish(gIn, g, (out) => (out.reactions.length === g.reactions.length ? null : 'could not add the reaction'));
}

/** Fuses a second shape into a body (mode 0 union, 1 morph, 2 region). */
export function addFuse(gIn: Genome, b: number, shape: ShapeKind, mode = 0): EditResult {
  const body = gIn.bodies[b];
  if (!body) return fail(gIn, 'no such body');
  if (!FUSE_SHAPE_KINDS.includes(shape)) return fail(gIn, `${shape} has no distance field to fuse`);
  if (UNIQUE_SHAPES.includes(shape) && gIn.bodies.some((x) => x.shape.kind === shape || x.fuse?.shape.kind === shape)) return fail(gIn, `only one ${shape} per preset`);
  const g = cloneGenome(gIn);
  g.bodies[b].fuse = { shape: repairShape({ kind: shape }), p: { ...defaultParams(FUSE_SCHEMA), mode } };
  g.bodies[b] = repairBody(g.bodies[b]);
  return finish(gIn, g, (out) => (out.bodies[b].fuse?.shape.kind === shape ? null : `${shape} cannot fuse into this ${body.shape.kind}`));
}

// ------------------------------------------------------------------ saving

/** Parses shared genome JSON (a genome, or a member / object holding one) into a valid genome. */
export function parseGenome(text: string): EditResult & { genome: Genome } {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return { genome: repair({}), ok: false, reason: 'not valid JSON' };
  }
  const obj = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : null;
  const src = obj && obj.genome && typeof obj.genome === 'object' ? obj.genome : obj;
  const s = src as Record<string, unknown> | null;
  if (!s || !Array.isArray(s.chain) || !(Array.isArray(s.bodies) || Array.isArray(s.emitters))) return { genome: repair({}), ok: false, reason: 'no genome in that JSON' };
  const g = repairKeeping(s as unknown as Genome);
  const errs = validate(g);
  return errs.length ? { genome: g, ok: false, reason: errs[0] } : { genome: g, ok: true };
}

/**
 * Saves an edited genome as a new member: repaired, a child of the original (generation parent + 1,
 * id G{gen}-nnnn, a descriptive name inherited from the parent) tagged 'edited'.
 */
export function saveEdited(pop: Population, parent: Member | null, g: Genome, now = Date.now()): Member {
  const child = pop.addChild(repairKeeping(g), parent ? [parent] : [], now);
  child.cross = 'edited';
  return child;
}



// Gene chat edits: the compact edit list the model replies with, its JSON schema
// (built from the current genome, so paths and kinds are enums the decoder
// cannot leave), and applying it through the gene editor's own edit functions
// (geneEdit.ts), which keep every limit: body count, reactions, chain and
// deform-op lengths, a parameter never driven twice, unique shapes, repair.
// Pure logic, no DOM: runs in the Node tests.

import {
  COST_BUDGET_MS, DRAW_OPS, LOCI, LOCUS_KINDS, MAX_BODIES, OP_KINDS, PALETTE_KINDS, CARRIER_KINDS, SHAPE_KINDS,
  SIGNALS, MAX_CHAIN, OP_SCHEMAS, PALETTE_SCHEMAS, cloneGenome, estimateCost, locusSchema, repair, validate,
  type GeneGroup, type Genome, type Locus, type OpKind, type ParamSpec, type ShapeKind, type Signal,
} from '../v2/genome';
import * as E from '../v2/geneEdit';

// ------------------------------------------------------------ edit types

export type Edit =
  | { op: 'set'; path: string; value: number | string }
  | { op: 'mul'; path: string; by: number }
  | { op: 'kind'; path: string; kind: string }
  | { op: 'add_body'; shape: string; place?: string; material?: string; emit?: string }
  | { op: 'remove_body'; body: number }
  | { op: 'add_op'; kind: string; stage?: string }
  | { op: 'remove_op'; index: number }
  | { op: 'move_op'; index: number; dir: number }
  | { op: 'add_deform_op'; body: number; kind: string }
  | { op: 'remove_deform_op'; body: number; index: number }
  | { op: 'add_reaction'; signal: string; path: string; gain: number }
  | { op: 'remove_reaction'; index: number }
  | { op: 'add_xform'; body: number }
  | { op: 'remove_xform'; body: number; index: number }
  | { op: 'fuse'; body: number; shape: string; mode?: string }
  | { op: 'unfuse'; body: number }
  | { op: 'express'; body: number; locus: string };

export interface Reply {
  say: string;
  edits: Edit[];
}

// ------------------------------------------------------------ addressing

/**
 * A parameter path is the editor's section id plus the key: b0.shape.r, b1.material.gain, b0.fuse.k,
 * b0.fuseShape.r, b0.drawOp1.amt, b0.xform0.weight, b0.xform0.var.julia, op2.rate, op2.w,
 * carrier.halfLife, palette.hue, tone.exposure, reaction0.gain, energy.lo.
 */
export function parsePath(path: string): { target: E.Target; key: string } | null {
  let m = /^b(\d)\.(shape|place|motion|deform|material|emit|feel|color)\.([A-Za-z0-9]+)$/.exec(path);
  if (m) return { target: { t: 'locus', b: +m[1], locus: m[2] as Locus }, key: m[3] };
  m = /^b(\d)\.(fuse|fuseShape)\.([A-Za-z0-9]+)$/.exec(path);
  if (m) return { target: { t: m[2] as 'fuse' | 'fuseShape', b: +m[1] }, key: m[3] };
  m = /^b(\d)\.(drawOp|xform)(\d)\.((?:var\.)?[A-Za-z0-9]+)$/.exec(path);
  if (m) return { target: { t: m[2] as 'drawOp' | 'xform', b: +m[1], j: +m[3] }, key: m[4] };
  m = /^(op|reaction)(\d)\.([A-Za-z0-9]+)$/.exec(path);
  if (m) return { target: { t: m[1] as 'op' | 'reaction', j: +m[2] }, key: m[3] };
  m = /^(carrier|palette|tone|energy)\.([A-Za-z0-9]+)$/.exec(path);
  if (m) return { target: { t: m[1] as 'carrier' }, key: m[2] };
  return null;
}

/** A kind path: a body locus (b0.shape), b0.fuseShape, b0.drawOp1, opN, palette, carrier, reactionN (its signal). */
export function parseKindPath(path: string): E.Target | null {
  let m = /^b(\d)\.(shape|place|motion|deform|material|emit|feel|color)$/.exec(path);
  if (m) return { t: 'locus', b: +m[1], locus: m[2] as Locus };
  m = /^b(\d)\.fuseShape$/.exec(path);
  if (m) return { t: 'fuseShape', b: +m[1] };
  m = /^b(\d)\.drawOp(\d)$/.exec(path);
  if (m) return { t: 'drawOp', b: +m[1], j: +m[2] };
  m = /^(op|reaction)(\d)$/.exec(path);
  if (m) return { t: m[1] as 'op' | 'reaction', j: +m[2] };
  if (path === 'palette' || path === 'carrier') return { t: path };
  return null;
}

/** Every parameter path the genome has now, with its spec. */
export function paramPaths(g: Genome): { path: string; spec: ParamSpec; value: number }[] {
  const out: { path: string; spec: ParamSpec; value: number }[] = [];
  for (const sec of E.buildModel(g)) {
    const add = (t: E.Target | undefined, params: E.ParamControl[]) => {
      if (!t) return;
      const id = E.targetId(t);
      for (const c of params) out.push({ path: `${id}.${c.key}`, spec: c.spec, value: c.value });
    };
    add(sec.target, sec.params);
    for (const it of sec.items ?? []) add(it.target, it.params);
  }
  return out;
}

/**
 * Paths a set / mul may name: the current ones plus those a kind switch or an added op / body in the
 * same reply would create (every kind's parameters at each body locus, every op's at the next slot).
 */
export function settablePaths(g: Genome): string[] {
  const set = new Set(paramPaths(g).map((p) => p.path));
  const nb = Math.min(g.bodies.length + 1, MAX_BODIES);
  for (let b = 0; b < nb; b++) {
    for (const locus of LOCI) for (const kind of LOCUS_KINDS[locus]) for (const k of Object.keys(locusSchema(locus, kind))) set.add(`b${b}.${locus}.${k}`);
  }
  for (let j = 0; j <= Math.min(g.chain.length, MAX_CHAIN - 1); j++) {
    set.add(`op${j}.w`);
    for (const kind of OP_KINDS) for (const k of Object.keys(OP_SCHEMAS[kind])) set.add(`op${j}.${k}`);
  }
  for (const kind of PALETTE_KINDS) for (const k of Object.keys(PALETTE_SCHEMAS[kind])) set.add(`palette.${k}`);
  return [...set].sort();
}

// ------------------------------------------------------------ colours

/** HSV hue (0..1) of everyday colour names. */
export const COLOUR_HUES: Record<string, number> = {
  red: 0, crimson: 0.97, scarlet: 0.01, orange: 0.07, amber: 0.11, gold: 0.13, yellow: 0.16, lime: 0.24, green: 0.33, emerald: 0.4,
  teal: 0.47, cyan: 0.5, turquoise: 0.48, aqua: 0.5, sky: 0.56, blue: 0.62, navy: 0.64, indigo: 0.7, violet: 0.76, purple: 0.78,
  magenta: 0.84, pink: 0.92, rose: 0.95,
};
const HUE_KEYS = new Set(['hue']);

function wrap01(x: number): number {
  return ((x % 1) + 1) % 1;
}

/** Name of the colour nearest an absolute hue. */
export function colourName(h: number): string {
  let best = 'red';
  let bd = 9;
  for (const [n, v] of Object.entries(COLOUR_HUES)) {
    if (['crimson', 'scarlet', 'amber', 'gold', 'emerald', 'turquoise', 'aqua', 'sky', 'navy', 'rose', 'lime', 'indigo'].includes(n)) continue;
    const d = Math.min(Math.abs(v - h), 1 - Math.abs(v - h));
    if (d < bd) {
      bd = d;
      best = n;
    }
  }
  return best;
}

// ------------------------------------------------------------ applying

export interface Applied {
  genome: Genome;
  /** One line per edit that failed, with the reason (fed back to the model). */
  errors: string[];
  /** Human-readable changes that happened. */
  changes: string[];
  /** Editor section ids / parameter paths touched (controls to flash). */
  touched: string[];
  /** True when the structure changed (a recompile). */
  structural: boolean;
}

export interface ApplyContext {
  /** The song's key hue now (palette hues are offsets from it). */
  keyHue: number;
}

const describeEdit = (e: Edit): string => JSON.stringify(e);

function choiceValue(g: Genome, t: E.Target, key: string, label: string): number | null {
  const sec = [...E.buildModel(g)].flatMap((s) => [{ t: s.target, params: s.params }, ...(s.items ?? []).map((i) => ({ t: i.target, params: i.params }))])
    .find((x) => x.t && E.targetId(x.t) === E.targetId(t));
  const c = sec?.params.find((p) => p.key === key);
  const o = c?.options?.find((x) => x.label.toLowerCase() === label.toLowerCase());
  return o ? o.value : null;
}

const MODE_NAMES: Record<string, number> = { union: 0, morph: 1, region: 2 };
/** Most edits one reply may hold (keeps replies targeted and short). */
export const MAX_EDITS = 6;

/** Applies the edits in order; each failure is recorded and skipped, the rest still apply. */
export function applyEdits(gIn: Genome, edits: Edit[], ctx: ApplyContext): Applied {
  let g = gIn;
  const errors: string[] = [];
  const changes: string[] = [];
  const touched: string[] = [];
  const res = (e: Edit, r: E.EditResult, touch: string, what: string) => {
    if (!r.ok) {
      errors.push(`${describeEdit(e)} failed: ${r.reason ?? 'not possible'}`);
      return;
    }
    g = r.genome;
    touched.push(touch);
    changes.push(what);
  };
  for (const e of edits) {
    if (!e || typeof e !== 'object' || typeof (e as { op?: unknown }).op !== 'string') {
      errors.push(`${JSON.stringify(e)} is not an edit`);
      continue;
    }
    switch (e.op) {
      case 'set': case 'mul': {
        const pp = parsePath(e.path);
        const spec = pp && E.schemaAt(g, pp.target)?.[pp.key];
        if (!pp || !spec) {
          errors.push(`${describeEdit(e)} failed: ${pp && E.schemaAt(g, pp.target) ? `${e.path.replace(/\.[^.]+$/, '')} has no parameter "${pp.key}"` : `no such path ${e.path}`}`);
          continue;
        }
        let v: number;
        const old = E.getParam(g, pp.target, pp.key);
        if (e.op === 'mul') {
          if (typeof e.by !== 'number' || !Number.isFinite(e.by)) {
            errors.push(`${describeEdit(e)} failed: "by" must be a number`);
            continue;
          }
          v = old * e.by;
          // A zero parameter cannot be scaled: nudge it off zero toward the side asked for.
          if (old === 0 && e.by > 1) v = spec.min + (spec.max - spec.min) * 0.15 * Math.min(4, e.by - 1 + 0.5);
        } else if (typeof e.value === 'string') {
          const name = e.value.trim().toLowerCase();
          if (HUE_KEYS.has(pp.key) && name in COLOUR_HUES) {
            const abs = COLOUR_HUES[name];
            if (pp.target.t === 'palette') v = wrap01(abs - ctx.keyHue);
            else if (pp.target.t === 'locus' && pp.target.locus === 'color') v = wrap01(abs - (ctx.keyHue + g.palette.p.hue));
            else v = abs;
          } else {
            const c = choiceValue(g, pp.target, pp.key, name);
            if (c === null && !Number.isFinite(Number(name))) {
              errors.push(`${describeEdit(e)} failed: "${e.value}" is not a value of ${e.path}`);
              continue;
            }
            v = c ?? Number(name);
          }
        } else if (typeof e.value === 'number' && Number.isFinite(e.value)) v = e.value;
        else {
          errors.push(`${describeEdit(e)} failed: value must be a number`);
          continue;
        }
        const clamped = E.clampTo(v, spec);
        const r = E.editParam(cloneGenome(g), pp.target, pp.key, clamped);
        g = r.genome;
        const now = E.getParam(g, pp.target, pp.key);
        if (now === old) break;
        touched.push(e.path);
        changes.push(`${e.path} ${E.formatValue(old, spec)} → ${E.formatValue(now, spec)}${Math.abs(clamped - v) > 1e-9 && !spec.choices ? ` (clamped to ${spec.min}..${spec.max})` : ''}`);
        if (Math.abs(clamped - v) > 1e-9 && !spec.choices && e.op === 'set') errors.push(`note: ${e.path} range is ${spec.min}..${spec.max}; ${v} was clamped`);
        break;
      }
      case 'kind': {
        const t = parseKindPath(e.path);
        if (!t) {
          errors.push(`${describeEdit(e)} failed: no such kind path ${e.path}`);
          continue;
        }
        res(e, E.switchKind(g, t, e.kind), E.targetId(t), `${e.path} → ${e.kind}`);
        break;
      }
      case 'add_body': res(e, E.addBody(g, { shape: e.shape as ShapeKind, place: e.place, material: e.material, emit: e.emit }), `b${g.bodies.length}.shape`, `new body: ${e.shape}`); break;
      case 'remove_body': res(e, E.removeBody(g, e.body), 'bodies', `removed body ${e.body}`); break;
      case 'add_op': {
        const r = E.addOp(g, e.kind as OpKind);
        if (r.ok && e.stage === 'view') {
          const j = r.genome.chain.length - 1;
          const s = E.setStage(r.genome, j, 'view');
          res(e, s.ok ? s : r, 'chain', `space op ${e.kind}${s.ok ? ' (view)' : ''}`);
        } else res(e, r, 'chain', `space op ${e.kind}`);
        break;
      }
      case 'remove_op': res(e, E.removeOp(g, e.index), 'chain', `removed space op ${e.index}`); break;
      case 'move_op': res(e, E.moveOp(g, e.index, e.dir < 0 ? -1 : 1), 'chain', `moved space op ${e.index}`); break;
      case 'add_deform_op': res(e, E.addDrawOp(g, e.body, e.kind as OpKind), `b${e.body}.deform`, `body ${e.body} bends through ${e.kind}`); break;
      case 'remove_deform_op': res(e, E.removeDrawOp(g, e.body, e.index), `b${e.body}.deform`, `body ${e.body} drops deform op ${e.index}`); break;
      case 'add_reaction': {
        const pp = parsePath(e.path);
        const grp = pp && reactionGroup(g, pp.target);
        if (!pp || !grp) {
          errors.push(`${describeEdit(e)} failed: ${e.path} cannot be driven by a reaction`);
          continue;
        }
        res(e, E.addReactionTo(g, e.signal as Signal, { ...grp, k: pp.key }, typeof e.gain === 'number' ? e.gain : 0.3), 'reactions', `${e.signal} drives ${e.path}`);
        break;
      }
      case 'remove_reaction': res(e, E.removeReaction(g, e.index), 'reactions', `removed reaction ${e.index}`); break;
      case 'add_xform': res(e, E.addXform(g, e.body), `b${e.body}.shape`, `flame transform added`); break;
      case 'remove_xform': res(e, E.removeXform(g, e.body, e.index), `b${e.body}.shape`, `flame transform ${e.index} removed`); break;
      case 'fuse': res(e, E.addFuse(g, e.body, e.shape as ShapeKind, MODE_NAMES[e.mode ?? 'union'] ?? 0), `b${e.body}.fuse`, `body ${e.body} fuses a ${e.shape}`); break;
      case 'unfuse': res(e, E.removeFuse(g, e.body), `b${e.body}.shape`, `body ${e.body} unfused`); break;
      case 'express': res(e, E.expressAllele(g, e.body, e.locus as Locus), `b${e.body}.${e.locus}`, `body ${e.body} expresses its silent ${e.locus}`); break;
      default: errors.push(`${describeEdit(e)} failed: unknown op`);
    }
  }
  const cost = estimateCost(g);
  if (cost > COST_BUDGET_MS) errors.push(`the preset now costs ${cost.toFixed(1)} ms per frame, over the ${COST_BUDGET_MS} ms budget: use fewer copies, particles or bodies`);
  const bad = validate(g);
  if (bad.length) {
    g = repair(g);
    changes.push('repaired to a valid genome');
  }
  return { genome: g, errors, changes, touched, structural: false };
}

/** Reaction group / index of a parameter target (null when reactions cannot address it). */
function reactionGroup(g: Genome, t: E.Target): { g: GeneGroup; i: number } | null {
  const BG: Record<Locus, GeneGroup> = { shape: 'sh', place: 'pl', motion: 'mo', deform: 'de', material: 'ma', emit: 'em', feel: 'fe', color: 'cm' };
  switch (t.t) {
    case 'locus': return g.bodies[t.b] ? { g: BG[t.locus], i: t.b } : null;
    case 'fuse': return { g: 'fu', i: t.b };
    case 'fuseShape': return { g: 'fs', i: t.b };
    case 'drawOp': return { g: 'dr', i: t.b * 3 + t.j };
    case 'op': return { g: 'op', i: t.j };
    case 'carrier': return { g: 'car', i: 0 };
    case 'tone': return { g: 'col', i: 0 };
    case 'palette': return { g: 'pal', i: 0 };
    default: return null;
  }
}

// ------------------------------------------------------------ schema

const NUM = { type: 'number' };
const INT = { type: 'integer' };
const str = (values: readonly string[]) => ({ type: 'string', enum: [...values] });
function obj(op: string, props: Record<string, object>, optional: string[] = []): object {
  return {
    type: 'object',
    properties: { op: { type: 'string', enum: [op] }, ...props },
    required: ['op', ...Object.keys(props).filter((k) => !optional.includes(k))],
    additionalProperties: false,
  };
}

/** JSON schema of a reply for this genome: every path and kind an enum. */
export function replySchema(g: Genome): object {
  const paths = settablePaths(g);
  // Multiplying only makes sense for continuous parameters (not switches or coded choices).
  const specs = new Map(paramPaths(g).map((p) => [p.path, p.spec]));
  const scalable = paths.filter((p) => !specs.get(p)?.choices);
  const reactable = E.reactionTargets(g).map((t) => {
    const id = t.g === 'op' ? `op${t.i}` : t.g === 'car' ? 'carrier' : t.g === 'col' ? 'tone' : t.g === 'pal' ? 'palette'
      : t.g === 'dr' ? `b${Math.floor(t.i / 3)}.drawOp${t.i % 3}` : t.g === 'fu' ? `b${t.i}.fuse` : t.g === 'fs' ? `b${t.i}.fuseShape`
      : `b${t.i}.${({ sh: 'shape', pl: 'place', mo: 'motion', de: 'deform', ma: 'material', em: 'emit', fe: 'feel', cm: 'color' } as Record<string, string>)[t.g]}`;
    return `${id}.${t.k}`;
  });
  const kindPaths: string[] = ['palette', 'carrier'];
  g.bodies.forEach((b, bi) => {
    for (const l of LOCI) kindPaths.push(`b${bi}.${l}`);
    if (b.fuse) kindPaths.push(`b${bi}.fuseShape`);
    b.deform.ops?.forEach((_o, j) => kindPaths.push(`b${bi}.drawOp${j}`));
  });
  g.chain.forEach((_o, j) => kindPaths.push(`op${j}`));
  g.reactions.forEach((_r, j) => kindPaths.push(`reaction${j}`));
  const allKinds = [...new Set([...SHAPE_KINDS, ...LOCI.flatMap((l) => LOCUS_KINDS[l]), ...OP_KINDS, ...PALETTE_KINDS, ...CARRIER_KINDS, ...SIGNALS])];
  const bodyIdx = { type: 'integer', minimum: 0, maximum: Math.max(0, g.bodies.length - 1) };
  const variants = [
    obj('set', { path: str(paths), value: { anyOf: [NUM, { type: 'string' }] } }),
    obj('mul', { path: str(scalable.length ? scalable : paths), by: NUM }),
    obj('kind', { path: str(kindPaths), kind: str(allKinds) }),
    obj('add_body', { shape: str(SHAPE_KINDS), place: str(LOCUS_KINDS.place), material: str(LOCUS_KINDS.material), emit: str(LOCUS_KINDS.emit) }, ['place', 'material', 'emit']),
    obj('remove_body', { body: bodyIdx }),
    obj('add_op', { kind: str(OP_KINDS), stage: str(['warp', 'view']) }, ['stage']),
    obj('remove_op', { index: INT }),
    obj('move_op', { index: INT, dir: INT }),
    obj('add_deform_op', { body: bodyIdx, kind: str(DRAW_OPS) }),
    obj('remove_deform_op', { body: bodyIdx, index: INT }),
    obj('add_reaction', { signal: str(SIGNALS), path: str(reactable.length ? reactable : ['tone.exposure']), gain: NUM }),
    obj('remove_reaction', { index: INT }),
    obj('fuse', { body: bodyIdx, shape: str(E.FUSE_SHAPE_KINDS), mode: str(Object.keys(MODE_NAMES)) }, ['mode']),
    obj('unfuse', { body: bodyIdx }),
  ];
  if (g.bodies.some((b) => b.shape.kind === 'flame')) variants.push(obj('add_xform', { body: bodyIdx }), obj('remove_xform', { body: bodyIdx, index: INT }));
  if (g.bodies.some((b) => b.alt)) variants.push(obj('express', { body: bodyIdx, locus: str(LOCI) }));
  return {
    type: 'object',
    properties: { say: { type: 'string' }, edits: { type: 'array', items: { anyOf: variants }, maxItems: MAX_EDITS } },
    required: ['say', 'edits'],
    additionalProperties: false,
  };
}


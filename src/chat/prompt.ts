// The gene chat's prompts: a static system prompt (edit format, gene glossary, everyday-words
// lexicon, rules) that stays byte-identical across turns so the engine keeps it cached, and per-turn
// user messages carrying the preset (in full when it is new to the conversation, as a diff after
// manual edits) plus what the screen looks like now.

import {
  CARRIER_KINDS, LOCUS_KINDS, MOTION_OPS, FOLD_OPS, PALETTE_KINDS, SIGNALS, LOCI, MAX_BODIES, MAX_CHAIN, MAX_DRAW, MAX_REACTIONS, COST_BUDGET_MS, estimateCost,
  type Gene, type Genome, type ParamSpec,
} from '../v2/genome';
import * as E from '../v2/geneEdit';
import { colourName, paramPaths } from './edits';
import { GLOSSARY, LEXICON, RULES } from './glossary';

/**
 * Kinds the schemas define that the glossary does not mention (genes added since it was written).
 * Built from the schemas at runtime, so new genes show up here until the glossary covers them.
 */
export function glossaryGaps(): string[] {
  const text = GLOSSARY.toLowerCase();
  const has = (w: string) => new RegExp(`(^|[^a-z_])${w.toLowerCase()}([^a-z_]|$)`).test(text);
  const kinds = [...LOCI.flatMap((l) => LOCUS_KINDS[l].map((k) => [l, k] as const)), ...[...MOTION_OPS, ...FOLD_OPS].map((k) => ['op', k] as const),
    ...CARRIER_KINDS.map((k) => ['carrier', k] as const), ...PALETTE_KINDS.map((k) => ['palette', k] as const), ...SIGNALS.map((k) => ['signal', k] as const)];
  return kinds.filter(([, k]) => k !== 'none' && !has(k)).map(([l, k]) => `${l} ${k}`);
}
const GAPS = glossaryGaps();
if (GAPS.length) console.warn(`[chat] genes without a glossary entry: ${GAPS.join(', ')}`);

export const SYSTEM_PROMPT = `You edit a live music visualizer preset for the user. The preset is a genome: 1-3 bodies (light sources) plus a space chain, a carrier (how light moves and fades), a palette, a tone and reactions (music signals driving parameters). Each user message shows the preset and a request. Reply with JSON only:
{"say": "<one short friendly sentence about what you changed>", "edits": [<edit>, ...]}
Edits (applied in order):
{"op":"set","path":"b0.shape.r","value":0.2}  set a parameter; value is a number in the range shown, a choice name, or for any hue a colour name ("blue")
{"op":"mul","path":"carrier.halfLife","by":2}  multiply a parameter (use for relative requests: bigger, faster, a bit less)
{"op":"kind","path":"b0.material","kind":"glow"}  switch a gene to another kind (paths: bN.<locus>, bN.fuseShape, bN.drawOpJ, opJ, palette, carrier, reactionJ = its signal)
{"op":"add_body","shape":"dot","place":"grid","material":"glow","emit":"trail"}  add a body (place, material, emit optional)
{"op":"remove_body","body":1}
{"op":"add_op","kind":"swirl","stage":"warp"}  add a space-chain op; {"op":"remove_op","index":0}; {"op":"move_op","index":1,"dir":-1}
{"op":"add_deform_op","body":0,"kind":"twist"}; {"op":"remove_deform_op","body":0,"index":0}
{"op":"add_reaction","signal":"bass","path":"b0.material.gain","gain":0.5}  gain -1..1; {"op":"remove_reaction","index":0}
{"op":"fuse","body":0,"shape":"star","mode":"morph"}  (modes union, morph, region); {"op":"unfuse","body":0}
{"op":"add_xform","body":0}; {"op":"remove_xform","body":0,"index":1}  (flame transforms)
{"op":"express","body":0,"locus":"shape"}  swap in a body's silent allele
Paths: bN.<locus>.<param> (loci: shape place motion deform material emit feel color), bN.fuse.<p>, bN.fuseShape.<p>, bN.drawOpJ.<p> (w = strength), bN.xformJ.<p> (var.<name> = variation weight), opJ.<p> (w = strength), carrier.<p>, palette.<p>, tone.<p>, reactionJ.<p>.
Colours: to change the overall colour set palette.hue to a colour name, e.g. {"op":"set","path":"palette.hue","value":"blue"}; a body's colour offset is bN.color.hue.
Use only paths shown in the preset (after a kind switch, the new kind's params). Keep edits few and targeted: usually 1-4, at most 6, each path once. Change what the request is about and nothing else. If the request is unclear or impossible, explain in "say" and send no edits. Never invent parameters.

${GLOSSARY}

Everyday words:
${LEXICON}

${RULES}
${GAPS.length ? `Genes not described above (newer; judge them by their parameter names): ${GAPS.join(', ')}.\n` : ''}Limits: ${MAX_BODIES} bodies, ${MAX_REACTIONS} reactions, ${MAX_CHAIN} chain ops, ${MAX_DRAW} deform ops per body, cost budget ${COST_BUDGET_MS} ms.`;

// ------------------------------------------------------------ genome text

const r3 = (v: number) => {
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  return String(Number(v.toPrecision(a >= 1 ? 3 : 2)));
};
function rangeText(s: ParamSpec): string {
  if (s.choices) return s.choices.length <= 8 ? `{${s.choices.map(r3).join(',')}}` : `{${r3(s.min)}..${r3(s.max)}}`;
  return `[${r3(s.min)}..${r3(s.max)}]`;
}

/** Parameters of one target as "key=value[range]" (choice names where they exist). */
function paramsText(g: Genome, t: E.Target, ctx: string): string {
  const s = E.schemaAt(g, t);
  if (!s) return '';
  const controls = E.paramControls(s, (k) => E.getParam(g, t, k), ctx);
  return controls.map((c) => {
    if (c.options) {
      const named = c.options.find((o) => o.value === c.value)?.label;
      const labels = c.options.map((o) => o.label);
      const numeric = labels.every((l, i) => l === String(c.options![i].value));
      return numeric ? `${c.key}=${r3(c.value)}${rangeText(c.spec)}` : `${c.key}=${named}{${labels.join('|')}}`;
    }
    return `${c.key}=${r3(c.value)}${rangeText(c.spec)}`;
  }).join(' ');
}

export interface LookMetrics {
  /** Mean luma 0..1. */
  brightness: number;
  /** Fraction of the screen lit. */
  coverage: number;
  /** Mean change between samples (0 still .. ~0.1 very busy). */
  motion: number;
  /** Mean saturation of lit pixels 0..1. */
  colourfulness: number;
  /** Mean hue of lit pixels (0..1), -1 when grey. */
  hue: number;
}

function word(v: number, cuts: number[], words: string[]): string {
  let i = 0;
  while (i < cuts.length && v > cuts[i]) i++;
  return words[i];
}

/** What the screen looks like now, in numbers and words. */
export function lookText(m: LookMetrics | null): string {
  if (!m) return 'Screen: not measured yet.';
  return `Screen now: brightness ${m.brightness.toFixed(2)} (${word(m.brightness, [0.05, 0.12, 0.3, 0.5], ['very dark', 'dark', 'medium', 'bright', 'very bright'])}), `
    + `coverage ${m.coverage.toFixed(2)} (${word(m.coverage, [0.08, 0.25, 0.6], ['mostly empty', 'sparse', 'half filled', 'filled'])}), `
    + `motion ${m.motion.toFixed(3)} (${word(m.motion, [0.004, 0.015, 0.04], ['still', 'gentle', 'lively', 'frantic'])}), `
    + `colourfulness ${m.colourfulness.toFixed(2)} (${word(m.colourfulness, [0.15, 0.4, 0.7], ['grey', 'muted', 'colourful', 'vivid'])})`
    + `${m.hue >= 0 && m.colourfulness > 0.15 ? `, mostly ${colourName(m.hue)}` : ''}.`;
}

/** The whole preset, expressed genes only, one line per gene. */
export function genomeText(g: Genome, keyHue: number): string {
  const lines: string[] = [];
  g.bodies.forEach((b, bi) => {
    lines.push(`Body b${bi}:`);
    for (const locus of LOCI) {
      const gene = b[locus] as Gene;
      const t: E.Target = { t: 'locus', b: bi, locus };
      const ps = paramsText(g, t, gene.kind);
      lines.push(`  b${bi}.${locus}=${gene.kind}${ps ? ` ${ps}` : ''}`);
      if (locus === 'shape' && b.fuse) {
        lines.push(`  b${bi}.fuse ${paramsText(g, { t: 'fuse', b: bi }, 'fuse')}`);
        lines.push(`  b${bi}.fuseShape=${b.fuse.shape.kind} ${paramsText(g, { t: 'fuseShape', b: bi }, b.fuse.shape.kind)}`);
      }
      if (locus === 'shape') b.shape.xforms?.forEach((_x, j) => lines.push(`  b${bi}.xform${j} ${paramsText(g, { t: 'xform', b: bi, j }, 'xform')}`));
      if (locus === 'deform') b.deform.ops?.forEach((o, j) => lines.push(`  b${bi}.drawOp${j}=${o.op} ${paramsText(g, { t: 'drawOp', b: bi, j }, o.op)}`));
      if (locus === 'color') {
        const abs = (keyHue + g.palette.p.hue + (gene.p.hue ?? 0)) % 1;
        lines[lines.length - 1] += ` (base colour on screen: ${colourName(abs)})`;
      }
    }
    if (b.alt) lines.push(`  silent alleles: ${Object.entries(b.alt).map(([l, a]) => `${l}=${a!.kind}`).join(', ')}`);
  });
  lines.push(g.chain.length ? 'Space chain:' : 'Space chain: empty');
  g.chain.forEach((o, j) => lines.push(`  op${j}=${o.op} stage=${o.stage} ${paramsText(g, { t: 'op', j }, o.op)}`));
  lines.push(`carrier=${g.carrier.kind} ${paramsText(g, { t: 'carrier' }, 'carrier')}`);
  const first = (keyHue + g.palette.p.hue) % 1;
  lines.push(`palette=${g.palette.kind} ${paramsText(g, { t: 'palette' }, 'palette')} (key hue now ${colourName(keyHue)}; first palette colour on screen: ${colourName(first)})`);
  lines.push(`tone ${paramsText(g, { t: 'tone' }, 'tone')}`);
  if (!g.reactions.length) lines.push('Reactions: none');
  g.reactions.forEach((r, j) => {
    const target = paramPaths(g).find((p) => p.path.endsWith(`.${r.k}`) && E.targetLabel(g, r).length > 0);
    void target;
    lines.push(`  reaction${j}: ${r.src} -> ${reactionPath(g, r)} gain=${r3(r.gain)}[-1..1] atk=${r3(r.atk)} rel=${r3(r.rel)} thr=${r3(r.thr)}`);
  });
  lines.push(`GPU cost ${estimateCost(g).toFixed(1)} of ${COST_BUDGET_MS} ms.`);
  return lines.join('\n');
}

const GROUP_LOCUS: Record<string, string> = { sh: 'shape', pl: 'place', mo: 'motion', de: 'deform', ma: 'material', em: 'emit', fe: 'feel', cm: 'color' };
export function reactionPath(_g: Genome, r: { g: string; i: number; k: string }): string {
  if (r.g === 'op') return `op${r.i}.${r.k}`;
  if (r.g === 'car') return `carrier.${r.k}`;
  if (r.g === 'col') return `tone.${r.k}`;
  if (r.g === 'pal') return `palette.${r.k}`;
  if (r.g === 'dr') return `b${Math.floor(r.i / MAX_DRAW)}.drawOp${r.i % MAX_DRAW}.${r.k}`;
  if (r.g === 'fu') return `b${r.i}.fuse.${r.k}`;
  if (r.g === 'fs') return `b${r.i}.fuseShape.${r.k}`;
  return `b${r.i}.${GROUP_LOCUS[r.g]}.${r.k}`;
}

/** Changes between two genomes as short lines (for manual edits made between turns). */
export function genomeDiff(a: Genome, b: Genome, keyHue: number): string[] | null {
  if (JSON.stringify(a) === JSON.stringify(b)) return [];
  // Structure changed: the diff would be long and confusing, send the whole preset instead.
  const shape = (g: Genome) => JSON.stringify([g.bodies.map((x) => LOCI.map((l) => (x[l] as Gene).kind).concat(x.fuse?.shape.kind ?? '', String(x.deform.ops?.length ?? 0), String(x.shape.xforms?.length ?? 0))), g.chain.map((o) => o.op + o.stage), g.carrier.kind, g.palette.kind, g.reactions.map((r) => r.src + reactionPath(g, r))]);
  if (shape(a) !== shape(b)) return null;
  const pa = new Map(paramPaths(a).map((p) => [p.path, p.value]));
  const out: string[] = [];
  for (const p of paramPaths(b)) {
    const old = pa.get(p.path);
    if (old !== undefined && Math.abs(old - p.value) > 1e-9) out.push(`${p.path} ${r3(old)} -> ${r3(p.value)}`);
  }
  void keyHue;
  return out;
}

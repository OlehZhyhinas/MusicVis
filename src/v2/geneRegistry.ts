// Genome-wide genes beside the fixed ones (chain, carrier, palette, tone, reactions): objects on
// the genome under their own key, each described once here (title, kinds, schemas, a plain-language
// description), so the gene editor shows them and the gene chat can edit them without code of its
// own. A gene module calls registerGenomeGene() when it is imported; genome.ts keeps registered
// genes through repair() / validate() with repairGenomeGenes() / validateGenomeGenes().
//
// This module imports only types from genome.ts, so genome.ts can import it without a cycle.

import type { ParamSpec, Params, Schema } from './genome';

/** The value stored on the genome under a registered key. */
export interface GenomeGeneValue {
  /** Present when the gene has kinds. */
  kind?: string;
  p: Params;
}

export interface GenomeGeneSpec {
  /** Property on the genome, e.g. 'choreo'. Must not collide with the fixed genome fields. */
  key: string;
  /** Section title in the gene editor, e.g. 'Choreography'. */
  title: string;
  /** Kinds the gene may take; absent for a gene with one fixed schema (no kind selector). */
  kinds?: readonly string[];
  /** Schema per kind (keyed by kind), or the schema of a kind-less gene. */
  schemas: Record<string, Schema> | Schema;
  /** true: a genome may lack it (the editor offers Add / Remove); false: repair adds the default. */
  optional: boolean;
  /** Kind a new gene starts as (default: the first kind). */
  defaultKind?: string;
  /** Plain-language description for the gene chat: what it does on screen, the key params. */
  glossary?: string;
  /** Where the editor lists it: 'genome' sections after the tone (default). */
  order?: number;
}

const FIXED_KEYS = new Set(['v', 'chain', 'bodies', 'carrier', 'palette', 'tone', 'reactions', 'energy']);
const REGISTRY = new Map<string, GenomeGeneSpec>();

/** Registers a genome-wide gene (idempotent per key: the last registration wins). */
export function registerGenomeGene(spec: GenomeGeneSpec): void {
  if (FIXED_KEYS.has(spec.key)) throw new Error(`genome gene key "${spec.key}" collides with a fixed field`);
  if (!/^[a-z][A-Za-z0-9]*$/.test(spec.key)) throw new Error(`genome gene key "${spec.key}" must be a lowerCamel identifier`);
  REGISTRY.set(spec.key, spec);
}

/** Every registered genome-wide gene, in editor order. */
export function genomeGenes(): GenomeGeneSpec[] {
  return [...REGISTRY.values()].sort((a, b) => (a.order ?? 0) - (b.order ?? 0) || a.key.localeCompare(b.key));
}

export function genomeGene(key: string): GenomeGeneSpec | undefined {
  return REGISTRY.get(key);
}

const isSchemaMap = (s: GenomeGeneSpec): s is GenomeGeneSpec & { schemas: Record<string, Schema> } => !!s.kinds;

/** The schema of a gene of this spec with this kind (kind ignored for kind-less genes). */
export function genomeGeneSchema(spec: GenomeGeneSpec, kind?: string): Schema {
  if (!isSchemaMap(spec)) return spec.schemas as Schema;
  const k = kind && spec.kinds!.includes(kind) ? kind : (spec.defaultKind ?? spec.kinds![0]);
  return spec.schemas[k] ?? {};
}

function clampSpec(v: number, s: ParamSpec): number {
  if (!Number.isFinite(v)) v = s.def;
  if (s.choices) {
    let best = s.choices[0];
    for (const c of s.choices) if (Math.abs(c - v) < Math.abs(best - v)) best = c;
    return best;
  }
  v = Math.min(s.max, Math.max(s.min, v));
  return s.int ? Math.round(v) : v;
}

/** A default gene of a spec (its default kind, default params). */
export function createGenomeGene(spec: GenomeGeneSpec, kind?: string): GenomeGeneValue {
  const k = spec.kinds ? (kind && spec.kinds.includes(kind) ? kind : (spec.defaultKind ?? spec.kinds[0])) : undefined;
  const schema = genomeGeneSchema(spec, k);
  const p: Params = {};
  for (const key of Object.keys(schema)) p[key] = schema[key].def;
  return k ? { kind: k, p } : { p };
}

/** One registered gene made valid (known kind, params clamped into the schema, unknown keys dropped). */
export function repairGenomeGene(spec: GenomeGeneSpec, raw: unknown): GenomeGeneValue {
  const r = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};
  const kind = spec.kinds ? (spec.kinds.includes(r.kind as string) ? (r.kind as string) : (spec.defaultKind ?? spec.kinds[0])) : undefined;
  const schema = genomeGeneSchema(spec, kind);
  const src = r.p && typeof r.p === 'object' ? (r.p as Record<string, unknown>) : {};
  const p: Params = {};
  for (const key of Object.keys(schema)) p[key] = clampSpec(typeof src[key] === 'number' ? (src[key] as number) : schema[key].def, schema[key]);
  return kind ? { kind, p } : { p };
}

/**
 * For genome.ts repair(): copies every registered gene from `src` into `out`, repaired (optional
 * genes only when present; required genes get their default when missing).
 */
export function repairGenomeGenes(src: Record<string, unknown>, out: Record<string, unknown>): void {
  for (const spec of REGISTRY.values()) {
    const raw = src[spec.key];
    if (raw && typeof raw === 'object') out[spec.key] = repairGenomeGene(spec, raw);
    else if (!spec.optional) out[spec.key] = createGenomeGene(spec);
  }
}

/** For genome.ts validate(): the rules registered genes break (appended to `errs`). */
export function validateGenomeGenes(g: Record<string, unknown>, errs: string[]): void {
  for (const spec of REGISTRY.values()) {
    const v = g[spec.key] as GenomeGeneValue | undefined;
    if (v === undefined) {
      if (!spec.optional) errs.push(`${spec.key} missing`);
      continue;
    }
    if (!v || typeof v !== 'object' || !v.p || typeof v.p !== 'object') {
      errs.push(`${spec.key} params`);
      continue;
    }
    if (spec.kinds && !spec.kinds.includes(v.kind ?? '')) errs.push(`${spec.key} kind`);
    const schema = genomeGeneSchema(spec, v.kind);
    for (const k of Object.keys(schema)) {
      const x = v.p[k];
      const s = schema[k];
      const ok = typeof x === 'number' && Number.isFinite(x) && (s.choices ? s.choices.includes(x) : x >= s.min - 1e-9 && x <= s.max + 1e-9 && (!s.int || Number.isInteger(x)));
      if (!ok) errs.push(`${spec.key}.${k}=${x} out of range`);
    }
    for (const k of Object.keys(v.p)) if (!(k in schema)) errs.push(`${spec.key}.${k} unknown`);
  }
}

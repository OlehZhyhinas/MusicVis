// Node-side loaders for harness output (.testdata/avq/).
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { parseClip, type CfResult, type Clip, type ClipHeader } from './format';
export type { CfResult };
import type { SongData, Hook, Moment } from './music';
import { OUT } from './cdp';

export function loadClip(base: string, dir = join(OUT, 'clips')): Clip {
  const header = JSON.parse(readFileSync(join(dir, base + '.json'), 'utf8')) as ClipHeader;
  const bin = readFileSync(join(dir, base + '.bin'));
  return parseClip(header, new Uint8Array(bin.buffer, bin.byteOffset, bin.byteLength));
}

/** Clip bases (preset/song__label) present on disk, optionally filtered. */
export function listClips(filter: { preset?: string; song?: string; label?: string } = {}, dir = join(OUT, 'clips')): string[] {
  if (!existsSync(dir)) return [];
  const out: string[] = [];
  for (const p of readdirSync(dir).sort()) {
    if (filter.preset && p !== filter.preset) continue;
    const pd = join(dir, p);
    for (const f of readdirSync(pd).sort()) {
      if (!f.endsWith('.json') || f.endsWith('.inst.json') || f.endsWith('.emb.json')) continue;
      const b = f.slice(0, -5);
      const [song, label] = b.split('__');
      if (filter.song && song !== filter.song) continue;
      if (filter.label && label !== filter.label) continue;
      out.push(`${p}/${b}`);
    }
  }
  return out;
}

function f32(b64: string): Float32Array {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

export interface SongDump extends SongData {
  path: string;
  slug: string;
  hooks: Hook[];
  moments: Moment[];
  complexity: Float32Array;
}

export function loadSong(slug: string): SongDump {
  const j = JSON.parse(readFileSync(join(OUT, 'songs', slug + '.json'), 'utf8'));
  const dec = (r: Record<string, string>) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, f32(v)]));
  return { ...j, chroma: f32(j.chroma), loudness: f32(j.loudness), complexity: f32(j.complexity), stems: dec(j.stems), stemOnsets: dec(j.stemOnsets), stemPresence: dec(j.stemPresence) };
}

/** Engine instrumentation recorded with a clip (per-frame columns by name), or null. */
export function loadInst(base: string, dir = join(OUT, 'clips')): { names: string[]; cols: (number | null)[][] } | null {
  const p = join(dir, base + '.inst.json');
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null;
}


export function loadCf(base: string): CfResult | null {
  const p = join(OUT, 'cf', base + '.json');
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null;
}

/** DINOv2 frame embeddings recorded with a clip, or null. */
export function loadEmb(base: string): { idx: number[]; vecs: number[][]; every: number } | null {
  const p = join(OUT, 'clips', base + '.emb.json');
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null;
}

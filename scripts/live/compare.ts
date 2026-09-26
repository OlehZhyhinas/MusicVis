// Before / after gate for live-analysis changes: compares two live-parity runs channel by channel.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/live/compare.ts <before> <after> [--keys a,b] [--all]
// <before> / <after>: run names (.testdata/live/parity-<name>/, 'parity' for the default run).
// Prints mean quality and median latency over the single songs, the change, and the worst song.

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { OUT } from './common';

interface S { key: string; group: string; quality: number | null; latency: number | null; num: Record<string, number | null> }

function load(name: string): Map<string, Map<string, S>> {
  const dir = join(OUT, name === 'parity' ? 'parity' : `parity-${name}`);
  if (!existsSync(dir)) throw new Error('no run ' + dir);
  const out = new Map<string, Map<string, S>>();
  for (const f of readdirSync(dir)) {
    const j = JSON.parse(readFileSync(join(dir, f), 'utf8')) as { kind: string; name: string; scores: S[] };
    if (j.kind !== 'song') continue;
    out.set(j.name, new Map(j.scores.map((s) => [(s.group === 'signal' ? 'signal:' : '') + s.key, s])));
  }
  return out;
}

const mean = (a: number[]) => (a.length ? a.reduce((p, q) => p + q, 0) / a.length : NaN);
const med = (a: number[]) => (a.length ? [...a].sort((p, q) => p - q)[a.length >> 1] : NaN);
const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : ' n/a');
const sg = (x: number) => (Number.isFinite(x) ? (x >= 0 ? '+' : '') + x.toFixed(2) : '  n/a');

const argv = process.argv.slice(2);
const [before, after] = argv.filter((x) => !x.startsWith('--'));
const ki = argv.indexOf('--keys');
const keys = ki >= 0 ? argv[ki + 1].split(',') : null;
const A = load(before), B = load(after);
const songs = [...A.keys()].filter((s) => B.has(s));
const all = new Set<string>();
for (const s of songs) for (const k of A.get(s)!.keys()) all.add(k);
const rows: { k: string; qa: number; qb: number; la: number; lb: number; worst: string }[] = [];
for (const k of all) {
  if (keys && !keys.includes(k)) continue;
  const qa: number[] = [], qb: number[] = [], la: number[] = [], lb: number[] = [];
  let worst = '', wd = Infinity;
  for (const s of songs) {
    const a = A.get(s)!.get(k), b = B.get(s)!.get(k);
    if (!a || !b) continue;
    if (a.quality !== null && b.quality !== null) {
      qa.push(a.quality);
      qb.push(b.quality);
      if (b.quality - a.quality < wd) (wd = b.quality - a.quality), (worst = `${s} ${sg(wd)}`);
    }
    if (a.latency !== null) la.push(a.latency);
    if (b.latency !== null) lb.push(b.latency);
  }
  rows.push({ k, qa: mean(qa), qb: mean(qb), la: med(la), lb: med(lb), worst });
}
rows.sort((x, y) => (y.qb - y.qa || 0) - (x.qb - x.qa || 0));
console.log(`${songs.length} songs: ${before} -> ${after}`);
console.log('channel'.padEnd(26) + 'before  after  change   lat ms before > after   worst song');
for (const r of rows) {
  if (!argv.includes('--all') && !keys && Math.abs(r.qb - r.qa) < 0.01 && Math.abs((r.lb - r.la) * 1000) < 10) continue;
  console.log(`${r.k.padEnd(26)}${f2(r.qa).padStart(6)} ${f2(r.qb).padStart(6)}  ${sg(r.qb - r.qa).padStart(6)}   ${String(Number.isFinite(r.la) ? Math.round(r.la * 1000) : '').padStart(6)} > ${String(Number.isFinite(r.lb) ? Math.round(r.lb * 1000) : '').padEnd(6)}        ${r.worst}`);
}

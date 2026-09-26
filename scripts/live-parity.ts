// Live-vs-offline signal parity (CLI). Runs each test song (and the synthetic DJ mixes from
// scripts/live/mix.ts) through the offline analysis and through the live path exactly as
// LiveInput feeds it, compares every preset signal and genome input, and prints a ranked gap
// table weighted by how many seeds use each one. Audio is decoded with ffmpeg and never played.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/live-parity.ts [--songs test|slug,slug] [--mixes] [--newsong] [--no-songs] [--seconds N]
//
// --out NAME: write .testdata/live/parity-NAME/ and parity-table-NAME.md instead.
// --newsong: new-song detection on whole songs back to back (no gap, 2 s, 6 s, a 15 s ad) and on the mixes.
//
// Writes .testdata/live/parity/<input>.json (scores per input) and .testdata/live/parity-table.md.

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadEngine } from './live/bundle';
import { decode, offlineCached, OUT, slugOf, testSongs } from './live/common';
import { buildMixes, mixTruthReport, type Mix } from './live/mix';
import { newSongReport } from './live/newsong';
import { record, scoreAll, signalChannels, stateChannels, type Score } from './live/parity';
import { CORE_INPUTS, SIGNAL_SOURCE, usage } from './live/usage';

function parseArgs(argv: string[]): Record<string, string | boolean> {
  const a: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue;
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) a[argv[i].slice(2)] = true;
    else a[argv[i].slice(2)] = argv[++i];
  }
  return a;
}

interface Agg {
  key: string;
  group: string;
  kind: string;
  q: number[];
  lat: number[];
  flags: Map<string, number>;
  details: string[];
}

function aggregate(all: { name: string; scores: Score[] }[]): Map<string, Agg> {
  const m = new Map<string, Agg>();
  for (const { name, scores } of all) {
    for (const s of scores) {
      const id = `${s.group}|${s.key}`;
      let a = m.get(id);
      if (!a) m.set(id, (a = { key: s.key, group: s.group, kind: s.kind, q: [], lat: [], flags: new Map(), details: [] }));
      if (Number.isFinite(s.quality)) a.q.push(s.quality);
      if (Number.isFinite(s.latency)) a.lat.push(s.latency);
      for (const f of s.flags) a.flags.set(f, (a.flags.get(f) ?? 0) + 1);
      a.details.push(`${name}: ${s.detail}`);
    }
  }
  return m;
}

const avg = (a: number[]) => (a.length ? a.reduce((p, q) => p + q, 0) / a.length : NaN);
const med = (a: number[]) => {
  if (!a.length) return NaN;
  const s = [...a].sort((p, q) => p - q);
  return s[s.length >> 1];
};
const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : 'n/a');
const ms = (x: number) => (Number.isFinite(x) ? `${Math.round(x * 1000)}` : '');

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const E = await loadEngine();
  const U = usage(E);
  const channels = [...signalChannels(E.SIGNALS), ...stateChannels()];
  const inputs: { name: string; kind: 'song' | 'mix'; run: () => { scores: Score[]; extra?: unknown } }[] = [];
  const seconds = a.seconds ? Number(a.seconds) : undefined;
  if (!a['no-songs']) {
    let songs = testSongs();
    if (typeof a.songs === 'string' && a.songs !== 'test') {
      const want = a.songs.split(',');
      songs = songs.filter((p) => want.some((w) => slugOf(p).includes(w)));
    }
    for (const path of songs) {
      const name = slugOf(path);
      inputs.push({
        name, kind: 'song',
        run: () => {
          const pcm = decode(path);
          const res = offlineCached(name, pcm);
          const rec = record(E, pcm, res, channels, { t1: seconds });
          console.error(`  live analyzer ${(rec.seconds / (rec.liveProcessMs / 1000)).toFixed(0)}x real time`);
          return { scores: scoreAll(rec) };
        },
      });
    }
  }
  const mixes: Mix[] = a.mixes || a.newsong ? buildMixes() : [];
  if (a.mixes) {
    for (const mix of mixes) {
      inputs.push({
        name: mix.name, kind: 'mix',
        run: () => {
          const res = offlineCached(mix.name + '-' + mix.hash, mix.pcm);
          const rec = record(E, mix.pcm, res, channels, { t1: seconds });
          return { scores: scoreAll(rec), extra: mixTruthReport(mix, rec, res) };
        },
      });
    }
  }
  const pdir = join(OUT, typeof a.out === 'string' ? `parity-${a.out}` : 'parity');
  mkdirSync(pdir, { recursive: true });
  const results: { name: string; kind: string; scores: Score[]; extra?: unknown }[] = [];
  for (const inp of inputs) {
    const t0 = Date.now();
    console.error(`${inp.name} ...`);
    const r = inp.run();
    results.push({ name: inp.name, kind: inp.kind, ...r });
    writeFileSync(join(pdir, inp.name + '.json'), JSON.stringify({ name: inp.name, kind: inp.kind, ...r }, (_, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v), 1));
    console.error(`  done in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }

  const lines: string[] = [];
  const out = (s = '') => {
    lines.push(s);
    console.log(s);
  };
  for (const kind of ['song', 'mix'] as const) {
    const set = results.filter((r) => r.kind === kind);
    if (!set.length) continue;
    const agg = aggregate(set);
    out(`\n## ${kind === 'song' ? `Single songs (${set.length})` : `DJ mixes (${set.length})`}: live vs offline, ranked by priority`);
    out('');
    out('priority = gap x presets affected; gap = 1 - quality (quality 1 = live matches offline). Presets: seeds using the signal in a reaction, or reading the input through a gene; core inputs every preset reads through the engine count as 0.25 x all seeds on top.');
    out('');
    out('| # | channel | kind | quality | gap | lat ms | presets | priority | flags | what live does |');
    out('|---|---|---|---|---|---|---|---|---|---|');
    const rows = [...agg.values()].map((g) => {
      const q = avg(g.q);
      const gap = Number.isFinite(q) ? 1 - q : NaN;
      let presets: number;
      if (g.group === 'signal') presets = U.signalSeeds.get(g.key) ?? 0;
      else presets = (U.inputSeeds.get(g.key) ?? 0) + (CORE_INPUTS.has(g.key) ? 0.25 * U.seeds : 0);
      const pr = Number.isFinite(gap) ? gap * presets : 0;
      return { g, q, gap, presets, pr };
    });
    rows.sort((x, y) => y.pr - x.pr || (y.gap || 0) - (x.gap || 0));
    rows.forEach((r, i) => {
      const flags = [...r.g.flags].map(([f, c]) => `${f} ${c}/${set.length}`).join(', ');
      const name = r.g.group === 'signal' ? `signal **${r.g.key}** (${SIGNAL_SOURCE[r.g.key] ?? ''})` : `${r.g.group}: ${r.g.key}`;
      // A representative detail: the median-quality input.
      const d = r.g.details[Math.floor(r.g.details.length / 2)] ?? '';
      out(`| ${i + 1} | ${name} | ${r.g.kind} | ${f2(r.q)} | ${f2(r.gap)} | ${ms(med(r.g.lat))} | ${r.presets.toFixed(0)} | ${r.pr.toFixed(1)} | ${flags} | ${d.replace(/\|/g, '/')} |`);
    });
  }
  const mixExtras = results.filter((r) => r.kind === 'mix' && r.extra);
  if (mixExtras.length) {
    out('\n## DJ mixes: against the construction truth');
    for (const r of mixExtras) {
      out(`\n### ${r.name}\n`);
      out(String(r.extra));
    }
  }
  if (a.newsong) {
    out('\n## New-song detection (live)\n');
    out(newSongReport(mixes));
  }
  out('\n## Seed usage (reactions)\n');
  out([...U.signalSeeds].sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${v}`).join(', '));
  out('\n## Seed usage (genes)\n');
  out([...U.inputSeeds].sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${v}`).join(', '));
  writeFileSync(join(OUT, typeof a.out === 'string' ? `parity-table-${a.out}.md` : 'parity-table.md'), lines.join('\n') + '\n');
}

await main();

// Live vs offline report cards (CLI): compares each preset's cards rendered from the live analysis
// path (<P>/..., render.ts's default) with the same clips rendered from the offline reference
// (<P>@offline/..., render.ts --offline), both scored by report.ts, and ranks the presets that
// lose the most reactivity live.
// "Why" lists the preset's signals (reactions) and gene inputs whose live parity gap is large,
// from scripts/live-parity.ts output (.testdata/live/parity/*.json), and the reactions that are
// dead only live.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/liveCompare.ts [--md out.md]

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { OUT } from './cdp';
import type { ReportCard } from './metrics';

const CARDS = join(OUT, 'cards');
const PARITY = join(OUT, '../live/parity');
const COLS = ['overall', 'sync', 'events', 'coupling', 'hookRhyme', 'melody', 'structure'] as const;

function loadCards(dir: string): Map<string, ReportCard> {
  const m = new Map<string, ReportCard>();
  if (!existsSync(dir)) return m;
  for (const f of readdirSync(dir)) if (f.endsWith('.json')) m.set(f.slice(0, -5), JSON.parse(readFileSync(join(dir, f), 'utf8')));
  return m;
}

/** Mean live parity quality per channel over the single songs. */
function parityGaps(): Map<string, number> {
  const acc = new Map<string, number[]>();
  if (!existsSync(PARITY)) return new Map();
  for (const f of readdirSync(PARITY)) {
    const j = JSON.parse(readFileSync(join(PARITY, f), 'utf8')) as { kind: string; scores: { key: string; group: string; quality: number | null }[] };
    if (j.kind !== 'song') continue;
    for (const s of j.scores) {
      if (s.quality === null) continue;
      const k = (s.group === 'signal' ? 'signal:' : '') + s.key;
      (acc.get(k) ?? acc.set(k, []).get(k)!).push(s.quality);
    }
  }
  return new Map([...acc].map(([k, v]) => [k, 1 - v.reduce((a, b) => a + b, 0) / v.length]));
}

const num = (x: number | null | undefined) => (typeof x === 'number' && Number.isFinite(x) ? x : NaN);
const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : ' -- ');
const sgn = (x: number) => (Number.isFinite(x) ? (x >= 0 ? '+' : '') + x.toFixed(2) : ' -- ');
const mean = (a: number[]) => {
  const b = a.filter(Number.isFinite);
  return b.length ? b.reduce((p, q) => p + q, 0) / b.length : NaN;
};

async function main() {
  const argv = process.argv.slice(2);
  const mdPath = argv[argv.indexOf('--md') + 1] && argv.includes('--md') ? argv[argv.indexOf('--md') + 1] : null;
  const gaps = parityGaps();
  // Seed signals / gene inputs, via the same bundle the parity tool uses.
  const { loadEngine } = await import('../live/bundle');
  const { geneInputs } = await import('../live/usage');
  const E = await loadEngine();
  const seedOf = new Map(E.SEEDS.map((s) => [s.origin, s]));
  const presets = existsSync(CARDS) ? readdirSync(CARDS).filter((p) => !p.includes('@') && existsSync(join(CARDS, p + '@offline'))) : [];
  interface Row { preset: string; name: string; n: number; off: Record<string, number>; live: Record<string, number>; d: Record<string, number>; foot: [number, number]; dead: [number, number]; deadLive: string[]; why: string[] }
  const rows: Row[] = [];
  for (const p of presets) {
    const off = loadCards(join(CARDS, p + '@offline'));
    const live = loadCards(join(CARDS, p));
    const keys = [...off.keys()].filter((k) => live.has(k));
    if (!keys.length) continue;
    const o: Record<string, number> = {}, l: Record<string, number> = {}, d: Record<string, number> = {};
    for (const c of COLS) {
      o[c] = mean(keys.map((k) => num(off.get(k)!.headline[c])));
      l[c] = mean(keys.map((k) => num(live.get(k)!.headline[c])));
      d[c] = l[c] - o[c];
    }
    const foot = (m: Map<string, ReportCard>) => mean(keys.map((k) => {
      const s = m.get(k)!.counterfactual?.stems;
      return s ? Object.values(s).reduce((a, b) => a + num(b), 0) : NaN;
    }));
    const deadCount = (m: Map<string, ReportCard>) => mean(keys.map((k) => num(m.get(k)!.counterfactual?.dead)));
    const deadLive = new Set<string>();
    for (const k of keys) {
      const ro = off.get(k)!.counterfactual?.reactions ?? [];
      const rl = live.get(k)!.counterfactual?.reactions ?? [];
      for (const r of rl) if (r.dead && !ro.find((x) => x.index === r.index)?.dead) deadLive.add(`r${r.index} ${r.src}>${r.target}`);
    }
    const seed = seedOf.get(p);
    const why: string[] = [];
    if (seed) {
      const sigs = [...new Set(seed.genome.reactions.map((r) => r.src))];
      for (const s of sigs) {
        const g = gaps.get('signal:' + s);
        if (g !== undefined && g >= 0.35) why.push(`${s} (gap ${g.toFixed(2)})`);
      }
      const ins = [...geneInputs(seed.genome)];
      const byGroup = new Map<string, number[]>();
      for (const k of ins) {
        const g = gaps.get(k);
        if (g === undefined) continue;
        const grp = k.split('.')[0];
        (byGroup.get(grp) ?? byGroup.set(grp, []).get(grp)!).push(g);
      }
      for (const [grp, gs] of byGroup) {
        const g = mean(gs);
        if (g >= 0.35) why.push(`gene input ${grp} (gap ${g.toFixed(2)})`);
      }
    }
    rows.push({ preset: p, name: seed?.name ?? '', n: keys.length, off: o, live: l, d, foot: [foot(off), foot(live)], dead: [deadCount(off), deadCount(live)], deadLive: [...deadLive], why });
  }
  rows.sort((a, b) => (a.d.overall || 0) - (b.d.overall || 0));
  const lines: string[] = [];
  const out = (s: string) => {
    lines.push(s);
    console.log(s);
  };
  out(`Presets with both offline and live cards: ${rows.length}. Mean delta (live - offline): ` + COLS.map((c) => `${c} ${sgn(mean(rows.map((r) => r.d[c])))}`).join(', ') + `; stem footprint ${sgn(mean(rows.map((r) => r.foot[1] - r.foot[0])))}; dead reactions ${sgn(mean(rows.map((r) => r.dead[1] - r.dead[0])))}.`);
  out('');
  out('| preset | clips | overall off > live | sync | events | coupling | hook | melody | structure | stem footprint off > live | dead reactions off > live | dead only live | why (live parity gaps it depends on) |');
  out('|---|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const r of rows) {
    out(`| ${r.preset} ${r.name} | ${r.n} | ${f2(r.off.overall)} > ${f2(r.live.overall)} (${sgn(r.d.overall)}) | ${sgn(r.d.sync)} | ${sgn(r.d.events)} | ${sgn(r.d.coupling)} | ${sgn(r.d.hookRhyme)} | ${sgn(r.d.melody)} | ${sgn(r.d.structure)} | ${f2(r.foot[0])} > ${f2(r.foot[1])} | ${f2(r.dead[0])} > ${f2(r.dead[1])} | ${r.deadLive.join('; ')} | ${r.why.join(', ')} |`);
  }
  if (mdPath) writeFileSync(mdPath, lines.join('\n') + '\n');
}

await main();

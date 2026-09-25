// Seed sweep (CLI): renders every seed on the test songs (drop + hook windows, DINOv2
// embeddings), runs counterfactuals on the hook windows, computes report cards and writes a
// ranked summary with the weakest presets and why -> .testdata/avq/report/SUMMARY.md (+ .json).
// Resumable: jobs whose outputs exist are skipped unless --force.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/sweep.ts [--jobs 3] [--preset E14,M01] [--song test]
//        [--cf-songs alors,avicii,t-a-t-u] [--no-cf] [--report-only] [--showcase 10] [--publish]
//
// --publish also writes public/avq/seed-metrics.json: each seed's mean judge vector, which the
// in-app duel page uses as priors and to rank all seeds with the trained judge.
//
// --showcase N re-renders the N weakest and 5 strongest presets with frames on each song's hook
// window and draws their sheets and MP4s (deterministic, so the numbers do not change).

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ensureServers, OUT, Tab } from './cdp';
import { encodeClip, ffmpegPath } from './clips';
import { listClips } from './load';
import type { ReportCard } from './metrics';
import { parseArgs, runJobs, testSongs, type Job } from './render';
import { cardFor } from './report';
import { drawSheets } from './sheets';
import { JUDGE_FEATURES, judgeVector } from '../../src/v2/judge';

const slugOf = (path: string) => path.split('/').pop()!.replace(/\.[^.]+$/, '').toLowerCase().replace(/\(.*?\)/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'song';

const HEAD = ['overall', 'sync', 'events', 'coupling', 'hookRhyme', 'melody', 'structure', 'flow', 'interest', 'correspond'] as const;

/** Map a report-card note to a failure-mode category. */
export function noteCategory(n: string): string {
  if (n.startsWith('STROBE')) return 'strobe';
  if (n.startsWith('still')) return 'mostly still';
  if (n.startsWith('drum hits barely land')) return 'drum hits do not land';
  if (n.startsWith('hook rhyme')) return 'no hook rhyme';
  if (n.startsWith('stems not distinguishable')) return 'stems not distinguishable';
  if (n.startsWith('chaotic:')) return 'chaotic (counterfactuals masked)';
  if (n.startsWith('steady motion')) return 'steady motion (constant spin / sweep)';
  if (n.startsWith('clock-driven motion')) return 'clock-driven motion (metronome gives the same motion)';
  if (n.startsWith('clock-locked')) return 'bar-locked motion';
  if (n.startsWith('chaotic')) return 'chaotic motion';
  if (n.startsWith('section changes barely visible')) return 'section changes invisible';
  if (n.startsWith('desync-blind')) return 'desync-blind';
  if (n.startsWith('no visual footprint from')) return 'stem with no footprint: ' + n.replace('no visual footprint from ', '');
  const m = n.match(/^reaction r\d+ (\S+) (idle source|pinned|tiny travel|invisible)/);
  if (m) return `reaction ${m[2]}`;
  if (n.startsWith('dead reaction')) return 'reaction invisible';
  return 'other';
}

interface PresetRow {
  id: string;
  name: string;
  n: number;
  mean: Record<string, number>;
  bySong: Record<string, Record<string, number>>;
  cfSync: number;
  deadReactions: string[];
  notes: string[];
  categories: Record<string, number>;
}

function aggregate(cards: ReportCard[], names: Record<string, string>): PresetRow[] {
  const by = new Map<string, ReportCard[]>();
  for (const c of cards) by.set(c.preset, [...(by.get(c.preset) ?? []), c]);
  const rows: PresetRow[] = [];
  for (const [id, cs] of by) {
    const mean: Record<string, number> = {};
    for (const k of HEAD) {
      const v = cs.map((c) => (c.headline as unknown as Record<string, number>)[k]).filter((x) => Number.isFinite(x));
      mean[k] = v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
    }
    const bySong: PresetRow['bySong'] = {};
    for (const c of cs) {
      const s = (bySong[c.song] ??= {});
      for (const k of HEAD) {
        const v = (c.headline as unknown as Record<string, number>)[k];
        if (Number.isFinite(v)) s[`${c.clip}.${k}`] = v;
      }
    }
    const cfs = cs.map((c) => c.counterfactual?.score).filter((x): x is number => Number.isFinite(x));
    const dead = new Set<string>();
    for (const c of cs) for (const r of c.readout?.reactions ?? []) if (r.verdict !== 'visible' && r.verdict !== 'masked') dead.add(`r${r.index} ${r.src}>${r.target} (${r.verdict})`);
    const categories: Record<string, number> = {};
    const notes: string[] = [];
    for (const c of cs) {
      for (const n of c.notes) {
        const cat = noteCategory(n);
        categories[cat] = (categories[cat] ?? 0) + 1;
        notes.push(`${c.song} ${c.clip}: ${n}`);
      }
    }
    rows.push({ id, name: names[id] ?? id, n: cs.length, mean, bySong, cfSync: cfs.length ? cfs.reduce((a, b) => a + b, 0) / cfs.length : NaN, deadReactions: [...dead], notes, categories });
  }
  return rows.sort((a, b) => b.mean.overall - a.mean.overall);
}

const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '--');

/** Short "why" for a weak preset: its lowest headline components plus its most frequent failure notes. */
function why(r: PresetRow): string {
  const comps = HEAD.filter((k) => k !== 'overall' && Number.isFinite(r.mean[k])).sort((a, b) => r.mean[a] - r.mean[b]).slice(0, 3).map((k) => `${k} ${f2(r.mean[k])}`);
  const cats = Object.entries(r.categories).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} (${v}x)`);
  return `lowest: ${comps.join(', ')}; ${cats.join('; ')}`;
}

function markdown(rows: PresetRow[], songs: string[], showcase: Record<string, { sheets: string[]; mp4: string[] }>): string {
  const L: string[] = [];
  L.push('# AV quality: seed report cards', '');
  L.push(`${rows.length} presets x ${songs.length} songs (${songs.join(', ')}), drop and hook windows at 320x180, 30 fps. Scores 0..1 (higher is better); cfSync = desync sensitivity from counterfactuals on hook windows.`, '');
  L.push('## Ranking', '');
  L.push('| # | preset | ' + HEAD.join(' | ') + ' | cfSync |');
  L.push('|' + '---|'.repeat(HEAD.length + 3));
  rows.forEach((r, i) => L.push(`| ${i + 1} | ${r.id} ${r.name} | ` + HEAD.map((k) => f2(r.mean[k])).join(' | ') + ` | ${f2(r.cfSync)} |`));
  L.push('', '## Weakest 15 and why', '');
  for (const r of rows.slice(-15).reverse()) {
    L.push(`- **${r.id} ${r.name}** overall ${f2(r.mean.overall)}: ${why(r)}`);
    const alors = Object.entries(r.bySong).find(([s]) => s.startsWith('stromae'));
    if (alors) L.push(`  - Alors on danse hook rhyme ${f2(alors[1]['hook.hookRhyme'])}, hook-window sync ${f2(alors[1]['hook.sync'])}`);
    if (r.deadReactions.length) L.push(`  - reactions without visible effect: ${r.deadReactions.slice(0, 4).join('; ')}`);
    const sc = showcase[r.id];
    if (sc) L.push(`  - sheets: ${sc.sheets.map((p) => '`' + p + '`').join(', ')}`, `  - clips: ${sc.mp4.map((p) => '`' + p + '`').join(', ')}`);
  }
  L.push('', '## Strongest 5', '');
  for (const r of rows.slice(0, 5)) {
    L.push(`- **${r.id} ${r.name}** overall ${f2(r.mean.overall)} (sync ${f2(r.mean.sync)}, hook rhyme ${f2(r.mean.hookRhyme)}, structure ${f2(r.mean.structure)})`);
    const sc = showcase[r.id];
    if (sc) L.push(`  - sheets: ${sc.sheets.map((p) => '`' + p + '`').join(', ')}`, `  - clips: ${sc.mp4.map((p) => '`' + p + '`').join(', ')}`);
  }
  // Failure modes across seeds.
  const cat: Record<string, Set<string>> = {};
  for (const r of rows) for (const k of Object.keys(r.categories)) (cat[k] ??= new Set()).add(r.id);
  L.push('', '## Failure modes across seeds (presets affected)', '');
  for (const [k, v] of Object.entries(cat).sort((a, b) => b[1].size - a[1].size)) L.push(`- ${k}: ${v.size} presets (${[...v].slice(0, 12).join(', ')}${v.size > 12 ? ', ...' : ''})`);
  // Per-song hook rhyme leaders.
  L.push('', '## Hook rhyme by song (hook window, top 5 / bottom 5)', '');
  for (const s of songs) {
    const list = rows.map((r) => ({ id: r.id, v: r.bySong[s]?.['hook.hookRhyme'] })).filter((x) => Number.isFinite(x.v)).sort((a, b) => b.v! - a.v!);
    if (!list.length) continue;
    L.push(`- ${s}: top ${list.slice(0, 5).map((x) => `${x.id} ${f2(x.v!)}`).join(', ')}; bottom ${list.slice(-5).map((x) => `${x.id} ${f2(x.v!)}`).join(', ')}`);
  }
  return L.join('\n') + '\n';
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const songs = !a.song || a.song === 'test' ? testSongs() : String(a.song).split(',');
  const par = Math.min(3, Number(a.jobs ?? 3));
  const servers = await ensureServers();
  const log = (s: string) => console.log(`[${new Date().toISOString().slice(11, 19)}] ${s}`);
  try {
    const tab = await Tab.open();
    await tab.load();
    const seeds = await tab.eval<string[]>('avq.seeds()');
    const names = Object.fromEntries(await tab.eval<[string, string][]>('avq.seedNames()'));
    await tab.close();
    const presets = a.preset ? String(a.preset).split(',') : seeds;
    if (!a['report-only'] && !a['cf-only']) {
      // 1. Renders (drop + hook windows, embeddings).
      const jobs: Job[] = [];
      for (const s of songs) for (const p of presets) if (a.force || !existsSync(join(OUT, 'clips', p, `${slugOf(s)}__hook.emb.json`))) jobs.push({ preset: p, song: s, opts: { clips: 'auto', embedEvery: 5 } });
      log(`renders: ${jobs.length} jobs on ${par} tabs`);
      let done = 0;
      const t0 = Date.now();
      await runJobs(jobs, par, (j, _r, err) => {
        done++;
        if (err) log(`FAIL render ${j.preset} ${slugOf(j.song)}: ${err.message.slice(0, 200)}`);
        if (done % 25 === 0) log(`renders ${done}/${jobs.length}, ${((Date.now() - t0) / done / 1000).toFixed(1)} s/job`);
      });
    }
    if (!a['report-only']) {
      // 2. Counterfactuals on hook windows.
      if (!a['no-cf']) {
        const cfSongs = songs.filter((s) => (a['cf-songs'] ? String(a['cf-songs']).split(',').some((k) => slugOf(s).includes(k)) : true));
        const cjobs: Job[] = [];
        for (const s of cfSongs) {
          const t = await Tab.open();
          await t.load();
          const info = await t.eval<{ windows: { label: string; start: number; end: number }[] }>(`avq.song(${JSON.stringify(s)})`);
          await t.close();
          const hook = info.windows.find((w) => w.label === 'hook') ?? info.windows[0];
          for (const p of presets) if (a.force || a['force-cf'] || !existsSync(join(OUT, 'cf', p, `${slugOf(s)}__${hook.label}.json`))) cjobs.push({ preset: p, song: s, method: 'counterfactual', opts: { clips: [hook] } });
        }
        log(`counterfactuals: ${cjobs.length} jobs`);
        let done = 0;
        const t1 = Date.now();
        await runJobs(cjobs, par, (j, _r, err) => {
          done++;
          if (err) log(`FAIL cf ${j.preset} ${slugOf(j.song)}: ${err.message.slice(0, 200)}`);
          if (done % 25 === 0) log(`cf ${done}/${cjobs.length}, ${((Date.now() - t1) / done / 1000).toFixed(1)} s/job`);
        });
      }
    }
    // 3. Report cards + ranking.
    const cards: ReportCard[] = [];
    for (const p of presets) for (const s of songs) for (const b of listClips({ preset: p, song: slugOf(s) })) {
      try {
        cards.push(cardFor(b));
      } catch (e) {
        log(`card failed ${b}: ${(e as Error).message}`);
      }
    }
    let rows = aggregate(cards, names);
    const showcase: Record<string, { sheets: string[]; mp4: string[] }> = {};
    const N = Number(a.showcase ?? 0);
    if (N > 0) {
      const pick = [...rows.slice(-N).map((r) => r.id), ...rows.slice(0, 5).map((r) => r.id)];
      const fjobs: Job[] = [];
      for (const s of songs) {
        const t = await Tab.open();
        await t.load();
        const info = await t.eval<{ windows: { label: string; start: number; end: number }[] }>(`avq.song(${JSON.stringify(s)})`);
        await t.close();
        for (const p of pick) fjobs.push({ preset: p, song: s, opts: { clips: info.windows, frames: true, embedEvery: 5 } });
      }
      log(`showcase renders with frames: ${fjobs.length}`);
      await runJobs(fjobs, par, (j, _r, err) => err && log(`FAIL showcase ${j.preset}: ${err.message.slice(0, 200)}`));
      const tab2 = await Tab.open();
      await tab2.load();
      const ff = ffmpegPath();
      for (const p of pick) {
        const bases = songs.flatMap((s) => listClips({ preset: p, song: slugOf(s), label: 'hook' }));
        const drops = songs.flatMap((s) => listClips({ preset: p, song: slugOf(s), label: 'drop' }));
        const sheets = await drawSheets([...bases, ...drops], tab2);
        showcase[p] = { sheets, mp4: [...bases, ...drops].map((b) => encodeClip(b, ff)) };
      }
      await tab2.close();
      rows = aggregate(cards, names);
    }
    const dir = join(OUT, 'report');
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'SUMMARY.md'), markdown(rows, songs.map(slugOf), showcase));
    writeFileSync(join(dir, 'ranking.json'), JSON.stringify(rows, null, 1));
    // Mean judge vector per seed (JUDGE_FEATURES order), for the duel page.
    const byPreset = new Map<string, number[][]>();
    for (const c of cards) byPreset.set(c.preset, [...(byPreset.get(c.preset) ?? []), judgeVector(c)]);
    const seedsOut = [...byPreset].map(([preset, vs]) => ({
      preset,
      v: JUDGE_FEATURES.map((_, k) => {
        const col = vs.map((v) => v[k]).filter(Number.isFinite);
        return col.length ? Math.round((col.reduce((x, y) => x + y, 0) / col.length) * 1e3) / 1e3 : null;
      }),
    }));
    const metrics = JSON.stringify({ version: 1, features: [...JUDGE_FEATURES], songs: songs.map(slugOf), seeds: seedsOut });
    writeFileSync(join(dir, 'seed-metrics.json'), metrics);
    if (a.publish) {
      const pub = join(import.meta.dirname, '../../public/avq');
      mkdirSync(pub, { recursive: true });
      writeFileSync(join(pub, 'seed-metrics.json'), metrics);
      log(`published ${join(pub, 'seed-metrics.json')} (${(metrics.length / 1024).toFixed(1)} KB)`);
    }
    log(`wrote ${join(dir, 'SUMMARY.md')} (${rows.length} presets, ${cards.length} cards)`);
  } finally {
    if (!a.keep) servers.stop();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) await main();

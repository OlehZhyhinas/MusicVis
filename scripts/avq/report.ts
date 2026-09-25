// Report cards (CLI): computes the reactivity metrics for rendered clips and prints a table.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/report.ts [--preset E14] [--song slug] [--label hook] [--json]
// Writes .testdata/avq/cards/<preset>/<song>__<label>.json per clip.

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { OUT } from './cdp';
import type { Clip } from './format';
import { listClips, loadCf, loadClip, loadInst } from './load';
import { cfSummary, readoutStats, reportCard, type ClipLike, type ReportCard } from './metrics';

export function asClipLike(c: Clip): ClipLike {
  const h = c.header;
  return {
    fps: h.render.fps, n: c.n,
    col: (name) => c.col(name as never),
    has: (name) => h.fields.includes(name),
    thumb: (i) => c.thumb(i),
    thumbW: h.thumb.w, thumbH: h.thumb.h,
    beats: h.beats, downbeats: h.downbeats, beatsPerBar: h.song.beatsPerBar, bpm: h.song.bpm,
    sections: h.sections, moments: h.moments, hooks: h.hooks,
  };
}

/** Song time of the frame before row 0. */
export const clipT0 = (c: Clip) => c.header.clip.start;

export function cardFor(base: string): ReportCard {
  const c = loadClip(base);
  const card = reportCard(asClipLike(c), { preset: c.header.preset.id, song: c.header.song.slug, clip: c.header.clip.label, clipT0: clipT0(c) });
  const cf = loadCf(base);
  if (cf) {
    const s = cfSummary(cf);
    card.counterfactual = s;
    const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : 'n/a');
    if (s.syncSensitivity < 0.15) card.notes.push(`desync-blind: a half-bar shift changes the picture by ${f2(s.desync)} of its own motion (chaos floor ${f2(s.chaos)})`);
    for (const r of s.reactions) if (r.dead) card.notes.push(`dead reaction r${r.index} ${r.src}>${r.target}: removing it changes ${f2(r.rel)} of the motion`);
    if (s.chaos >= 0.3) card.notes.push(`chaotic: an inaudible 2% level change already moves the picture by ${f2(s.chaos)} of its motion, so single-change counterfactuals are masked`);
    const silent = Object.entries(s.stems).filter(([, v]) => v < 0.03).map(([k]) => k);
    if (silent.length && s.chaos < 0.3) card.notes.push(`no visual footprint from ${silent.join(', ')}`);
  }
  const inst = loadInst(base);
  if (inst) {
    const ro = readoutStats(asClipLike(c), inst, card.counterfactual);
    card.readout = ro;
    for (const r of ro.reactions) {
      if (r.verdict === 'visible' || r.verdict === 'masked') continue;
      const why = r.verdict === 'idle source' ? `its source is active in only ${(r.srcActive * 100).toFixed(0)}% of frames`
        : r.verdict === 'pinned' ? 'the driven value sits at a parameter limit'
        : r.verdict === 'tiny travel' ? `the driven value moves ${(r.travel * 100).toFixed(1)}% of its range`
        : `the value moves ${(r.travel * 100).toFixed(0)}% of its range but nothing visible follows (r ${r.visible.toFixed(2)}${r.footprint !== undefined ? `, footprint ${r.footprint.toFixed(2)}` : ''})`;
      card.notes.push(`reaction r${r.index} ${r.src}>${r.target} ${r.verdict}: ${why}`);
    }
  }
  const p = join(OUT, 'cards', base + '.json');
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify(card, (_, v) => (typeof v === 'number' ? (Number.isFinite(v) ? +v.toFixed(4) : null) : v)));
  return card;
}

const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : ' -- ');

async function main() {
  const a: Record<string, string | boolean> = {};
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) if (argv[i].startsWith('--')) a[argv[i].slice(2)] = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true;
  const bases = listClips({ preset: a.preset as string, song: a.song as string, label: a.label as string });
  const cols = ['overall', 'sync', 'coupling', 'hookRhyme', 'melody', 'structure', 'flow', 'interest', 'correspond'] as const;
  if (!a.json) console.log('clip'.padEnd(48) + cols.map((c) => c.slice(0, 8).padStart(9)).join('') + '  cfSync');
  for (const b of bases) {
    const card = cardFor(b);
    if (a.json) console.log(JSON.stringify(card));
    else {
      console.log(b.padEnd(48) + cols.map((c) => f2(card.headline[c]).padStart(9)).join('') + (card.counterfactual ? f2(card.counterfactual.score).padStart(8) : ''));
      for (const n of card.notes) console.log('    - ' + n);
    }
  }
}

if (import.meta.url === `file://${process.argv[1]}`) await main();

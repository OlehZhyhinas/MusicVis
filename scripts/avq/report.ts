// Report cards (CLI): computes the reactivity metrics for rendered clips and prints a table.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/report.ts [--preset E14] [--song slug] [--label hook] [--json]
// Writes .testdata/avq/cards/<preset>/<song>__<label>.json per clip.

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { OUT } from './cdp';
import type { Clip } from './format';
import { listClips, loadClip } from './load';
import { reportCard, type ClipLike, type ReportCard } from './metrics';

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
  if (!a.json) console.log('clip'.padEnd(48) + cols.map((c) => c.slice(0, 8).padStart(9)).join(''));
  for (const b of bases) {
    const card = cardFor(b);
    if (a.json) console.log(JSON.stringify(card));
    else {
      console.log(b.padEnd(48) + cols.map((c) => f2(card.headline[c]).padStart(9)).join(''));
      for (const n of card.notes) console.log('    - ' + n);
    }
  }
}

if (import.meta.url === `file://${process.argv[1]}`) await main();

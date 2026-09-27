// Drive the real live chord/key path and the offline analysis over a set of songs by reusing
// scripts/live/parity.ts's own record()/stateChannels()/score() (no reimplementation of the
// scoring, so results are directly comparable to the repo's "0.66 agreement" parity number),
// and dump the raw per-frame chord/key indices to JSON for scoring against madmom teacher
// labels in Python (mir_eval.chord).
//
// Run:
//   node --import ../../../analysis-test.hooks.mjs scripts/ml/instruments/chords/live_dump.ts [--corpus=test|own] [--songs=a,b]
//
// Only imports from scripts/live/ and src/ (nothing there is edited).
import { existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { decode, offlineCached, slugOf, TEST_DIR, type Pcm } from '../../../live/common';
import { loadEngine } from '../../../live/bundle';
import { record, score, stateChannels } from '../../../live/parity';

const OUT_DIR = join(import.meta.dirname, '../../../../.testdata/instr/chords/live');
mkdirSync(OUT_DIR, { recursive: true });

function ownEvalTracks(): { id: string; path: string }[] {
  const root = '/Users/oleh/personal/MusicVis-data/own';
  const out: { id: string; path: string }[] = [];
  if (!existsSync(root)) return out;
  for (const pl of readdirSync(root).sort()) {
    const d = join(root, pl);
    for (const f of readdirSync(d).sort()) {
      if (!/\.(m4a|mp3)$/i.test(f)) continue;
      const m = /^(\d+)/.exec(f);
      if (!m || Number(m[1]) % 5 !== 0) continue; // held-out only, per common.is_eval()
      out.push({ id: slugOf(f.replace(/^\d+\s*-\s*/, '')), path: join(d, f) });
    }
  }
  return out;
}

async function main() {
  const args = process.argv.slice(2);
  const corpusArg = (args.find((a) => a.startsWith('--corpus=')) || '--corpus=test').split('=')[1];
  const songsArg = args.find((a) => a.startsWith('--songs='));

  let items: { id: string; path: string; corpus: string }[] = [];
  if (corpusArg === 'test' || corpusArg === 'both') {
    for (const f of readdirSync(TEST_DIR).filter((f) => /\.(mp3|m4a|wav|flac|ogg)$/i.test(f) && !/\(\d+\)\.[a-z0-9]+$/i.test(f)).sort()) {
      items.push({ id: slugOf(f), path: join(TEST_DIR, f), corpus: 'test' });
    }
  }
  if (corpusArg === 'own' || corpusArg === 'both') {
    for (const { id, path } of ownEvalTracks()) items.push({ id, path, corpus: 'own' });
  }
  if (songsArg) {
    const want = new Set(songsArg.split('=')[1].split(','));
    items = items.filter((p) => want.has(p.id));
  }

  console.error(`loading engine bundle...`);
  const E = await loadEngine();
  const channels = stateChannels().filter((c) => c.key === 'chord' || c.key === 'key');

  console.error(`driving live path over ${items.length} songs`);
  const summary: Record<string, unknown>[] = [];
  for (const { id, path, corpus } of items) {
    const outPath = join(OUT_DIR, `${corpus}__${id}.json`);
    console.error(`${id}: decoding`);
    const pcm: Pcm = decode(path);
    const result = offlineCached(id, pcm);
    const rec = record(E, pcm, result, channels, {});
    const scores = channels.map((c, i) => score(c, rec.off[i], rec.live[i], rec.fps));
    for (const s of scores) summary.push({ id, corpus, channel: s.key, quality: s.num?.agree ?? s.quality, changesOff: s.num?.changesOff, changesLive: s.num?.changesLive });
    writeFileSync(
      outPath,
      JSON.stringify({
        id,
        corpus,
        fps: rec.fps,
        n: rec.n,
        channels: channels.map((c) => c.key),
        off: channels.map((_, i) => Array.from(rec.off[i])),
        live: channels.map((_, i) => Array.from(rec.live[i])),
        parityScores: scores.map((s) => ({ key: s.key, quality: s.quality, detail: s.detail })),
      })
    );
    const chordS = scores.find((s) => s.key === 'chord');
    const keyS = scores.find((s) => s.key === 'key');
    console.error(`${id}: chord agree=${chordS?.quality.toFixed(3)} key agree=${keyS?.quality.toFixed(3)}`);
  }
  writeFileSync(join(OUT_DIR, '_summary.json'), JSON.stringify(summary, null, 2));
  console.error('done');
}

main();

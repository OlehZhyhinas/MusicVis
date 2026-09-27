// Drive the app's real live path (and the offline analysis) over the songs that have MIDI / Hooktheory
// ground truth, and dump per-frame beat / bar / chord / key state for scoring in eval_teachers.py.
// Reuses scripts/live/parity.ts record() and stateChannels(); nothing under src/ or scripts/live/ is edited.
// Audio is decoded with ffmpeg and never played.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/groundtruth/live_dump.ts <list.json>
//
// list.json: [{"corpus": "test", "id": "<path id>", "path": "/abs/audio"}]. Output:
// .testdata/gt/live/<corpus>/<id>.json {fps, n, channels, off: [[...]], live: [[...]]}.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { decode, offlineCached, type Pcm } from '../../live/common';
import { loadEngine } from '../../live/bundle';
import { record, stateChannels } from '../../live/parity';

const OUT_DIR = join(import.meta.dirname, '../../../.testdata/gt/live');
const WANT = new Set(['onBeat', 'onBar', 'chord', 'key']);

async function main() {
  const items: { corpus: string; id: string; path: string }[] = JSON.parse(readFileSync(process.argv[2], 'utf8'));
  const E = await loadEngine();
  const channels = stateChannels().filter((c) => WANT.has(c.key));
  for (const { corpus, id, path } of items) {
    const outPath = join(OUT_DIR, corpus, `${id}.json`);
    if (existsSync(outPath)) continue;
    mkdirSync(join(OUT_DIR, corpus), { recursive: true });
    const pcm: Pcm = decode(path);
    const result = offlineCached(`gt__${corpus}__${id}`.replace(/[^A-Za-z0-9_.-]+/g, '_').slice(0, 120), pcm);
    const rec = record(E, pcm, result, channels, {});
    writeFileSync(
      outPath,
      JSON.stringify({
        id, corpus, fps: rec.fps, n: rec.n,
        channels: channels.map((c) => c.key),
        off: channels.map((_, i) => Array.from(rec.off[i], (v) => (Number.isFinite(v) ? v : -1))),
        live: channels.map((_, i) => Array.from(rec.live[i], (v) => (Number.isFinite(v) ? v : -1))),
      }),
    );
    console.error(`${corpus}/${id}: done`);
  }
}

main();

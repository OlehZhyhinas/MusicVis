// The live path's stems.* (exactly as the app sees them: LivePath = LiveInput's block feeding +
// RealtimeSampler at the render clock) sampled once per 512-sample hop, with the stem network on
// (MUSICVIS_NEURAL_STEMS=1) or off (=0), the same sampling as the beat agent's stems-eval.ts.
// Writes .testdata/instr/presence/live/<tag>/<slug>.f32 (T x 4: drums bass vocals other) and the
// per-block CPU time. Score with live_stems_score.py.
//   MUSICVIS_NEURAL_STEMS=1 node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/presence/live_stems_dump.ts neural
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { OfflineLive } from '../../../avq/audio';
import { decode, LivePath, mono, slugOf, testSongs } from '../../../live/common';

const tag = process.argv[2] ?? 'run';
const OUT = join(import.meta.dirname, '../../../../.testdata/instr/presence/live', tag);
mkdirSync(OUT, { recursive: true });
const STEMS = ['drums', 'bass', 'vocals', 'other'] as const;
const HOP = 512;
let ms = 0, blocks = 0;
for (const path of testSongs()) {
  const slug = slugOf(path);
  const pcm = decode(path);
  const T = Math.floor(pcm.left.length / HOP);
  const lp = new LivePath(pcm);
  const live = new OfflineLive(mono(pcm), pcm.sr);
  const out = new Float32Array(T * 4);
  const dt = HOP / pcm.sr;
  for (let k = 0; k < T; k++) {
    const t = (k + 1) * dt;
    const st = lp.sample(t, dt, live.read(t, dt));
    for (let s = 0; s < 4; s++) out[k * 4 + s] = st.stems[STEMS[s]];
  }
  ms += lp.processMs;
  blocks += T;
  writeFileSync(join(OUT, slug + '.f32'), Buffer.from(out.buffer));
  console.error(slug, (lp.processMs / T).toFixed(3), 'ms per block (analyzer + beat RNN + stems)');
}
console.log(`${tag}: ${(ms / blocks).toFixed(3)} ms per 512-sample block for the whole live analysis`);

// Checks src/analysis/beatRnn.ts against madmom itself: features and ensemble activation in madmom's
// file mode (centred frames) from scripts/ml/export-beat-lstm.py, then times the causal port.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/beatrnn-check.ts [slug]

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { BeatRnn, parseBeatRnn } from '../../src/analysis/beatRnn';
import { decode, mono, slugOf, testSongs } from '../live/common';

const slug = process.argv[2] ?? 'inna-morenito';
const root = join(import.meta.dirname, '../..');
const bin = readFileSync(join(root, 'public/models/beat-lstm.bin'));
const model = parseBeatRnn(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const rd = (f: string) => { const b = readFileSync(join(root, '.testdata/ml/beatrnn-ref', f)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const refFeat = rd(slug + '.feat.f32'), refAct = rd(slug + '.act.f32');
const pcm = decode(testSongs().find((p) => slugOf(p) === slug)!);
const x = mono(pcm).subarray(0, 44100 * 30);

const rnn = new BeatRnn(model, { sampleRate: 44100, lookahead: 1023 });
const acts: number[] = [];
let fmax = 0;
rnn.push(x, (a) => {
  const k = acts.length;
  if (k < refAct.length) for (let i = 0; i < 162; i++) fmax = Math.max(fmax, Math.abs(rnn.feat[i] - refFeat[k * 162 + i]));
  acts.push(a);
});
let amax = 0;
const n = Math.min(acts.length, refAct.length - 2);
for (let k = 0; k < n; k++) amax = Math.max(amax, Math.abs(acts[k] - refAct[k]));
console.log(`${slug}: ${acts.length} frames (madmom ${refAct.length}); max |feature diff| ${fmax.toExponential(2)}, max |activation diff| ${amax.toExponential(2)}`);

// Causal timing over the whole song, in 512-sample blocks as the capture worklet posts them.
for (const nets of [8, 4, 2, 1]) {
  const all = mono(pcm);
  const c = new BeatRnn(model, { sampleRate: 44100, nets });
  const t0 = performance.now();
  for (let i = 0; i + 512 <= all.length; i += 512) c.push(all.subarray(i, i + 512));
  const ms = performance.now() - t0;
  const blocks = Math.floor(all.length / 512);
  console.log(`${nets} nets: ${(ms / blocks).toFixed(3)} ms per 512-sample block (${((ms / 1000) / (all.length / 44100) * 100).toFixed(2)} % of real time)`);
}

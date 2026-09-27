// The SHIPPED stem network (src/analysis/stemNet.ts + public/models/stems.bin) over the 13 test songs,
// fed exactly like LiveInput (44.1 kHz stereo blocks of 512 -> (L+R)/2). Writes absolute stem levels
// (dB rel. to mix + the frame's mix dB) per network frame: .testdata/instr/presence/srcnet/<slug>.f32
// (T x 6: drums bass vocals guitar piano other). Score with srcnet_score.py.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/presence/srcnet_dump.ts
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { StemNet } from '../../../../src/analysis/stemNet';
import { decode, slugOf, stemNetModelSync, testSongs } from '../../../live/common';

const OUT = join(import.meta.dirname, '../../../../.testdata/instr/presence/srcnet');
mkdirSync(OUT, { recursive: true });
const model = stemNetModelSync();
const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
let ms = 0, frames = 0;
for (const path of testSongs()) {
  const slug = slugOf(path);
  const pcm = decode(path);
  const net = new StemNet(model, pcm.sr);
  const S = model.header.stems.length;
  const rows: number[] = [];
  const mono = new Float32Array(512);
  const c0 = cpu();
  for (let p = 0; p + 512 <= pcm.left.length; p += 512) {
    for (let i = 0; i < 512; i++) mono[i] = 0.5 * (pcm.left[p + i] + pcm.right[p + i]);
    net.push(mono, (s) => {
      const mixDb = 10 * Math.log10((net as any).mixP + 1e-10);
      for (let i = 0; i < S; i++) rows.push(s.db[i] + mixDb);
    });
  }
  ms += cpu() - c0;
  frames += rows.length / S;
  writeFileSync(join(OUT, slug + '.f32'), Buffer.from(Float32Array.from(rows).buffer));
}
console.log(`${(model.header as any).model}: ${(ms / frames).toFixed(4)} ms CPU per 512-sample block (front end + network), stems.bin weights ${model.header.C}/${model.header.H}`);

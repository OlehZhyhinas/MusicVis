// The beat RNN's front-end features (src/analysis/beatRnn.ts: 2048-Hann at 100 fps, 81 log bands +
// positive diff = 162 dims, frames ending at the newest sample), computed by the shipped TS code itself
// so a stem student trained on them matches the browser exactly. One net is kept (its output is unused).
// Writes .testdata/instr/presence/beatfeat/<corpus>/<id>.f16 (T x 162, float16 as uint16).
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/presence/beatfeat_dump.ts <corpus> <listfile>
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { BeatRnn } from '../../../../src/analysis/beatRnn';
import { beatRnnModelSync, ffmpegPath } from '../../../live/common';

const [corpus, list] = process.argv.slice(2);
const OUT = join(import.meta.dirname, '../../../../.testdata/instr/presence/beatfeat', corpus);
mkdirSync(OUT, { recursive: true });
const model = beatRnnModelSync();
const f2h = (() => {
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  return (v: number) => {
    f[0] = v;
    const x = u[0], s = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 127 + 15, m = x & 0x7fffff;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7c00;
    return s | (e << 10) | ((m + 0x1000) >>> 13);
  };
})();
for (const line of readFileSync(list, 'utf8').split('\n').filter(Boolean)) {
  const i = line.indexOf('=');
  const id = line.slice(0, i), path = line.slice(i + 1);
  const op = join(OUT, id + '.f16');
  if (existsSync(op)) continue;
  const buf = execFileSync(ffmpegPath(), ['-v', 'error', '-i', path, '-f', 'f32le', '-ac', '2', '-ar', '44100', '-'], { maxBuffer: 2 ** 31 - 1 });
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const n = all.length >> 1;
  const x = new Float32Array(n);
  for (let k = 0; k < n; k++) x[k] = 0.5 * (all[2 * k] + all[2 * k + 1]);
  const rnn = new BeatRnn(model, { sampleRate: 44100, nets: 1 });
  const D = rnn.feat.length;
  const frames: Uint16Array[] = [];
  for (let p = 0; p < n; p += 512) {
    rnn.push(x.subarray(p, Math.min(n, p + 512)), () => {
      const h = new Uint16Array(D);
      for (let d = 0; d < D; d++) h[d] = f2h(rnn.feat[d]);
      frames.push(h);
    });
  }
  const out = new Uint16Array(frames.length * D);
  frames.forEach((h, k) => out.set(h, k * D));
  writeFileSync(op, Buffer.from(out.buffer));
  console.log(id, frames.length, 'frames');
}

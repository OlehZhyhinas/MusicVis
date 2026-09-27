// DSP baseline (candidate c): drive RealtimeAnalyzer offline exactly as LiveInput does (512-sample
// blocks) and dump, per analyzer frame (~86 fps), its 4 crude stems:
//   env[4] (0..1 running-normalised dB envelopes), pres[4] (stemPresence), rw[4] (weighted
//   linear stem power, the private `rw`), for drums, bass, vocals, other.
// Output: .testdata/instr/presence/dsp/<id>.f32 (T x 12 float32) + <id>.json {fps, cols, ms}.
// Audio is decoded with ffmpeg to a pipe and never played.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/presence/dsp_dump.ts <id=path> ...
//   (or a single arg @listfile with one id=path per line)
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { RealtimeAnalyzer } from '../../../../src/analysis/RealtimeAnalyzer';
import { ffmpegPath } from '../../../live/common';

const OUT = join(import.meta.dirname, '../../../../.testdata/instr/presence/dsp');
mkdirSync(OUT, { recursive: true });
const SR = 44100, BLOCK = 512;
const S = ['drums', 'bass', 'vocals', 'other'] as const;
let args = process.argv.slice(2);
if (args.length === 1 && args[0].startsWith('@')) args = readFileSync(args[0].slice(1), 'utf8').split('\n').filter(Boolean);
for (const a of args) {
  const i = a.indexOf('=');
  const id = a.slice(0, i), path = a.slice(i + 1);
  if (existsSync(join(OUT, id + '.json'))) continue;
  const buf = execFileSync(ffmpegPath(), ['-v', 'error', '-i', path, '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], { maxBuffer: 2 ** 31 - 1 });
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const n = all.length >> 1;
  const L = new Float32Array(n), R = new Float32Array(n);
  for (let k = 0; k < n; k++) (L[k] = all[2 * k]), (R[k] = all[2 * k + 1]);
  const an = new RealtimeAnalyzer(SR);
  const hop = BLOCK; // sampled once per 512-sample input block (86 fps), as LiveInput's worklet posts them
  const T = Math.floor(n / hop);
  const out = new Float32Array(T * 12);
  let f = 0, pos = 0, ms = 0;
  while (pos + BLOCK <= n) {
    const t0 = performance.now();
    an.process(L.subarray(pos, pos + BLOCK), R.subarray(pos, pos + BLOCK));
    ms += performance.now() - t0;
    pos += BLOCK;
    {
      const rw = (an as any).rw;
      for (let s = 0; s < 4; s++) {
        out[f * 12 + s] = an.stems[S[s]];
        out[f * 12 + 4 + s] = an.stemPresence[S[s]];
        out[f * 12 + 8 + s] = rw[S[s]];
      }
      f++;
    }
  }
  writeFileSync(join(OUT, id + '.f32'), Buffer.from(out.buffer, 0, f * 12 * 4));
  writeFileSync(join(OUT, id + '.json'), JSON.stringify({ fps: SR / hop, hop, frames: f, cols: [...S.map((s) => 'env.' + s), ...S.map((s) => 'pres.' + s), ...S.map((s) => 'rw.' + s)], msPerSecAudio: ms / (n / SR) }));
  console.log(id, f, 'frames', (ms / (n / SR)).toFixed(2), 'ms per s audio');
}

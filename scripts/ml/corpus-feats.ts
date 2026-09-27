// Student features for a training corpus (FMA, the owner's own tracks, the test songs): the beat
// RNN's own frontend (src/analysis/beatRnn.ts: 81 log-filtered bands + positive diff at 100 fps,
// causal 2048 frames), so a student can share the STFT with the beat RNN in the app.
// Audio is decoded with ffmpeg to a pipe and never played.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/corpus-feats.ts <audio dir> <out dir> [--shard i/N] [--max-seconds 30]
//
// Writes <out dir>/<id>.f16 (float16 LE, frames x 162) where <id> is the file's path under <audio dir>
// with '/' -> '__' and no extension; skips existing outputs. Prints one line per file.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { BeatRnn } from '../../src/analysis/beatRnn';
import { beatRnnModelSync, ffmpegPath } from '../live/common';

const [dir, out] = process.argv.slice(2).filter((x) => !x.startsWith('--'));
const arg = (k: string) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : undefined; };
const [si, sn] = (arg('shard') ?? '0/1').split('/').map(Number);
const maxS = Number(arg('max-seconds') ?? '600');
mkdirSync(out, { recursive: true });

function walk(d: string, acc: string[] = []): string[] {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    const s = statSync(p);
    if (s.isDirectory()) walk(p, acc);
    else if (/\.(mp3|m4a|wav|flac|ogg|opus)$/i.test(f)) acc.push(p);
  }
  return acc;
}
export const idOf = (root: string, p: string) => relative(root, p).replace(/\.[^.]+$/, '').replace(/[\\/]/g, '__');

function f16(x: number): number {
  // float32 -> IEEE half (round to nearest, no subnormal care needed for these features)
  const f = new Float32Array([x]);
  const u = new Uint32Array(f.buffer)[0];
  const s = (u >>> 16) & 0x8000;
  let e = ((u >>> 23) & 0xff) - 127 + 15;
  let m = u & 0x7fffff;
  if (e <= 0) return s;
  if (e >= 31) return s | 0x7c00;
  m += 0x1000;
  if (m & 0x800000) { m = 0; e++; if (e >= 31) return s | 0x7c00; }
  return s | (e << 10) | (m >>> 13);
}

const files = walk(dir).sort().filter((_, i) => i % sn === si);
const model = beatRnnModelSync();
let done = 0;
for (const p of files) {
  const id = idOf(dir, p);
  const o = join(out, id + '.f16');
  if (existsSync(o)) continue;
  let buf: Buffer;
  try {
    buf = execFileSync(ffmpegPath(), ['-v', 'error', '-i', p, '-t', String(maxS), '-f', 'f32le', '-ac', '1', '-ar', '44100', '-'], { maxBuffer: 1 << 30 });
  } catch {
    console.log(`${id}: decode failed`);
    continue;
  }
  const x = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (x.length < 44100 * 5) { console.log(`${id}: too short`); continue; }
  const rnn = new BeatRnn(model, { sampleRate: 44100, nets: 1 });
  const rows: Uint16Array[] = [];
  rnn.push(x, () => rows.push(Uint16Array.from(rnn.feat, f16)));
  const all = new Uint16Array(rows.length * 162);
  rows.forEach((r, i) => all.set(r, i * 162));
  writeFileSync(o, Buffer.from(all.buffer));
  done++;
  if (done % 100 === 0) console.log(`${done} done (${id}, ${rows.length} frames)`);
}
console.log(`shard ${si}/${sn}: ${done} new files`);

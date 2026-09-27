// Checks src/analysis/downbeatBlstm.ts against madmom on a 5 s window (scripts/ml/export-downbeat-blstm.py
// reference) and times one analysis per ensemble size.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/downbeat-check.ts [slug]

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { downbeatFeatures, parseDownbeat, runNets } from '../../src/analysis/downbeatBlstm';
import { decode, mono, slugOf, testSongs } from '../live/common';

const slug = process.argv[2] ?? 'inna-morenito';
const root = join(import.meta.dirname, '../..');
const bin = readFileSync(join(root, 'public/models/downbeat-blstm.bin'));
const m = parseDownbeat(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const rd = (f: string) => { const b = readFileSync(join(root, '.testdata/ml/beatrnn-ref', f)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const rf = rd(slug + '.db.feat.f32'), ra = rd(slug + '.db.act.f32');
const x = mono(decode(testSongs().find((p) => slugOf(p) === slug)!)).slice(44100 * 20, 44100 * 25);
const f = downbeatFeatures(m, x);
let fd = 0;
for (let i = 0; i < Math.min(f.length, rf.length); i++) fd = Math.max(fd, Math.abs(f[i] - rf[i]));
const run = (nets: number) => { const g = runNets(m, f, f.length / m.nfeat, nets); let r = g.next(); while (!r.done) r = g.next(); return r.value; };
const a = run(8);
{
  const c0 = process.cpuUsage();
  for (let k = 0; k < 3; k++) downbeatFeatures(m, x);
  const c = process.cpuUsage(c0);
  console.log(`features: ${((c.user + c.system) / 3000 / 5).toFixed(1)} ms CPU per second of audio`);
}
let ad = 0;
for (let i = 0; i < Math.min(a.length, ra.length); i++) ad = Math.max(ad, Math.abs(a[i] - ra[i]));
console.log(`${slug}: frames ${f.length / m.nfeat} (madmom ${rf.length / m.nfeat}), max |feature diff| ${fd.toExponential(2)}, max |activation diff| ${ad.toExponential(2)}`);
for (const nets of [1, 2, 4, 8]) {
  // CPU time (the machine may be busy): networks only; features are computed once per frame live.
  const c0 = process.cpuUsage();
  for (let k = 0; k < 3; k++) run(nets);
  const c = process.cpuUsage(c0);
  console.log(`${nets} nets: ${((c.user + c.system) / 3000).toFixed(1)} ms CPU per 5 s analysis`);
}

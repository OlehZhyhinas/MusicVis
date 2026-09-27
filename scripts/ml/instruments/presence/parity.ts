// Parity + speed of the plain-TS student (stemStudent.ts) against the PyTorch model.
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/presence/parity.ts <name>
// (1) model parity: the Python features of the first 30 s of owl-city-fireflies through step(), vs torch outputs
// (2) front-end parity: the TS 44.1 kHz front end on the decoded song vs the Python 22.05 kHz features
// (3) ms per 512-sample block for process() (front end + model), single thread
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { StemStudent, type StudentWeights } from './stemStudent';
import { decode, mono } from '../../../live/common';

const name = process.argv[2] ?? 'e48';
const MD = join(import.meta.dirname, '../../../../.testdata/instr/presence/models');
const f32 = (p: string) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const w = JSON.parse(readFileSync(join(MD, name + '.json'), 'utf8')) as StudentWeights;
const st = new StemStudent(w);
const M = w.meta.front.n_mel;
const fin = f32(join(MD, name + '.ref_in.f32'));
const renv = f32(join(MD, name + '.ref_env.f32'));
const ract = f32(join(MD, name + '.ref_act.f32'));
const T = fin.length / M;
let me = 0, ma = 0;
for (let t = 0; t < T; t++) {
  st.step(fin.subarray(t * M, (t + 1) * M));
  for (let s = 0; s < st.S; s++) me = Math.max(me, Math.abs(st.envRel[s] - renv[t * st.S + s]));
  for (let g = 0; g < st.G; g++) ma = Math.max(ma, Math.abs(st.act[g] - ract[t * st.G + g]));
}
console.log(`model parity over ${T} hops: max |env diff| ${me.toExponential(2)} dB, max |act diff| ${ma.toExponential(2)}`);

// front end vs Python features, and end-to-end outputs
const pcm = decode('/Users/oleh/Downloads/YoutubeToMp3/Owl City - Fireflies (Official Music Video).mp3');
const x = mono(pcm);
const st2 = new StemStudent(w);
let fd = 0, fdn = 0, e2e = 0;
const t0 = performance.now();
const nb = Math.min(T, Math.floor(x.length / 512));
for (let k = 0; k < nb; k++) {
  st2.process(x.subarray(k * 512, (k + 1) * 512));
  if (k >= 8) {
    for (let j = 0; j < M; j++) (fd += Math.abs(st2.mel[j] - fin[k * M + j])), fdn++;
    for (let s = 0; s < st2.S; s++) e2e = Math.max(e2e, k > 200 ? Math.abs(st2.envRel[s] - renv[k * st2.S + s]) : 0);
  }
}
const ms = (performance.now() - t0) / nb;
console.log(`front end: mean |log10 mel diff| vs Python (ffmpeg 22.05 kHz resample) ${(fd / fdn).toFixed(4)} (= ${((10 * fd) / fdn).toFixed(2)} dB); end-to-end max |envRel diff| ${e2e.toFixed(2)} dB`);
// speed on a longer run (warm)
const st3 = new StemStudent(w);
const blocks = Math.floor(x.length / 512);
// CPU time (process.cpuUsage, user+system) rather than wall time: the machine is shared and loaded
const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
const t1 = cpu();
for (let k = 0; k < blocks; k++) st3.process(x.subarray(k * 512, (k + 1) * 512));
const ms2 = (cpu() - t1) / blocks;
const t2 = cpu();
for (let k = 0; k < blocks; k++) st3.frontEnd(x.subarray(k * 512, (k + 1) * 512));
const msFe = (cpu() - t2) / blocks;
console.log(`speed: ${ms2.toFixed(4)} ms per 512-sample block total (${msFe.toFixed(4)} front end), = ${((ms2 * 86.13) / 10).toFixed(2)}% of one core; params ${(w.meta as any).params} (${(((w.meta as any).params * 4) / 1e6).toFixed(2)} MB f32)`);
void ms;

// (4) the shipped src/analysis/stemNet.ts with public/models/stems.bin (must be exported from the same model)
{
  const { parseStemNet, StemNet } = await import('../../../../src/analysis/stemNet');
  const b = readFileSync(join(import.meta.dirname, '../../../../public/models/stems.bin'));
  const sm = parseStemNet(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const net = new StemNet(sm, 44100);
  let md = 0, k = 0;
  const nb2 = Math.min(T, Math.floor(x.length / 512));
  const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
  const c0 = cpu();
  net.push(x.subarray(0, nb2 * 512), (s) => {
    if (k > 200) for (let i = 0; i < s.db.length; i++) md = Math.max(md, Math.abs(s.db[i] - renv[k * s.db.length + i]));
    k++;
  });
  const cms = (cpu() - c0) / nb2;
  console.log(`src StemNet (${(sm.header as any).model}): ${k} frames, max |db diff| vs torch (after warm-up) ${md.toFixed(3)} dB; ${cms.toFixed(4)} ms CPU per block; stems.bin ${(b.byteLength / 1e6).toFixed(3)} MB`);
}

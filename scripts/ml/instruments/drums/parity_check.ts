// TS-vs-torch parity + timing for a trained drum-transcription student.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/drums/parity_check.ts <size> <corpus> <tid>
//
// Reads .testdata/instr/drums/parity/<size>-<corpus>-<tid>.{feat,ref}.f32 (from dump_torch_ref.py)
// and .testdata/instr/drums/models/<size>.{bin,header.json}, runs the plain-TS forward pass
// frame by frame, reports max abs error vs the torch reference and ms/hop (Node, single thread).

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DrumStudentTS, type DrumHeader } from './forward';

const WORK = join(import.meta.dirname, '../../../../.testdata/instr/drums');

async function main() {
  const [size, corpus, tid] = process.argv.slice(2);
  const header = JSON.parse(readFileSync(join(WORK, 'models', `${size}.header.json`), 'utf8')) as DrumHeader;
  const binBuf = readFileSync(join(WORK, 'models', `${size}.bin`));
  const weights = new Float32Array(binBuf.buffer, binBuf.byteOffset, binBuf.byteLength / 4);

  const featBuf = readFileSync(join(WORK, 'parity', `${size}-${corpus}-${tid}.feat.f32`));
  const feat = new Float32Array(featBuf.buffer, featBuf.byteOffset, featBuf.byteLength / 4);
  const refBuf = readFileSync(join(WORK, 'parity', `${size}-${corpus}-${tid}.ref.f32`));
  const ref = new Float32Array(refBuf.buffer, refBuf.byteOffset, refBuf.byteLength / 4);

  const T = feat.length / header.in_feat;
  const model = new DrumStudentTS(weights, header);
  model.reset();

  const activations = new Float32Array(T * header.n_classes);
  const t0 = performance.now();
  for (let k = 0; k < T; k++) {
    const f = feat.subarray(k * header.in_feat, (k + 1) * header.in_feat);
    const a = model.push(f);
    activations.set(a, k * header.n_classes);
  }
  const t1 = performance.now();
  const msPerHop = (t1 - t0) / T;
  const hopMs = (512 / 44100) * 1000;
  const cpuPct = (100 * msPerHop) / hopMs;

  let maxErr = 0, sumErr = 0;
  for (let i = 0; i < activations.length; i++) {
    const e = Math.abs(activations[i] - ref[i]);
    if (e > maxErr) maxErr = e;
    sumErr += e;
  }
  console.log(JSON.stringify({
    size, corpus, tid, T,
    max_abs_error: maxErr,
    mean_abs_error: sumErr / activations.length,
    ms_per_hop: msPerHop,
    hop_ms: hopMs,
    cpu_pct_of_one_core: cpuPct,
  }, null, 1));
}

main();

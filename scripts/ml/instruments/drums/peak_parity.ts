// Checks that the streaming TS peak picker (peakpick.ts) emits exactly the same onset frames as
// peakpick.py's peak_pick() on a torch reference activation dump.
//   <torch venv>/bin/python scripts/ml/instruments/drums/peak_parity_ref.py <tag> <corpus> <tid>
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/drums/peak_parity.ts <tag> <corpus> <tid>
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { StreamPeakPicker, type PeakParams } from './peakpick';

const WORK = join(import.meta.dirname, '../../../../.testdata/instr/drums');
const [tag, corpus, tid] = process.argv.slice(2);
const ref = readFileSync(join(WORK, 'parity', `${tag}-${corpus}-${tid}.ref.f32`));
const act = new Float32Array(ref.buffer, ref.byteOffset, ref.byteLength / 4);
const py = JSON.parse(readFileSync(join(WORK, 'parity', `${tag}-${corpus}-${tid}.peaks.json`), 'utf8')) as {
  classes: string[]; params: Record<string, PeakParams>; events: Record<string, number[]>;
};
const C = py.classes.length;
const T = act.length / C;
let ok = true;
for (let c = 0; c < C; c++) {
  const cls = py.classes[c];
  const pp = new StreamPeakPicker(py.params[cls]);
  const ev: number[] = [];
  for (let k = 0; k < T + py.params[cls].lookahead; k++) {
    const e = pp.push(k < T ? act[k * C + c] : -Infinity);
    if (e >= 0) ev.push(e);
  }
  const want = py.events[cls];
  const same = ev.length === want.length && ev.every((v, i) => v === want[i]);
  ok &&= same;
  console.log(cls, same ? 'MATCH' : 'DIFF', 'ts', ev.length, 'py', want.length);
}
process.exit(ok ? 0 : 1);

// How strong the engine's drum hits (Signals F.hit, the value 'hit' reactions and motion 'hits'
// see at each hit) come out offline vs live: hits per minute, and the share of hits above 0.5
// (what hit-gated motion needs).
//   node --import ./scripts/analysis-test.hooks.mjs scripts/live/hitScale.ts [--songs a,b]

import { loadEngine } from './bundle';
import { decode, offlineCached, slugOf, testSongs } from './common';
import { record, signalChannels } from './parity';

const argv = process.argv.slice(2);
const want = argv.includes('--songs') ? argv[argv.indexOf('--songs') + 1].split(',') : null;
const E = await loadEngine();
const ch = signalChannels(['hit', 'drums']);
const pct = (a: number[], q: number) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(q * (a.length - 1))] : NaN);
const tot = { off: [] as number[], live: [] as number[] };
let minutes = 0;
console.log('song'.padEnd(44) + 'hits/min off  live   >0.5 off  live   median off  live');
for (const p of testSongs()) {
  const name = slugOf(p);
  if (want && !want.some((w) => name.includes(w))) continue;
  const pcm = decode(p);
  const rec = record(E, pcm, offlineCached(name, pcm), ch);
  const vals = (a: Float32Array) => {
    const out: number[] = [];
    // A hit sets hitPulse to F.hit (>= its decayed value): a rise marks it.
    for (let i = 1; i < a.length; i++) if (a[i] > a[i - 1] + 0.02) out.push(a[i]);
    return out;
  };
  const o = vals(rec.off[0]), l = vals(rec.live[0]);
  tot.off.push(...o);
  tot.live.push(...l);
  const m = rec.seconds / 60;
  minutes += m;
  const above = (a: number[]) => (a.length ? a.filter((x) => x > 0.5).length / a.length : NaN);
  console.log(`${name.padEnd(44)}${(o.length / m).toFixed(0).padStart(8)} ${(l.length / m).toFixed(0).padStart(5)}   ${above(o).toFixed(2).padStart(8)} ${above(l).toFixed(2).padStart(5)}   ${pct(o, 0.5).toFixed(2).padStart(10)} ${pct(l, 0.5).toFixed(2).padStart(5)}`);
}
const above = (a: number[]) => a.filter((x) => x > 0.5).length / a.length;
console.log(`ALL: hits/min off ${(tot.off.length / minutes).toFixed(0)} live ${(tot.live.length / minutes).toFixed(0)}; share > 0.5 off ${above(tot.off).toFixed(2)} live ${above(tot.live).toFixed(2)}; p50 off ${pct(tot.off, 0.5).toFixed(2)} live ${pct(tot.live, 0.5).toFixed(2)}; p90 off ${pct(tot.off, 0.9).toFixed(2)} live ${pct(tot.live, 0.9).toFixed(2)}`);

// How the engine's drum hits (Signals F.hit) come out offline vs live, and how many of them pass
// the hit-triggered motion gates: the old fixed thresholds (F.hit rising past 0.5 / 0.7) and the
// relative hit gate (F.hitGate: strength against the song's recent hits, an eighth note apart).
//   node --import ./scripts/analysis-test.hooks.mjs scripts/live/hitScale.ts [--songs a,b]

import { loadEngine } from './bundle';
import { decode, offlineCached, slugOf, testSongs } from './common';
import { record, type Channel } from './parity';

const argv = process.argv.slice(2);
const want = argv.includes('--songs') ? argv[argv.indexOf('--songs') + 1].split(',') : null;
const E = await loadEngine();
const ch: Channel[] = [
  { key: 'hit', group: 'engine', kind: { t: 'cont' }, read: (_s, _sig, F) => F.hit as number },
  { key: 'gate', group: 'engine', kind: { t: 'cont' }, read: (_s, _sig, F) => (F.hitGate ? 1 : 0) },
];
const pct = (a: number[], q: number) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(q * (a.length - 1))] : NaN);
const tot = { off: [] as number[], live: [] as number[], gOff: 0, gLive: 0 };
let minutes = 0;
const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : 'n/a');
console.log('song'.padEnd(44) + 'hits/min off live | share >0.5 off live | >0.7 off live | gate off live | gated/min off live');
for (const p of testSongs()) {
  const name = slugOf(p);
  if (want && !want.some((w) => name.includes(w))) continue;
  const pcm = decode(p);
  const rec = record(E, pcm, offlineCached(name, pcm), ch);
  const hits = (a: Float32Array) => Array.from(a).filter((x) => x > 0);
  const gates = (a: Float32Array) => Array.from(a).filter((x) => x > 0).length;
  const o = hits(rec.off[0]), l = hits(rec.live[0]);
  const go = gates(rec.off[1]), gl = gates(rec.live[1]);
  tot.off.push(...o);
  tot.live.push(...l);
  tot.gOff += go;
  tot.gLive += gl;
  const m = rec.seconds / 60;
  minutes += m;
  const above = (a: number[], t: number) => (a.length ? a.filter((x) => x > t).length / a.length : NaN);
  console.log(`${name.padEnd(44)}${(o.length / m).toFixed(0).padStart(8)} ${(l.length / m).toFixed(0).padStart(4)} | ${f2(above(o, 0.5)).padStart(14)} ${f2(above(l, 0.5))} | ${f2(above(o, 0.7)).padStart(8)} ${f2(above(l, 0.7))} | ${f2(go / o.length).padStart(8)} ${f2(gl / l.length)} | ${(go / m).toFixed(0).padStart(13)} ${(gl / m).toFixed(0).padStart(4)}`);
}
const above = (a: number[], t: number) => a.filter((x) => x > t).length / a.length;
console.log(`ALL: hits/min off ${(tot.off.length / minutes).toFixed(0)} live ${(tot.live.length / minutes).toFixed(0)}; share > 0.5 off ${f2(above(tot.off, 0.5))} live ${f2(above(tot.live, 0.5))}; > 0.7 off ${f2(above(tot.off, 0.7))} live ${f2(above(tot.live, 0.7))}; hit gate off ${f2(tot.gOff / tot.off.length)} live ${f2(tot.gLive / tot.live.length)}; p50 off ${f2(pct(tot.off, 0.5))} live ${f2(pct(tot.live, 0.5))}`);

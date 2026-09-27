// Beat clock experiments on the neural beat activation: BeatTracker (src/analysis/rtBeat.ts) fed the
// beat RNN's activation, scored against Beat This! (beatPhase in-phase share, onBeat F1) on the test
// songs and the DJ mixes, with a per-10 s timeline for the mixes (re-lock after cuts and ramps).
// Activations are cached in .testdata/ml/rnnact/. Audio is read from the decoded PCM cache, never played.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/beat-eval.ts [--set k=v,...] [--timeline] [--only a,b]
// --set: BeatTracker fields to override (numbers or true / false), e.g. --set relatedSwitch=true,switchEvals=4
// --ext: also feed the downbeat network's delayed evidence (src/analysis/downbeatBlstm.ts, 2 nets, 5 s
//        windows every 1 s, 1.5 s right context; cached in .testdata/ml/dbev/) at its arrival time:
//        downbeats to downbeatEvidence() (--ext bar).

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { BeatRnn } from '../../src/analysis/beatRnn';
import { BeatTracker } from '../../src/analysis/rtBeat';
import { DownbeatTracker } from '../../src/analysis/downbeatBlstm';
import { beatRnnModelSync, downbeatModelSync, OUT } from '../live/common';
import { FPS, score, type Channel } from '../live/parity';

const argv = process.argv.slice(2);
const arg = (k: string) => (argv.includes('--' + k) ? argv[argv.indexOf('--' + k) + 1] : undefined);
const sets: Record<string, number | boolean> = {};
for (const kv of (arg('set') ?? '').split(',').filter(Boolean)) {
  const [k, v] = kv.split('=');
  sets[k] = v === 'true' ? true : v === 'false' ? false : Number(v);
}
const only = arg('only')?.split(',');
const ext = arg('ext');
const DBEV = join(import.meta.dirname, '../../.testdata/ml/dbev');
mkdirSync(DBEV, { recursive: true });

/** Downbeat network evidence rows [frame time, arrival time, beat, downbeat] for an input. */
function evidence(key: string): Float32Array {
  const f = join(DBEV, key + '.f32');
  if (existsSync(f)) { const b = readFileSync(f); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
  const b = readFileSync(join(OUT, 'pcm', key + '.f32'));
  const st = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  const x = new Float32Array(st.length >> 1);
  for (let i = 0; i < x.length; i++) x[i] = 0.5 * (st[2 * i] + st[2 * i + 1]);
  const tr = new DownbeatTracker(downbeatModelSync(), { sampleRate: 44100, nets: 2, window: 5, hop: 1, delay: 1.5 });
  const rows: number[] = [];
  let now = 0;
  tr.onEvidence = (t, d, bb) => rows.push(t, now, bb, d);
  for (let i = 0; i + 512 <= x.length; i += 512) {
    tr.push(x.subarray(i, i + 512));
    now = (i + 512) / 44100;
    tr.work(Infinity);
  }
  const out = Float32Array.from(rows);
  writeFileSync(f, Buffer.from(out.buffer));
  return out;
}
const ACT = join(import.meta.dirname, '../../.testdata/ml/rnnact');
mkdirSync(ACT, { recursive: true });
const frac = (x: number) => x - Math.floor(x);
const PH: Channel = { key: 'beatPhase', group: 'beat', kind: { t: 'phase', tol: 0.1 }, read: () => 0 };
const ON: Channel = { key: 'onBeat', group: 'beat', kind: { t: 'event', tol: 0.07, jump: 0.5 }, read: () => 0 };

function activation(key: string): { act: Float32Array; energy: Float32Array } {
  const fa = join(ACT, key + '.act.f32'), fe = join(ACT, key + '.en.f32');
  const rd = (f: string) => { const b = readFileSync(f); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
  if (existsSync(fa) && existsSync(fe)) return { act: rd(fa), energy: rd(fe) };
  const b = readFileSync(join(OUT, 'pcm', key + '.f32'));
  const st = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  const x = new Float32Array(st.length >> 1);
  for (let i = 0; i < x.length; i++) x[i] = 0.5 * (st[2 * i] + st[2 * i + 1]);
  const rnn = new BeatRnn(beatRnnModelSync(), { sampleRate: 44100 });
  const acts: number[] = [], en: number[] = [];
  let k = 0;
  rnn.push(x, (a) => {
    acts.push(a);
    let e = 0;
    for (let i = Math.max(0, k * 441 - 440); i <= k * 441 && i < x.length; i += 4) e += x[i] * x[i];
    en.push(e / 110);
    k++;
  });
  const act = Float32Array.from(acts), energy = Float32Array.from(en);
  writeFileSync(fa, Buffer.from(act.buffer));
  writeFileSync(fe, Buffer.from(energy.buffer));
  return { act, energy };
}

const keys = readdirSync(join(OUT, 'beatref')).map((f) => f.replace(/\.json$/, '')).filter((k) => existsSync(join(OUT, 'pcm', k + '.f32')) && (!only || only.some((o) => k.includes(o))));
const rows: { key: string; ph: number; on: number }[] = [];
for (const key of keys.sort()) {
  const ref = JSON.parse(readFileSync(join(OUT, 'beatref', key + '.json'), 'utf8')) as { beats: number[] };
  const { act, energy } = activation(key);
  const bt = new BeatTracker(100);
  Object.assign(bt, sets);
  const T = act.length;
  const ev = ext ? evidence(key) : new Float32Array(0);
  let ei = 0;
  const refPos = new Float64Array(T), refTime = new Float64Array(T), period = new Float64Array(T), bpm = new Float32Array(T);
  for (let k = 0; k < T; k++) {
    const t = (k * 441 + 1) / 44100;
    bt.push(4 * act[k], t, energy[k] > 1e-6, { kick: 0, bass: 0, snare: 0 }, null);
    while (ei < ev.length && ev[ei + 1] <= t) {
      bt.downbeatEvidence(ev[ei], ev[ei + 3]);
      ei += 4;
    }
    refPos[k] = bt.refPos; refTime[k] = bt.refTime; period[k] = bt.period; bpm[k] = bt.bpm;
  }
  const n = Math.floor((T / 100) * FPS) - 2;
  const live = new Float32Array(n), on = new Float32Array(n), rph = new Float32Array(n), ron = new Float32Array(n);
  let last = NaN, bi = 0, lo = 0;
  const g = ref.beats;
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / FPS;
    const k = Math.min(T - 1, Math.floor(t * 100));
    const p = refPos[k] + (Math.min(t + 0.02, (k * 441 + 1) / 44100 + 0.07) - refTime[k]) / period[k];
    live[i] = frac(p);
    on[i] = Number.isFinite(last) && Math.floor(p) > last ? 1 : 0;
    last = Math.max(Number.isFinite(last) ? last : -Infinity, Math.floor(p));
    while (lo + 1 < g.length && g[lo + 1] <= t) lo++;
    rph[i] = g.length > 1 && t >= g[0] && t < g[g.length - 1] ? (t - g[lo]) / (g[lo + 1] - g[lo]) : NaN;
    while (bi < g.length && g[bi] <= t) (ron[i] = 1), bi++;
  }
  const sp = score(PH, rph, live, FPS), so = score(ON, ron, on, FPS);
  rows.push({ key, ph: sp.quality, on: so.quality });
  let line = `${key.slice(0, 28).padEnd(28)} beatPhase ${sp.quality.toFixed(2)} onBeat ${so.quality.toFixed(2)}`;
  if (argv.includes('--timeline') && (key.startsWith('mix') || only)) {
    const seg: string[] = [];
    for (let s0 = 0; s0 + 10 <= n / FPS; s0 += 10) {
      let ok = 0, c = 0;
      for (let i = s0 * FPS; i < (s0 + 10) * FPS; i++) {
        if (!Number.isFinite(rph[i])) continue;
        c++;
        let e = live[i] - rph[i];
        e -= Math.round(e);
        if (Math.abs(e) < 0.1) ok++;
      }
      const k = Math.min(T - 1, (s0 + 5) * 100);
      const rb = g.filter((b) => b >= s0 && b < s0 + 10);
      const refBpm = rb.length > 2 ? (60 * (rb.length - 1)) / (rb[rb.length - 1] - rb[0]) : NaN;
      seg.push(`${String(s0).padStart(3)}s ${c ? (ok / c).toFixed(2) : ' -- '} bpm ${bpm[k].toFixed(0)}/${Number.isFinite(refBpm) ? refBpm.toFixed(0) : '--'}`);
    }
    line += '\n  ' + seg.join('\n  ');
  }
  console.log(line);
}
const mean = (a: number[]) => a.reduce((p, q) => p + q, 0) / a.length;
const songs = rows.filter((r) => !r.key.startsWith('mix'));
const mixes = rows.filter((r) => r.key.startsWith('mix'));
console.log(`songs (${songs.length}): beatPhase ${mean(songs.map((r) => r.ph)).toFixed(3)} onBeat ${mean(songs.map((r) => r.on)).toFixed(3)} | mixes (${mixes.length}): beatPhase ${mixes.length ? mean(mixes.map((r) => r.ph)).toFixed(3) : 'n/a'} onBeat ${mixes.length ? mean(mixes.map((r) => r.on)).toFixed(3) : 'n/a'}  ${JSON.stringify(sets)}`);

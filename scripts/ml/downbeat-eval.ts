// Causal downbeat prototypes on a causal beat clock, scored against Beat This! beats / downbeats with
// the parity metrics (beatPhase / barPhase in-phase share, onBar F1). Audio is never played.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/downbeat-eval.ts <variant> [<variant> ...]
//
// Variants:
//   win:<tag>          madmom's downbeat BLSTM on sliding windows of past audio (scripts/ml/downbeat-window.py,
//                      .testdata/ml/downbeat/<tag>/), known D + H s late, on the beat RNN's clock
//   student:<name>     a student's downbeat activation (train-student.py, .testdata/ml/models/<name>.onnx) on
//                      the beat RNN's clock
//   student-all:<name> the student's beat activation through BeatTracker, and its downbeats
//   ts:<nets>:<window>:<hop>:<delay> the TypeScript port (src/analysis/downbeatBlstm.ts) streaming, as live
// The bar slot: downbeat evidence at each beat (peak within +-50 ms) accumulated per slot of 4 with
// BeatTracker's decay and hysteresis. 'live' reads the live path from .testdata/live/parity-rnn-ref.

import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { basename, join } from 'node:path';
import { BeatRnn } from '../../src/analysis/beatRnn';
import { DownbeatTracker, parseDownbeat, type DownbeatModel } from '../../src/analysis/downbeatBlstm';
import { BeatTracker } from '../../src/analysis/rtBeat';
import { beatRnnModelSync, decode, mono, OUT, slugOf, testSongs } from '../live/common';
import { FPS, score, type Channel } from '../live/parity';

const require = createRequire(import.meta.url);
const variants = process.argv.slice(2);
const ML = join(import.meta.dirname, '../../.testdata/ml');
const DATA = join(process.env.HOME!, 'personal/MusicVis-data');
const frac = (x: number) => x - Math.floor(x);
const PH: Channel = { key: 'phase', group: 'beat', kind: { t: 'phase', tol: 0.1 }, read: () => 0 };
const ONBAR: Channel = { key: 'onBar', group: 'beat', kind: { t: 'event', tol: 0.1, jump: 0.5 }, read: () => 0 };

function refRows(beats: number[], downbeats: number[], n: number) {
  const ph = (g: number[], t: number) => {
    if (g.length < 2 || t < g[0] || t >= g[g.length - 1]) return NaN;
    let lo = 0, hi = g.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (g[m] <= t) lo = m; else hi = m; }
    return (t - g[lo]) / (g[hi] - g[lo]);
  };
  const beat = new Float32Array(n), bar = new Float32Array(n), on = new Float32Array(n);
  let di = 0;
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / FPS;
    beat[i] = ph(beats, t);
    bar[i] = ph(downbeats, t);
    while (di < downbeats.length && downbeats[di] <= t) (on[i] = 1), di++;
  }
  return { beat, bar, on };
}

/** A causal beat clock from a 100 fps onset activation through BeatTracker. */
function clockOf(act: Float32Array, energy: Float32Array) {
  const bt = new BeatTracker(100);
  const T = act.length;
  const refPos = new Float64Array(T), refTime = new Float64Array(T), period = new Float64Array(T);
  for (let k = 0; k < T; k++) {
    const t = (k * 441 + 1) / 44100;
    bt.push(4 * act[k], t, energy[k] > 1e-6, { kick: 0, bass: 0, snare: 0 }, null);
    refPos[k] = bt.refPos; refTime[k] = bt.refTime; period[k] = bt.period;
  }
  return (t: number) => {
    const k = Math.min(T - 1, Math.floor(t * 100));
    if (k < 0) return NaN;
    return refPos[k] + (Math.min(t + 0.02, (k * 441 + 1) / 44100 + 0.07) - refTime[k]) / period[k];
  };
}

/** Bar phase and onBar from a beat clock and downbeat evidence known `delay` s after its frame. */
function barOf(posAt: (t: number) => number, db: Float32Array, delay: number, n: number) {
  const slotScore = new Float64Array(4);
  let slot = 0, lastFl = NaN, prevBar = NaN;
  const pending: { beat: number; t: number }[] = [];
  const beat = new Float32Array(n), bar = new Float32Array(n), on = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / FPS;
    const p = posAt(t);
    if (!Number.isFinite(p)) { beat[i] = bar[i] = NaN; continue; }
    const fl = Math.floor(p);
    if (fl !== lastFl) {
      if (Number.isFinite(lastFl)) pending.push({ beat: fl, t });
      lastFl = fl;
    }
    while (pending.length && pending[0].t + delay <= t) {
      const pb = pending.shift()!;
      let v = 0;
      const c = Math.round(pb.t * 100);
      for (let j = c - 5; j <= c + 5; j++) if (j >= 0 && j < db.length) v = Math.max(v, db[j]);
      const s = ((pb.beat % 4) + 4) % 4;
      for (let k = 0; k < 4; k++) slotScore[k] *= 0.985;
      slotScore[s] += v;
      let best = slot;
      for (let k = 0; k < 4; k++) if (slotScore[k] > slotScore[best]) best = k;
      const cur = slotScore[slot];
      if (best !== slot && slotScore[best] > cur + 0.3 * Math.abs(cur) + 0.5) slot = best;
    }
    beat[i] = frac(p);
    bar[i] = frac((p - slot) / 4);
    on[i] = Number.isFinite(prevBar) && bar[i] < prevBar - 0.5 ? 1 : 0;
    prevBar = bar[i];
  }
  return { beat, bar, on };
}

const sessions = new Map<string, { s: import('onnxruntime-node').InferenceSession; meta: { hidden: number; layers: number } }>();
async function student(name: string, feat: Float32Array): Promise<{ beat: Float32Array; down: Float32Array }> {
  const ort = require('onnxruntime-node') as typeof import('onnxruntime-node');
  if (!sessions.has(name)) {
    sessions.set(name, {
      s: await ort.InferenceSession.create(join(ML, 'models', name + '.onnx'), { intraOpNumThreads: 1 }),
      meta: JSON.parse(readFileSync(join(ML, 'models', name + '.json'), 'utf8')),
    });
  }
  const { s, meta } = sessions.get(name)!;
  const T = feat.length / 162;
  const z = () => new ort.Tensor('float32', new Float32Array(meta.layers * meta.hidden), [meta.layers, 1, meta.hidden]);
  let h = z(), c = z();
  const beat = new Float32Array(T), down = new Float32Array(T);
  for (let k = 0; k < T; k++) {
    const o = await s.run({ x: new ort.Tensor('float32', feat.slice(k * 162, (k + 1) * 162), [1, 1, 162]), h, c });
    const y = o.y.data as Float32Array;
    beat[k] = y[0]; down[k] = y[1];
    h = o.h_out as typeof h; c = o.c_out as typeof c;
  }
  return { beat, down };
}

let dbm: DownbeatModel | null = null;
function dbModel(): DownbeatModel {
  if (!dbm) {
    const b = readFileSync(join(import.meta.dirname, '../../public/models/downbeat-blstm.bin'));
    dbm = parseDownbeat(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  }
  return dbm;
}

const res: Record<string, { beat: number[]; bar: number[]; on: number[] }> = {};
const put = (k: string, b: number, r: number, o: number) => {
  const a = (res[k] ??= { beat: [], bar: [], on: [] });
  a.beat.push(b); a.bar.push(r); a.on.push(o);
};
for (const path of testSongs()) {
  const slug = slugOf(path);
  const ref = JSON.parse(readFileSync(join(OUT, 'beatref', slug + '.json'), 'utf8')) as { beats: number[]; downbeats: number[] };
  const pcm = decode(path);
  const x = mono(pcm);
  const n = Math.floor((x.length / pcm.sr) * FPS) - 2;
  const R = refRows(ref.beats, ref.downbeats, n);
  // The beat RNN (8 nets), its features and frame energies at 100 fps.
  const rnn = new BeatRnn(beatRnnModelSync(), { sampleRate: pcm.sr });
  const acts: number[] = [], feats: number[] = [], en: number[] = [];
  let k = 0;
  rnn.push(x, (a) => {
    acts.push(a);
    for (let i = 0; i < 162; i++) feats.push(rnn.feat[i]);
    let e = 0;
    for (let i = Math.max(0, k * 441 - 440); i <= k * 441 && i < x.length; i += 4) e += x[i] * x[i];
    en.push(e / 110);
    k++;
  });
  const act = Float32Array.from(acts), feat = Float32Array.from(feats), energy = Float32Array.from(en);
  const rnnClock = clockOf(act, energy);
  const scoreOf = (name: string, v: ReturnType<typeof barOf>) => {
    const sb = score(PH, R.beat, v.beat, FPS), sr = score(PH, R.bar, v.bar, FPS), so = score(ONBAR, R.on, v.on, FPS);
    put(name, sb.quality, sr.quality, so.quality);
    return `${name} beat ${sb.quality.toFixed(2)} bar ${sr.quality.toFixed(2)} onBar ${so.quality.toFixed(2)}`;
  };
  const line: string[] = [scoreOf('rnn, no downbeat', barOf(rnnClock, new Float32Array(act.length), 0, n))];
  for (const v of variants) {
    const [kind, tag] = v.split(':');
    if (kind === 'ts') {
      const [, ...rest] = v.split(':');
      const [n2, w2, h2, d2] = rest.map(Number);
      const tr = new DownbeatTracker(dbModel(), { sampleRate: 44100, nets: n2, window: w2, hop: h2, delay: d2 });
      const db = new Float32Array(act.length);
      tr.onEvidence = (t, d) => {
        const f = Math.round(t * 100);
        if (f >= 0 && f < db.length) db[f] = d;
      };
      for (let i = 0; i + 512 <= x.length; i += 512) {
        tr.push(x.subarray(i, i + 512));
        tr.work(Infinity);
      }
      line.push(scoreOf(v, barOf(rnnClock, db, h2 + d2 + 0.05, n)));
    } else if (kind === 'win') {
      const f = join(ML, 'downbeat', tag, slug + '.f32');
      if (!existsSync(f)) continue;
      const b = readFileSync(f);
      const m = /h([\d.]+)-d([\d.]+)/.exec(tag)!;
      line.push(scoreOf(v, barOf(rnnClock, new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)), Number(m[1]) + Number(m[2]), n)));
    } else if (kind === 'student' || kind === 'student-all') {
      const s = await student(tag, feat);
      const clock = kind === 'student' ? rnnClock : clockOf(s.beat, energy);
      // A downbeat head trained with a lag (dbLag frames of right context) reports frame k at k + lag.
      const lag = (JSON.parse(readFileSync(join(ML, 'models', tag + '.json'), 'utf8')) as { dbLag?: number }).dbLag ?? 0;
      const down = lag ? Float32Array.from({ length: s.down.length }, (_, k) => s.down[k + lag] ?? 0) : s.down;
      line.push(scoreOf(v, barOf(clock, down, 0.06 + lag / 100, n)));
    }
  }
  console.error(`${slug.slice(0, 22).padEnd(22)} ${line.join(' | ')}`);
  const lp = join(OUT, 'parity-rnn-ref', slug + '.json');
  if (existsSync(lp)) {
    const j = JSON.parse(readFileSync(lp, 'utf8')) as { scores: { key: string; group: string; quality: number }[] };
    const q = (key: string) => j.scores.find((s) => s.group === 'beat' && s.key === key)!.quality;
    put('live (rnn + DSP accents)', q('beatPhase'), q('barPhase'), q('onBar'));
  }
}
void basename;
const mean = (a: number[]) => a.reduce((p, q) => p + q, 0) / a.length;
console.log('\n| variant | beatPhase | barPhase | onBar F1 | songs |\n|---|---|---|---|---|');
for (const [k, v] of Object.entries(res)) console.log(`| ${k} | ${mean(v.beat).toFixed(2)} | ${mean(v.bar).toFixed(2)} | ${mean(v.on).toFixed(2)} | ${v.bar.length} |`);

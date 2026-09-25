// Tests for the AV quality metrics (scripts/avq/metrics.ts) on synthetic clips with known answers.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/avq-test.ts

import {
  cfSummary, correspondences, flowStats, hookRhyme, interestStats, melodyStats, onsetEvents, reportCard, rhymeInputs, structureStats,
  syncStats, thumbChange, visualResponse, couplingStats, featureSeries, type ClipLike,
} from './avq/metrics';
import type { Hook } from './avq/music';

let failed = 0;
function check(name: string, ok: boolean, detail = ''): void {
  if (!ok) failed++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
}

const FPS = 30;
const BPM = 120;
const BEAT = 60 / BPM;
const TW = 32, TH = 18;

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Synth {
  n: number;
  cols: Record<string, Float32Array>;
  /** Grey level 0..1 per frame for uniform thumbnails, or a custom thumb function. */
  thumb?: (i: number) => Uint8Array;
  hooks?: Hook[];
  moments?: ClipLike['moments'];
}

function clip(s: Synth): ClipLike {
  const z = () => new Float32Array(s.n);
  const cols = new Map(Object.entries(s.cols));
  const lum = cols.get('lum') ?? z();
  const thumbs = Array.from({ length: s.n }, (_, i) => s.thumb?.(i) ?? new Uint8Array(TW * TH * 3).fill(Math.round(lum[i] * 255)));
  const beats: number[] = [];
  for (let t = 0; t < s.n / FPS + 1; t += BEAT) beats.push(t);
  return {
    fps: FPS, n: s.n,
    col: (name) => cols.get(name) ?? (cols.set(name, name === 'cx' || name === 'cy' ? z().fill(0.5) : z()), cols.get(name)!),
    has: (name) => cols.has(name),
    thumb: (i) => thumbs[i],
    thumbW: TW, thumbH: TH,
    beats, downbeats: beats.filter((_, i) => i % 4 === 0), beatsPerBar: 4, bpm: BPM,
    sections: [], moments: s.moments ?? [], hooks: s.hooks ?? [],
  };
}

/** Onset strength with spikes at the given frames. */
function spikes(n: number, at: number[], width = 1): Float32Array {
  const x = new Float32Array(n);
  for (const a of at) for (let d = 0; d < width; d++) if (a + d < n) x[a + d] = Math.max(x[a + d], 1 - d / width);
  return x;
}

/** Visual "flash" decays starting at the given frames. */
function flashes(n: number, at: number[], amp = 0.3): Float32Array {
  const x = new Float32Array(n).fill(0.1);
  for (const a of at) for (let d = 0; d < 6 && a + d < n; d++) x[a + d] += amp * Math.exp(-d / 1.5);
  return x;
}

// ---------------------------------------------------------------- sync

{
  const n = 900;
  const r = rng(1);
  const onsets: number[] = [];
  for (let f = 20; f < n - 20; f += 11 + Math.floor(r() * 12)) onsets.push(f);
  const on = spikes(n, onsets);
  const mk = (lagF: number) => {
    const lum = flashes(n, onsets.map((o) => o + lagF));
    return clip({ n, cols: { onDrums: on, lum, diff: Float32Array.from(lum, (v, i) => (i ? Math.abs(v - lum[i - 1]) : 0)) } });
  };
  const ev = onsetEvents(on, FPS);
  check('onsetEvents finds every spike', ev.length === onsets.length, `${ev.length}/${onsets.length}`);
  const inSync = mk(2);
  const R = visualResponse(inSync);
  const s = syncStats(ev, R.peaks, FPS);
  check('sync: responses 67 ms late are hits', s.hitRate > 0.9 && s.lift > 0.7, `hit ${s.hitRate.toFixed(2)} chance ${s.chance.toFixed(2)} lift ${s.lift.toFixed(2)}`);
  check('sync: mean lag ~ 67 ms', Math.abs(s.meanLagMs - 66.7) < 20, `${s.meanLagMs.toFixed(0)} ms`);
  const late = syncStats(ev, visualResponse(mk(9)).peaks, FPS);
  check('sync: responses 300 ms late are misses', late.hitRate < 0.25 && late.lift < 0.15, `hit ${late.hitRate.toFixed(2)} lift ${late.lift.toFixed(2)}`);
  const early = syncStats(ev, visualResponse(mk(-3)).peaks, FPS);
  check('sync: responses 100 ms early fall outside the (asymmetric) window', early.lift < 0.3, `lift ${early.lift.toFixed(2)}`);
  const rand: number[] = [];
  for (let f = 15; f < n - 15; f += 11 + Math.floor(r() * 12)) rand.push(f);
  const lumR = flashes(n, rand);
  const cr = clip({ n, cols: { onDrums: on, lum: lumR } });
  const sr = syncStats(ev, visualResponse(cr).peaks, FPS);
  check('sync: unrelated responses score ~ chance', Math.abs(sr.lift) < 0.25, `hit ${sr.hitRate.toFixed(2)} chance ${sr.chance.toFixed(2)} lift ${sr.lift.toFixed(2)}`);
}

// ---------------------------------------------------------------- coupling

{
  const n = 1200;
  const r = rng(2);
  const smoothNoise = (seed: number) => {
    const rr = rng(seed);
    const x = new Float32Array(n);
    let v = 0.5;
    for (let i = 0; i < n; i++) x[i] = v = Math.max(0, Math.min(1, v + (rr() - 0.5) * 0.15));
    return x;
  };
  const drums = smoothNoise(10), bass = smoothNoise(11), vocals = smoothNoise(12), other = smoothNoise(13);
  const lag = (x: Float32Array, k: number) => Float32Array.from(x, (_, i) => x[Math.max(0, i - k)]);
  const pres = new Float32Array(n).fill(0.8);
  const base = { drums, bass, vocals, other, prDrums: pres, prBass: pres, prVocals: pres, prOther: pres };
  const noise = () => Float32Array.from({ length: n }, () => r() * 0.05);
  const specific = clip({ n, cols: { ...base, lum: lag(drums, 2), coverage: lag(vocals, 1), curl: lag(other, 3), lumStd: noise(), colorful: noise() } });
  const cs = couplingStats(specific, featureSeries(specific, thumbChange(specific)));
  check('coupling: drums -> lum found', cs.top.drums.feature === 'lum' && cs.top.drums.r > 0.8, `${cs.top.drums.feature} ${cs.top.drums.r.toFixed(2)}`);
  check('coupling: vocals -> coverage found', cs.top.vocals.feature === 'coverage' && cs.top.vocals.r > 0.8, `${cs.top.vocals.feature} ${cs.top.vocals.r.toFixed(2)}`);
  check('coupling: bass (unused) stays low', Math.abs(cs.top.bass.r) < 0.35, `${cs.top.bass.feature} ${cs.top.bass.r.toFixed(2)}`);
  check('coupling: 3 distinct drivers -> specificity 0.75', Math.abs(cs.specificity - 0.75) < 1e-9, `${cs.drivers.join(',')}`);
  const mix = Float32Array.from(drums, (v, i) => v + bass[i] + vocals[i] + other[i]);
  const same = clip({ n, cols: { ...base, lum: mix, coverage: mix, curl: mix } });
  const cm = couplingStats(same, featureSeries(same, thumbChange(same)));
  check('coupling: one mixed drive -> low specificity', cm.specificity <= 0.5, `spec ${cm.specificity.toFixed(2)} drivers ${cm.drivers.join(',')}`);
}

// ---------------------------------------------------------------- hook rhyme

{
  // 2-bar hook (8 beats = 4 s = 120 frames) repeated 6 times from t = 2 s.
  const n = 900;
  const hookLen = 8 * BEAT;
  const occ = Array.from({ length: 6 }, (_, k) => ({ start: 2 + k * hookLen, end: 2 + (k + 1) * hookLen, sim: 1 }));
  const hook: Hook = { id: 0, bars: 2, len: hookLen, occurrences: occ, salience: 1, distinct: 0.5, score: 1 };
  const r = rng(3);
  const pattern = Array.from({ length: 120 }, () => r());
  const clipT0 = -1 / FPS; // row i is at time i / fps
  const mk = (f: (i: number) => number, noise: number) => {
    const rr = rng(4);
    const diff = Float32Array.from({ length: n }, (_, i) => Math.max(0, f(i) + noise * (rr() - 0.5)));
    return clip({ n, cols: { diff, flowMag: diff, lum: Float32Array.from(diff, (v) => 0.2 + 0.1 * v) }, hooks: [hook] });
  };
  const follows = mk((i) => pattern[(i - 60 + 1200) % 120], 0.3);
  const rf = hookRhyme(follows, rhymeInputs(follows, thumbChange(follows)), hook, clipT0);
  check('rhyme: response repeating with the motif rhymes', rf.rhyme > 0.5, `S_hook ${rf.sHook.toFixed(2)} S_base ${rf.sBase.toFixed(2)} rhyme ${rf.rhyme.toFixed(2)}`);
  const beatOnly = mk((i) => Math.exp(-((i % 15) / 3)), 0.3);
  const rb = hookRhyme(beatOnly, rhymeInputs(beatOnly, thumbChange(beatOnly)), hook, clipT0);
  check('rhyme: beat-only flashing does not rhyme', Math.abs(rb.rhyme) < 0.2, `S_hook ${rb.sHook.toFixed(2)} S_base ${rb.sBase.toFixed(2)} rhyme ${rb.rhyme.toFixed(2)}`);
  const rnd = mk(() => 0, 1);
  const rr = hookRhyme(rnd, rhymeInputs(rnd, thumbChange(rnd)), hook, clipT0);
  check('rhyme: noise does not rhyme', Math.abs(rr.rhyme) < 0.2, `rhyme ${rr.rhyme.toFixed(2)}`);
}

// ---------------------------------------------------------------- melody + correspondences

{
  const n = 900;
  const r = rng(5);
  const midi = new Float32Array(n);
  let p = 64;
  for (let i = 0; i < n; i++) {
    if (i % 8 === 0) p = 60 + Math.floor(r() * 12);
    midi[i] = p;
  }
  const sal = new Float32Array(n).fill(0.8);
  const follow = clip({ n, cols: { melMidi: midi, melSal: sal, cy: Float32Array.from(midi, (v) => 0.3 + (v - 60) / 30) } });
  const m1 = melodyStats(follow);
  check('melody: height following pitch is found', m1.best.target === 'height' && m1.best.r > 0.8, `${m1.best.target} ${m1.best.r.toFixed(2)}`);
  const none = clip({ n, cols: { melMidi: midi, melSal: sal, cy: Float32Array.from({ length: n }, () => 0.5 + 0.1 * r()) } });
  const m2 = melodyStats(none);
  check('melody: unrelated height scores low', m2.score < 0.4, `${m2.best.target} ${m2.best.r.toFixed(2)}`);
  const loud = Float32Array.from({ length: n }, (_, i) => 0.5 + 0.4 * Math.sin(i / 40));
  const cor = correspondences(clip({ n, cols: { loud, coverage: Float32Array.from(loud, (v) => v * 0.5), lum: Float32Array.from(loud, (v) => 1 - v) } }));
  check('correspondence: loud -> size positive, loud -> bright negative', cor.rules['loud>size'] > 0.9 && cor.rules['loud>bright'] < -0.9, JSON.stringify(cor.rules));
}

// ---------------------------------------------------------------- structure

{
  const n = 600;
  const clipT0 = -1 / FPS;
  const r = rng(6);
  const grey = (i: number, level: number) => Uint8Array.from({ length: TW * TH * 3 }, (_, k) => Math.round(255 * Math.max(0, Math.min(1, level + 0.05 * Math.sin(k + i * 0.3)))));
  const moments = [{ t: 10, kind: 'drop' as const, label: 'build>drop' }];
  const changes = clip({ n, cols: {}, thumb: (i) => grey(i, i < 300 ? 0.2 : 0.7), moments });
  const s1 = structureStats(changes, clipT0);
  check('structure: a look change at the drop scores high', s1.score > 0.9, `ratio ${s1.boundaries[0]?.ratio.toFixed(1)} score ${s1.score.toFixed(2)}`);
  const flat = clip({ n, cols: {}, thumb: (i) => grey(i, 0.4 + 0.02 * r()), moments });
  const s2 = structureStats(flat, clipT0);
  check('structure: no change at the drop scores low', s2.score < 0.4, `ratio ${s2.boundaries[0]?.ratio.toFixed(2)} score ${s2.score.toFixed(2)}`);
}

// ---------------------------------------------------------------- flow

{
  const n = 300;
  const strobeLum = Float32Array.from({ length: n }, (_, i) => (Math.floor(i / 3) % 2 ? 0.9 : 0.1)); // 5 Hz full-frame
  const st = clip({ n, cols: { lum: strobeLum } });
  const fs = flowStats(st, thumbChange(st));
  check('flow: 5 Hz full-frame flashing is flagged as strobe', fs.strobe && fs.score === 0, `area ${fs.flashArea.toFixed(2)} rate ${fs.flashesPerSec.toFixed(1)}`);
  const slow = clip({ n, cols: { lum: Float32Array.from({ length: n }, (_, i) => 0.5 + 0.3 * Math.sin(i / 30)) } });
  const fsl = flowStats(slow, thumbChange(slow));
  check('flow: slow pulsing is not a strobe', !fsl.strobe, `area ${fsl.flashArea.toFixed(2)}`);
  const still = clip({ n, cols: { lum: new Float32Array(n).fill(0.3) } });
  const fst = flowStats(still, thumbChange(still));
  check('flow: a frozen frame is still', fst.stillness > 0.95, `still ${fst.stillness.toFixed(2)}`);
  const smoothMot = Float32Array.from({ length: n }, (_, i) => Math.sin(i / 20));
  const jitMot = Float32Array.from({ length: n }, (_, i) => (i % 2 ? 1 : -1));
  const js = flowStats(clip({ n, cols: { flowX: smoothMot } }), new Float32Array(n)).jerk;
  const jj = flowStats(clip({ n, cols: { flowX: jitMot } }), new Float32Array(n)).jerk;
  check('flow: jittery motion is jerkier than smooth motion', jj > js * 3, `jitter ${jj.toFixed(2)} smooth ${js.toFixed(2)}`);
}

// ---------------------------------------------------------------- interest

{
  const n = 600;
  const r = rng(8);
  const still = clip({ n, cols: {} , thumb: () => new Uint8Array(TW * TH * 3).fill(80) });
  const i1 = interestStats(still, visualResponse(still).resp, thumbChange(still));
  const noise = clip({ n, cols: { diff: Float32Array.from({ length: n }, () => r()) }, thumb: () => Uint8Array.from({ length: TW * TH * 3 }, () => Math.floor(r() * 255)) });
  const i2 = interestStats(noise, visualResponse(noise).resp, thumbChange(noise));
  const period = 15;
  const groove = clip({
    n, cols: { diff: Float32Array.from({ length: n }, (_, i) => Math.exp(-(i % period) / 3) * 0.02) },
    thumb: (i) => Uint8Array.from({ length: TW * TH * 3 }, (_, k) => 100 + Math.round(30 * Math.exp(-(i % period) / 3) * Math.sin(k * 0.7 + i * 0.05))),
  });
  const i3 = interestStats(groove, visualResponse(groove).resp, thumbChange(groove));
  check('interest: static scores ~0', i1.score < 0.05, i1.score.toFixed(2));
  check('interest: chaos scores low', i2.score < 0.15, `act ${i2.activity.toFixed(3)} pred ${i2.predictability.toFixed(2)} score ${i2.score.toFixed(2)}`);
  check('interest: moderate periodic motion scores highest', i3.score > i1.score && i3.score > i2.score && i3.score > 0.4, `act ${i3.activity.toFixed(3)} pred ${i3.predictability.toFixed(2)} score ${i3.score.toFixed(2)}`);
}

// ---------------------------------------------------------------- report card smoke

{
  const n = 600;
  const on = spikes(n, Array.from({ length: 38 }, (_, k) => 15 + k * 15));
  const lum = flashes(n, Array.from({ length: 38 }, (_, k) => 16 + k * 15));
  const rc = reportCard(clip({ n, cols: { onDrums: on, lum, diff: Float32Array.from(lum, (v, i) => (i ? Math.abs(v - lum[i - 1]) : 0)), prDrums: new Float32Array(n).fill(0.8) } }), { preset: 'X', song: 's', clip: 'c', clipT0: -1 / FPS });
  const finite = Object.entries(rc.headline).filter(([k]) => k !== 'hookRhyme' && k !== 'structure').every(([, v]) => Number.isFinite(v) && v >= 0 && v <= 1);
  check('report card: headline scores in 0..1', finite, JSON.stringify(rc.headline, (_, v) => (typeof v === 'number' ? +v.toFixed(2) : v)));
  check('report card: beat-locked flashes score sync > 0.6', rc.headline.sync > 0.6, rc.headline.sync.toFixed(2));
}

// ---------------------------------------------------------------- counterfactual summary

{
  const cf = {
    motion: 0.05,
    reactions: [{ src: 'bass', target: 'sh0.scale', gain: 0.3 }, { src: 'vocals', target: 'sh0.hue', gain: 0.2 }],
    variants: [
      { id: 'shift2b', kind: 'shift', beats: 2, rel: 0.9 },
      { id: 'gain0.98', kind: 'gain', rel: 0.1 },
      { id: 'offset80s', kind: 'offset', rel: 1.2 },
      { id: 'mute-drums', kind: 'mute', stem: 'drums', rel: 0.6 },
      { id: 'mute-other', kind: 'mute', stem: 'other', rel: 0.11 },
      { id: 'ablate-r0', kind: 'ablate', reaction: 0, rel: 0.4 },
      { id: 'ablate-r1', kind: 'ablate', reaction: 1, rel: 0.12 },
    ],
  };
  const s = cfSummary(cf);
  check('cf: sync sensitivity = desync - chaos floor', Math.abs(s.syncSensitivity - 0.8) < 1e-9, s.syncSensitivity.toFixed(2));
  check('cf: stem footprints subtract the floor', Math.abs(s.stems.drums - 0.5) < 1e-9 && s.stems.other < 0.02, JSON.stringify(s.stems));
  check('cf: a reaction within the chaos floor is dead', s.reactions[1].dead && !s.reactions[0].dead && s.dead === 1, JSON.stringify(s.reactions));
}

console.log(failed ? `FAILED: ${failed} check(s)` : 'PASSED: 0 failing check(s)');
process.exit(failed ? 1 : 0);

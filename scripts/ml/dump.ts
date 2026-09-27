// Training data for the ML prototypes: causal streaming features (scripts/ml/features.ts) and
// labels from the OFFLINE analysis (the reference the live path is measured against), per test song.
// Audio is decoded with ffmpeg and never played.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/dump.ts
//
// Writes .testdata/ml/data/<slug>.f32 (features, T x 160), <slug>.lab.f32 (labels, T x L) and index.json.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { decimate } from '../../src/analysis/dsp';
import { bandPower, stemBands, stftPower } from '../../src/analysis/stft';
import { computeStems } from '../../src/analysis/stems';
import type { AnalysisResult } from '../../src/types';
import { decode, offlineCached, slugOf, testSongs, type Pcm } from '../live/common';
import { featuresOf, frameTime, ML_FEAT } from './features';

export const LABELS = [
  'beat', 'downbeat', 'beatCos', 'beatSin', 'barCos', 'barSin',
  'stems.drums', 'stems.bass', 'stems.vocals', 'stems.other',
  'onset.drums', 'onset.bass', 'onset.vocals', 'onset.other',
  'kick', 'snare', 'hat', 'beatValid',
] as const;
export const OUT_ML = join(import.meta.dirname, '../../.testdata/ml');

/** Offline kick / snare / hat onset strengths (the HPSS stems' internals, not in AnalysisResult). */
export function drumOnsets(pcm: Pcm): { kick: Float32Array; snare: Float32Array; hat: Float32Array; frameRate: number } {
  const factor = Math.max(1, Math.round(pcm.sr / 22050));
  const sr = pcm.sr / factor;
  const l2 = decimate(pcm.left, factor), r2 = decimate(pcm.right, factor);
  const mid = new Float32Array(l2.length), side = new Float32Array(l2.length);
  for (let i = 0; i < l2.length; i++) (mid[i] = 0.5 * (l2[i] + r2[i])), (side[i] = 0.5 * (l2[i] - r2[i]));
  let hop = 1;
  while (hop * 2 <= sr / 86 * 1.414) hop *= 2;
  const n = hop * 8, frameRate = sr / hop;
  const T = Math.max(1, Math.floor((mid.length - 1) / hop) + 1);
  const bands = stemBands(n, sr);
  const B = bands.count;
  const midB = new Float32Array(T * B), sideB = new Float32Array(T * B);
  stftPower(mid, n, hop, T, (f, pow) => bandPower(pow, bands, midB, f * B));
  stftPower(side, n, hop, T, (f, pow) => bandPower(pow, bands, sideB, f * B));
  const st = computeStems(midB, sideB, bands, T, frameRate);
  const ph = st.raw.percHigh;
  const hat = new Float32Array(T);
  for (let i = 1; i < T; i++) hat[i] = Math.max(0, Math.log10(ph[i] + 1e-12) - Math.log10(ph[i - 1] + 1e-12));
  const s = [...hat].sort((a, b) => a - b)[Math.floor(0.99 * T)] || 1;
  for (let i = 0; i < T; i++) hat[i] = Math.min(1, hat[i] / s);
  return { kick: st.kickOnset, snare: st.snareOnset, hat, frameRate };
}

const lerpAt = (x: ArrayLike<number>, pos: number) => {
  const i = Math.floor(pos);
  if (i < 0) return x[0] ?? 0;
  if (i >= x.length - 1) return x[x.length - 1] ?? 0;
  const f = pos - i;
  return x[i] * (1 - f) + x[i + 1] * f;
};

/** Phase in [0,1) of t within a sorted event grid, NaN outside it. */
function gridPhase(grid: ArrayLike<number>, t: number): number {
  if (grid.length < 2 || t < grid[0] || t >= grid[grid.length - 1]) return NaN;
  let lo = 0, hi = grid.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (grid[m] <= t) lo = m;
    else hi = m;
  }
  return (t - grid[lo]) / (grid[hi] - grid[lo]);
}

/** speed: playback-rate of the (resampled) audio the frames come from; the song's time at frame time t is t * speed. */
export function labelsOf(res: AnalysisResult, drums: ReturnType<typeof drumOnsets>, T: number, speed = 1): Float32Array {
  const L = LABELS.length;
  const lab = new Float32Array(T * L);
  const put = (k: number, name: (typeof LABELS)[number], v: number) => (lab[k * L + LABELS.indexOf(name)] = v);
  const spike = (times: Float32Array, name: (typeof LABELS)[number]) => {
    for (const t of times) {
      const kf = t / speed / frameTime(0) - 1; // frame whose end time is t
      const k0 = Math.round(kf);
      for (let d = -1; d <= 1; d++) {
        const k = k0 + d;
        if (k < 0 || k >= T) continue;
        const v = d === 0 ? 1 : 0.5;
        const i = k * L + LABELS.indexOf(name);
        lab[i] = Math.max(lab[i], v);
      }
    }
  };
  spike(res.beats, 'beat');
  spike(res.downbeats, 'downbeat');
  const fr = res.frameRate;
  for (let k = 0; k < T; k++) {
    const t = frameTime(k) * speed;
    const bp = gridPhase(res.beats, t);
    const br = gridPhase(res.downbeats, t);
    const valid = Number.isFinite(bp) && Number.isFinite(br) ? 1 : 0;
    put(k, 'beatValid', valid);
    put(k, 'beatCos', valid ? Math.cos(2 * Math.PI * bp) : 0);
    put(k, 'beatSin', valid ? Math.sin(2 * Math.PI * bp) : 0);
    put(k, 'barCos', valid ? Math.cos(2 * Math.PI * br) : 0);
    put(k, 'barSin', valid ? Math.sin(2 * Math.PI * br) : 0);
    const p = t * fr;
    for (const s of ['drums', 'bass', 'vocals', 'other'] as const) {
      put(k, `stems.${s}`, lerpAt(res.stems[s], p));
      put(k, `onset.${s}`, lerpAt(res.stemOnsets[s], p));
    }
    const pd = t * drums.frameRate;
    put(k, 'kick', lerpAt(drums.kick, pd));
    put(k, 'snare', lerpAt(drums.snare, pd));
    put(k, 'hat', lerpAt(drums.hat, pd));
  }
  return lab;
}

/** Resample by linear interpolation: speed > 1 plays faster and higher (vinyl-style tempo + pitch shift). */
function resample(x: Float32Array, speed: number): Float32Array {
  const n = Math.floor((x.length - 1) / speed);
  const o = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * speed, j = Math.floor(p), f = p - j;
    o[i] = x[j] * (1 - f) + x[j + 1] * f;
  }
  return o;
}

async function main() {
  // --teacher beat-this: beat / downbeat labels from .testdata/ml/pretrained/beat-this/<slug>.json and
  // kick / snare / hat as spikes from pretrained/adtof/<slug>.json (instead of our offline analysis). Writes data-<teacher>/.
  const ti = process.argv.indexOf('--teacher');
  const teacher = ti > 0 ? process.argv[ti + 1] : '';
  const dir = join(OUT_ML, teacher ? 'data-' + teacher : 'data');
  mkdirSync(dir, { recursive: true });
  // --speeds 0.94,1.06: extra augmented copies (tempo and pitch shifted together), named <slug>@<speed>.
  const si = process.argv.indexOf('--speeds');
  const speeds = [1, ...(si > 0 ? process.argv[si + 1].split(',').map(Number) : [])];
  const index: { slug: string; song: string; speed: number; T: number }[] = [];
  for (const path of testSongs()) {
    const song = slugOf(path);
    const pcm0 = decode(path);
    let res = offlineCached(song, pcm0);
    let drums = drumOnsets(pcm0);
    if (teacher) {
      const j = JSON.parse(readFileSync(join(OUT_ML, 'pretrained', teacher, song + '.json'), 'utf8')) as { beats: number[]; downbeats: number[] };
      res = { ...res, beats: Float32Array.from(j.beats), downbeats: Float32Array.from(j.downbeats) };
      const ad = join(OUT_ML, 'pretrained', 'adtof', song + '.json');
      if (existsSync(ad)) {
        const a = JSON.parse(readFileSync(ad, 'utf8')) as Record<string, number[]>;
        const fr = drums.frameRate, T0 = drums.kick.length;
        const spikes = (ts: number[] = []) => {
          const o = new Float32Array(T0);
          for (const t of ts) for (let d = -1; d <= 1; d++) { const i = Math.round(t * fr) + d; if (i >= 0 && i < T0) o[i] = Math.max(o[i], d === 0 ? 1 : 0.5); }
          return o;
        };
        drums = { kick: spikes(a.kick), snare: spikes(a.snare), hat: spikes(a.hat), frameRate: fr };
      }
    }
    for (const speed of speeds) {
      const t0 = Date.now();
      const slug = speed === 1 ? song : `${song}@${speed}`;
      const pcm = speed === 1 ? pcm0 : { sr: pcm0.sr, left: resample(pcm0.left, speed), right: resample(pcm0.right, speed) };
      const feat = featuresOf(pcm.left, pcm.right);
      const T = feat.length / ML_FEAT;
      const lab = labelsOf(res, drums, T, speed);
      writeFileSync(join(dir, slug + '.f32'), Buffer.from(feat.buffer));
      writeFileSync(join(dir, slug + '.lab.f32'), Buffer.from(lab.buffer));
      index.push({ slug, song, speed, T });
      console.error(`${slug}: ${T} frames, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    }
  }
  writeFileSync(join(dir, 'index.json'), JSON.stringify({ feat: ML_FEAT, labels: LABELS, songs: index }, null, 1));
}

if (process.argv[1]?.endsWith('dump.ts')) await main();

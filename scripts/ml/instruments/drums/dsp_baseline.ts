// Baseline (a): the current live DSP path (RealtimeAnalyzer), driven offline exactly as
// LiveInput.ts feeds it (512-sample blocks in order), scored the same way as the student model.
//
// RealtimeAnalyzer only exposes a single combined "drums" onset-strength signal per frame
// (stemOnsets.drums; it does not separate kick/snare/hat internally in its public output), so
// this baseline predicts ONE onset stream per song and scores it against every class's
// reference times (it cannot discriminate instrument type, unlike the ADTOF-trained student).
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/drums/dsp_baseline.ts
//
// Writes .testdata/instr/drums/dsp_onsets/*.f32; then score with
//   .testdata/instr/venvs/torch/bin/python scripts/ml/instruments/drums/eval.py --dsp

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { RealtimeAnalyzer } from '../../../../src/analysis/RealtimeAnalyzer';
import { execFileSync } from 'node:child_process';
import { decode as decodeCached, ffmpegPath, SR } from '../../../live/common';

// Test songs use the shared cached decode; every other track is decoded straight from ffmpeg to a
// pipe WITHOUT writing a 44.1 kHz stereo cache file (disk is tight).
function decode(path: string) {
  if (path.startsWith('/Users/oleh/Downloads/YoutubeToMp3/')) return decodeCached(path);
  const buf = execFileSync(ffmpegPath(), ['-v', 'error', '-i', path, '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], { maxBuffer: 1 << 30 });
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const n = all.length >> 1;
  const left = new Float32Array(n), right = new Float32Array(n);
  for (let i = 0; i < n; i++) { left[i] = all[2 * i]; right[i] = all[2 * i + 1]; }
  return { sr: SR, left, right };
}

const REPO = join(import.meta.dirname, '../../../..');
const WORK = join(REPO, '.testdata/instr/drums');
const MANIFEST = JSON.parse(readFileSync(join(WORK, 'manifest.json'), 'utf8')) as {
  corpus: string; tid: string; path: string; eval: boolean; split: string; label_path: string;
}[];

const BLOCK = 512;
const CLASSES = ['kick', 'snare', 'hat', 'tom', 'cymbal'] as const;

function analyzerOnsets(path: string): { onset: Float32Array; frameRate: number } {
  const pcm = decode(path);
  const analyzer = new RealtimeAnalyzer(pcm.sr);
  const T = Math.floor(pcm.left.length / BLOCK);
  const onset = new Float32Array(T);
  for (let k = 0; k < T; k++) {
    const l = pcm.left.subarray(k * BLOCK, (k + 1) * BLOCK);
    const r = pcm.right.subarray(k * BLOCK, (k + 1) * BLOCK);
    analyzer.process(l, r);
    onset[k] = analyzer.stemOnsets.drums;
  }
  return { onset, frameRate: pcm.sr / BLOCK };
}

function peakPick(act: Float32Array, threshold: number, refractory: number, lookahead = 1): number[] {
  const T = act.length;
  const events: number[] = [];
  let last = -refractory - 1;
  for (let k = 0; k < T; k++) {
    if (act[k] < threshold) continue;
    const end = Math.min(T, k + lookahead + 1);
    let isMax = true;
    for (let j = k; j < end; j++) if (act[j] > act[k]) { isMax = false; break; }
    if (!isMax) continue;
    if (k - last <= refractory) continue;
    events.push(k);
    last = k;
  }
  return events;
}

function scorePRF(pred: number[], ref: number[], tolLo = -0.01, tolHi = 0.05): { p: number; r: number; f1: number; delays: number[] } {
  const used = new Array(ref.length).fill(false);
  let tp = 0;
  const delays: number[] = [];
  // ref and pred are both sorted ascending; a pred at t can only match ref in [t-tolHi, t-tolLo].
  // Binary-search the window bounds instead of scanning every ref for every pred (O(n log n)).
  const lowerBound = (arr: number[], x: number) => {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
    return lo;
  };
  for (const t of pred) {
    const lo_t = t - tolHi, hi_t = t - tolLo;
    let i0 = lowerBound(ref, lo_t);
    let best = -1, bestD = Infinity;
    for (let i = i0; i < ref.length && ref[i] <= hi_t; i++) {
      if (used[i]) continue;
      const d = Math.abs(t - ref[i]);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) { used[best] = true; tp++; delays.push(t - ref[best]); }
  }
  const fp = pred.length - tp, fn = ref.length - tp;
  const p = tp + fp > 0 ? tp / (tp + fp) : ref.length === 0 ? 1 : 0;
  const r = tp + fn > 0 ? tp / (tp + fn) : pred.length === 0 ? 1 : 0;
  const f1 = p + r > 0 ? (2 * p * r) / (p + r) : 0;
  return { p, r, f1, delays };
}

// Dumps the per-frame stemOnsets.drums stream (frame k = after block k, time (k+1)/fps) for every
// val + held-out track to .testdata/instr/drums/dsp_onsets/<corpus>__<tid>.f32. Scoring is done by
// eval.py --dsp with the same (fixed) peak picker and threshold grid as the student, tuned per class
// on the validation split. The peakPick()/scorePRF() above are the old in-TS scorer, kept for reference.
async function main() {
  const out = join(WORK, 'dsp_onsets');
  mkdirSync(out, { recursive: true });
  const rows = MANIFEST.filter((r) => r.split === 'eval' || r.split === 'val');
  let n = 0;
  for (const r of rows) {
    const p = join(out, `${r.corpus}__${r.tid}.f32`);
    if (existsSync(p)) continue;
    try {
      const t0 = performance.now();
      const { onset } = analyzerOnsets(r.path);
      console.error(r.corpus, r.tid, `${((performance.now() - t0) / 1000).toFixed(1)}s`);
      writeFileSync(p, Buffer.from(onset.buffer, onset.byteOffset, onset.byteLength));
      n++;
    } catch (e) {
      console.error('FAILED', r.corpus, r.tid, String(e));
    }
  }
  console.error(`dumped ${n} DSP onset streams (${rows.length} val+eval rows) -> ${out}`);
  void scorePRF; void peakPick; void CLASSES;
}

main();

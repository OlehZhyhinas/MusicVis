// Pure analysis pipeline: PCM in, AnalysisResult out. No DOM / worker APIs,
// so it runs in the worker and in Node tests alike.

import type { AnalysisResult, KeySegment, Section } from '../types';
import { decimate } from './dsp';
import { stftPower, stemBands, bandPower } from './stft';
import { computeStems, TIMBRE_BANDS } from './stems';
import { onsetEnvelope, estimateTempo, trackBeats, refineBeats, downbeatPhase } from './beats';
import { computeChroma, resampleChroma, detectKeys } from './key';
import { detectSections } from './structure';
import { detectRepeats } from './repetition';
import { computeComplexity, type ComplexityFeatures } from './complexity';
import { analyzeGroove } from './groove';
import { TimbreRecorder } from './timbre';
import { NoteRecorder } from './notes';
import { findHooks } from './hooks';

export type ProgressFn = (stage: string, progress: number) => void;

const BEATS_PER_BAR = 4;

function pow2Near(x: number): number {
  return Math.pow(2, Math.max(5, Math.round(Math.log2(Math.max(32, x)))));
}

/** Optional diagnostics sink (tests / calibration scripts). */
export interface AnalysisDebug {
  complexity?: ComplexityFeatures;
}

export function analyzePcm(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number,
  onProgress?: ProgressFn,
  debug?: AnalysisDebug,
): AnalysisResult {
  const report = (stage: string, p: number) => onProgress?.(stage, Math.max(0, Math.min(1, p)));
  const duration = sampleRate > 0 ? left.length / sampleRate : 0;
  report('Preparing audio', 0);

  // --- Downmix + decimate to ~22 kHz ---
  const factor = Math.max(1, Math.round(sampleRate / 22050));
  const sr = sampleRate / factor;
  const r = right && right.length === left.length ? right : left;
  const stereo = r !== left;
  // Decimate each channel first, then form mid / side at the low rate.
  const l2 = decimate(sanitize(left), factor);
  const r2 = stereo ? decimate(sanitize(r), factor) : l2;
  const mid = new Float32Array(l2.length);
  const side = new Float32Array(l2.length);
  for (let i = 0; i < l2.length; i++) {
    mid[i] = 0.5 * (l2[i] + r2[i]);
    side[i] = 0.5 * (l2[i] - r2[i]);
  }
  report('Preparing audio', 0.05);

  const hop = pow2Near(sr / 86);
  const n = hop * 8;
  const frameRate = sr / hop;
  const T = Math.max(1, Math.floor((mid.length - 1) / hop) + 1);

  // --- Main STFT -> banded mid / side power ---
  const bands = stemBands(n, sr);
  const B = bands.count;
  const midB = new Float32Array(T * B);
  const sideB = new Float32Array(T * B);
  const every = Math.max(1, Math.floor(T / 20));
  const timbre = new TimbreRecorder(n, sr, frameRate, T);
  const noteRec = new NoteRecorder(n, sr, frameRate, T);
  stftPower(mid, n, hop, T, (f, pow) => {
    bandPower(pow, bands, midB, f * B);
    timbre.frame(f, pow);
    noteRec.frame(f, pow);
    if (f % every === 0) report('Spectrum', 0.05 + 0.1 * (f / T));
  });
  if (stereo) stftPower(side, n, hop, T, (f, pow) => bandPower(pow, bands, sideB, f * B));
  report('Spectrum', 0.2);

  // --- Stems ---
  const st = computeStems(midB, sideB, bands, T, frameRate, (p) => report('Separating stems', 0.2 + 0.35 * p));
  report('Separating stems', 0.55);

  // --- Chroma ---
  const chromaRaw = computeChroma(mid, sr, n * 2, hop * 4);
  const chroma = resampleChroma(chromaRaw, T, frameRate);
  report('Harmony', 0.62);

  // --- Absolute presence / complexity ---
  const cx = computeComplexity({ raw: st.raw, T, frameRate, fftSize: n, chroma });
  if (debug) debug.complexity = cx;
  report('Harmony', 0.65);

  // --- Beats ---
  const env = onsetEnvelope(mid, sr, n / 2, hop, T);
  report('Beats', 0.7);
  const tempo = estimateTempo(env, frameRate);
  let beatFrames = trackBeats(env, tempo.period);
  beatFrames = refineBeats(beatFrames, env, tempo.period);
  let beatTimes = Array.from(beatFrames, (f) => f / frameRate);
  // Trim beats in leading / trailing silence.
  const loudAt = (t: number) => st.loudness[Math.max(0, Math.min(T - 1, Math.round(t * frameRate)))];
  const actLvl = 0.03;
  while (beatTimes.length > 0 && loudAt(beatTimes[0]) < actLvl && loudAt(beatTimes[0] + 0.2) < actLvl) beatTimes.shift();
  while (beatTimes.length > 0 && loudAt(beatTimes[beatTimes.length - 1]) < actLvl) beatTimes.pop();
  let bpm = tempo.bpm;
  if (beatTimes.length >= 8) {
    // Least-squares fit of beat time vs index.
    const m = beatTimes.length;
    let sx = 0,
      sy = 0;
    for (let i = 0; i < m; i++) {
      sx += i;
      sy += beatTimes[i];
    }
    sx /= m;
    sy /= m;
    let num = 0,
      den = 0;
    for (let i = 0; i < m; i++) {
      num += (i - sx) * (beatTimes[i] - sy);
      den += (i - sx) ** 2;
    }
    const p = num / den;
    if (p > 0.2 && p < 1.5) bpm = 60 / p;
  } else {
    // Synthetic grid so the visuals always have a pulse.
    const p = 60 / bpm;
    beatTimes = [];
    for (let t = 0; t < duration; t += p) beatTimes.push(t);
    if (beatTimes.length === 0) beatTimes.push(0);
  }
  const beats = Float32Array.from(beatTimes);
  report('Beats', 0.8);

  // --- Downbeats ---
  const beatPosFrames = Float64Array.from(beatTimes, (t) => t * frameRate);
  const phase = downbeatPhase(beatPosFrames, st.kickOnset, st.stemOnsets.bass, st.snareOnset, chroma, BEATS_PER_BAR);
  const dbList: number[] = [];
  for (let i = phase; i < beatTimes.length; i += BEATS_PER_BAR) dbList.push(beatTimes[i]);
  if (dbList.length === 0 && beatTimes.length > 0) dbList.push(beatTimes[0]);
  const downbeats = Float32Array.from(dbList);
  report('Key', 0.85);

  // --- Key ---
  let keys: KeySegment[] = detectKeys(chromaRaw, dbList.length >= 2 ? dbList : fixedGrid(duration, 2), duration);
  if (keys.length === 0 || !(duration > 0)) keys = [{ start: 0, end: Math.max(duration, 1e-3), tonic: 0, mode: 'major', confidence: 0 }];
  report('Structure', 0.9);

  // --- Structure ---
  let sections: Section[] = detectSections({
    duration,
    frameRate,
    numFrames: T,
    beats: Float64Array.from(beatTimes),
    downbeats: Float64Array.from(dbList),
    chroma,
    timbre: st.timbre,
    timbreBands: TIMBRE_BANDS,
    loudness: st.loudness,
    drums: st.stems.drums,
    bass: st.stems.bass,
    drumOnsets: st.stemOnsets.drums,
    complexity: cx.complexity,
    drumsPresence: cx.stemPresence.drums,
    bassPresence: cx.stemPresence.bass,
    songComplexity: cx.songComplexity,
  });
  if (sections.length === 0) sections = [{ start: 0, end: Math.max(duration, 1e-3), label: 'verse', energy: 0 }];
  const groove = analyzeGroove(env, frameRate, beats, downbeats, sections, st.raw.total);
  report('Done', 1);

  const result: AnalysisResult = {
    duration,
    frameRate,
    numFrames: T,
    stems: st.stems,
    stemOnsets: st.stemOnsets,
    stemPresence: cx.stemPresence,
    complexity: cx.complexity,
    songComplexity: cx.songComplexity,
    loudness: st.loudness,
    chroma,
    bpm,
    beats,
    downbeats,
    beatsPerBar: BEATS_PER_BAR,
    sections,
    keys,
    groove,
    timbre: timbre.track,
    notes: noteRec.build(),
  };
  result.repeats = detectRepeats(result);
  result.hooks = findHooks(result);
  return result;
}

/** Replace non-finite samples with 0 (copies only when needed). */
function sanitize(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i])) {
      const y = Float32Array.from(x, (v) => (Number.isFinite(v) ? v : 0));
      return y;
    }
  }
  return x;
}

function fixedGrid(duration: number, step: number): number[] {
  const out: number[] = [];
  for (let t = step; t < duration; t += step) out.push(t);
  return out;
}

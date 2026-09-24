// Tests for section repetition detection (src/analysis/repetition.ts) on feature-level synthetic
// songs (no audio), and the MusicState fields the TimelineSampler derives from it.
// Called from analysis-test.ts.

import { detectRepeats } from '../src/analysis/repetition';
import { TimelineSampler } from '../src/analysis/TimelineSampler';
import type { AnalysisResult, LiveAudioFrame, Section, SectionLabel } from '../src/types';

type Check = (name: string, ok: boolean, detail: string) => void;

/** A section recipe: its chord cycle (pitch classes per bar), loudness, drums and bass level. */
export interface Recipe {
  label: SectionLabel;
  bars: number;
  chords: number[][];
  loud: number;
  drums: number;
  bass: number;
  shift?: number; // transposition in semitones
}

const FR = 20;
const BPM = 120;
export const FEATURE_BAR = (60 / BPM) * 4;
const BAR = FEATURE_BAR;

/** A feature-level AnalysisResult built from section recipes (deterministic small noise). */
export function featureSong(parts: Recipe[]): AnalysisResult {
  const duration = parts.reduce((s, p) => s + p.bars * BAR, 0);
  const T = Math.ceil(duration * FR);
  const chroma = new Float32Array(T * 12);
  const loud = new Float32Array(T);
  const drums = new Float32Array(T);
  const bass = new Float32Array(T);
  const zero = () => new Float32Array(T);
  const sections: Section[] = [];
  let seed = 12345;
  const rnd = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296;
  let t0 = 0;
  for (const p of parts) {
    sections.push({ start: t0, end: t0 + p.bars * BAR, label: p.label, energy: p.loud });
    for (let f = Math.floor(t0 * FR); f < Math.min(T, Math.floor((t0 + p.bars * BAR) * FR)); f++) {
      const bar = Math.floor((f / FR - t0) / BAR);
      const ch = p.chords[bar % p.chords.length];
      for (const pc of ch) chroma[f * 12 + ((pc + (p.shift ?? 0)) % 12)] = 1;
      for (let k = 0; k < 12; k++) chroma[f * 12 + k] = Math.max(chroma[f * 12 + k], 0.1 * rnd());
      const beatPh = (((f / FR) * BPM) / 60) % 1;
      loud[f] = p.loud * (0.9 + 0.1 * rnd());
      drums[f] = p.drums * (beatPh < 0.2 ? 1 : 0.3);
      bass[f] = p.bass * (0.8 + 0.2 * rnd());
    }
    t0 += p.bars * BAR;
  }
  const beats: number[] = [];
  for (let t = 0; t < duration; t += 60 / BPM) beats.push(t);
  const downbeats = beats.filter((_, i) => i % 4 === 0);
  return {
    duration, frameRate: FR, numFrames: T,
    stems: { drums, bass, vocals: zero(), other: zero() },
    stemOnsets: { drums: zero(), bass: zero(), vocals: zero(), other: zero() },
    stemPresence: { drums, bass, vocals: zero(), other: zero() },
    complexity: loud, songComplexity: 0.5, loudness: loud, chroma, bpm: BPM,
    beats: Float32Array.from(beats), downbeats: Float32Array.from(downbeats), beatsPerBar: 4,
    sections, keys: [{ start: 0, end: duration, tonic: 9, mode: 'minor', confidence: 1 }],
  };
}

const VERSE = [[9, 0, 4], [5, 9, 0], [7, 11, 2], [9, 0, 4]];
const CHORUS = [[0, 4, 7], [7, 11, 2], [9, 0, 4], [5, 9, 0]];
const BRIDGE = [[2, 5, 9], [10, 2, 5], [4, 8, 11], [4, 8, 11]];

export const RV = (bars = 16): Recipe => ({ label: 'verse', bars, chords: VERSE, loud: 0.55, drums: 0.5, bass: 0.5 });
export const RC = (bars = 16, shift = 0): Recipe => ({ label: 'chorus', bars, chords: CHORUS, loud: 0.9, drums: 1, bass: 0.9, shift });
export const RB = (): Recipe => ({ label: 'breakdown', bars: 8, chords: BRIDGE, loud: 0.3, drums: 0, bass: 0.2 });
export const RI = (): Recipe => ({ label: 'intro', bars: 8, chords: [[9, 0, 4]], loud: 0.15, drums: 0, bass: 0 });

export function repetitionTests(check: Check): void {
  // Pop form with a bridge and a last chorus a step up: I V C V C B C'.
  const song = featureSong([RI(), RV(), RC(), RV(), RC(), RB(), RC(16, 2)]);
  const r1 = detectRepeats(song);
  const txt = r1.map((x, i) => `${i}:g${x.group}${x.of >= 0 ? `<-${x.of}` : ''}(${x.sim.toFixed(2)})`).join(' ');
  check('repeat.verse-returns', r1[3].of === 1 && r1[3].n === 1 && r1[3].sim >= 0.6, txt);
  check('repeat.chorus-returns', r1[4].of === 2 && r1[4].n === 1 && r1[4].group === r1[2].group, txt);
  check('repeat.transposed-chorus', r1[6].of === 2 && r1[6].n === 2, txt);
  check('repeat.novel-sections', r1[0].of === -1 && r1[1].of === -1 && r1[2].of === -1 && r1[5].of === -1 && r1[2].group !== r1[1].group, txt);
  check('repeat.returns-flag', r1[1].returnSim >= 0.6 && r1[2].returnSim >= 0.6 && r1[5].returnSim === 0 && r1[6].returnSim === 0, r1.map((x) => x.returnSim.toFixed(2)).join(','));
  check('repeat.groups-ordered', r1.every((x, i) => x.group <= Math.max(-1, ...r1.slice(0, i).map((y) => y.group)) + 1), txt);
  check('repeat.deterministic', JSON.stringify(detectRepeats(song)) === JSON.stringify(r1), 'same input, same groups');

  // Boundaries off by a bar: the second verse starts one bar late and the chorus runs a bar short.
  const shifted = featureSong([RI(), RV(), RC(), { ...RV(), bars: 17 }, RC(15)]);
  const r2 = detectRepeats(shifted);
  check('repeat.misaligned', r2[3].of === 1 && r2[4].of === 2, r2.map((x) => `${x.group}<-${x.of}`).join(' '));

  // Nothing repeats: every section its own group.
  const through = featureSong([RI(), RV(), RC(), RB()]);
  const r3 = detectRepeats(through);
  check('repeat.through-composed', r3.every((x) => x.of === -1 && x.returnSim === 0), r3.map((x) => x.group).join(','));

  // Degenerate inputs never throw.
  let ok = true;
  try {
    detectRepeats({ ...song, sections: [] });
    detectRepeats({ ...song, sections: [song.sections[0]] });
    detectRepeats({ ...song, numFrames: 0, chroma: new Float32Array(0) });
    detectRepeats({ ...song, beats: new Float32Array(0) });
  } catch {
    ok = false;
  }
  check('repeat.degenerate', ok, 'empty / single / no-feature / no-beat inputs');

  // Sampler: the MusicState carries the repetition of the current section, detected on the fly when
  // the result has none (results analysed before repetition existed).
  const live: LiveAudioFrame = { bass: 0, mid: 0, treb: 0, bassAtt: 0, midAtt: 0, trebAtt: 0, waveform: new Float32Array(1024), spectrum: new Float32Array(512) };
  const smp = new TimelineSampler(song);
  const seen: string[] = [];
  for (let t = 0; t < song.duration; t += 1) {
    const s = smp.sample(t, 1 / 60, true, live);
    const x = r1[s.sectionIndex];
    if (s.repeatGroup !== x.group || s.repeatOf !== x.of || s.repeatIndex !== x.n || s.repeatSim !== x.sim || s.repeatReturnSim !== x.returnSim) seen.push(`t=${t}`);
  }
  check('repeat.sampler-fields', !seen.length, seen.slice(0, 4).join(',') || 'fields match the detected repeats');
  const stored = new TimelineSampler({ ...song, repeats: r1.map((x) => ({ ...x, sim: 0.77 })) }).sample(BAR * 45, 0.016, true, live);
  check('repeat.sampler-prefers-stored', stored.repeatSim === 0.77, `sim ${stored.repeatSim}`);
}

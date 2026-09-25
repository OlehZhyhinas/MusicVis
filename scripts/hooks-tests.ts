// Tests for the hook finder (src/analysis/hooks.ts) on a feature-level synthetic song with a
// repeating one-bar riff in the 'other' stem, and the hook fields TimelineSampler derives from it.
// Called from analysis-test.ts.

import { LiveHooks, findHooks, hookTimeline, sampleHook, type HookSample } from '../src/analysis/hooks';
import { TimelineSampler } from '../src/analysis/TimelineSampler';
import type { AnalysisResult, LiveAudioFrame, SongHook } from '../src/types';
import { FEATURE_BAR, RC, RI, RV, featureSong } from './repetition-tests';

type Check = (name: string, ok: boolean, detail: string) => void;

const RIFF = [
  { at: 0, pc: 9 },
  { at: 3 / 8, pc: 0 },
  { at: 6 / 8, pc: 4 },
];

/** The song with a riff (three notes in the bar at 0, 3/8, 6/8) in every chorus bar, and scattered other-stem noise elsewhere. */
function riffSong(): { r: AnalysisResult; riffBars: number[] } {
  const r = featureSong([RI(), RV(), RC(), RV(), RC()]);
  const fr = r.frameRate;
  const T = r.numFrames;
  const other = new Float32Array(T);
  const on = new Float32Array(T);
  let seed = 99;
  const rnd = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296;
  const riffBars: number[] = [];
  for (const sec of r.sections) {
    const bars = Math.round((sec.end - sec.start) / FEATURE_BAR);
    for (let b = 0; b < bars; b++) {
      const t0 = sec.start + b * FEATURE_BAR;
      if (sec.label === 'chorus') {
        riffBars.push(t0);
        for (const n of RIFF) {
          const f0 = Math.round((t0 + n.at * FEATURE_BAR) * fr);
          for (let f = f0; f < Math.min(T, f0 + Math.round(0.3 * fr)); f++) {
            other[f] = 0.8;
            for (let k = 0; k < 12; k++) r.chroma[f * 12 + k] *= 0.3;
            r.chroma[f * 12 + n.pc] = 1;
          }
          if (f0 < T) on[f0] = 1;
        }
      } else {
        // Wandering notes: melodic, but no bar repeats another.
        for (let k = 0; k < 3; k++) {
          const f0 = Math.round((t0 + rnd() * FEATURE_BAR) * fr);
          const pc = Math.floor(rnd() * 12);
          for (let f = f0; f < Math.min(T, f0 + Math.round(0.25 * fr)); f++) {
            other[f] = 0.5;
            r.chroma[f * 12 + pc] = Math.max(r.chroma[f * 12 + pc], 0.8);
          }
          if (f0 < T) on[f0] = 0.7;
        }
      }
    }
  }
  r.stems.other = other;
  r.stemOnsets.other = on;
  r.stemPresence.other = other;
  return { r, riffBars };
}

const LIVE: LiveAudioFrame = { bass: 0, mid: 0, treb: 0, bassAtt: 0, midAtt: 0, trebAtt: 0, waveform: new Float32Array(1024), spectrum: new Float32Array(512) };

export function hooksTests(check: Check): void {
  const { r, riffBars } = riffSong();
  const hooks = findHooks(r);
  const h = hooks[0];
  const detail = hooks.map((x) => `${x.bars}bar x${x.occurrences.length} notes ${x.notes?.map((n) => n.toFixed(2)).join(',')}`).join('; ');
  const onRiff = h ? h.occurrences.filter((o) => riffBars.some((t) => Math.abs(t - o.start) < 0.05)).length : 0;
  check('hooks.finds-riff', !!h && h.occurrences.length >= 6 && onRiff >= h.occurrences.length * 0.8, detail);
  const notes = h?.notes ?? [];
  const near = (x: number) => notes.some((n) => Math.abs(n - x / (h?.bars ?? 1)) < 0.07);
  check('hooks.motif-notes', notes[0] === 0 && near(3 / 8) && near(6 / 8), notes.map((n) => n.toFixed(3)).join(','));
  check('hooks.deterministic', JSON.stringify(findHooks(r)) === JSON.stringify(hooks), 'same input, same hooks');

  // Timeline + sampling.
  const fake: SongHook[] = [
    { id: 0, bars: 1, len: 2, salience: 1, distinct: 0.2, score: 1, notes: [0, 0.5], occurrences: [{ start: 10, end: 12, sim: 1 }, { start: 20, end: 22, sim: 0.9 }] },
    { id: 1, bars: 1, len: 2, salience: 1, distinct: 0.2, score: 0.5, occurrences: [{ start: 11, end: 13, sim: 1 }, { start: 30, end: 32, sim: 0.8 }] },
  ];
  const tl = hookTimeline(fake);
  check('hooks.timeline', tl.length === 3 && tl[0].start === 10 && tl[1].start === 20 && tl[2].hook === 1 && tl[2].notes.length === 1, tl.map((x) => `${x.hook}@${x.start}`).join(' '));
  const o: HookSample = { on: 0, phase: 0, pulse: 0, notePulse: 0, note: -1, hook: -1, index: -1 };
  sampleHook(tl[0], 11.02, o);
  const mid = { ...o };
  sampleHook(tl[0], 12.5, o);
  check('hooks.sample', mid.on === 1 && Math.abs(mid.phase - 0.51) < 1e-6 && mid.note === 1 && mid.notePulse > 0.8 && mid.pulse < 0.01 && o.on === 0 && o.phase === 0 && o.hook === -1, JSON.stringify(mid));

  // Sampler: hook fields over the riff song, and seeking lands on the same values.
  const s = new TimelineSampler(r);
  const dt = 1 / 30;
  let inside = 0, outside = 0, pulses = 0, prevPulse = 0;
  const first = h?.occurrences[0];
  for (let t = 0; t < r.duration; t += dt) {
    const m = s.sample(t, dt, true, LIVE);
    const inOcc = !!h?.occurrences.some((x) => t >= x.start && t < x.end);
    if (m.hookOn === 1) inside += inOcc ? 1 : 0;
    else outside += inOcc ? 1 : 0;
    if ((m.hookPulse ?? 0) > 0.7 && (m.hookPulse ?? 0) > prevPulse + 0.3) pulses++;
    prevPulse = m.hookPulse ?? 0;
  }
  check('hooks.sampler-on', inside > 0 && outside === 0, `inside ${inside} frames, missed ${outside}`);
  check('hooks.sampler-pulses', !!h && pulses >= h.occurrences.length - 1, `${pulses} repeat pulses for ${h?.occurrences.length} repeats`);
  if (first) {
    const t = first.start + 0.4 * (first.end - first.start);
    const a = { ...s.sample(t, dt, true, LIVE) };
    s.reset();
    s.sample(t - 30, dt, true, LIVE);
    const b = s.sample(t, dt, true, LIVE);
    check('hooks.sampler-seek', a.hookPhase === b.hookPhase && a.hookNotePulse === b.hookNotePulse && Math.abs((a.hookPhase ?? 0) - 0.4) < 0.01, `${a.hookPhase?.toFixed(3)} vs ${b.hookPhase?.toFixed(3)}`);
  }
  // A result without other/vocals features (or any hooks) leaves the fields undefined and never throws.
  const plain = featureSong([RI(), RV()]);
  const s2 = new TimelineSampler({ ...plain, hooks: [] });
  const m2 = s2.sample(5, dt, true, LIVE);
  check('hooks.none', m2.hookOn === undefined, String(m2.hookOn));
}

/** Live hooks on synthetic frames: a riff that repeats is flagged from its third bar, wandering bars are not. */
export function liveHooksTests(check: Check): void {
  const lh = new LiveHooks();
  const bar = 2;
  const fps = 60;
  let seed = 7;
  const rnd = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296;
  const onBars: number[] = [];
  const noteHits: number[] = [];
  const chroma = new Float32Array(12);
  for (let f = 0; f < 16 * bar * fps; f++) {
    const t = f / fps;
    const bi = Math.floor(t / bar);
    const ph = t / bar - bi;
    const riff = bi >= 4 && bi < 12;
    chroma.fill(0.05);
    let onset = 0;
    let pres = 0.1;
    if (riff) {
      for (const n of RIFF) if (ph >= n.at && ph < n.at + 0.1) (chroma[n.pc] = 1), (pres = 0.8), (onset = ph - n.at < 1 / (fps * bar) ? 1 : 0);
    } else {
      const pc = Math.floor(rnd() * 12);
      chroma[pc] = 1;
      pres = 0.5;
      onset = rnd() < 0.05 ? 0.8 : 0;
    }
    const o = lh.update(t, bi, ph, bar, chroma, pres, onset);
    if (o.on && !onBars.includes(bi)) onBars.push(bi);
    if (o.on && o.notePulse > 0.95) noteHits.push(+(ph).toFixed(2));
  }
  check('hooks.live-riff', onBars.length >= 4 && onBars.every((b) => b >= 6 && b <= 12), `on in bars ${onBars.join(',')}`);
  check('hooks.live-notes', [0, 0.38, 0.75].every((x) => noteHits.some((h) => Math.abs(h - x) < 0.03)), [...new Set(noteHits)].join(','));
}

// Music-side helpers for the AV quality harness, pure TS on a serialisable subset of the
// analysis (SongData): drops and section boundaries, repeated short motifs ("hooks"), and
// clip windows around them.
//
// The hook finder lives in the analysis (src/analysis/hooks.ts), so the renderer's hook signals
// and the harness's hook rhyme are measured against the same hooks; it is re-exported here.

export interface SectionLite {
  start: number;
  end: number;
  label: string;
  energy: number;
}

export interface SongData {
  duration: number;
  frameRate: number;
  numFrames: number;
  bpm: number;
  beats: number[];
  downbeats: number[];
  beatsPerBar: number;
  sections: SectionLite[];
  repeats?: { group: number; of: number; n: number; sim: number }[];
  /** numFrames*12 */
  chroma: Float32Array;
  loudness: Float32Array;
  stems: Record<'drums' | 'bass' | 'vocals' | 'other', Float32Array>;
  stemOnsets: Record<'drums' | 'bass' | 'vocals' | 'other', Float32Array>;
  stemPresence: Record<'drums' | 'bass' | 'vocals' | 'other', Float32Array>;
}

export type HookOccurrence = SongHookOccurrence;
export type Hook = SongHook;

export interface Moment {
  t: number;
  kind: 'drop' | 'lift' | 'section';
  label: string;
}

export interface ClipWindow {
  label: string;
  start: number;
  end: number;
}

import type { AnalysisResult, SongHook, SongHookOccurrence } from '../../types';
import { findHooks as findHooksImpl } from '../../analysis/hooks';

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));

/** The SongData subset of an analysis result. */
export function songDataOf(r: AnalysisResult): SongData {
  return {
    duration: r.duration, frameRate: r.frameRate, numFrames: r.numFrames, bpm: r.bpm,
    beats: Array.from(r.beats), downbeats: Array.from(r.downbeats), beatsPerBar: r.beatsPerBar,
    sections: r.sections.map((s) => ({ ...s })), repeats: r.repeats?.map((x) => ({ group: x.group, of: x.of, n: x.n, sim: x.sim })),
    chroma: r.chroma, loudness: r.loudness, stems: r.stems, stemOnsets: r.stemOnsets, stemPresence: r.stemPresence,
  };
}

/**
 * Short review windows for side-by-side comparisons: the densest stretch of the top hook,
 * the main drop (or lift), and up to two other section changes, each `len` seconds with the
 * event `lead` seconds in.
 */
export function reviewWindows(s: SongData, hooks: Hook[], len = 12, lead = 5): ClipWindow[] {
  const out: ClipWindow[] = [];
  const fit = (w: ClipWindow): ClipWindow => {
    const start = clamp(w.start, 0, Math.max(0, s.duration - len));
    return { label: w.label, start, end: Math.min(s.duration, start + len) };
  };
  if (hooks.length) {
    const w = densestHookWindow(hooks[0], len);
    if (w) out.push(fit({ label: 'hook', start: w.start, end: w.end }));
  }
  const d = mainDrop(s);
  if (d !== null) out.push(fit({ label: 'drop', start: d - lead, end: d - lead + len }));
  const others = songMoments(s).filter((m) => m.kind === 'section' && (d === null || Math.abs(m.t - d) > len));
  for (const m of others.filter((_, i) => i % Math.max(1, Math.floor(others.length / 2)) === 0).slice(0, 2)) out.push(fit({ label: `section ${m.label}`, start: m.t - lead, end: m.t - lead + len }));
  return out;
}

function meanRange(x: Float32Array, fr: number, a: number, b: number): number {
  const i0 = clamp(Math.floor(a * fr), 0, x.length - 1);
  const i1 = clamp(Math.ceil(b * fr), i0 + 1, x.length);
  let s = 0;
  for (let i = i0; i < i1; i++) s += x[i];
  return s / (i1 - i0);
}

/** Drops (sections labelled drop, or a chorus straight after a build, like TimelineSampler) or, when there are none, the biggest loudness lift at a section boundary; plus every section boundary. */
export function songMoments(s: SongData): Moment[] {
  const out: Moment[] = [];
  const secs = s.sections;
  for (let i = 1; i < secs.length; i++) {
    const sec = secs[i];
    const isDrop = sec.label === 'drop' || (sec.label === 'chorus' && secs[i - 1].label === 'build');
    out.push({ t: sec.start, kind: isDrop ? 'drop' : 'section', label: `${secs[i - 1].label}>${sec.label}` });
  }
  if (!out.some((m) => m.kind === 'drop')) {
    let best: Moment | null = null;
    let bestLift = 0.05;
    for (const m of out) {
      const lift = meanRange(s.loudness, s.frameRate, m.t, m.t + 4) - meanRange(s.loudness, s.frameRate, m.t - 4, m.t);
      if (lift > bestLift) {
        bestLift = lift;
        best = m;
      }
    }
    if (best) best.kind = 'lift';
  }
  return out;
}

/** The main drop (or lift) time, or null. */
export function mainDrop(s: SongData): number | null {
  const ms = songMoments(s).filter((m) => m.kind !== 'section');
  if (!ms.length) return null;
  // The drop with the biggest loudness lift.
  let best = ms[0];
  let bl = -Infinity;
  for (const m of ms) {
    const lift = meanRange(s.loudness, s.frameRate, m.t, m.t + 4) - meanRange(s.loudness, s.frameRate, m.t - 4, m.t);
    if (lift > bl) {
      bl = lift;
      best = m;
    }
  }
  return best.t;
}

/** Repeated short motifs, best first (src/analysis/hooks.ts). */
export function findHooks(s: SongData, maxHooks = 2): Hook[] {
  return findHooksImpl(s, maxHooks);
}

/** The `len`-second window holding the most occurrences of a hook (start snapped to the first occurrence inside). */
export function densestHookWindow(h: Hook, len: number, avoid?: ClipWindow): ClipWindow | null {
  if (!h.occurrences.length) return null;
  let best: ClipWindow | null = null;
  let bestN = -1;
  for (const o of h.occurrences) {
    const start = o.start - 1;
    const end = start + len;
    let n = h.occurrences.filter((p) => p.start >= start && p.end <= end).length;
    if (avoid) {
      const ov = Math.max(0, Math.min(end, avoid.end) - Math.max(start, avoid.start));
      if (ov > len * 0.5) n -= 100;
    }
    if (n > bestN) {
      bestN = n;
      best = { label: `hook${h.id}`, start, end };
    }
  }
  return best;
}

/** Default review windows: the main drop (+-10 s) and the densest 20 s of the top hook. */
export function autoWindows(s: SongData, hooks: Hook[], dropHalf = 10, hookLen = 20): ClipWindow[] {
  const out: ClipWindow[] = [];
  const d = mainDrop(s);
  if (d !== null) out.push({ label: 'drop', start: d - dropHalf, end: d + dropHalf });
  if (hooks.length) {
    const w = densestHookWindow(hooks[0], hookLen, out[0]);
    if (w) out.push({ ...w, label: 'hook' });
  }
  if (!out.length) out.push({ label: 'mid', start: s.duration / 2 - 10, end: s.duration / 2 + 10 });
  for (const w of out) {
    const len = w.end - w.start;
    w.start = clamp(w.start, 0, Math.max(0, s.duration - len));
    w.end = Math.min(s.duration, w.start + len);
  }
  return out;
}

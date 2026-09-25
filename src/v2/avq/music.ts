// Music-side helpers for the AV quality harness, pure TS on a serialisable subset of the
// analysis (SongData): drops and section boundaries, repeated short motifs ("hooks"), and
// clip windows around them.
//
// Hook finder: every downbeat starts a candidate unit of 1 or 2 bars. Each unit is cut into
// eighth-note slots; a slot's descriptor is the analysis chroma weighted by how present the
// melodic stems (other + vocals) are, plus their onset strength. Two units are similar when
// both the raw slot chroma and the unit-centred chroma (the contour, with the unit's chord
// removed) line up, and their melodic onset patterns agree. A hook is a prototype unit with
// many non-overlapping near copies that stand out against the rest of the song
// (distinctiveness) and carry melodic energy (salience).

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

export interface HookOccurrence {
  start: number;
  end: number;
  /** Similarity to the prototype, 0..1 (1 for the prototype itself). */
  sim: number;
}

export interface Hook {
  id: number;
  bars: number;
  /** Seconds per occurrence (mean). */
  len: number;
  occurrences: HookOccurrence[];
  /** Mean melodic presence over the occurrences, 0..1. */
  salience: number;
  /** Mean similarity within the group minus the mean similarity of the prototype to all units. */
  distinct: number;
  score: number;
}

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

import type { AnalysisResult } from '../../types';

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

interface Unit {
  start: number;
  end: number;
  raw: Float32Array; // slots*12, L2-normalised overall
  ctr: Float32Array; // centred per chroma bin across slots, L2-normalised
  ons: Float32Array; // slots
  mel: number; // mean melodic presence
}

function cos(a: Float32Array, b: Float32Array): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na > 1e-12 && nb > 1e-12 ? d / Math.sqrt(na * nb) : 0;
}

function buildUnits(s: SongData, bars: number): Unit[] {
  const db = s.downbeats;
  const fr = s.frameRate;
  const bpb = Math.max(2, s.beatsPerBar || 4);
  const slots = bpb * 2 * bars;
  const units: Unit[] = [];
  for (let i = 0; i + bars < db.length; i++) {
    const start = db[i];
    const end = db[i + bars];
    if (end - start <= 0.2) continue;
    const raw = new Float32Array(slots * 12);
    const ons = new Float32Array(slots);
    let mel = 0;
    for (let k = 0; k < slots; k++) {
      const a = start + ((end - start) * k) / slots;
      const b = start + ((end - start) * (k + 1)) / slots;
      const f0 = clamp(Math.floor(a * fr), 0, s.numFrames - 1);
      const f1 = clamp(Math.ceil(b * fr), f0 + 1, s.numFrames);
      let m = 0;
      for (let f = f0; f < f1; f++) {
        const w = s.stemPresence.other[f] + s.stemPresence.vocals[f];
        m += w;
        ons[k] += s.stemOnsets.other[f] + s.stemOnsets.vocals[f];
        for (let c = 0; c < 12; c++) {
          const v = s.chroma[f * 12 + c];
          raw[k * 12 + c] += w * v * v; // squared: emphasise the loudest notes
        }
      }
      ons[k] /= f1 - f0;
      mel += m / (f1 - f0);
    }
    const ctr = new Float32Array(slots * 12);
    for (let c = 0; c < 12; c++) {
      let mu = 0;
      for (let k = 0; k < slots; k++) mu += raw[k * 12 + c];
      mu /= slots;
      for (let k = 0; k < slots; k++) ctr[k * 12 + c] = raw[k * 12 + c] - mu;
    }
    units.push({ start, end, raw, ctr, ons, mel: mel / slots / 2 });
  }
  return units;
}

function unitSim(a: Unit, b: Unit): number {
  const r = cos(a.raw, b.raw);
  const c = Math.max(0, cos(a.ctr, b.ctr));
  const o = Math.max(0, cos(a.ons, b.ons));
  return 0.35 * r + 0.45 * c + 0.2 * o;
}

function percentileOf(v: number[], p: number): number {
  if (!v.length) return 0;
  const s = [...v].sort((a, b) => a - b);
  return s[clamp(Math.round(p * (s.length - 1)), 0, s.length - 1)];
}

function findHooksFor(s: SongData, bars: number, maxHooks: number): Hook[] {
  const units = buildUnits(s, bars);
  const U = units.length;
  if (U < 6) return [];
  const M = new Float32Array(U * U);
  const off: number[] = [];
  for (let i = 0; i < U; i++) {
    for (let j = i + 1; j < U; j++) {
      const v = unitSim(units[i], units[j]);
      M[i * U + j] = M[j * U + i] = v;
      if (j - i >= bars) off.push(v);
    }
  }
  const tau = Math.max(0.66, percentileOf(off, 0.8));
  const melMax = Math.max(1e-6, ...units.map((u) => u.mel));
  const used = new Uint8Array(U);
  const hooks: Hook[] = [];
  for (let h = 0; h < maxHooks; h++) {
    let best: Hook | null = null;
    let bestProto = -1;
    for (let i = 0; i < U; i++) {
      if (used[i] || units[i].mel < 0.15 * melMax) continue;
      const cand = [] as number[];
      let meanAll = 0;
      for (let j = 0; j < U; j++) {
        if (j === i) continue;
        meanAll += M[i * U + j];
        if (!used[j] && M[i * U + j] >= tau) cand.push(j);
      }
      meanAll /= U - 1;
      cand.sort((a, b) => M[i * U + b] - M[i * U + a]);
      const occ = [i];
      for (const j of cand) if (occ.every((k) => Math.abs(k - j) >= bars)) occ.push(j);
      if (occ.length < 3) continue;
      let within = 0;
      let mel = 0;
      for (const k of occ) {
        mel += units[k].mel;
        if (k !== i) within += M[i * U + k];
      }
      within /= occ.length - 1;
      mel /= occ.length;
      const distinct = within - meanAll;
      if (distinct < 0.04) continue;
      // Density: repeats that come back to back (a riff) are what a 20 s review window can show.
      let dense = 0;
      for (const k of occ) dense = Math.max(dense, occ.filter((j) => units[j].start >= units[k].start && units[j].end <= units[k].start + 20).length);
      const score = Math.log2(occ.length) * distinct * (mel / melMax) * Math.sqrt(bars) * (0.4 + 0.6 * Math.min(1, dense / 4));
      if (!best || score > best.score) {
        occ.sort((a, b) => a - b);
        best = {
          id: h, bars, len: 0, salience: mel, distinct, score,
          occurrences: occ.map((k) => ({ start: units[k].start, end: units[k].end, sim: k === i ? 1 : M[i * U + k] })),
        };
        bestProto = i;
      }
    }
    if (!best) break;
    best.len = best.occurrences.reduce((a, o) => a + o.end - o.start, 0) / best.occurrences.length;
    hooks.push(best);
    // Units overlapping the chosen occurrences are taken.
    for (let j = 0; j < U; j++) {
      for (const o of best.occurrences) if (units[j].start < o.end - 0.05 && units[j].end > o.start + 0.05) used[j] = 1;
    }
    used[bestProto] = 1;
  }
  return hooks;
}

/** Repeated short motifs, best first (1- and 2-bar candidates compete on score). */
export function findHooks(s: SongData, maxHooks = 2): Hook[] {
  const all = [...findHooksFor(s, 1, maxHooks), ...findHooksFor(s, 2, maxHooks)].sort((a, b) => b.score - a.score);
  // Keep hooks whose occurrences mostly do not coincide with an already kept hook.
  const kept: Hook[] = [];
  for (const h of all) {
    const overl = (o: HookOccurrence) => kept.some((k) => k.occurrences.some((p) => p.start < o.end && p.end > o.start));
    const share = h.occurrences.filter(overl).length / h.occurrences.length;
    if (share < 0.5) kept.push(h);
    if (kept.length >= maxHooks) break;
  }
  kept.forEach((h, i) => (h.id = i));
  return kept;
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

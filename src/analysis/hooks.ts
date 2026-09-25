// Hooks: repeated short melodic motifs (a riff, a sung hook), found once per song offline.
//
// Every downbeat starts a candidate unit of 1 or 2 bars. Each unit is cut into eighth-note slots;
// a slot's descriptor is the analysis chroma weighted by how present the melodic stems (other +
// vocals) are, plus their onset strength. Two units are similar when both the raw slot chroma and
// the unit-centred chroma (the contour, with the unit's chord removed) line up, and their melodic
// onset patterns agree. A hook is a prototype unit with many non-overlapping near copies that stand
// out against the rest of the song (distinctiveness) and carry melodic energy (salience).
//
// The AV quality harness measures "hook rhyme" against these same hooks (src/v2/avq/music.ts
// re-exports this module), and TimelineSampler turns them into the hook signals (hook, hookphase,
// hookon) the renderer and reactions use. Pure TS, no DOM; a 4-minute song takes a few ms.

import type { AnalysisResult, SongHook, SongHookOccurrence } from '../types';

/** The part of an analysis result the hook finder reads. */
export interface HookInput {
  duration: number;
  frameRate: number;
  numFrames: number;
  downbeats: ArrayLike<number>;
  beatsPerBar: number;
  /** numFrames*12 */
  chroma: Float32Array;
  stemOnsets: Record<'vocals' | 'other', Float32Array>;
  stemPresence: Record<'vocals' | 'other', Float32Array>;
}

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));

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

function buildUnits(s: HookInput, bars: number): Unit[] {
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

/**
 * The motif's note starts as fractions of an occurrence: peaks of the melodic onset pattern
 * averaged over the occurrences (so one sloppy repeat does not move them). Always includes 0.
 */
function motifNotes(units: Unit[], occ: number[]): number[] {
  const slots = units[occ[0]].ons.length;
  const mean = new Float32Array(slots);
  for (const k of occ) for (let s = 0; s < slots; s++) mean[s] += units[k].ons[s] / occ.length;
  let mx = 0;
  for (const v of mean) mx = Math.max(mx, v);
  const out = [0];
  if (mx <= 1e-6) return out;
  for (let s = 1; s < slots; s++) {
    const prev = mean[s - 1];
    const next = s + 1 < slots ? mean[s + 1] : 0;
    if (mean[s] >= 0.45 * mx && mean[s] >= prev && mean[s] >= next) out.push(s / slots);
  }
  return out;
}

function findHooksFor(s: HookInput, bars: number, maxHooks: number): SongHook[] {
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
  let melMax = 1e-6;
  for (const u of units) melMax = Math.max(melMax, u.mel);
  const used = new Uint8Array(U);
  const hooks: SongHook[] = [];
  for (let h = 0; h < maxHooks; h++) {
    let best: SongHook | null = null;
    let bestProto = -1;
    let bestOcc: number[] = [];
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
        bestOcc = occ;
      }
    }
    if (!best) break;
    best.len = best.occurrences.reduce((a, o) => a + o.end - o.start, 0) / best.occurrences.length;
    best.notes = motifNotes(units, bestOcc);
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
export function findHooks(s: HookInput, maxHooks = 2): SongHook[] {
  const all = [...findHooksFor(s, 1, maxHooks), ...findHooksFor(s, 2, maxHooks)].sort((a, b) => b.score - a.score);
  // Keep hooks whose occurrences mostly do not coincide with an already kept hook.
  const kept: SongHook[] = [];
  for (const h of all) {
    const overl = (o: SongHookOccurrence) => kept.some((k) => k.occurrences.some((p) => p.start < o.end && p.end > o.start));
    const share = h.occurrences.filter(overl).length / h.occurrences.length;
    if (share < 0.5) kept.push(h);
    if (kept.length >= maxHooks) break;
  }
  kept.forEach((h, i) => (h.id = i));
  return kept;
}

/** The hooks of an analysis result (stored ones, else found now for results analysed without them). */
export function hooksOf(r: AnalysisResult): SongHook[] {
  if (Array.isArray(r.hooks)) return r.hooks;
  if (!r.stemOnsets?.other || !r.stemPresence?.other || !r.chroma?.length) return [];
  return findHooks(r);
}

/** One occurrence on the song's hook timeline (hooks merged, the better hook winning overlaps). */
export interface HookSpan {
  start: number;
  end: number;
  /** Hook id (0 = the best hook). */
  hook: number;
  /** Occurrence number within its hook (0 = first). */
  index: number;
  /** Motif note starts as fractions of the span (always starts with 0). */
  notes: number[];
}

/** Every hook occurrence in time order; a weaker hook's occurrences that overlap a better one's are dropped. */
export function hookTimeline(hooks: readonly SongHook[]): HookSpan[] {
  const out: HookSpan[] = [];
  for (const h of [...hooks].sort((a, b) => a.id - b.id)) {
    h.occurrences.forEach((o, index) => {
      if (!(o.end > o.start)) return;
      if (out.some((p) => p.start < o.end - 0.05 && p.end > o.start + 0.05)) return;
      out.push({ start: o.start, end: o.end, hook: h.id, index, notes: h.notes?.length ? h.notes : [0] });
    });
  }
  return out.sort((a, b) => a.start - b.start);
}

/** What the hook signals read at one moment. */
export interface HookSample {
  /** 1 inside a hook repeat, else 0. */
  on: number;
  /** 0..1 through the current repeat (0 outside). */
  phase: number;
  /** 1 at a repeat's start, decaying (~0.2 s). */
  pulse: number;
  /** 1 at each motif note of the repeat, decaying (~0.12 s). */
  notePulse: number;
  /** Index of the motif note last started (-1 outside a repeat). */
  note: number;
  /** Hook id of the current repeat (-1 outside). */
  hook: number;
  /** Occurrence number of the current repeat within its hook (-1 outside). */
  index: number;
}

export const HOOK_PULSE_TAU = 0.2;
export const HOOK_NOTE_TAU = 0.12;

/** The hook signals at time t, given the span at or before t (a pure function of time, so seeking is exact). */
export function sampleHook(span: HookSpan | undefined, t: number, out: HookSample): HookSample {
  out.on = 0;
  out.phase = 0;
  out.pulse = 0;
  out.notePulse = 0;
  out.note = -1;
  out.hook = -1;
  out.index = -1;
  if (!span || t < span.start) return out;
  const since = t - span.start;
  const len = span.end - span.start;
  // The pulses finish decaying after the span ends; the rest is only inside it.
  out.pulse = Math.exp(-since / HOOK_PULSE_TAU);
  if (out.pulse < 1e-3) out.pulse = 0;
  if (t >= span.end) {
    const lastNote = span.start + span.notes[span.notes.length - 1] * len;
    const np = Math.exp(-(t - lastNote) / HOOK_NOTE_TAU);
    out.notePulse = np < 1e-3 ? 0 : np;
    return out;
  }
  out.on = 1;
  out.phase = since / len;
  out.hook = span.hook;
  out.index = span.index;
  let k = 0;
  while (k + 1 < span.notes.length && span.notes[k + 1] <= out.phase) k++;
  out.note = k;
  out.notePulse = Math.exp(-(since - span.notes[k] * len) / HOOK_NOTE_TAU);
  return out;
}

// ------------------------------------------------------------------ live input (realtime-lite)

const LIVE_SLOTS = 8;
const LIVE_HISTORY = 4;
/** Similarity a bar needs to count as a repeat of a recent one. */
const LIVE_SIM = 0.8;

interface LiveBar {
  raw: Float32Array;
  ctr: Float32Array;
  ons: Float32Array;
  mel: number;
}

/**
 * Hooks from live input, without look-ahead: each finished bar is compared with the last few (the
 * same descriptor as the offline finder, one-bar units); when it repeats at least two of them, a
 * riff is going and the next bar is treated as a hook repeat, with the motif notes from the matching
 * bars' onset pattern. The first repeats of a riff are missed (it has to be heard twice first) and a
 * riff ends one bar late; the signals are otherwise the same as for songs.
 */
export class LiveHooks {
  private raw = new Float32Array(LIVE_SLOTS * 12);
  private ons = new Float32Array(LIVE_SLOTS);
  private cnt = new Float32Array(LIVE_SLOTS);
  private mel = 0;
  private frames = 0;
  private history: LiveBar[] = [];
  private bar = -1;
  private barStart = 0;
  private barLen = 2;
  /** Motif of the running riff (null when none). */
  private notes: number[] | null = null;
  private span: HookSpan | null = null;
  private index = 0;
  readonly out: HookSample = { on: 0, phase: 0, pulse: 0, notePulse: 0, note: -1, hook: -1, index: -1 };

  reset(): void {
    this.history = [];
    this.bar = -1;
    this.notes = null;
    this.span = null;
    this.clearBar();
  }

  private clearBar(): void {
    this.raw.fill(0);
    this.ons.fill(0);
    this.cnt.fill(0);
    this.mel = 0;
    this.frames = 0;
  }

  /**
   * One analysis frame: time (s), bar index and phase from the beat clock, seconds per bar, the
   * frame's chroma (12), and the melodic stems' presence and onset strength (other + vocals).
   */
  update(time: number, barIndex: number, barPhase: number, barSeconds: number, chroma: ArrayLike<number>, presence: number, onset: number): HookSample {
    if (barIndex !== this.bar) {
      if (this.bar >= 0 && barIndex === this.bar + 1 && this.frames > 4) this.finishBar();
      else if (barIndex !== this.bar + 1) (this.history = []), (this.notes = null);
      this.bar = barIndex;
      this.barStart = time - barPhase * barSeconds;
      this.barLen = barSeconds;
      this.clearBar();
      this.span = this.notes && barIndex >= 0 ? { start: this.barStart, end: this.barStart + this.barLen, hook: 0, index: this.index++, notes: this.notes } : null;
    }
    if (barIndex >= 0) {
      const k = Math.min(LIVE_SLOTS - 1, Math.max(0, Math.floor(barPhase * LIVE_SLOTS)));
      for (let c = 0; c < 12; c++) {
        const v = chroma[c] ?? 0;
        this.raw[k * 12 + c] += presence * v * v;
      }
      this.ons[k] += onset;
      this.cnt[k]++;
      this.mel += presence;
      this.frames++;
    }
    return sampleHook(this.span ?? undefined, time, this.out);
  }

  private finishBar(): void {
    const raw = new Float32Array(this.raw);
    const ons = new Float32Array(LIVE_SLOTS);
    for (let k = 0; k < LIVE_SLOTS; k++) {
      const n = Math.max(1, this.cnt[k]);
      ons[k] = this.ons[k] / n;
      for (let c = 0; c < 12; c++) raw[k * 12 + c] /= n;
    }
    const ctr = new Float32Array(LIVE_SLOTS * 12);
    for (let c = 0; c < 12; c++) {
      let mu = 0;
      for (let k = 0; k < LIVE_SLOTS; k++) mu += raw[k * 12 + c];
      mu /= LIVE_SLOTS;
      for (let k = 0; k < LIVE_SLOTS; k++) ctr[k * 12 + c] = raw[k * 12 + c] - mu;
    }
    const cur: LiveBar = { raw, ctr, ons, mel: this.mel / Math.max(1, this.frames) };
    const matches = cur.mel >= 0.12 ? this.history.filter((b) => 0.35 * cos(b.raw, cur.raw) + 0.45 * Math.max(0, cos(b.ctr, cur.ctr)) + 0.2 * Math.max(0, cos(b.ons, cur.ons)) >= LIVE_SIM) : [];
    if (matches.length >= 2) {
      const group = [...matches, cur];
      const mean = new Float32Array(LIVE_SLOTS);
      for (const b of group) for (let k = 0; k < LIVE_SLOTS; k++) mean[k] += b.ons[k] / group.length;
      let mx = 0;
      for (const v of mean) mx = Math.max(mx, v);
      const notes = [0];
      for (let k = 1; k < LIVE_SLOTS; k++) if (mx > 1e-6 && mean[k] >= 0.45 * mx && mean[k] >= mean[k - 1] && mean[k] >= (mean[k + 1] ?? 0)) notes.push(k / LIVE_SLOTS);
      if (!this.notes) this.index = 0;
      this.notes = notes;
    } else this.notes = null;
    this.history.push(cur);
    if (this.history.length > LIVE_HISTORY) this.history.shift();
  }
}

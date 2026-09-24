// Visual deja vu: a genome-wide gene that makes a repeated section recognisably the same scene.
// Songs are analysed before playback, including which sections come back (src/analysis/repetition.ts).
// At a returning section's first appearance the renderer takes a snapshot: the feedback image
// (downscaled), the camera framing, the palette hue and the free-running motion phases. When the
// section returns, the snapshot is folded back in over a few bars (the image blended into the
// feedback buffer so the chain carries it on, the framing eased back, the colours and phases pulled
// home), evolved a little more on every return.
//
// This file is pure (no GL): the gene, its breeding, and the planner that decides from the music's
// timeline when to snapshot and how strongly to recall, so seeking and the tests are deterministic.
// The GPU side (snapshot textures, the recall blend) is genes/dejavuGpu.ts.
// Imports only types from genome.ts so genome.ts can import it without a cycle.

import type { ParamSpec, Params, Schema } from '../genome';
import type { MusicState } from '../../types';
import { registerGenomeGene } from '../geneRegistry';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * recall: how strongly the remembered image returns (0 none .. 1 the picture cuts back to it);
 * blend: bars the recall takes to fold in and fade (it then lives on in the feedback);
 * snap: where in the first appearance the snapshot is taken (fraction of the section);
 * frame: how far the camera goes back to the remembered framing, with a small flashback push;
 * hue: how far the palette returns to the remembered colours (holds through the section);
 * motion: how far the free-running motion phases (spins, rotations, drifts) rewind to the remembered ones;
 * evolve: how much each return differs (the memory comes back turned, pushed and hue-shifted, more each time);
 * keep: 0 remembers the first appearance, 1 re-remembers every appearance (changes compound);
 * res: snapshot resolution (fraction of the screen; lower = softer, dreamier memory, less memory used);
 * cap: most snapshots held at once (the least recently used is forgotten);
 * min: how similar a return must be (repetition similarity) to be remembered and recalled.
 */
export const DEJAVU_SCHEMA: Schema = {
  recall: P(0, 1, 0.6),
  blend: C([0.5, 1, 2, 4], 2),
  snap: P(0.3, 0.95, 0.8),
  frame: P(0, 1, 0.5),
  hue: P(0, 1, 0.5),
  motion: P(0, 1, 0.3),
  evolve: P(0, 1, 0.35),
  keep: C([0, 1], 0),
  res: C([0.125, 0.25, 0.5], 0.25),
  cap: C([2, 3, 4], 3),
  min: P(0.6, 0.95, 0.7),
};

registerGenomeGene({
  key: 'dejavu',
  title: 'Deja vu',
  schemas: DEJAVU_SCHEMA,
  optional: true,
  order: 1,
  glossary: 'makes a repeated section (the second chorus, a returning riff) recognisably the same scene: at its first appearance the picture is remembered (at snap through the section), and when it returns the memory folds back in over blend bars (recall = how strongly the image returns), the camera eases back to the remembered framing (frame), the palette returns to the remembered colours (hue) and spins and drifts rewind to where they were (motion); each return comes back more turned, pushed and hue-shifted (evolve); keep 1 re-remembers each appearance so changes compound; res = memory sharpness; cap = scenes remembered at once; min = how alike a return must be',
});

export interface DejaVuGene {
  p: Params;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object';
const TAU = Math.PI * 2;
const smooth = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

/**
 * Estimated GPU cost at 1440p: the recall pass (one full-screen blend of a small texture) counted
 * every frame, plus the snapshot downsample amortised (it runs once per remembered section).
 */
export function dejavuCost(p: Params): number {
  return 0.06 + 0.08 * (p.res ?? 0.25);
}

/** Memory held by the snapshots at a screen size (bytes; RGBA16F). */
export function dejavuMemoryBytes(p: Params, w: number, h: number): number {
  const sw = Math.max(2, Math.round(w * p.res));
  const sh = Math.max(2, Math.round(h * p.res));
  return sw * sh * 8 * p.cap;
}

function clampSpec(v: unknown, s: ParamSpec): number {
  let x = typeof v === 'number' && Number.isFinite(v) ? v : s.def;
  if (s.choices) {
    let best = s.choices[0];
    for (const c of s.choices) if (Math.abs(c - x) < Math.abs(best - x)) best = c;
    return best;
  }
  x = clamp(x, s.min, s.max);
  return s.int ? Math.round(x) : x;
}

/** A valid deja vu gene from anything (missing or broken values take the defaults). */
export function repairDejaVu(raw: unknown): DejaVuGene {
  const src = isObj(raw) && isObj(raw.p) ? raw.p : {};
  const p: Params = {};
  for (const k of Object.keys(DEJAVU_SCHEMA)) p[k] = clampSpec(src[k], DEJAVU_SCHEMA[k]);
  return { p };
}

/** Rule violations of a deja vu gene (empty when valid). */
export function validateDejaVu(c: DejaVuGene): string[] {
  const errs: string[] = [];
  if (!isObj(c) || !isObj(c.p)) return ['dejavu params'];
  for (const k of Object.keys(DEJAVU_SCHEMA)) {
    const s = DEJAVU_SCHEMA[k];
    const v = c.p[k];
    const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min - 1e-9 && v <= s.max + 1e-9);
    if (!ok) errs.push(`dejavu.${k}`);
  }
  for (const k of Object.keys(c.p)) if (!(k in DEJAVU_SCHEMA)) errs.push(`dejavu.${k} unknown`);
  return errs;
}

// ------------------------------------------------------------ breeding

type Rng = () => number;

function gauss(rng: Rng): number {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function randomValue(s: ParamSpec, rng: Rng): number {
  if (s.choices) return s.choices[Math.floor(rng() * s.choices.length)];
  return clampSpec(s.min + (s.max - s.min) * rng(), s);
}

/** A new random deja vu (most settings near their defaults, a few pushed further). */
export function randomDejaVu(rng: Rng): DejaVuGene {
  const p: Params = {};
  for (const k of Object.keys(DEJAVU_SCHEMA)) {
    const s = DEJAVU_SCHEMA[k];
    p[k] = rng() < 0.5 ? randomValue(s, rng) : s.def;
  }
  // A memory nobody can see is no memory: keep at least one recall channel audible.
  if (p.recall < 0.25 && p.frame < 0.25 && p.hue < 0.25) p.recall = 0.25 + 0.5 * rng();
  return repairDejaVu({ p });
}

/** Nudges some settings of a deja vu in place. */
export function jitterDejaVu(c: DejaVuGene, rng: Rng, amt = 1): void {
  const keys = Object.keys(DEJAVU_SCHEMA);
  let touched = false;
  for (const k of keys) {
    if (rng() >= 0.3) continue;
    touched = true;
    const s = DEJAVU_SCHEMA[k];
    c.p[k] = s.choices ? (rng() < 0.5 ? randomValue(s, rng) : c.p[k]) : clampSpec(c.p[k] + gauss(rng) * (s.max - s.min) * 0.15 * amt, s);
  }
  if (!touched) {
    const k = keys[Math.floor(rng() * keys.length)];
    c.p[k] = randomValue(DEJAVU_SCHEMA[k], rng);
  }
}

/** Probability a child inherits the deja vu when only one parent has one. */
export const DEJAVU_CARRY = 0.5;

/**
 * Crossover of the deja vu. Draws from the rng only when a parent has one, so children of parents
 * without it come out exactly as before the gene existed.
 */
export function crossDejaVu(d: DejaVuGene | undefined, r: DejaVuGene | undefined, rng: Rng): DejaVuGene | undefined {
  if (!d && !r) return undefined;
  if (!d || !r) {
    const only = (d ?? r)!;
    return rng() < DEJAVU_CARRY ? repairDejaVu(only) : undefined;
  }
  const p: Params = {};
  for (const k of Object.keys(DEJAVU_SCHEMA)) {
    const s = DEJAVU_SCHEMA[k];
    const x = d.p[k] ?? s.def;
    const y = r.p[k] ?? s.def;
    const t = rng();
    p[k] = s.choices ? (t < 0.5 ? x : y) : x + (y - x) * t;
  }
  return repairDejaVu({ p });
}

// --------------------------------------------------------------- timeline

/** The timeline position deja vu needs (repetition from the offline analysis). */
export interface DejaVuCue {
  /** False for live input or songs without repetition data: the gene stays idle. */
  valid: boolean;
  /** Repetition group of the current section, its occurrence (0 = first), similarity, and how closely it returns later. */
  group: number;
  index: number;
  sim: number;
  returnSim: number;
  /** Start of the current section (identifies this appearance), seconds since it began, and its length. */
  sectionStart: number;
  sinceSection: number;
  sectionLen: number;
  barSeconds: number;
  /** Song position (seconds), for least-recently-used bookkeeping. */
  time: number;
}

/** The cue a MusicState carries (sampled songs only). */
export function dejavuCueOf(state: MusicState): DejaVuCue {
  const bpm = state.bpm > 0 && Number.isFinite(state.bpm) ? state.bpm : 120;
  const bar = state.barSeconds && state.barSeconds > 0 ? state.barSeconds : 240 / bpm;
  const sec = state.section;
  const start = sec && Number.isFinite(sec.start) ? sec.start : 0;
  const since = state.time - start;
  const len = sec && Number.isFinite(sec.end) && sec.end > start ? sec.end - start : 16 * bar;
  const g = state.repeatGroup;
  return {
    valid: typeof g === 'number' && g >= 0 && Number.isFinite(state.time),
    group: typeof g === 'number' ? g : -1,
    index: state.repeatIndex ?? 0,
    sim: state.repeatSim ?? 0,
    returnSim: state.repeatReturnSim ?? 0,
    sectionStart: start,
    sinceSection: Number.isFinite(since) && since > 0 ? since : 0,
    sectionLen: len,
    barSeconds: bar,
    time: state.time,
  };
}

/** Camera framing as deja vu remembers it (a subset of the choreography pose). */
export interface Framing {
  zoom: number;
  roll: number;
  tx: number;
  ty: number;
}

/** What a snapshot holds besides its texture (the GPU side keeps the image by group). */
export interface MemoryRecord {
  group: number;
  /** Occurrence of the group it was taken in, and that appearance's start (seconds). */
  index: number;
  sectionStart: number;
  /** Last song time it was written or recalled (least recently used goes first). */
  used: number;
  framing: Framing;
  /** Palette hue (turns) the slot used when it was taken. */
  hue: number;
  /** Free-running phases from the slot's memory (see isPhaseKey). */
  phases: Record<string, number>;
}

/** The recall in progress this frame. */
export interface Recall {
  group: number;
  /** Returns since the remembered appearance (1 = the first return of that memory). */
  k: number;
  /** 0..1 progress through the blend window (1 = done). */
  t: number;
  /** Image fold envelope 0..1 (a quick rise, a long fade), framing envelope, colour hold. */
  image: number;
  framing: number;
  colour: number;
  /** Motion rewind progress 0..1 (monotonic; the phase offset is applied in increments of it). */
  motion: number;
  record: MemoryRecord;
}

/** What the planner decided this frame. */
export interface DejaVuPlan {
  /** Take a snapshot of this group this frame (-1: none). */
  snap: number;
  /** Forget this group's snapshot first to stay within the cap (-1: none). */
  evict: number;
  recall: Recall | null;
}

/** Phase-like slot memory keys: free body clocks, draw-op angles and phases, particle phases. */
export function isPhaseKey(k: string): boolean {
  return /\.c(sp|tb)$/.test(k) || /(ang|ph)$/.test(k) || /^(rp|np)\d+$/.test(k);
}

/** Period used to wrap a phase offset the short way round (Infinity: never wrap, just cap). */
function phasePeriod(k: string): number {
  if (/(sp|ang)$/.test(k) || /^rp\d+$/.test(k)) return TAU;
  if (/\.ctb$/.test(k)) return 16; // bars: keeps sway / circle periods (1-16 bars) in step
  return Infinity;
}

/** Largest offset applied to a phase without a period (noise phases): a quick rewind, not a spin-up. */
const MAX_FREE_OFFSET = 24;

/** The short-way offset from `cur` to `target` for a phase key (0 when it would be too far). */
export function phaseOffset(k: string, cur: number, target: number): number {
  const per = phasePeriod(k);
  let d = target - cur;
  if (per === Infinity) return Math.abs(d) <= MAX_FREE_OFFSET ? d : 0;
  d -= Math.round(d / per) * per;
  return d;
}

/** A 0..1 hash (stable across runs and platforms). */
function hash01(a: number, b: number): number {
  let h = Math.imul(Math.floor(a * 9973) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 1, 0xc2b2ae35);
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12;
  h = Math.imul(h, 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/** How a recalled memory is evolved on its k-th return: turned, pushed and hue-shifted. */
export interface Evolution {
  zoom: number; // >= 1
  rot: number; // radians
  hue: number; // turns
}

export function evolution(c: DejaVuGene, group: number, k: number, out: Evolution = { zoom: 1, rot: 0, hue: 0 }): Evolution {
  const e = c.p.evolve * Math.max(0, k);
  const sign = hash01(group + 0.5, 7) < 0.5 ? -1 : 1;
  out.zoom = 1 + 0.07 * e;
  out.rot = sign * 0.045 * TAU * e;
  out.hue = (hash01(group + 0.5, 3) < 0.5 ? -1 : 1) * 0.06 * e;
  return out;
}

/** Envelopes over the blend window, t = 0..1. */
export function recallEnvelopes(t: number): { image: number; framing: number; colour: number; motion: number } {
  if (!(t >= 0)) t = 0;
  return {
    image: t >= 1 ? 0 : smooth(0, 0.08, t) * (1 - smooth(0.3, 1, t)),
    framing: t >= 1.6 ? 0 : smooth(0, 0.15, t) * (1 - smooth(0.6, 1.6, t)),
    colour: smooth(0, 0.3, t),
    motion: smooth(0, 0.5, t),
  };
}

/** Image recall rate at full envelope (per second): recall 1 all but replaces the picture within ~0.2 s. */
const RECALL_RATE = 18;

/** The blend alpha of the recalled image into the feedback buffer this frame. */
export function imageAlpha(c: DejaVuGene, env: number, sdt: number): number {
  const rate = RECALL_RATE * Math.pow(c.p.recall, 1.5) * env;
  return rate > 0 && sdt > 0 ? 1 - Math.exp(-rate * Math.min(sdt, 0.1)) : 0;
}

/** Flashback push-in at full framing recall. */
const FLASH_ZOOM = 0.08;

/**
 * The planner: one per rendering slot. Tracks which groups are remembered (the images themselves live
 * on the GPU) and decides per frame whether to snapshot, what to forget, and how far a recall is.
 */
export class DejaVuPlanner {
  readonly records = new Map<number, MemoryRecord>();
  /** Appearance (section start) already snapshotted, so a section is remembered once per visit. */
  private snapped = new Set<number>();
  private lastSection = NaN;
  /** Motion rewind already applied in this recall (fraction), and its per-key offsets. */
  private motionDone = 0;
  private offsets: Record<string, number> = {};

  gene: DejaVuGene;

  constructor(gene: DejaVuGene) {
    this.gene = gene;
  }

  /** Forgets everything (new song). */
  reset(): void {
    this.records.clear();
    this.snapped.clear();
    this.lastSection = NaN;
    this.motionDone = 0;
    this.offsets = {};
  }

  /** Decides this frame (does not modify records: commit a snapshot with remember()). */
  plan(cue: DejaVuCue): DejaVuPlan {
    const p = this.gene.p;
    const out: DejaVuPlan = { snap: -1, evict: -1, recall: null };
    if (!cue.valid) return out;
    if (cue.sectionStart !== this.lastSection) {
      this.lastSection = cue.sectionStart;
      this.motionDone = 0;
      this.offsets = {};
    }
    const rec = this.records.get(cue.group);
    // Recall: a return of a remembered group, from an earlier appearance than this one.
    if (cue.index > 0 && cue.sim >= p.min - 1e-9 && rec && rec.sectionStart < cue.sectionStart - 1e-6) {
      const t = cue.sinceSection / Math.max(1e-3, p.blend * cue.barSeconds);
      const env = recallEnvelopes(t);
      out.recall = { group: cue.group, k: Math.max(1, cue.index - rec.index), t: Math.min(1, t), ...env, record: rec };
    }
    // Snapshot: this group comes back later, and this appearance has not been remembered yet.
    const wants = cue.returnSim >= p.min - 1e-9 && (!rec || p.keep >= 0.5) && !this.snapped.has(cue.sectionStart);
    const due = cue.sinceSection >= p.snap * cue.sectionLen;
    // With keep=1 a recall still in progress finishes before the memory is rewritten.
    if (wants && due && !(out.recall && out.recall.t < 1)) {
      out.snap = cue.group;
      if (!rec && this.records.size >= p.cap) {
        let lru: MemoryRecord | null = null;
        for (const r of this.records.values()) if (!lru || r.used < lru.used) lru = r;
        if (lru) out.evict = lru.group;
      }
    }
    return out;
  }

  /** Commits a snapshot the GPU took (after an eviction, if planned). */
  remember(cue: DejaVuCue, plan: DejaVuPlan, framing: Framing, hue: number, mem: Record<string, number>): MemoryRecord {
    if (plan.evict >= 0) this.records.delete(plan.evict);
    const phases: Record<string, number> = {};
    for (const k of Object.keys(mem)) if (isPhaseKey(k) && Number.isFinite(mem[k])) phases[k] = mem[k];
    const rec: MemoryRecord = { group: cue.group, index: cue.index, sectionStart: cue.sectionStart, used: cue.time, framing: { ...framing }, hue, phases };
    this.records.set(cue.group, rec);
    this.snapped.add(cue.sectionStart);
    return rec;
  }

  /**
   * Applies a recall to this frame's camera and palette (in place) and rewinds the slot's phases.
   * `pose` is the slot's pose before deja vu (identity or the choreography's); `hueNow` the palette
   * hue the slot would use (the key hue plus the palette offset).
   */
  apply(r: Recall, pose: Framing & { hue: number }, hueNow: number, mem: Record<string, number>, time: number): void {
    const p = this.gene.p;
    const rec = r.record;
    rec.used = time;
    const ev = evolution(this.gene, r.group, r.k);
    // Framing: back toward the remembered shot, with a flashback push that settles.
    const w = p.frame * r.framing;
    if (w > 0) {
      pose.zoom = (pose.zoom + (rec.framing.zoom - pose.zoom) * w) * (1 + FLASH_ZOOM * w);
      pose.roll = pose.roll + (rec.framing.roll + ev.rot * 0.25 - pose.roll) * w;
      pose.tx = pose.tx + (rec.framing.tx - pose.tx) * w;
      pose.ty = pose.ty + (rec.framing.ty - pose.ty) * w;
    }
    // Colour: the remembered palette (the short way round the hue circle), plus the evolution's shift.
    let dh = rec.hue - hueNow;
    dh -= Math.round(dh);
    pose.hue += (dh + ev.hue) * p.hue * r.colour;
    // Motion: phases rewind toward the remembered ones over the first half of the blend window.
    if (p.motion > 0) {
      if (this.motionDone === 0 && r.motion > 0) {
        this.offsets = {};
        for (const k of Object.keys(rec.phases)) if (k in mem) this.offsets[k] = phaseOffset(k, mem[k], rec.phases[k]);
      }
      const step = Math.max(0, r.motion - this.motionDone);
      if (step > 0) {
        for (const k of Object.keys(this.offsets)) if (k in mem) mem[k] += this.offsets[k] * step * p.motion;
        this.motionDone = r.motion;
      }
    }
  }
}

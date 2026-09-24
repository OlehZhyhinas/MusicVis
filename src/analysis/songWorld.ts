// The song as a landscape: the offline analysis turned into a small "world map" of the song's
// timeline, which the 'landscape' shape (src/v2/genes/landscape.ts) builds its terrain from.
// Along-track distance is song time, so what the camera sees on the horizon is the music that is
// still to come: the climb to a drop, the drop itself as a mountain pass, breakdowns as valleys.
//
// The map is n samples x 2 rows of 4 floats (uploaded as an RGBA32F texture, n x 2):
//   row 0: alt     macro altitude 0..~1.6 (smoothed energy, the climb to each drop, breakdown dips)
//          summit  1 at a drop's start, falling off over a couple of seconds (the mountain pass)
//          motif   the section type (index into SECTION_LABELS), so every chorus reuses one terrain
//          since   seconds since the section began
//   row 1: key     hue of the key there (circle of fifths, as TimelineSampler.keyHue), 0..1
//          minor   1 in a minor key
//          until   seconds until the section ends
//          cx      musical density (complexity), 0..1
// Sample i sits at time t0 + i * span / (n - 1).
//
// Songs get the whole map once (buildSongWorld). Live input has no future, so LiveWorld builds a
// moving window: the part behind the camera is what happened, and each sample entering the window at
// the far end is generated once from recent history and then kept, so the ground never shifts.
//
// Pure (no DOM, no GL): the tests run it in node.

import type { AnalysisResult, KeySegment, MusicState, SectionLabel } from '../types';

export const SECTION_LABELS: SectionLabel[] = ['intro', 'verse', 'build', 'chorus', 'drop', 'breakdown', 'outro'];

export interface WorldMark {
  /** Song time of the section start (seconds). */
  t: number;
  /** Section type (index into SECTION_LABELS). */
  label: number;
  /** Section energy 0..1. */
  energy: number;
}

export interface SongWorld {
  t0: number;
  span: number;
  n: number;
  /** Row 0 (n * 4 floats) then row 1 (n * 4 floats): exactly the texture's layout. */
  data: Float32Array;
  /** True for the live window (no look-ahead). */
  live: boolean;
  /** Section starts (after the first section): the landmarks. */
  marks: WorldMark[];
  /** Drop starts (the sections that fire dropPulse, as TimelineSampler). */
  drops: number[];
  /** Bumped whenever data changes (the texture re-uploads). */
  version: number;
}

export const WORLD_SAMPLES = 1024;
export const LIVE_SAMPLES = 256;
/** Live window: this many seconds behind the camera and the rest ahead. */
export const LIVE_BACK = 4;
export const LIVE_SPAN = 48;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const smooth = (x: number) => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };

function keyHueOf(k: KeySegment): number {
  const major = k.mode === 'minor' ? (k.tonic + 3) % 12 : k.tonic;
  return ((major * 7) % 12) / 12;
}

function gaussBlur(a: Float32Array, sigma: number): Float32Array {
  if (sigma < 0.3) return a.slice();
  const r = Math.ceil(sigma * 3);
  const w: number[] = [];
  for (let k = -r; k <= r; k++) w.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    let s = 0, ws = 0;
    for (let k = -r; k <= r; k++) {
      const j = i + k;
      if (j < 0 || j >= a.length) continue;
      s += a[j] * w[k + r];
      ws += w[k + r];
    }
    out[i] = ws > 0 ? s / ws : 0;
  }
  return out;
}

/** Mean of a per-frame envelope over [ta, tb). */
function meanOver(env: Float32Array | undefined, rate: number, ta: number, tb: number): number {
  if (!env || !env.length) return 0;
  const a = clamp(Math.floor(ta * rate), 0, env.length - 1);
  const b = clamp(Math.ceil(tb * rate), a + 1, env.length);
  let s = 0;
  for (let i = a; i < b; i++) s += env[i];
  return s / (b - a);
}

/** Seconds of climb before a drop (the build before it, 4..16 s). */
function climbLead(prev: { start: number; end: number; label: SectionLabel } | undefined): number {
  if (prev && prev.label === 'build') return clamp(prev.end - prev.start, 4, 16);
  return 8;
}

/** Climb toward a drop at d: 0 far before, rising to 1 at d, then a cliff back to 0 within 0.6 s. */
export function climbAt(t: number, d: number, lead: number): number {
  if (t > d) return 1 - smooth((t - d) / 0.6);
  if (t < d - lead) return 0;
  return Math.pow((t - (d - lead)) / lead, 1.5);
}

/** The whole song as a world map (once per song). */
export function buildSongWorld(r: AnalysisResult, n = WORLD_SAMPLES): SongWorld {
  const dur = Math.max(r.duration, 1);
  const dt = dur / (n - 1);
  const rate = r.frameRate > 0 ? r.frameRate : 1;
  const sections = r.sections.length ? r.sections : [{ start: 0, end: dur, label: 'verse' as SectionLabel, energy: 0.5 }];
  const keys = r.keys.length ? r.keys : [{ start: 0, end: dur, tonic: 0, mode: 'major' as const, confidence: 0 }];
  const drops: number[] = [];
  const leads: number[] = [];
  for (let i = 1; i < sections.length; i++) {
    const s = sections[i];
    if (s.label === 'drop' || (s.label === 'chorus' && sections[i - 1].label === 'build')) {
      drops.push(s.start);
      leads.push(climbLead(sections[i - 1]));
    }
  }
  // Energy: loudness per sample, smoothed over ~1.5 s and scaled so the loud end reaches ~1.
  const raw = new Float32Array(n);
  const cx = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    raw[i] = meanOver(r.loudness, rate, t - dt / 2, t + dt / 2);
    cx[i] = meanOver(r.complexity, rate, t - dt / 2, t + dt / 2);
  }
  const energy = gaussBlur(raw, 1.5 / dt);
  const sorted = Array.from(energy).sort((a, b) => a - b);
  const top = Math.max(sorted[Math.floor(sorted.length * 0.95)] ?? 0, 1e-3);
  // Breakdowns dig valleys (soft-edged over ~2 s).
  const dip = new Float32Array(n);
  let si = 0, ki = 0;
  const secIdx = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    while (si < sections.length - 1 && t >= sections[si].end) si++;
    secIdx[i] = si;
    dip[i] = sections[si].label === 'breakdown' ? 1 : 0;
  }
  const dipS = gaussBlur(dip, 2 / dt);
  const data = new Float32Array(n * 8);
  const row1 = n * 4;
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    let climb = 0, summit = 0;
    for (let j = 0; j < drops.length; j++) {
      climb = Math.max(climb, climbAt(t, drops[j], leads[j]));
      summit = Math.max(summit, Math.exp(-(((t - drops[j]) / 1.5) ** 2)));
    }
    const s = sections[secIdx[i]];
    while (ki < keys.length - 1 && t >= keys[ki].end) ki++;
    const e = clamp(energy[i] / top, 0, 1.2);
    data[i * 4] = Math.max(0, 0.15 + 0.85 * e + 0.45 * climb - 0.3 * dipS[i]);
    data[i * 4 + 1] = summit;
    data[i * 4 + 2] = Math.max(0, SECTION_LABELS.indexOf(s.label));
    data[i * 4 + 3] = Math.max(0, t - s.start);
    data[row1 + i * 4] = keyHueOf(keys[ki]);
    data[row1 + i * 4 + 1] = keys[ki].mode === 'minor' ? 1 : 0;
    data[row1 + i * 4 + 2] = Math.max(0, s.end - t);
    data[row1 + i * 4 + 3] = clamp(cx[i], 0, 1);
  }
  const marks: WorldMark[] = sections.slice(1).map((s) => ({ t: s.start, label: Math.max(0, SECTION_LABELS.indexOf(s.label)), energy: clamp(s.energy, 0, 1) }));
  return { t0: 0, span: dur, n, data, live: false, marks, drops, version: 1 };
}

/** A channel of the map at time t (row 0 channels 0-3, row 1 channels 4-7), linear between samples. */
export function worldAt(w: SongWorld, t: number, ch: number): number {
  const u = clamp(((t - w.t0) / w.span) * (w.n - 1), 0, w.n - 1);
  const i = Math.min(Math.floor(u), w.n - 2);
  const f = u - i;
  const o = ch < 4 ? ch : w.n * 4 + ch - 4;
  const a = w.data[o + i * 4], b = w.data[o + (i + 1) * 4];
  return a + (b - a) * f;
}

/** A flat, calm world (before any song or live input). */
export function emptyWorld(): SongWorld {
  const n = 2;
  const data = new Float32Array(n * 8);
  for (let i = 0; i < n; i++) {
    data[i * 4] = 0.3;
    data[i * 4 + 2] = 1;
    data[i * 4 + 3] = 1e3;
    data[n * 4 + i * 4 + 2] = 1e3;
  }
  return { t0: 0, span: 1, n, data, live: true, marks: [], drops: [], version: 1 };
}

/**
 * Live input: a moving window of the world. Behind the camera it is what happened; each sample
 * entering at the far end is generated once (from the loudness of a window-length ago and of the
 * last few seconds, the current section and key) and kept, so the ground ahead stays put as the
 * camera reaches it. A build in progress raises a predicted climb toward a pass a few seconds ahead.
 */
export class LiveWorld {
  readonly world: SongWorld;
  private readonly dt = LIVE_SPAN / (LIVE_SAMPLES - 1);
  /** Generated samples by absolute index (ring of LIVE_SAMPLES, 8 floats each). */
  private ring = new Float32Array(LIVE_SAMPLES * 8);
  private gen = -1;
  /** Loudness history at 10 Hz (60 s ring). */
  private hist = new Float32Array(600);
  private histN = 0;
  private histAcc = 0;
  private histT = 0;
  private histCnt = 0;
  private peak = 0.3;
  private build = 0;
  private lastTime = -1;
  private sinceWrite = 1;
  private secStart = 0;
  private secLabel = 1;
  private lastSec = -1;

  constructor() {
    const n = LIVE_SAMPLES;
    this.world = { t0: 0, span: LIVE_SPAN, n, data: new Float32Array(n * 8), live: true, marks: [], drops: [], version: 0 };
  }

  reset(): void {
    this.gen = -1;
    this.histN = 0;
    this.histAcc = 0;
    this.histT = 0;
    this.histCnt = 0;
    this.peak = 0.3;
    this.build = 0;
    this.lastSec = -1;
    this.world.marks.length = 0;
    this.world.drops.length = 0;
  }

  private histAgo(sec: number): number {
    if (!this.histN) return 0.3;
    const k = clamp(Math.round(sec * 10), 0, Math.min(this.histN, this.hist.length) - 1);
    return this.hist[(this.histN - 1 - k) % this.hist.length];
  }

  private histMean(sec: number): number {
    const m = Math.min(this.histN, Math.round(sec * 10));
    if (!m) return 0.3;
    let s = 0;
    for (let k = 0; k < m; k++) s += this.hist[(this.histN - 1 - k) % this.hist.length];
    return s / m;
  }

  /** Feeds a frame; returns true when the window was rewritten (about 10 times a second). */
  update(state: MusicState, dt: number): boolean {
    const time = Number.isFinite(state.time) ? state.time : 0;
    if (time < this.lastTime - 1) this.reset();
    this.lastTime = time;
    const loud = clamp(Number.isFinite(state.loudness) ? state.loudness : 0, 0, 2);
    this.peak = Math.max(loud, this.peak * Math.exp(-dt / 20), 0.1);
    this.build += ((state.buildIntensity ?? 0) - this.build) * (1 - Math.exp(-dt * 2));
    // Section changes (landmarks behind the camera) and drops.
    const sec = state.sectionIndex ?? 0;
    if (sec !== this.lastSec) {
      if (this.lastSec >= 0) this.world.marks.push({ t: time, label: Math.max(0, SECTION_LABELS.indexOf(state.section?.label ?? 'verse')), energy: clamp(loud / this.peak, 0, 1) });
      if (this.world.marks.length > 16) this.world.marks.shift();
      this.lastSec = sec;
      this.secStart = time;
    }
    this.secLabel = Math.max(0, SECTION_LABELS.indexOf(state.section?.label ?? 'verse'));
    if ((state.dropPulse ?? 0) > 0.95 && (!this.world.drops.length || time - this.world.drops[this.world.drops.length - 1] > 4)) {
      this.world.drops.push(time);
      if (this.world.drops.length > 8) this.world.drops.shift();
    }
    this.histAcc += loud / this.peak;
    this.histCnt++;
    this.histT += dt;
    if (this.histT >= 0.1) {
      this.hist[this.histN % this.hist.length] = this.histAcc / Math.max(1, this.histCnt);
      this.histN++;
      this.histAcc = 0;
      this.histCnt = 0;
      this.histT = 0;
    }
    this.sinceWrite += dt;
    if (this.sinceWrite < 0.1 && this.gen >= 0) return false;
    this.sinceWrite = 0;
    this.write(time, state);
    return true;
  }

  private write(time: number, state: MusicState): void {
    const n = LIVE_SAMPLES, dt = this.dt, R = this.ring;
    const i0 = Math.floor((time - LIVE_BACK) / dt);
    const iEnd = i0 + n - 1;
    if (this.gen < i0 - 1 || this.gen > iEnd) this.gen = i0 - 1;
    const recent = this.histMean(6);
    const minor = state.keyMode === 'minor' ? 1 : 0;
    const key = Number.isFinite(state.keyHue) ? state.keyHue : 0;
    const cx = clamp(state.complexity ?? 0.5, 0, 1);
    for (let i = this.gen + 1; i <= iEnd; i++) {
      const ts = i * dt;
      const ahead = ts - time;
      // Ahead: replay the loudness of one window-length ago, pulled toward the last few seconds.
      const e = ahead <= 0 ? this.histAgo(-ahead) : 0.5 * this.histAgo(Math.max(0, LIVE_SPAN - ahead)) + 0.5 * recent;
      const o = (((i % n) + n) % n) * 8;
      R[o] = 0.15 + 0.85 * clamp(e, 0, 1.2);
      R[o + 1] = 0;
      R[o + 2] = this.secLabel;
      R[o + 3] = Math.max(0, ts - this.secStart);
      R[o + 4] = key;
      R[o + 5] = minor;
      R[o + 6] = 1e3;
      R[o + 7] = cx;
    }
    this.gen = iEnd;
    // Copy the window out of the ring, adding the predicted climb of a build in progress and the
    // passes of drops already heard.
    const w = this.world;
    w.t0 = i0 * dt;
    const bi = this.build;
    const td = time + Math.max(3, (1 - bi) * 12);
    for (let k = 0; k < n; k++) {
      const i = i0 + k;
      const ts = i * dt;
      const o = (((i % n) + n) % n) * 8;
      const climb = bi > 0.2 ? smooth((bi - 0.2) / 0.5) * climbAt(ts, td, 8) : 0;
      let summit = 0;
      for (const d of w.drops) summit = Math.max(summit, Math.exp(-(((ts - d) / 1.5) ** 2)));
      w.data[k * 4] = R[o] + 0.45 * climb;
      w.data[k * 4 + 1] = summit;
      w.data[k * 4 + 2] = R[o + 2];
      w.data[k * 4 + 3] = R[o + 3];
      w.data[n * 4 + k * 4] = R[o + 4];
      w.data[n * 4 + k * 4 + 1] = R[o + 5];
      w.data[n * 4 + k * 4 + 2] = R[o + 6];
      w.data[n * 4 + k * 4 + 3] = R[o + 7];
    }
    w.version++;
  }
}

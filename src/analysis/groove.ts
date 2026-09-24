// Groove: the timing feel of the playing, measured against the tracked beat grid.
//
//   swing     where the off-beat of each 8th (or 16th) pair lands: 0 straight (1:1), 1 triplet swing (2:1)
//   push      the backbeat's lean against the strong beats: > 0 laid back (late), < 0 pushing (early),
//             1 = 40 ms
//   humanity  how loosely onsets sit on their (swung) grid slots: 0 machine-exact, 1 ~ 20 ms spread
//   synco     syncopation density: onsets on weak slots whose next stronger slot stays silent
//
// Offline: onset peaks are picked from the onset envelope with sub-frame (parabolic) timing and
// read against a local ideal grid (a least-squares line through the 9 tracked beats around each
// beat, so tempo drift and the tracker's own snapping cancel). Every beat gives its evidence; a
// windowed estimate over +-8 beats becomes the per-frame track, with a median per section and one
// for the song. Cost: one pass over the envelope plus O(beats) work (well under a millisecond per
// minute of audio).
//
// Realtime: GrooveTracker runs the same estimator every half second over the last 12 s of live
// onset strength and beat clock. Its swing is dependable; push, humanity and syncopation read the
// live clock's own wobble as well, so they are rougher than the offline values.

import type { GrooveStats, GrooveTrack, Section } from '../types';
import { percentiles } from './dsp';

/** Beats either side of a beat in the local grid fit and in the windowed estimates. */
const FIT_HALF = 4;
const WIN_HALF = 8;
/** Push of 1 = the backbeat this many seconds behind the strong beats. */
export const PUSH_SCALE = 0.04;
/** Onset spread (robust std, seconds) mapped to humanity 0 and 1. */
export const HUMAN_LO = 0.003;
export const HUMAN_HI = 0.02;

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

export const NEUTRAL_GROOVE: Readonly<GrooveStats> = { swing: 0, push: 0, humanity: 0, synco: 0 };

/** Off-beat position within its pair (0.5 straight .. 2/3 triplet) -> swing amount 0..1. */
export function swingOf(frac: number): number {
  return clamp((frac - 0.5) / (2 / 3 - 0.5), 0, 1);
}

/** Swing amount 0..1 -> the off-beat's position within its pair (0.5 .. 2/3). */
export function fracOf(swing: number): number {
  return 0.5 + clamp(swing, 0, 1.5) * (1 / 6);
}

export function humanityOf(spreadSec: number): number {
  return clamp((spreadSec - HUMAN_LO) / (HUMAN_HI - HUMAN_LO), 0, 1);
}

interface Onset {
  t: number; // seconds
  s: number; // strength, 0..1 (relative to the song's strong onsets)
}

/**
 * Local maxima of the envelope above an adaptive floor, timed with parabolic interpolation.
 * power (optional, linear frame power): scales each onset's strength by the loudness it reaches
 * (the log-flux envelope alone rates a quiet tick after silence as high as a loud hit).
 */
export function pickOnsets(env: Float32Array, frameRate: number, power?: ArrayLike<number>): Onset[] {
  const T = env.length;
  const out: Onset[] = [];
  if (T < 5) return out;
  const [p98] = percentiles(env, [0.98]);
  const top = Math.max(p98, 1e-9);
  const floor = 0.12 * top;
  const pw = power && power.length >= T ? power : null;
  const pTop = pw ? Math.max(percentiles(pw, [0.98])[0], 1e-20) : 1;
  const reach = Math.max(1, Math.round(0.04 * frameRate));
  for (let t = 2; t < T - 2; t++) {
    const b = env[t];
    if (b < floor) continue;
    if (!(b > env[t - 1] && b >= env[t + 1] && b >= env[t - 2] && b >= env[t + 2])) continue;
    const a = env[t - 1];
    const c = env[t + 1];
    const den = a - 2 * b + c;
    const d = den < 0 ? clamp((0.5 * (a - c)) / den, -0.5, 0.5) : 0;
    let s = Math.min(1, b / top);
    if (pw) {
      let pk = 0;
      for (let k = t; k <= Math.min(T - 1, t + reach); k++) if (pw[k] > pk) pk = pw[k];
      s *= Math.min(1, Math.sqrt(pk / pTop));
    }
    out.push({ t: (t + d) / frameRate, s });
  }
  return out;
}

/** Weighted median of values (weights > 0); NaN when empty. */
function wmedian(v: number[], w: number[]): number {
  const n = v.length;
  if (!n) return NaN;
  const idx = Array.from({ length: n }, (_, i) => i).sort((x, y) => v[x] - v[y]);
  let tot = 0;
  for (const x of w) tot += x;
  let acc = 0;
  for (const i of idx) {
    acc += w[i];
    if (acc >= tot / 2) return v[i];
  }
  return v[idx[n - 1]];
}

/** Robust spread (1.4826 x median absolute deviation from 0) of deviations. */
function spread(dev: number[]): number {
  if (dev.length < 3) return NaN;
  const a = dev.map(Math.abs).sort((x, y) => x - y);
  return 1.4826 * a[a.length >> 1];
}

/**
 * The 16th slot of an onset at beat phase ph (-0.125..0.875), allowing for swing: the off-8th
 * window reaches past triplet swing (0.4..0.72), the off-16ths sit either side of it; -1 between.
 */
export function slotOf(ph: number): number {
  if (ph > -0.1 && ph < 0.1) return 0;
  if (ph > 0.4 && ph < 0.72) return 2;
  if (ph > 0.17 && ph < 0.36) return 1;
  if (ph >= 0.72 && ph < 0.86) return 3;
  return -1;
}

/** Everything one beat contributes. */
interface BeatEvidence {
  t: number; // grid time of the beat
  P: number; // local beat period
  pos: number; // position in the bar (0 = downbeat), -1 unknown
  dev0: number; // on-beat onset deviation from the grid, seconds (NaN: none)
  f8: number; // off-8th position in the beat (NaN: none)
  w8: number;
  f16: number; // off-16th position within its 8th, averaged over the two pairs (NaN: none)
  w16: number;
  slots: number[]; // strongest onset strength near each 16th slot (straight positions)
  slotPh: number[]; // their phases in the beat
}

/**
 * The groove of a song from its onset envelope and tracked beats. `sections` (optional) gets one
 * summary each. Returns neutral tracks when there are too few beats to read a grid.
 */
export function analyzeGroove(
  env: Float32Array,
  frameRate: number,
  beats: ArrayLike<number>,
  downbeats: ArrayLike<number>,
  sections: Section[] = [],
  power?: ArrayLike<number>,
  align = false,
): GrooveTrack {
  const T = env.length;
  const track: GrooveTrack = {
    swing: new Float32Array(T),
    push: new Float32Array(T),
    humanity: new Float32Array(T),
    synco: new Float32Array(T),
    song: { ...NEUTRAL_GROOVE },
    sections: sections.map(() => ({ ...NEUTRAL_GROOVE })),
    sub: 8,
  };
  const nb = beats.length;
  if (nb < 2 * FIT_HALF + 2 || T < 8 || !(frameRate > 0)) return track;

  const onsets = pickOnsets(env, frameRate, power);
  if (onsets.length < 8) return track;

  // Live clocks: shift the grid to the circular mean phase of the strongest onsets.
  if (align) {
    const P0 = (beats[nb - 1] - beats[0]) / (nb - 1);
    let top = 0;
    for (const o of onsets) top = Math.max(top, o.s);
    let c = 0, sn = 0;
    let bi = 0;
    for (const o of onsets) {
      if (o.s < 0.5 * top) continue;
      while (bi + 1 < nb && beats[bi + 1] <= o.t) bi++;
      const ph = (o.t - beats[bi]) / P0;
      const w = o.s * o.s;
      c += w * Math.cos(2 * Math.PI * ph);
      sn += w * Math.sin(2 * Math.PI * ph);
    }
    const shift = c * c + sn * sn > 0 ? (Math.atan2(sn, c) / (2 * Math.PI)) * P0 : 0;
    const shifted = new Float64Array(nb);
    for (let i = 0; i < nb; i++) shifted[i] = beats[i] + shift;
    const sd = Float64Array.from(downbeats as ArrayLike<number>, (d) => d + shift);
    beats = shifted;
    downbeats = sd;
  }

  // Bar position of each beat (downbeats are a subset of the beats).
  const pos = new Int32Array(nb).fill(-1);
  {
    let di = 0;
    let last = -1;
    for (let k = 0; k < nb; k++) {
      while (di < downbeats.length && downbeats[di] < beats[k] - 1e-3) di++;
      if (di < downbeats.length && Math.abs(downbeats[di] - beats[k]) < 1e-3) last = k;
      if (last >= 0) pos[k] = (k - last) % 4;
    }
    // Beats before the first downbeat count back from it.
    const first = pos.indexOf(0);
    if (first > 0) for (let k = 0; k < first; k++) pos[k] = (((k - first) % 4) + 4) % 4;
  }

  // Per beat: local grid, then the onsets inside its span.
  const ev: BeatEvidence[] = [];
  let oc = 0;
  for (let k = 0; k < nb; k++) {
    const a = Math.max(0, Math.min(nb - 1 - 2 * FIT_HALF, k - FIT_HALF));
    const b = a + 2 * FIT_HALF;
    let sx = 0, sy = 0;
    for (let i = a; i <= b; i++) {
      sx += i;
      sy += beats[i];
    }
    const m = b - a + 1;
    sx /= m;
    sy /= m;
    let num = 0, den = 0;
    for (let i = a; i <= b; i++) {
      num += (i - sx) * (beats[i] - sy);
      den += (i - sx) * (i - sx);
    }
    const P = den > 0 ? num / den : 0;
    if (!(P > 0.2 && P < 2)) continue;
    const g = sy + P * (k - sx);
    const e: BeatEvidence = { t: g, P, pos: pos[k], dev0: NaN, f8: NaN, w8: 0, f16: NaN, w16: 0, slots: [0, 0, 0, 0], slotPh: [0, 0.25, 0.5, 0.75] };
    // Onsets in [g - P/8, g + 7P/8).
    while (oc < onsets.length && onsets[oc].t < g - P / 8) oc++;
    let best0 = 0, best8 = 0, best16a = 0, best16b = 0;
    let f16a = NaN, f16b = NaN;
    for (let j = oc; j < onsets.length && onsets[j].t < g + (7 * P) / 8; j++) {
      const o = onsets[j];
      const ph = (o.t - g) / P;
      const slot = slotOf(ph);
      if (slot >= 0 && o.s > e.slots[slot]) {
        e.slots[slot] = o.s;
        e.slotPh[slot] = ph;
      }
      if (ph > -0.1 && ph < 0.1 && o.s > best0) {
        best0 = o.s;
        e.dev0 = ph * P;
      } else if (ph > 0.4 && ph < 0.72 && o.s > best8) {
        best8 = o.s;
        e.f8 = ph;
      } else if (ph > 0.17 && ph < 0.36 && o.s > best16a) {
        best16a = o.s;
        f16a = ph * 2;
      } else if (ph > 0.72 && ph < 0.86 && o.s > best16b) {
        best16b = o.s;
        f16b = (ph - 0.5) * 2;
      }
    }
    e.w8 = best8;
    if (best16a > 0 || best16b > 0) {
      const wa = best16a, wb = best16b;
      e.f16 = ((Number.isNaN(f16a) ? 0 : f16a * wa) + (Number.isNaN(f16b) ? 0 : f16b * wb)) / (wa + wb);
      e.w16 = wa + wb;
    }
    ev.push(e);
  }
  if (ev.length < 4) return track;

  // Song-level subdivision: 16ths when the off-16ths carry comparable evidence.
  let s8 = 0, s16 = 0;
  for (const e of ev) {
    s8 += e.w8;
    s16 += e.w16 * 0.5;
  }
  const sub: 8 | 16 = s16 > 0.6 * s8 && s16 > 0 ? 16 : 8;
  track.sub = sub;

  /** Groove stats over evidence ev[a..b] (inclusive). */
  const stats = (a: number, b: number, fallback: GrooveStats | null): GrooveStats & { conf: number } => {
    const fv: number[] = [], fw: number[] = [];
    for (let i = a; i <= b; i++) {
      const e = ev[i];
      const f = sub === 16 ? e.f16 : e.f8;
      const w = sub === 16 ? e.w16 : e.w8;
      if (!Number.isNaN(f) && w > 0) {
        fv.push(f);
        fw.push(w);
      }
    }
    const conf = Math.min(1, fv.length / Math.max(3, (b - a + 1) * 0.4));
    const fMed = fv.length ? wmedian(fv, fw) : fallback ? fracOf(fallback.swing) : 0.5;
    const swing = fv.length >= 2 ? swingOf(fMed) : fallback?.swing ?? 0;

    // Deviations from the (swung) grid: on-beats (less their bar position's mean, so a steady
    // backbeat lean is not counted as looseness), and off-beats from their median position.
    const dev: number[] = [];
    const pm = [0, 0, 0, 0, 0], pn = [0, 0, 0, 0, 0];
    for (let i = a; i <= b; i++) {
      const e = ev[i];
      if (Number.isNaN(e.dev0)) continue;
      const k = e.pos >= 0 ? e.pos : 4;
      pm[k] += e.dev0;
      pn[k]++;
    }
    let lb = 0, nlb = 0, ls = 0, nls = 0;
    for (let i = a; i <= b; i++) {
      const e = ev[i];
      if (!Number.isNaN(e.dev0)) {
        const k = e.pos >= 0 ? e.pos : 4;
        dev.push(e.dev0 - (pn[k] >= 2 ? pm[k] / pn[k] : 0));
        if (e.pos === 1 || e.pos === 3) {
          lb += e.dev0;
          nlb++;
        } else if (e.pos === 0 || e.pos === 2) {
          ls += e.dev0;
          nls++;
        }
      }
      if (!Number.isNaN(e.f8) && sub === 8) dev.push((e.f8 - fMed) * e.P);
      if (!Number.isNaN(e.f16) && sub === 16) dev.push(((e.f16 - fMed) * e.P) / 2);
    }
    // Subtract the systematic backbeat lean before measuring the spread.
    const lean = nlb && nls ? lb / nlb - ls / nls : 0;
    const sp = spread(dev);
    const humanity = Number.isNaN(sp) ? fallback?.humanity ?? 0 : humanityOf(sp);
    const push = nlb >= 2 && nls >= 2 ? clamp(lean / PUSH_SCALE, -1, 1) : fallback?.push ?? 0;

    // Syncopation: weak-slot onsets whose next stronger slot is silent (read on the swung grid).
    let sn = 0, sd = 0;
    for (let i = a; i <= b; i++) {
      const o = ev[i].slots;
      const next0 = i + 1 < ev.length ? ev[i + 1].slots[0] : o[0];
      sn += Math.max(0, o[2] - next0) + 0.5 * Math.max(0, o[1] - o[2]) + 0.5 * Math.max(0, o[3] - next0);
      sd += o[0] + o[1] + o[2] + o[3];
    }
    const synco = sd > 1e-6 ? clamp((1.6 * sn) / sd, 0, 1) : fallback?.synco ?? 0;
    return { swing, push, humanity, synco, conf };
  };

  const songS = stats(0, ev.length - 1, null);
  track.song = { swing: songS.swing, push: songS.push, humanity: songS.humanity, synco: songS.synco };

  // Sections: evidence whose beat falls inside.
  sections.forEach((sec, si) => {
    let a = -1, b = -1;
    for (let i = 0; i < ev.length; i++) {
      if (ev[i].t >= sec.start && ev[i].t < sec.end) {
        if (a < 0) a = i;
        b = i;
      }
    }
    if (a >= 0 && b - a >= 3) {
      const s = stats(a, b, track.song);
      track.sections[si] = { swing: s.swing, push: s.push, humanity: s.humanity, synco: s.synco };
    } else track.sections[si] = { ...track.song };
  });

  // Per beat: windowed estimate (falling back to the song where evidence is thin).
  const per = ev.map((_e, i) => stats(Math.max(0, i - WIN_HALF), Math.min(ev.length - 1, i + WIN_HALF), track.song));
  // Frames: linear between beat times, held outside.
  let j = 0;
  for (let t = 0; t < T; t++) {
    const time = t / frameRate;
    while (j + 1 < ev.length && ev[j + 1].t <= time) j++;
    let x: GrooveStats, y: GrooveStats, f: number;
    if (time <= ev[0].t) {
      x = y = per[0];
      f = 0;
    } else if (j + 1 >= ev.length) {
      x = y = per[ev.length - 1];
      f = 0;
    } else {
      x = per[j];
      y = per[j + 1];
      f = clamp((time - ev[j].t) / Math.max(1e-6, ev[j + 1].t - ev[j].t), 0, 1);
    }
    track.swing[t] = x.swing + (y.swing - x.swing) * f;
    track.push[t] = x.push + (y.push - x.push) * f;
    track.humanity[t] = x.humanity + (y.humanity - x.humanity) * f;
    track.synco[t] = x.synco + (y.synco - x.synco) * f;
  }
  return track;
}

// ------------------------------------------------------------ realtime

/** Seconds of live history the realtime estimate reads, and how often it re-reads it. */
const RT_WINDOW = 12;
const RT_EVERY = 0.5;

/**
 * Running groove estimate for live input: the offline estimator run every half second over the
 * last 12 s of the live onset strength and beat clock, eased into the output. Feed it every
 * analysis frame with the onset strength (any scale), the beat clock (phase 0..1, period in
 * seconds, beat-in-bar or -1), the frame spacing and the clock's confidence (below 0.3 the
 * estimate holds). A live clock can sit a fraction of a beat off the hits, so the window's grid is
 * first re-aligned to the circular mean phase of its strongest onsets.
 */
export class GrooveTracker {
  readonly out: GrooveStats = { swing: 0, push: 0, humanity: 0, synco: 0 };
  private env = new Float32Array(0);
  private head = 0;
  private filled = 0;
  private time = 0;
  private prevPh = -1;
  private beats: number[] = [];
  private bars: number[] = [];
  private since = 0;
  private fr = 0;

  reset(): void {
    Object.assign(this.out, NEUTRAL_GROOVE);
    this.head = this.filled = this.time = this.since = 0;
    this.prevPh = -1;
    this.beats = [];
    this.bars = [];
  }

  push(onset: number, phase: number, period: number, beatInBar: number, dt: number, confidence = 1): GrooveStats {
    if (!(dt > 0)) return this.out;
    if (!this.env.length || Math.abs(1 / dt - this.fr) > 0.5) {
      this.fr = 1 / dt;
      this.env = new Float32Array(Math.ceil(RT_WINDOW * this.fr));
      this.head = this.filled = 0;
    }
    this.time += dt;
    this.env[this.head] = Number.isFinite(onset) && onset > 0 ? onset : 0;
    this.head = (this.head + 1) % this.env.length;
    this.filled = Math.min(this.env.length, this.filled + 1);
    // A beat of the clock: the time its phase wrapped.
    if (this.prevPh >= 0 && phase < this.prevPh - 0.5 && period > 0.2) {
      const tb = this.time - phase * period;
      this.beats.push(tb);
      if (beatInBar === 0) this.bars.push(tb);
    }
    this.prevPh = phase;
    const t0 = this.time - this.filled * dt;
    while (this.beats.length && this.beats[0] < t0) this.beats.shift();
    while (this.bars.length && this.bars[0] < t0) this.bars.shift();
    this.since += dt;
    if (this.since >= RT_EVERY && confidence > 0.3 && this.beats.length >= 12) {
      this.since = 0;
      const n = this.filled;
      const e = new Float32Array(n);
      const start = (this.head - n + this.env.length) % this.env.length;
      for (let i = 0; i < n; i++) e[i] = this.env[(start + i) % this.env.length];
      const g = analyzeGroove(e, this.fr, this.beats.map((b) => b - t0), this.bars.map((b) => b - t0), [], undefined, true);
      const k = 0.35;
      const o = this.out;
      o.swing += (g.song.swing - o.swing) * k;
      o.push += (g.song.push - o.push) * k;
      o.humanity += (g.song.humanity - o.humanity) * k;
      o.synco += (g.song.synco - o.synco) * k;
    }
    return this.out;
  }
}

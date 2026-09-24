// Writes the harmony map into MusicState once per frame: from the precomputed HarmonyTrack for
// songs (HarmonyCursor, with look-ahead to the next resolution) and from the realtime
// HarmonyTracker for live input (LiveHarmony). Both fill the same optional fields, so visuals
// do not care where the harmony came from.

import type { MusicState } from '../types';
import { NO_CHORD, keyFifths, tonnetzXY, type HarmonyTrack, type HarmonyTracker } from './harmony';

const CHORD_PULSE_TAU = 0.35;
const RESOLVE_PULSE_TAU = 0.7;
const MOD_PULSE_TAU = 1.2;
const TENSION_UP_TAU = 0.5; // tension creeps in
const TENSION_DOWN_TAU = 0.25; // and lets go faster
const WALK_TAU = 0.18; // the Tonnetz position glides to the new chord

const XY: [number, number] = [0, 0];

function decay(s: MusicState, dt: number): void {
  s.chordPulse = (s.chordPulse ?? 0) * Math.exp(-dt / CHORD_PULSE_TAU);
  s.resolvePulse = (s.resolvePulse ?? 0) * Math.exp(-dt / RESOLVE_PULSE_TAU);
  s.modulationPulse = (s.modulationPulse ?? 0) * Math.exp(-dt / MOD_PULSE_TAU);
}

function glide(s: MusicState, tension: number, x: number, y: number, dt: number, snap: boolean): void {
  if (snap) {
    s.tension = tension;
    s.tonnetzX = x;
    s.tonnetzY = y;
    return;
  }
  const t0 = s.tension ?? 0;
  s.tension = t0 + (tension - t0) * (1 - Math.exp(-dt / (tension > t0 ? TENSION_UP_TAU : TENSION_DOWN_TAU)));
  const k = 1 - Math.exp(-dt / WALK_TAU);
  s.tonnetzX = (s.tonnetzX ?? 0) + (x - (s.tonnetzX ?? 0)) * k;
  s.tonnetzY = (s.tonnetzY ?? 0) + (y - (s.tonnetzY ?? 0)) * k;
}

/** Zeroed harmony fields for a fresh MusicState. */
export function initHarmonyState(s: MusicState): void {
  s.chord = NO_CHORD;
  s.tension = 0;
  s.chordPulse = 0;
  s.resolvePulse = 0;
  s.modulationPulse = 0;
  s.tonnetzX = 0;
  s.tonnetzY = 0;
  s.chordMove = 0;
  s.keyWalk = 0;
}

/** Cursor over a precomputed harmony map (amortized O(1) during playback). */
export class HarmonyCursor {
  private seg = 0;
  private res = -1;
  private mod = -1;

  readonly track: HarmonyTrack;

  constructor(track: HarmonyTrack) {
    this.track = track;
  }

  /** `jumped`: a seek or the first frame (re-sync without firing events). */
  sample(s: MusicState, time: number, dt: number, jumped: boolean): void {
    const tr = this.track;
    decay(s, dt);
    const segs = tr.segments;
    if (!segs.length) {
      initHarmonyState(s);
      return;
    }
    let i = jumped ? 0 : this.seg;
    if (i >= segs.length || segs[i].start > time) i = 0;
    while (i + 1 < segs.length && segs[i + 1].start <= time) i++;
    const changed = !jumped && i !== this.seg;
    this.seg = i;
    const seg = segs[i];

    // Tension: this beat's value (it varies inside a chord with the dissonance).
    let cell = 0;
    {
      let lo = 0;
      let hi = tr.times.length - 1;
      while (lo <= hi) {
        const m = (lo + hi) >> 1;
        if (tr.times[m] <= time) {
          cell = m;
          lo = m + 1;
        } else hi = m - 1;
      }
    }
    const x = seg.chord < 0 ? (s.tonnetzX ?? 0) : seg.x;
    const y = seg.chord < 0 ? (s.tonnetzY ?? 0) : seg.y;
    glide(s, tr.tension[cell] ?? 0, x, y, dt, jumped);
    s.chord = seg.chord;
    if (changed && seg.chord >= 0) {
      s.chordPulse = 1;
      s.chordMove = seg.move;
    } else if (jumped) s.chordMove = seg.move;

    // Resolutions and modulations crossed since the last frame.
    const lastLE = (arr: { time: number }[]): number => {
      let k = -1;
      for (let q = 0; q < arr.length && arr[q].time <= time; q++) k = q;
      return k;
    };
    const ri = lastLE(tr.resolutions);
    if (!jumped && ri > this.res) s.resolvePulse = Math.max(s.resolvePulse ?? 0, tr.resolutions[ri].strength);
    this.res = ri;
    const mi = lastLE(tr.modulations);
    if (!jumped && mi > this.mod) s.modulationPulse = 1;
    this.mod = mi;
    let walk = 0;
    for (let q = 0; q <= mi; q++) walk += tr.modulations[q].fifths;
    s.keyWalk = walk;
    const next = ri + 1 < tr.resolutions.length ? tr.resolutions[ri + 1].time - time : Infinity;
    s.timeToResolve = next;
  }
}

/** Realtime-lite: follows a HarmonyTracker's counters. */
export class LiveHarmony {
  private changes = 0;
  private resolves = 0;
  private mods = 0;
  private synced = false;
  private walk = 0;
  private key = -1;

  sample(s: MusicState, trk: HarmonyTracker, tonic: number, mode: 'major' | 'minor', dt: number, events: boolean): void {
    decay(s, dt);
    const snap = !events || !this.synced;
    const c = trk.chord;
    if (c >= 0) tonnetzXY(c, tonic, XY);
    glide(s, trk.tension, c >= 0 ? XY[0] : (s.tonnetzX ?? 0), c >= 0 ? XY[1] : (s.tonnetzY ?? 0), dt, snap);
    s.chord = c;
    if (!snap) {
      if (trk.changes !== this.changes) {
        s.chordPulse = 1;
        s.chordMove = trk.lastMove;
      }
      if (trk.resolves !== this.resolves) s.resolvePulse = Math.max(s.resolvePulse ?? 0, trk.lastResolve);
      if (trk.modulations !== this.mods) s.modulationPulse = 1;
    }
    this.changes = trk.changes;
    this.resolves = trk.resolves;
    this.mods = trk.modulations;
    this.synced = true;
    const key = tonic + (mode === 'minor' ? 12 : 0);
    if (this.key >= 0 && key !== this.key) this.walk += keyFifths({ tonic: this.key % 12, mode: this.key >= 12 ? 'minor' : 'major' }, { tonic, mode });
    this.key = key;
    s.keyWalk = this.walk;
    s.timeToResolve = undefined;
  }

  reset(): void {
    this.synced = false;
  }
}

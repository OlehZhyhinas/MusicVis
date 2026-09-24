// Turns a precomputed AnalysisResult plus live analyser data into a MusicState
// once per animation frame. Cursor-based lookups (amortized O(1) during normal
// playback, binary search after jumps); no per-frame allocation.

import type { AnalysisResult, KeySegment, LiveAudioFrame, MusicState, Section, StemName } from '../types';
import { STEM_NAMES } from '../types';

const BEAT_TAU = 0.15;
const BAR_TAU = 0.3;
const DROP_TAU = 1.5;
const KEY_PULSE_TAU = 0.6;
const KEY_HUE_TAU = 0.5; // ~95% settled after 1.5 s
const BUILD_DECAY_TAU = 0.4;
const COMPLEXITY_TAU = 2.0; // extra smoothing of the precomputed complexity
const SEEK_THRESHOLD = 0.5; // seconds; larger jumps are treated as seeks

/** Index of the last element <= t, or -1. */
function lastLE(arr: ArrayLike<number>, t: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] <= t) {
      ans = m;
      lo = m + 1;
    } else hi = m - 1;
  }
  return ans;
}

/** Advance or search a cursor so that arr[c] <= t < arr[c+1]. */
function moveCursor(arr: ArrayLike<number>, c: number, t: number): number {
  const n = arr.length;
  if (c >= -1 && c < n && (c < 0 || arr[c] <= t)) {
    let k = c;
    let steps = 0;
    while (k + 1 < n && arr[k + 1] <= t && steps < 8) {
      k++;
      steps++;
    }
    if (k + 1 >= n || arr[k + 1] > t) return k;
  }
  return lastLE(arr, t);
}

function segIndex(segs: { start: number }[], c: number, t: number): number {
  const n = segs.length;
  if (n === 0) return 0;
  if (c >= 0 && c < n && segs[c].start <= t && (c + 1 >= n || segs[c + 1].start > t)) return c;
  if (c >= 0 && c + 1 < n && segs[c + 1].start <= t && (c + 2 >= n || segs[c + 2].start > t)) return c + 1;
  let lo = 0;
  let hi = n - 1;
  let ans = 0;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (segs[m].start <= t) {
      ans = m;
      lo = m + 1;
    } else hi = m - 1;
  }
  return ans;
}

function keyHueOf(k: KeySegment): number {
  const major = k.mode === 'minor' ? (k.tonic + 3) % 12 : k.tonic;
  return ((major * 7) % 12) / 12;
}

function wrap01(x: number): number {
  return x - Math.floor(x);
}

export class TimelineSampler {
  private readonly r: AnalysisResult;
  private readonly state: MusicState;
  private readonly beatPeriod: number;
  private readonly dbBeatIdx: Int32Array; // beat index of each downbeat
  private readonly sections: Section[];
  private readonly keys: KeySegment[];
  /** Start times of the sections that fire dropPulse (drops, and choruses right after a build). */
  private readonly dropStarts: number[];

  private synced = false;
  private lastTime = 0;
  private beatCur = -1;
  private barCur = -1;
  private secCur = 0;
  private keyCur = 0;

  constructor(result: AnalysisResult) {
    this.r = result;
    const bpm = result.bpm > 0 && Number.isFinite(result.bpm) ? result.bpm : 120;
    this.beatPeriod = 60 / bpm;
    const beats = result.beats;
    this.dbBeatIdx = new Int32Array(result.downbeats.length);
    for (let i = 0; i < result.downbeats.length; i++) {
      const d = result.downbeats[i];
      let k = lastLE(beats, d + 1e-4);
      if (k < 0) k = 0;
      this.dbBeatIdx[i] = k;
    }
    const dur = Math.max(result.duration, 1e-3);
    this.sections = result.sections.length > 0 ? result.sections : [{ start: 0, end: dur, label: 'verse', energy: 0 }];
    this.dropStarts = [];
    for (let i = 1; i < this.sections.length; i++) {
      const sec = this.sections[i];
      if (sec.label === 'drop' || (sec.label === 'chorus' && this.sections[i - 1].label === 'build')) this.dropStarts.push(sec.start);
    }
    this.keys = result.keys.length > 0 ? result.keys : [{ start: 0, end: dur, tonic: 0, mode: 'major', confidence: 0 }];
    const zeroStems = (): Record<StemName, number> => ({ drums: 0, bass: 0, vocals: 0, other: 0 });
    this.state = {
      time: 0,
      dt: 0,
      playing: false,
      bass: 0,
      mid: 0,
      treb: 0,
      bassAtt: 0,
      midAtt: 0,
      trebAtt: 0,
      waveform: new Float32Array(1024),
      spectrum: new Float32Array(512),
      bpm,
      beatIndex: -1,
      barIndex: -1,
      beatPhase: 0,
      barPhase: 0,
      beatPulse: 0,
      barPulse: 0,
      onBeat: false,
      onBar: false,
      stems: zeroStems(),
      stemOnsets: zeroStems(),
      stemPresence: zeroStems(),
      loudness: 0,
      complexity: 0,
      songComplexity: Number.isFinite(result.songComplexity) ? result.songComplexity : 0,
      chroma: new Float32Array(12),
      keyTonic: this.keys[0].tonic,
      keyMode: this.keys[0].mode,
      keyHue: keyHueOf(this.keys[0]),
      keyChangePulse: 0,
      section: this.sections[0],
      sectionIndex: 0,
      sectionProgress: 0,
      sectionChanged: false,
      dropPulse: 0,
      buildIntensity: 0,
      timeToDrop: Infinity,
      sinceDrop: Infinity,
      barSeconds: this.beatPeriod * (result.beatsPerBar > 0 ? result.beatsPerBar : 4),
    };
  }

  /** Call after seeking: the next sample re-syncs cursors without firing events. */
  reset(): void {
    this.synced = false;
    const s = this.state;
    s.beatPulse = 0;
    s.barPulse = 0;
    s.dropPulse = 0;
    s.keyChangePulse = 0;
    s.buildIntensity = 0;
  }

  sample(time: number, dt: number, playing: boolean, live: LiveAudioFrame): MusicState {
    const s = this.state;
    const r = this.r;
    if (!Number.isFinite(time)) time = 0;
    dt = Number.isFinite(dt) ? Math.max(0, Math.min(0.1, dt)) : 0;
    const jumped = !this.synced || time < this.lastTime - 1e-6 || time - this.lastTime > SEEK_THRESHOLD;
    const events = !jumped;
    s.time = time;
    s.dt = dt;
    s.playing = playing;

    // --- Live ---
    s.bass = live.bass;
    s.mid = live.mid;
    s.treb = live.treb;
    s.bassAtt = live.bassAtt;
    s.midAtt = live.midAtt;
    s.trebAtt = live.trebAtt;
    s.waveform = live.waveform;
    s.spectrum = live.spectrum;

    // --- Decays ---
    const kBeat = Math.exp(-dt / BEAT_TAU);
    const kBar = Math.exp(-dt / BAR_TAU);
    s.beatPulse *= kBeat;
    s.barPulse *= kBar;
    s.dropPulse *= Math.exp(-dt / DROP_TAU);
    s.keyChangePulse *= Math.exp(-dt / KEY_PULSE_TAU);

    // --- Beats ---
    const beats = r.beats;
    const nb = beats.length;
    const P = this.beatPeriod;
    const prevBeat = this.beatCur;
    const bi = jumped ? lastLE(beats, time) : moveCursor(beats, this.beatCur, time);
    this.beatCur = bi;
    s.onBeat = events && bi > prevBeat;
    if (s.onBeat) s.beatPulse = 1;
    s.beatIndex = bi;
    let beatPos: number; // continuous beat position in beat-index units
    if (nb === 0) {
      beatPos = time / P;
      s.beatPhase = wrap01(beatPos);
    } else if (bi < 0) {
      beatPos = (time - beats[0]) / P; // negative
      s.beatPhase = wrap01(beatPos);
    } else if (bi >= nb - 1) {
      const x = (time - beats[nb - 1]) / P;
      beatPos = nb - 1 + x;
      s.beatPhase = wrap01(x);
    } else {
      const a = beats[bi];
      const b = beats[bi + 1];
      const ph = b > a ? (time - a) / (b - a) : 0;
      s.beatPhase = ph < 0 ? 0 : ph >= 1 ? 0.999999 : ph;
      beatPos = bi + s.beatPhase;
    }

    // --- Bars ---
    const dbs = r.downbeats;
    const nd = dbs.length;
    const bpb = r.beatsPerBar > 0 ? r.beatsPerBar : 4;
    const prevBar = this.barCur;
    const di = jumped ? lastLE(dbs, time) : moveCursor(dbs, this.barCur, time);
    this.barCur = di;
    s.onBar = events && di > prevBar;
    if (s.onBar) s.barPulse = 1;
    s.barIndex = di;
    let barPhase: number;
    if (nd === 0) {
      barPhase = wrap01(beatPos / bpb);
    } else if (di < 0) {
      const rel = beatPos - this.dbBeatIdx[0];
      barPhase = wrap01(rel / bpb);
    } else {
      const k = this.dbBeatIdx[di];
      const len = di + 1 < nd ? Math.max(1, this.dbBeatIdx[di + 1] - k) : bpb;
      const rel = (beatPos - k) / len;
      barPhase = di + 1 < nd ? Math.min(0.999999, Math.max(0, rel)) : wrap01(rel);
    }
    s.barPhase = barPhase;

    // --- Envelopes ---
    const fpos = time * r.frameRate;
    const T = r.numFrames;
    let i0 = Math.floor(fpos);
    let f = fpos - i0;
    if (i0 < 0) {
      i0 = 0;
      f = 0;
    }
    if (i0 >= T - 1) {
      i0 = Math.max(0, T - 1);
      f = 0;
    }
    const i1 = Math.min(T - 1, i0 + 1);
    for (let k = 0; k < STEM_NAMES.length; k++) {
      const name = STEM_NAMES[k];
      const e = r.stems[name];
      const o = r.stemOnsets[name];
      const pr = r.stemPresence?.[name];
      s.stems[name] = T > 0 ? e[i0] + (e[i1] - e[i0]) * f : 0;
      s.stemOnsets[name] = T > 0 ? o[i0] + (o[i1] - o[i0]) * f : 0;
      s.stemPresence[name] = T > 0 && pr && pr.length >= T ? pr[i0] + (pr[i1] - pr[i0]) * f : 0;
    }
    {
      const c = r.complexity;
      const target = T > 0 && c && c.length >= T ? c[i0] + (c[i1] - c[i0]) * f : 0;
      // Snap after seeks / on the first frame, otherwise follow over ~2 s.
      s.complexity = jumped ? target : s.complexity + (target - s.complexity) * (1 - Math.exp(-dt / COMPLEXITY_TAU));
    }
    s.loudness = T > 0 ? r.loudness[i0] + (r.loudness[i1] - r.loudness[i0]) * f : 0;
    if (T > 0) {
      const c = r.chroma;
      for (let k = 0; k < 12; k++) s.chroma[k] = c[i0 * 12 + k] + (c[i1 * 12 + k] - c[i0 * 12 + k]) * f;
    }

    // --- Key ---
    const prevKey = this.keyCur;
    const ki = segIndex(this.keys, jumped ? -1 : this.keyCur, time);
    this.keyCur = ki;
    const key = this.keys[ki];
    s.keyTonic = key.tonic;
    s.keyMode = key.mode;
    const target = keyHueOf(key);
    if (jumped) {
      s.keyHue = target;
    } else {
      if (ki !== prevKey) s.keyChangePulse = 1;
      let d = target - s.keyHue;
      d -= Math.round(d); // shortest way round, in [-0.5, 0.5]
      s.keyHue = wrap01(s.keyHue + d * (1 - Math.exp(-dt / KEY_HUE_TAU)));
    }

    // --- Sections ---
    const prevSec = this.secCur;
    const si = segIndex(this.sections, jumped ? -1 : this.secCur, time);
    this.secCur = si;
    const sec = this.sections[si];
    s.section = sec;
    s.sectionIndex = si;
    const span = sec.end - sec.start;
    const prog = span > 0 ? (time - sec.start) / span : 0;
    s.sectionProgress = prog < 0 ? 0 : prog > 1 ? 1 : prog;
    s.sectionChanged = events && si !== prevSec;
    if (s.sectionChanged) {
      const prevLabel = this.sections[prevSec]?.label;
      if (sec.label === 'drop' || (sec.label === 'chorus' && prevLabel === 'build')) s.dropPulse = 1;
    }
    if (sec.label === 'build') {
      const p = s.sectionProgress;
      s.buildIntensity = p * p;
    } else {
      s.buildIntensity = jumped ? 0 : s.buildIntensity * Math.exp(-dt / BUILD_DECAY_TAU);
    }

    // --- Look-ahead: the next and the last drop ---
    s.timeToDrop = Infinity;
    s.sinceDrop = Infinity;
    for (const d of this.dropStarts) {
      if (d > time) {
        s.timeToDrop = d - time;
        break;
      }
      s.sinceDrop = time - d;
    }

    this.synced = true;
    this.lastTime = time;
    return s;
  }
}

// Turns a precomputed AnalysisResult plus live analyser data into a MusicState
// once per animation frame. Cursor-based lookups (amortized O(1) during normal
// playback, binary search after jumps); no per-frame allocation.

import { neutralNotes, sampleNoteTrack } from './notes';
import type { AnalysisResult, KeySegment, LiveAudioFrame, MusicState, Section, SectionRepeat, StemName } from '../types';
import { STEM_NAMES } from '../types';
import { detectRepeats } from './repetition';
import { analyzeHarmony } from './harmony';
import { HarmonyCursor, initHarmonyState } from './harmonyState';
import { hookTimeline, hooksOf, sampleHook, type HookSample, type HookSpan } from './hooks';

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

const z = () => ({ bright: 0, noise: 0, rough: 0, attack: 0 });

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
  /** Repetition per section (from the result, or detected here for results analysed without it). */
  private readonly repeats: SectionRepeat[];

  private synced = false;
  private lastTime = 0;
  private beatCur = -1;
  private barCur = -1;
  private secCur = 0;
  private keyCur = 0;
  /** The harmony map (chords on the Tonnetz), computed once per song. */
  private readonly harmony: HarmonyCursor | null;
  /** Hook repeats in time order (from the result, or found here for results analysed without them). */
  private readonly hookSpans: HookSpan[];
  private readonly hookStarts: Float64Array;
  private hookCur = -1;
  private readonly hookOut: HookSample = { on: 0, phase: 0, pulse: 0, notePulse: 0, note: -1, hook: -1, index: -1 };

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
    const rep = result.repeats && result.repeats.length === this.sections.length ? result.repeats : null;
    this.repeats = rep ?? (result.sections.length > 0 ? detectRepeats(result) : [{ group: 0, of: -1, n: 0, sim: 0, returnSim: 0 }]);
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
    initHarmonyState(this.state);
    let harmony: HarmonyCursor | null = null;
    try {
      if (result.chroma && result.numFrames > 0) harmony = new HarmonyCursor(analyzeHarmony(result));
    } catch {
      harmony = null;
    }
    this.harmony = harmony;
    let spans: HookSpan[] = [];
    try {
      spans = hookTimeline(hooksOf(result));
    } catch {
      spans = [];
    }
    this.hookSpans = spans;
    this.hookStarts = Float64Array.from(spans, (x) => x.start);
    if (spans.length) Object.assign(this.state, { hookOn: 0, hookPhase: 0, hookPulse: 0, hookNotePulse: 0, hookNote: -1, hookId: -1 });
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
    s.chordPulse = 0;
    s.resolvePulse = 0;
    s.modulationPulse = 0;
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
    s.prevSectionLabel = si > 0 ? this.sections[si - 1].label : undefined;
    s.sectionChanged = events && si !== prevSec;
    const rp = this.repeats[si];
    if (rp) {
      s.repeatGroup = rp.group;
      s.repeatOf = rp.of;
      s.repeatIndex = rp.n;
      s.repeatSim = rp.sim;
      s.repeatReturnSim = rp.returnSim;
    }
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

    // --- Hooks ---
    if (this.hookSpans.length) {
      const hi = jumped ? lastLE(this.hookStarts, time) : moveCursor(this.hookStarts, this.hookCur, time);
      this.hookCur = hi;
      const h = sampleHook(hi >= 0 ? this.hookSpans[hi] : undefined, time, this.hookOut);
      s.hookOn = h.on;
      s.hookPhase = h.phase;
      s.hookPulse = h.pulse;
      s.hookNotePulse = h.notePulse;
      s.hookNote = h.note;
      s.hookId = h.hook;
    }

    // --- Harmony map ---
    this.harmony?.sample(s, time, dt, jumped);
    // --- Groove ---
    const gr = r.groove;
    if (gr && T > 0 && gr.swing.length >= T) {
      const g = (s.groove ??= { swing: 0, push: 0, humanity: 0, synco: 0 });
      g.swing = gr.swing[i0] + (gr.swing[i1] - gr.swing[i0]) * f;
      g.push = gr.push[i0] + (gr.push[i1] - gr.push[i0]) * f;
      g.humanity = gr.humanity[i0] + (gr.humanity[i1] - gr.humanity[i0]) * f;
      g.synco = gr.synco[i0] + (gr.synco[i1] - gr.synco[i0]) * f;
    }

    // --- Timbre ---
    const tt = r.timbre;
    if (tt && T > 0 && tt.mix.bright.length >= T) {
      const out = (s.timbre ??= { mix: z(), drums: z(), bass: z(), vocals: z(), other: z() });
      for (const key of ['mix', 'drums', 'bass', 'vocals', 'other'] as const) {
        const a = tt[key];
        const o = out[key];
        o.bright = a.bright[i0] + (a.bright[i1] - a.bright[i0]) * f;
        o.noise = a.noise[i0] + (a.noise[i1] - a.noise[i0]) * f;
        o.rough = a.rough[i0] + (a.rough[i1] - a.rough[i0]) * f;
        o.attack = a.attack[i0] + (a.attack[i1] - a.attack[i0]) * f;
      }
    }

    // --- Articulation (melody notes) ---
    if (r.notes && T > 0 && r.notes.on.length >= T) sampleNoteTrack(r.notes, r.frameRate, time, (s.notes ??= neutralNotes()));

    this.synced = true;
    this.lastTime = time;
    return s;
  }
}

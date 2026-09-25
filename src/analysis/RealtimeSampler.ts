// MusicState from live input: same interface as TimelineSampler, backed by a
// RealtimeAnalyzer instead of a precomputed AnalysisResult, so the
// visualizers need no changes.
//
// `time` passed to sample() is the analyzer's stream time (seconds of input
// consumed), extrapolated to the moment of rendering by the caller. The beat
// clock is extrapolated to that time; events (onBeat, onBar, sectionChanged)
// fire once per crossing and never twice, even when the PLL nudges the clock.

import type { LiveAudioFrame, MusicState, Section, StemName } from '../types';
import { STEM_NAMES } from '../types';
import type { RealtimeAnalyzer } from './RealtimeAnalyzer';
import { LiveHarmony, initHarmonyState } from './harmonyState';

const BEAT_TAU = 0.15;
const BAR_TAU = 0.3;
const DROP_TAU = 1.5;
const KEY_PULSE_TAU = 0.6;
const KEY_HUE_TAU = 0.5;
const COMPLEXITY_TAU = 1.0;
const ENV_TAU = 0.03;
/** Render slightly ahead of the analysis so pulses land on the audible beat (window latency). */
const LOOKAHEAD_S = 0.02;

function wrap01(x: number): number {
  return x - Math.floor(x);
}

function keyHueOf(tonic: number, mode: 'major' | 'minor'): number {
  const major = mode === 'minor' ? (tonic + 3) % 12 : tonic;
  return ((major * 7) % 12) / 12;
}

export class RealtimeSampler {
  private readonly a: RealtimeAnalyzer;
  private readonly state: MusicState;
  private synced = false;
  private beatIdx = -1;
  private barIdx = -1;
  private lastBeatFloor = -Infinity;
  private seenChanges = 0;
  private seenKey = -1;
  private readonly harmony = new LiveHarmony();

  constructor(analyzer: RealtimeAnalyzer) {
    this.a = analyzer;
    const zero = (): Record<StemName, number> => ({ drums: 0, bass: 0, vocals: 0, other: 0 });
    const section: Section = { start: 0, end: 30, label: 'intro', energy: 0 };
    this.state = {
      time: 0,
      dt: 0,
      playing: true,
      bass: 0,
      mid: 0,
      treb: 0,
      bassAtt: 0,
      midAtt: 0,
      trebAtt: 0,
      waveform: new Float32Array(1024),
      spectrum: new Float32Array(512),
      bpm: 120,
      beatIndex: -1,
      barIndex: -1,
      beatPhase: 0,
      barPhase: 0,
      beatPulse: 0,
      barPulse: 0,
      onBeat: false,
      onBar: false,
      stems: zero(),
      stemOnsets: zero(),
      stemPresence: zero(),
      loudness: 0,
      complexity: 0,
      songComplexity: analyzer.songComplexity,
      chroma: new Float32Array(12),
      keyTonic: 0,
      keyMode: 'major',
      keyHue: 0,
      keyChangePulse: 0,
      section,
      sectionIndex: 0,
      sectionProgress: 0,
      sectionChanged: false,
      dropPulse: 0,
      buildIntensity: 0,
      beatConfidence: 0,
    };
    initHarmonyState(this.state);
  }

  /** Re-sync without firing events (e.g. after the input device changed). */
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
    const a = this.a;
    if (!Number.isFinite(time)) time = a.streamTime;
    dt = Number.isFinite(dt) ? Math.max(0, Math.min(0.1, dt)) : 0;
    const events = this.synced;
    s.time = time;
    s.dt = dt;
    s.playing = playing;

    // --- Live analyser (MilkDrop-style levels, waveform, spectrum) ---
    s.bass = live.bass;
    s.mid = live.mid;
    s.treb = live.treb;
    s.bassAtt = live.bassAtt;
    s.midAtt = live.midAtt;
    s.trebAtt = live.trebAtt;
    s.waveform = live.waveform;
    s.spectrum = live.spectrum;

    // --- Decays ---
    s.beatPulse *= Math.exp(-dt / BEAT_TAU);
    s.barPulse *= Math.exp(-dt / BAR_TAU);
    s.dropPulse *= Math.exp(-dt / DROP_TAU);
    s.keyChangePulse *= Math.exp(-dt / KEY_PULSE_TAU);

    // --- Beats: extrapolated PLL clock ---
    const bt = a.beat;
    s.bpm = bt.bpm;
    s.beatConfidence = bt.confidence;
    const pos = bt.positionAt(time + LOOKAHEAD_S);
    const fl = Math.floor(pos);
    s.onBeat = false;
    s.onBar = false;
    if (!events) {
      this.lastBeatFloor = fl;
    } else if (fl > this.lastBeatFloor) {
      // One event per frame at most, even if the clock jumped over several beats.
      this.lastBeatFloor = fl;
      this.beatIdx++;
      const quiet = a.gate < 0.5;
      if (!quiet) {
        s.onBeat = true;
        // Before lock the pulse is softer (graceful, low confidence).
        s.beatPulse = Math.max(s.beatPulse, 0.35 + 0.65 * bt.confidence);
      }
      const slot = bt.downbeatSlot;
      if ((((fl - slot) % 4) + 4) % 4 === 0) {
        this.barIdx++;
        if (!quiet) {
          s.onBar = true;
          s.barPulse = Math.max(s.barPulse, 0.35 + 0.65 * bt.confidence);
        }
      }
    }
    // A clock that moved back a little (PLL correction) does not re-fire.
    s.beatIndex = this.beatIdx;
    s.barIndex = this.barIdx;
    s.beatPhase = Math.min(0.999999, Math.max(0, pos - fl));
    s.barPhase = Math.min(0.999999, wrap01((pos - bt.downbeatSlot) / 4));

    // --- Envelopes (light smoothing of the ~86 Hz analysis frames) ---
    const kEnv = 1 - Math.exp(-dt / ENV_TAU);
    for (let k = 0; k < STEM_NAMES.length; k++) {
      const n = STEM_NAMES[k];
      s.stems[n] += (a.stems[n] - s.stems[n]) * kEnv;
      s.stemOnsets[n] = Math.max(a.stemOnsets[n], s.stemOnsets[n] * Math.exp(-dt / 0.08));
      s.stemPresence[n] += (a.stemPresence[n] - s.stemPresence[n]) * kEnv;
    }
    s.loudness += (a.loudness - s.loudness) * kEnv;
    s.complexity = events ? s.complexity + (a.complexity - s.complexity) * (1 - Math.exp(-dt / COMPLEXITY_TAU)) : a.complexity;
    s.songComplexity = a.songComplexity;
    s.chroma.set(a.chroma);

    // --- Key ---
    const kt = a.key;
    const key = kt.key;
    if (key >= 0) {
      s.keyTonic = kt.tonic;
      s.keyMode = kt.mode;
      const target = keyHueOf(kt.tonic, kt.mode);
      if (!events || this.seenKey < 0) s.keyHue = target;
      else {
        if (key !== this.seenKey) s.keyChangePulse = 1;
        let d = target - s.keyHue;
        d -= Math.round(d);
        s.keyHue = wrap01(s.keyHue + d * (1 - Math.exp(-dt / KEY_HUE_TAU)));
      }
    }
    this.seenKey = key;

    // --- Structure ---
    const st = a.structure;
    const sec = st.section;
    s.section = sec;
    s.sectionIndex = st.sectionIndex;
    const span = sec.end - sec.start;
    const prog = span > 0 ? (time - sec.start) / span : 0;
    s.sectionProgress = prog < 0 ? 0 : prog > 1 ? 1 : prog;
    s.sectionChanged = events && st.changes !== this.seenChanges;
    if (s.sectionChanged) {
      if (sec.label === 'drop' || (sec.label === 'chorus' && st.prevLabel === 'build')) s.dropPulse = 1;
    }
    this.seenChanges = st.changes;
    s.buildIntensity = st.buildIntensity;

    // --- Harmony map (realtime-lite) ---
    this.harmony.sample(s, a.harmony, s.keyTonic, s.keyMode, dt, events);
    // --- Groove (running estimate) ---
    const gr = a.groove.out;
    const gs = (s.groove ??= { swing: 0, push: 0, humanity: 0, synco: 0 });
    gs.swing = gr.swing;
    gs.push = gr.push;
    gs.humanity = gr.humanity;
    gs.synco = gr.synco;

    // --- Timbre (running estimate) ---
    const tl = a.timbreLive.out;
    const ts = (s.timbre ??= { mix: { ...tl.mix }, drums: { ...tl.drums }, bass: { ...tl.bass }, vocals: { ...tl.vocals }, other: { ...tl.other } });
    for (const key of ['mix', 'drums', 'bass', 'vocals', 'other'] as const) Object.assign(ts[key], tl[key]);

    // --- Articulation (running note tracker) ---
    const nl = a.notesLive.out;
    const ns = (s.notes ??= { ...nl, recent: [] });
    Object.assign(ns, nl, { recent: ns.recent });
    ns.recent.length = nl.recent.length;
    for (let i = 0; i < nl.recent.length; i++) ns.recent[i] = Object.assign(ns.recent[i] ?? {}, nl.recent[i]);

    this.synced = true;
    return s;
  }
}

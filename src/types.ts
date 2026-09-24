// Shared contracts between analysis, playback, rendering and UI.
// Everything under src/analysis produces AnalysisResult; the TimelineSampler
// turns it plus live analyser data into a MusicState once per animation frame;
// the renderer consumes MusicState only.

export type StemName = 'drums' | 'bass' | 'vocals' | 'other';
export const STEM_NAMES: StemName[] = ['drums', 'bass', 'vocals', 'other'];

export type SectionLabel =
  | 'intro'
  | 'verse'
  | 'build'
  | 'chorus'
  | 'drop'
  | 'breakdown'
  | 'outro';

export interface Section {
  start: number; // seconds
  end: number; // seconds
  label: SectionLabel;
  energy: number; // 0..1, mean loudness relative to the song
}

export interface KeySegment {
  start: number;
  end: number;
  tonic: number; // pitch class, 0 = C .. 11 = B
  mode: 'major' | 'minor';
  confidence: number; // 0..1
}

/** Produced once per song by the analysis worker. All arrays are transferable. */
export interface AnalysisResult {
  duration: number; // seconds
  frameRate: number; // feature frames per second (frame i covers time i / frameRate)
  numFrames: number;

  /** Per-stem energy envelopes, 0..1, normalized per stem over the song. */
  stems: Record<StemName, Float32Array>;
  /** Per-stem onset strength, 0..1 (transients of each stem). */
  stemOnsets: Record<StemName, Float32Array>;
  /**
   * Per-stem ABSOLUTE presence, 0..1: how much of the mix this stem actually is
   * right now. Unlike `stems` (normalized per stem over the song), a stem that is
   * absent or faint in the whole song stays near 0 here.
   */
  stemPresence: Record<StemName, Float32Array>;
  /**
   * Musical density per frame, 0..1, absolute across songs: number of stems
   * meaningfully present, onset rate, spectral spread and absolute loudness.
   * Solo instrument / simple melody ~0.1-0.25, full band ~0.5-0.7, dense EDM drop ~0.9.
   */
  complexity: Float32Array;
  /** Mean complexity over the song, 0..1. */
  songComplexity: number;

  /** Overall loudness envelope, 0..1. */
  loudness: Float32Array;
  /** numFrames * 12, each frame's chroma normalized to max 1. */
  chroma: Float32Array;

  bpm: number;
  beats: Float32Array; // beat times in seconds, ascending
  downbeats: Float32Array; // bar start times in seconds, ascending, subset of beats
  beatsPerBar: number;

  sections: Section[]; // contiguous, cover [0, duration]
  keys: KeySegment[]; // contiguous, cover [0, duration]
  /** Repetition structure, one entry per section (src/analysis/repetition.ts); absent in older results. */
  repeats?: SectionRepeat[];
}

/** How a section relates to the rest of the song (src/analysis/repetition.ts). */
export interface SectionRepeat {
  /** Repetition group: sections that are the same music share it (0, 1, 2... by first appearance). */
  group: number;
  /** Index of the group's first section, or -1 when this section is the first appearance. */
  of: number;
  /** Occurrence within the group: 0 the first appearance, 1 the first return... */
  n: number;
  /** 0..1 how closely this section repeats an earlier one (0 for a first appearance). */
  sim: number;
  /** 0..1 how closely the group comes back later (the best later return; 0 when it never does). */
  returnSim: number;
}

/** Messages to/from the analysis worker. */
export type AnalysisRequest = {
  type: 'analyze';
  sampleRate: number;
  left: Float32Array;
  right: Float32Array; // same as left for mono files
};

export type AnalysisResponse =
  | { type: 'progress'; stage: string; progress: number } // progress 0..1
  | { type: 'done'; result: AnalysisResult }
  | { type: 'error'; message: string };

/** Everything the renderer needs for one frame. */
export interface MusicState {
  time: number; // playback position, seconds
  dt: number; // seconds since last frame (clamped to <= 0.1)
  playing: boolean;

  // --- Live, from an AnalyserNode (MilkDrop-compatible semantics) ---
  /** ~1.0 = average for this song so far, >1 = louder than usual. */
  bass: number;
  mid: number;
  treb: number;
  bassAtt: number; // smoothed versions
  midAtt: number;
  trebAtt: number;
  /** Time-domain samples, -1..1, length 1024. */
  waveform: Float32Array;
  /** Magnitude spectrum, 0..1, length 512. */
  spectrum: Float32Array;

  // --- Rhythm, from precomputed beats ---
  bpm: number;
  beatIndex: number; // index of the most recent beat (-1 before first)
  barIndex: number;
  beatPhase: number; // 0..1 progress through the current beat
  barPhase: number; // 0..1 progress through the current bar (drives 1 turn/bar)
  beatPulse: number; // 1 at a beat, decays exponentially toward 0
  barPulse: number; // 1 at a downbeat, decays
  /** True only on the frame a beat / downbeat is crossed. */
  onBeat: boolean;
  onBar: boolean;
  /**
   * 0..1 confidence of the beat grid. Only set in live-input mode (low before
   * the real-time tracker has locked); precomputed songs leave it undefined.
   */
  beatConfidence?: number;

  // --- Stems, sampled from the precomputed envelopes at `time` ---
  stems: Record<StemName, number>; // 0..1
  stemOnsets: Record<StemName, number>; // 0..1
  /** Absolute presence per stem (see AnalysisResult.stemPresence). */
  stemPresence: Record<StemName, number>;
  loudness: number;
  /** Musical density now, 0..1, smoothed over ~2 s (see AnalysisResult.complexity). */
  complexity: number;
  /** Mean complexity of the whole song, 0..1. */
  songComplexity: number;

  // --- Harmony ---
  chroma: Float32Array; // length 12
  keyTonic: number;
  keyMode: 'major' | 'minor';
  /** 0..1 hue for the current key, circle-of-fifths ordered so related keys have near hues. Smoothed across changes. */
  keyHue: number;
  keyChangePulse: number; // 1 at a key change, decays

  // --- Structure ---
  section: Section;
  sectionIndex: number;
  sectionProgress: number; // 0..1 through the current section
  /** True only on the frame a new section begins. */
  sectionChanged: boolean;
  /** 1 at the start of a 'drop' or 'chorus' section after a build, decays. */
  dropPulse: number;
  /** 0..1, rises through 'build' sections toward the boundary. */
  buildIntensity: number;

  // --- Look-ahead (precomputed songs only; live input leaves these undefined) ---
  /** Seconds until the next drop starts (a section that fires dropPulse); Infinity when none is coming. */
  timeToDrop?: number;
  /** Seconds since the most recent drop started; Infinity before the first one. */
  sinceDrop?: number;
  /** Seconds per bar at the song's tempo. */
  barSeconds?: number;
  /** Type of the section before the current one (undefined at the start or when unknown). */
  prevSectionLabel?: SectionLabel;
  /** Repetition of the current section (src/analysis/repetition.ts): group, first section of the group (-1 when this is it), occurrence (0 = first), similarity to the earlier one, and how closely it comes back later. */
  repeatGroup?: number;
  repeatOf?: number;
  repeatIndex?: number;
  repeatSim?: number;
  repeatReturnSim?: number;
}

/** Produced every frame by src/audio/LiveAnalyser.ts from an AnalyserNode. */
export interface LiveAudioFrame {
  bass: number;
  mid: number;
  treb: number;
  bassAtt: number;
  midAtt: number;
  trebAtt: number;
  waveform: Float32Array; // length 1024
  spectrum: Float32Array; // length 512
}

/**
 * Module contracts (who implements what):
 *
 * src/analysis/analyzeWorker.ts   Web Worker: AnalysisRequest -> AnalysisResponse
 * src/analysis/analyze.ts         export function analyzeAudio(buf: AudioBuffer, onProgress): Promise<AnalysisResult>
 *                                 (spawns the worker via new URL('./analyzeWorker.ts', import.meta.url))
 * src/analysis/TimelineSampler.ts export class TimelineSampler {
 *                                   constructor(result: AnalysisResult)
 *                                   sample(time: number, dt: number, playing: boolean, live: LiveAudioFrame): MusicState
 *                                   reset(): void   // call after seeking
 *                                 }
 * src/audio/LiveAnalyser.ts       export class LiveAnalyser { constructor(ctx: AudioContext, input: AudioNode); read(dt: number): LiveAudioFrame }
 * src/audio/Player.ts             AudioBuffer playback with play/pause/seek/currentTime, exposes `output: AudioNode`
 * src/v2/engine.ts               WebGL2 renderer for V2 genomes
 * src/v2/main.ts                  UI + glue
 */

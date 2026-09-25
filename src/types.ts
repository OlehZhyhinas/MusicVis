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

/** Timing feel (src/analysis/groove.ts). */
export interface GrooveStats {
  /** 0 straight .. 1 triplet swing (where the off-beat of each pair lands). */
  swing: number;
  /** Backbeat lean against the strong beats, -1 pushing (early) .. 1 laid back (late); 1 = 40 ms. */
  push: number;
  /** 0 machine-quantized .. 1 loose human timing (~20 ms spread around the grid). */
  humanity: number;
  /** 0..1 syncopation density (weak-slot onsets whose next stronger slot is silent). */
  synco: number;
}

/** Timbre of one stem (or the mix) now (src/analysis/timbre.ts), all 0..1, absolute across songs. */
export interface TimbreStats {
  /** Spectral centroid in the stem's range: dull 0 .. brilliant 1. */
  bright: number;
  /** Spectral flatness: pure tone 0 .. noise 1 (breath, distortion, cymbals). */
  noise: number;
  /** Sensory roughness: beating partials (a sine 0, a buzzy saw or a detuned chord high). */
  rough: number;
  /** Onset sharpness: a pluck or hit near 1, a slow swell near 0 (held briefly, then relaxing). */
  attack: number;
}

/** Per-frame timbre tracks for the mix and each stem. */
export type TimbreTrack = Record<'mix' | StemName, { bright: Float32Array; noise: Float32Array; rough: Float32Array; attack: Float32Array }>;

/** One tracked melody note (src/analysis/notes.ts). Times in seconds, pitch in MIDI (continuous). */
export interface NoteEvent {
  start: number;
  end: number;
  /** Median pitch over the note. */
  pitch: number;
  /** Pitch contour while held, one value per analysis frame from `start`. */
  contour: Float32Array;
  /** Loudness of the note's harmonics, 0..1 within the song. */
  strength: number;
  /** How legato this note is played: 0 short and detached .. 1 held into the next, gliding or with vibrato. */
  legato: number;
  /** Voice-likeness of the note: 0 steady on-grid pitch (a synth) .. 1 vibrato, drift, scoops (a singer). */
  voice: number;
}

/** A recent note as a mark (newest last in NoteStats.recent). */
export interface NoteMark {
  /** Seconds since the note started. */
  age: number;
  /** Seconds it has been (or was) held. */
  len: number;
  /** True once the note has ended. */
  ended: boolean;
  /** Pitch height 0..1 at the note's start (within the song's melodic range). */
  height: number;
  strength: number;
}

/** Articulation now: the melody line's notes (src/analysis/notes.ts). */
export interface NoteStats {
  /** Note-on pulse: 1 at a note start (scaled by its strength), decaying over ~0.15 s. */
  on: number;
  /** Strength of the note being held, 0 between notes. */
  held: number;
  /** 0 staccato (short detached notes, "tu tu tu") .. 1 legato (held, tied, gliding, vibrato), around now. */
  legato: number;
  /** Pitch slope of the held note, semitones per second (signed, vibrato removed). */
  glide: number;
  /** Vibrato depth of the held note, semitones (peak). */
  vibrato: number;
  /** Pitch of the held (or last) note, MIDI, continuous. */
  pitch: number;
  /** Pitch height 0..1 within the song's melodic range. */
  height: number;
  /** Voice-likeness of the melody: 0 steady synth lead .. 1 sung voice (vibrato, pitch drift, scoops). */
  voice: number;
  /** Latest notes, newest last (at most NOTE_RECENT). */
  recent: NoteMark[];
}

/** Per-frame articulation tracks plus the note events. */
export interface NoteTrack {
  notes: NoteEvent[];
  on: Float32Array;
  held: Float32Array;
  legato: Float32Array;
  glide: Float32Array;
  vibrato: Float32Array;
  pitch: Float32Array;
  height: Float32Array;
  voice: Float32Array;
  /** The song's melodic range used for height (MIDI). */
  lo: number;
  hi: number;
}

/** Per-frame groove tracks plus song and section summaries. */
export interface GrooveTrack {
  swing: Float32Array;
  push: Float32Array;
  humanity: Float32Array;
  synco: Float32Array;
  song: GrooveStats;
  /** One per AnalysisResult.sections entry. */
  sections: GrooveStats[];
  /** The swung subdivision: 8th or 16th pairs. */
  sub: 8 | 16;
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
  /** Timing feel (optional: older cached results lack it). */
  groove?: GrooveTrack;
  /** Timbre per stem and for the mix (optional: older cached results lack it). */
  timbre?: TimbreTrack;
  /** Melody notes and articulation (optional: older cached results lack it). */
  notes?: NoteTrack;
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

  // --- Harmony map (src/analysis/harmony.ts; undefined when not analysed) ---
  /** Current chord: 0..11 major triad on that root, 12..23 minor, -1 no chord. */
  chord?: number;
  /** 0..1 harmonic tension: the chord's pull away from the tonic plus the chroma's dissonance, smoothed. */
  tension?: number;
  /** 1 on a chord change, decays. */
  chordPulse?: number;
  /** On a resolution to the tonic (V-I, IV-I, a return home): the tension released, 0..1, decays. */
  resolvePulse?: number;
  /** 1 on a modulation (key change), decays. */
  modulationPulse?: number;
  /** The chord's position on the Tonnetz relative to the tonic (x along fifths, y along thirds), gliding. */
  tonnetzX?: number;
  tonnetzY?: number;
  /** How the last chord change moved (harmony.ts MOVE: 1 fifth, 2 third, 3 step, 4 tritone, 5 mode). */
  chordMove?: number;
  /** Signed circle-of-fifths steps the key has travelled from the song's first key. */
  keyWalk?: number;
  /** Seconds until the next resolution (precomputed songs only). */
  timeToResolve?: number;

  // --- Groove (timing feel; songs from the offline groove track, live input from a running estimate) ---
  groove?: GrooveStats;

  // --- Timbre (per stem and the mix; songs from the offline track, live input from the running estimate) ---
  timbre?: Record<'mix' | StemName, TimbreStats>;

  // --- Articulation (melody notes; songs from the offline note track, live input from a running tracker) ---
  notes?: NoteStats;

  // --- Lyrics (src/lyrics/sampler.ts; undefined when the song has no lyrics or for live input) ---
  /** The line being sung ('' between lines), and the next one. */
  lyricLine?: string;
  lyricNext?: string;
  /** Index of the line being sung, -1 between lines. */
  lyricIndex?: number;
  /** 0..1 through the current line. */
  lyricProgress?: number;
  /** What the words are about: imagery tags 0..1 in src/lyrics/lexicon.ts LYRIC_TAGS order, eased. */
  lyricTags?: Float32Array;
  /** 0 sad .. 1 happy, and 0 calm .. 1 intense, of the words (0.5 neutral), eased. */
  lyricValence?: number;
  lyricArousal?: number;
  /** 1 when a new line starts, decays. */
  lyricPulse?: number;
  /** 1 while lines are being sung, fading over a few seconds between them. */
  lyricPresence?: number;
  /** true: line times come from synced lyrics; false: plain lyrics spread over the vocals. */
  lyricSynced?: boolean;
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

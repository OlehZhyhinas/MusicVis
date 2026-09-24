// Harmony map: the chord progression of a song, placed on the Tonnetz.
//
// Offline (analyzeHarmony): chroma averaged over each beat (loudness weighted) is matched against
// 24 triad templates (12 major, 12 minor) plus a no-chord state, and a Viterbi pass over the beats
// smooths the labels (chords tend to last; chords of the local key are a little more likely).
// Realtime (HarmonyTracker): the same templates on a short exponential chroma window, with
// hysteresis instead of Viterbi.
//
// Every chord sits on the Tonnetz, the tonal lattice where a step right is a perfect fifth and a
// step up-right a major third (a step up-left is then a minor third). A triad is a triangle of
// three neighbouring notes: major triads point up, minor triads point down. From the chord and the
// key the map derives
//   tension     how far the harmony has wandered from home: the chord's function in the key
//               (tonic 0, dominant high, chromatic higher), its distance from the tonic triad on
//               the lattice, and the roughness of the chroma (energy outside the chord's notes),
//   motion      what kind of move each chord change is (a fifth, a third, a step, a tritone, or the
//               same root switching mode),
//   resolutions arrivals on the tonic triad after tension (V-I, IV-I, or any return home),
//   modulations key changes, with their direction around the circle of fifths.
//
// Pure functions and small classes, no allocation per frame in the hot paths.

import type { KeySegment } from '../types';

/** A chord index: 0..11 major triads on that root, 12..23 minor triads, -1 no chord. */
export type ChordIndex = number;
export const NO_CHORD = -1;
export const NUM_CHORDS = 24;

const NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

export const chordRoot = (c: ChordIndex): number => (c < 0 ? 0 : c % 12);
export const chordIsMinor = (c: ChordIndex): boolean => c >= 12;
export const makeChord = (root: number, minor: boolean): ChordIndex => (((root % 12) + 12) % 12) + (minor ? 12 : 0);

/** 'C', 'F#m', 'N'. */
export function chordName(c: ChordIndex): string {
  if (c < 0) return 'N';
  return NAMES[c % 12] + (c >= 12 ? 'm' : '');
}

const mod12 = (x: number) => ((x % 12) + 12) % 12;

// ------------------------------------------------------------ templates

/** Chord tones of each chord (root, third, fifth). */
const TONES: number[][] = [];
/** Mean-centred, unit-norm templates, 24 x 12. */
const TEMPLATES = new Float64Array(NUM_CHORDS * 12);
for (let c = 0; c < NUM_CHORDS; c++) {
  const r = c % 12;
  const tones = [r, mod12(r + (c >= 12 ? 3 : 4)), mod12(r + 7)];
  TONES.push(tones);
  const t = new Float64Array(12);
  // Root a little stronger than the fifth and third (it is usually doubled, often in the bass).
  t[tones[0]] = 1.15;
  t[tones[1]] = 1;
  t[tones[2]] = 0.95;
  let mean = 0;
  for (let k = 0; k < 12; k++) mean += t[k] / 12;
  let n = 0;
  for (let k = 0; k < 12; k++) {
    t[k] -= mean;
    n += t[k] * t[k];
  }
  n = Math.sqrt(n);
  for (let k = 0; k < 12; k++) TEMPLATES[c * 12 + k] = t[k] / n;
}

/** Pitch classes of a chord's triad. */
export function chordTones(c: ChordIndex): readonly number[] {
  return c < 0 ? [] : TONES[c];
}

/**
 * Correlation of a chroma vector with every triad template (out[24], each -1..1); returns the
 * chroma's total energy (0 means silence: all scores are 0).
 */
export function chordScores(ch: ArrayLike<number>, out: Float64Array | number[], o = 0): number {
  let mean = 0;
  let tot = 0;
  for (let k = 0; k < 12; k++) {
    mean += ch[o + k] / 12;
    tot += ch[o + k];
  }
  let n = 0;
  for (let k = 0; k < 12; k++) {
    const d = ch[o + k] - mean;
    n += d * d;
  }
  n = Math.sqrt(n);
  for (let c = 0; c < NUM_CHORDS; c++) {
    if (n < 1e-9) {
      out[c] = 0;
      continue;
    }
    let s = 0;
    for (let k = 0; k < 12; k++) s += TEMPLATES[c * 12 + k] * (ch[o + k] - mean);
    out[c] = s / n;
  }
  return tot;
}

/** Share of the chroma's energy outside the chord's three notes (0 = pure triad, ~0.75 = flat noise). */
export function roughness(ch: ArrayLike<number>, c: ChordIndex, o = 0): number {
  let tot = 0;
  let inside = 0;
  for (let k = 0; k < 12; k++) tot += ch[o + k] * ch[o + k];
  if (tot < 1e-12) return 0;
  if (c < 0) return 0.5;
  for (const k of TONES[c]) inside += ch[o + k] * ch[o + k];
  // Semitone clashes are the harshest part: count energy next to a chord tone a little extra.
  return Math.min(1, Math.max(0, 1 - inside / tot));
}

// ------------------------------------------------------------- Tonnetz

/**
 * Lattice coordinates (fifths, major thirds) of an interval: the shortest way to reach it with
 * fifth and major-third steps (7f + 4t = interval mod 12), fifths preferred on ties.
 */
const LATTICE: [number, number][] = [];
for (let iv = 0; iv < 12; iv++) {
  let best: [number, number] = [0, 0];
  let bestCost = Infinity;
  for (let t = -1; t <= 1; t++) {
    for (let f = -6; f <= 6; f++) {
      if (mod12(7 * f + 4 * t) !== iv) continue;
      // Lattice steps: fifth = 1, major third = 1, and a minor third (f+1, t-1) also 1.
      const cost = f * t < 0 ? Math.max(Math.abs(f), Math.abs(t)) : Math.abs(f) + Math.abs(t);
      if (cost < bestCost - 1e-9 || (Math.abs(cost - bestCost) < 1e-9 && Math.abs(t) < Math.abs(best[1]))) {
        best = [f, t];
        bestCost = cost;
      }
    }
  }
  LATTICE.push(best);
}

/** (fifths, thirds) lattice coordinates of a pitch class relative to a tonic. */
export function latticeOf(pc: number, tonic = 0): readonly [number, number] {
  return LATTICE[mod12(pc - tonic)];
}

/** Lattice steps between two pitch classes (a fifth, a major or a minor third is one step). */
export function latticeSteps(a: number, b: number): number {
  const [f, t] = LATTICE[mod12(b - a)];
  return f * t < 0 ? Math.max(Math.abs(f), Math.abs(t)) : Math.abs(f) + Math.abs(t);
}

/**
 * The chord's triangle centre on the plane, relative to the tonic (x along fifths, y up; a unit
 * lattice step is 1). Major triads (root, +fifth, +major third) point up, minor triads (root,
 * +fifth, +minor third) point down. The tonic major triad's centre is at (0.5, 0.29).
 */
export function tonnetzXY(c: ChordIndex, tonic: number, out: [number, number] = [0, 0]): [number, number] {
  if (c < 0) {
    out[0] = out[1] = 0;
    return out;
  }
  const [f, t] = LATTICE[mod12(chordRoot(c) - tonic)];
  // Triangle centroid in lattice coordinates: major (f + 1/3, t + 1/3), minor (f + 2/3, t - 1/3).
  const cf = f + (c >= 12 ? 2 / 3 : 1 / 3);
  const ct = t + (c >= 12 ? -1 / 3 : 1 / 3);
  out[0] = cf + 0.5 * ct;
  out[1] = 0.8660254 * ct;
  return out;
}

/** Triad adjacency on the Tonnetz (P, L, R: two shared notes), as graph distances 24 x 24. */
const PLR = new Uint8Array(NUM_CHORDS * NUM_CHORDS).fill(255);
{
  const nb = (c: number): number[] => {
    const r = c % 12;
    return c < 12
      ? [makeChord(r, true), makeChord(r + 9, true), makeChord(r + 4, true)] // P, R, L
      : [makeChord(r, false), makeChord(r + 3, false), makeChord(r + 8, false)];
  };
  for (let s = 0; s < NUM_CHORDS; s++) {
    const q = [s];
    PLR[s * NUM_CHORDS + s] = 0;
    for (let h = 0; h < q.length; h++) {
      const c = q[h];
      for (const n of nb(c)) {
        if (PLR[s * NUM_CHORDS + n] !== 255) continue;
        PLR[s * NUM_CHORDS + n] = PLR[s * NUM_CHORDS + c] + 1;
        q.push(n);
      }
    }
  }
}

/** Tonnetz (P/L/R) steps between two triads; 0 for the same chord, up to 6. */
export function triadDistance(a: ChordIndex, b: ChordIndex): number {
  if (a < 0 || b < 0) return 0;
  return PLR[a * NUM_CHORDS + b];
}

/** Kind of a chord change, by the root interval. */
export const MOVE = { none: 0, fifth: 1, third: 2, step: 3, tritone: 4, mode: 5 } as const;
export type MoveKind = (typeof MOVE)[keyof typeof MOVE];

export function moveKind(a: ChordIndex, b: ChordIndex): MoveKind {
  if (a < 0 || b < 0 || a === b) return MOVE.none;
  const iv = mod12(chordRoot(b) - chordRoot(a));
  if (iv === 0) return MOVE.mode;
  if (iv === 5 || iv === 7) return MOVE.fifth;
  if (iv === 3 || iv === 4 || iv === 8 || iv === 9) return MOVE.third;
  if (iv === 6) return MOVE.tritone;
  return MOVE.step;
}

/** +1 when the root falls a fifth (the resolving direction, e.g. G -> C), -1 when it rises one, else 0. */
export function fifthDirection(a: ChordIndex, b: ChordIndex): number {
  if (a < 0 || b < 0) return 0;
  const iv = mod12(chordRoot(b) - chordRoot(a));
  return iv === 5 ? 1 : iv === 7 ? -1 : 0;
}

// ------------------------------------------------------------ function

export interface KeyRef {
  tonic: number;
  mode: 'major' | 'minor';
}

/** The key's home triad. */
export const tonicChord = (k: KeyRef): ChordIndex => makeChord(k.tonic, k.mode === 'minor');

// Functional tension of each chord in a key: [root interval from the tonic][major, minor].
// Tonic 0, tonic substitutes low, subdominants mid, dominants high, chromatic chords higher still.
const FUNC_MAJOR: [number, number][] = [
  [0, 0.45], // I, i (borrowed)
  [0.8, 0.8], // bII
  [0.6, 0.35], // II (secondary dominant), ii
  [0.55, 0.75], // bIII (borrowed), biii
  [0.6, 0.3], // III (secondary dominant), iii
  [0.3, 0.45], // IV, iv (borrowed)
  [0.95, 0.9], // #IV / tritone
  [0.65, 0.5], // V, v
  [0.5, 0.75], // bVI (borrowed), bvi
  [0.55, 0.2], // VI (secondary dominant), vi
  [0.5, 0.7], // bVII (borrowed), bvii
  [0.85, 0.75], // VII, vii (leading-tone)
];
const FUNC_MINOR: [number, number][] = [
  [0.4, 0], // I (Picardy), i
  [0.8, 0.8], // bII (Neapolitan)
  [0.65, 0.55], // II, ii (half-diminished area)
  [0.2, 0.6], // III (relative major), iii
  [0.6, 0.75], // IV (dorian), #iii
  [0.45, 0.3], // IV, iv
  [0.95, 0.9], // tritone
  [0.65, 0.45], // V (harmonic minor), v
  [0.25, 0.7], // VI, vi
  [0.55, 0.75], // #VI, #vi
  [0.4, 0.65], // VII (subtonic), vii
  [0.85, 0.8], // leading-tone area
];

/** How much a chord pulls away from home in a key, 0..1 (the tonic triad is 0). */
export function functionalTension(c: ChordIndex, k: KeyRef): number {
  if (c < 0) return 0.3;
  const iv = mod12(chordRoot(c) - k.tonic);
  const row = (k.mode === 'minor' ? FUNC_MINOR : FUNC_MAJOR)[iv];
  return row[c >= 12 ? 1 : 0];
}

/** True when the chord's three notes all belong to the key's scale (natural minor plus the raised 7th). */
export function isDiatonic(c: ChordIndex, k: KeyRef): boolean {
  if (c < 0) return false;
  const scale = k.mode === 'minor' ? [0, 2, 3, 5, 7, 8, 10, 11] : [0, 2, 4, 5, 7, 9, 11];
  return TONES[c].every((pc) => scale.includes(mod12(pc - k.tonic)));
}

/**
 * Harmonic tension 0..1 of a chord in a key given the chroma's roughness (0..1): mostly the
 * chord's function, plus its lattice distance from the tonic triad, plus the dissonance.
 */
export function chordTension(c: ChordIndex, k: KeyRef, rough: number): number {
  const func = functionalTension(c, k);
  const lat = c < 0 ? 0.3 : Math.min(1, triadDistance(tonicChord(k), c) / 4);
  return Math.min(1, Math.max(0, 0.6 * func + 0.2 * lat + 0.35 * Math.max(0, rough - 0.15)));
}

/** Kind of an arrival on the tonic: 1 authentic (V or vii -> I), 2 plagal (IV -> I), 3 any other return. */
export const CADENCE = { none: 0, authentic: 1, plagal: 2, return: 3 } as const;

export function cadenceKind(prev: ChordIndex, next: ChordIndex, k: KeyRef): number {
  if (next < 0 || prev < 0 || chordRoot(next) !== k.tonic || chordRoot(prev) === k.tonic) return CADENCE.none;
  const iv = mod12(chordRoot(prev) - k.tonic);
  if (iv === 7 || iv === 11) return CADENCE.authentic;
  if (iv === 5) return CADENCE.plagal;
  return CADENCE.return;
}

/** Signed circle-of-fifths steps from one key to another (relative majors compared), -6..5. */
export function keyFifths(a: KeyRef, b: KeyRef): number {
  const ma = a.mode === 'minor' ? a.tonic + 3 : a.tonic;
  const mb = b.mode === 'minor' ? b.tonic + 3 : b.tonic;
  const s = mod12((mb - ma) * 7);
  return s > 5 ? s - 12 : s;
}

// -------------------------------------------------------------- offline

/** What the offline analysis needs (a subset of AnalysisResult). */
export interface HarmonyInput {
  duration: number;
  frameRate: number;
  numFrames: number;
  /** numFrames * 12, each frame normalized to max 1. */
  chroma: Float32Array;
  /** Per-frame loudness 0..1 (weights the beat average); optional. */
  loudness?: Float32Array;
  beats: ArrayLike<number>;
  keys: KeySegment[];
}

export interface ChordSegment {
  start: number;
  end: number;
  chord: ChordIndex;
  /** Mean template score over the segment, 0..1. */
  confidence: number;
  /** Mean tension over the segment. */
  tension: number;
  /** How this chord was reached (MOVE kind) and whether the root fell (+1) or rose (-1) a fifth. */
  move: MoveKind;
  fifths: number;
  /** Tonnetz triangle centre relative to the local tonic (see tonnetzXY). */
  x: number;
  y: number;
}

export interface Resolution {
  time: number;
  /** CADENCE kind. */
  kind: number;
  /** 0..1: the tension released (the peak since the last time home), scaled by the cadence kind. */
  strength: number;
}

export interface Modulation {
  time: number;
  from: KeyRef;
  to: KeyRef;
  /** Signed circle-of-fifths steps (keyFifths). */
  fifths: number;
}

export interface HarmonyTrack {
  /** Beat-grid cell start times (the analysis unit), and each cell's chord and tension. */
  times: Float32Array;
  chords: Int8Array;
  tension: Float32Array;
  segments: ChordSegment[];
  resolutions: Resolution[];
  modulations: Modulation[];
}

/** Viterbi settings (exported for the tests). */
export const HARMONY_PARAMS = {
  /** Emission sharpness: log-likelihood per unit template score. */
  sharp: 10,
  /** Probability a chord continues into the next beat. */
  stay: 0.75,
  /** Log bonus for chords diatonic to the local key. */
  diatonic: 0.6,
  /** Template score the no-chord state competes with. */
  noChord: 0.35,
  /** Grid used when the song has no beats, seconds. */
  gridSeconds: 0.5,
};

/** Beat-grid cells: start times covering [0, duration). */
function gridTimes(inp: HarmonyInput): number[] {
  const out: number[] = [];
  const b = inp.beats;
  const dur = Math.max(inp.duration, 1e-3);
  if (b.length >= 2) {
    if (b[0] > 0.05) out.push(0);
    for (let i = 0; i < b.length; i++) if (b[i] < dur && (out.length === 0 || b[i] - out[out.length - 1] > 0.05)) out.push(b[i]);
  } else {
    for (let t = 0; t < dur; t += HARMONY_PARAMS.gridSeconds) out.push(t);
  }
  if (out.length === 0) out.push(0);
  return out;
}

/** The key segment in force at a time. */
function keyAt(keys: KeySegment[], t: number): KeyRef {
  if (!keys.length) return { tonic: 0, mode: 'major' };
  let k = keys[0];
  for (const s of keys) if (s.start <= t + 1e-6) k = s;
  return k;
}

/** The whole harmony map of a song. Cost: O(beats * 24 * 25), a few ms for a four-minute song. */
export function analyzeHarmony(inp: HarmonyInput): HarmonyTrack {
  const P = HARMONY_PARAMS;
  const times = gridTimes(inp);
  const n = times.length;
  const dur = Math.max(inp.duration, times[n - 1] + 1e-3);
  const T = inp.numFrames;
  const fr = inp.frameRate > 0 ? inp.frameRate : 86;

  // Beat-synchronous chroma, loudness weighted (the chroma frames are each normalized to max 1).
  const bc = new Float64Array(n * 12);
  const energy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, Math.floor(times[i] * fr));
    const b = Math.min(T, Math.max(a + 1, Math.floor((i + 1 < n ? times[i + 1] : dur) * fr)));
    let wsum = 0;
    for (let f = a; f < b; f++) {
      const w = inp.loudness && inp.loudness.length >= T ? 0.05 + inp.loudness[f] : 1;
      let any = 0;
      for (let k = 0; k < 12; k++) any += inp.chroma[f * 12 + k];
      if (any <= 0) continue;
      wsum += w;
      for (let k = 0; k < 12; k++) bc[i * 12 + k] += w * inp.chroma[f * 12 + k];
    }
    if (wsum > 0) for (let k = 0; k < 12; k++) bc[i * 12 + k] /= wsum;
    energy[i] = wsum / Math.max(1, b - a);
  }

  // Emissions: template scores, the no-chord state, the local key's diatonic prior.
  const S = NUM_CHORDS + 1; // state 24 = no chord
  const score = new Float64Array(n * S);
  const emit = new Float64Array(n * S);
  const tmp = new Float64Array(NUM_CHORDS);
  const keyOf: KeyRef[] = [];
  for (let i = 0; i < n; i++) {
    const tot = chordScores(bc, tmp, i * 12);
    const k = keyAt(inp.keys, times[i]);
    keyOf.push(k);
    for (let c = 0; c < NUM_CHORDS; c++) {
      score[i * S + c] = tot > 0 ? tmp[c] : 0;
      emit[i * S + c] = P.sharp * score[i * S + c] + (isDiatonic(c, k) ? P.diatonic : 0);
    }
    // Quiet or flat chroma: no chord.
    const quiet = tot <= 0 || energy[i] < 0.02;
    score[i * S + NUM_CHORDS] = quiet ? 1 : P.noChord;
    emit[i * S + NUM_CHORDS] = P.sharp * (quiet ? 1 : P.noChord) + P.diatonic * 0.5;
  }

  // Viterbi over the beats.
  const lStay = Math.log(P.stay);
  const lMove = Math.log((1 - P.stay) / (S - 1));
  let prev = new Float64Array(S);
  let cur = new Float64Array(S);
  const back = new Int8Array(n * S);
  for (let s = 0; s < S; s++) prev[s] = emit[s];
  for (let i = 1; i < n; i++) {
    let best = 0;
    for (let s = 1; s < S; s++) if (prev[s] > prev[best]) best = s;
    for (let s = 0; s < S; s++) {
      const stay = prev[s] + lStay;
      const move = prev[best] + lMove;
      if (stay >= move || best === s) {
        cur[s] = stay + emit[i * S + s];
        back[i * S + s] = s;
      } else {
        cur[s] = move + emit[i * S + s];
        back[i * S + s] = best;
      }
    }
    [prev, cur] = [cur, prev];
  }
  const path = new Int8Array(n);
  let last = 0;
  for (let s = 1; s < S; s++) if (prev[s] > prev[last]) last = s;
  path[n - 1] = last;
  for (let i = n - 1; i > 0; i--) path[i - 1] = back[i * S + path[i]];

  const chords = new Int8Array(n);
  const tension = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const c = path[i] === NUM_CHORDS ? NO_CHORD : path[i];
    chords[i] = c;
    const rough = energy[i] > 0.02 ? roughness(bc, c, i * 12) : 0;
    tension[i] = c < 0 && energy[i] <= 0.02 ? 0 : chordTension(c, keyOf[i], rough);
  }

  // Segments, motion, resolutions.
  const segments: ChordSegment[] = [];
  const resolutions: Resolution[] = [];
  const xy: [number, number] = [0, 0];
  let peak = 0; // highest tension since the last time home
  let prevChord: ChordIndex = NO_CHORD;
  for (let i = 0; i < n; ) {
    let j = i + 1;
    while (j < n && chords[j] === chords[i]) j++;
    const c = chords[i];
    let conf = 0;
    let ten = 0;
    for (let q = i; q < j; q++) {
      conf += Math.max(0, score[q * S + (c < 0 ? NUM_CHORDS : c)]);
      ten += tension[q];
    }
    const k = keyOf[i];
    tonnetzXY(c, k.tonic, xy);
    const start = times[i];
    const seg: ChordSegment = {
      start, end: j < n ? times[j] : dur, chord: c, confidence: conf / (j - i), tension: ten / (j - i),
      move: moveKind(prevChord, c), fifths: fifthDirection(prevChord, c), x: xy[0], y: xy[1],
    };
    segments.push(seg);
    if (c >= 0) {
      const home = c === tonicChord(k) || (chordRoot(c) === k.tonic && k.mode === 'minor');
      if (home) {
        const kind = cadenceKind(prevChord, c, k);
        const w = kind === CADENCE.authentic ? 1 : kind === CADENCE.plagal ? 0.8 : 0.6;
        const strength = Math.min(1, Math.max(peak, kind === CADENCE.authentic ? 0.5 : 0) * w);
        if (kind !== CADENCE.none && strength > 0.2) resolutions.push({ time: start, kind, strength });
        peak = 0;
      } else peak = Math.max(peak, seg.tension);
      prevChord = c;
    }
    i = j;
  }

  const modulations: Modulation[] = [];
  for (let i = 1; i < inp.keys.length; i++) {
    const a = inp.keys[i - 1];
    const b = inp.keys[i];
    if (a.tonic === b.tonic && a.mode === b.mode) continue;
    const from = { tonic: a.tonic, mode: a.mode };
    const to = { tonic: b.tonic, mode: b.mode };
    modulations.push({ time: b.start, from, to, fifths: keyFifths(from, to) });
  }

  return { times: Float32Array.from(times), chords, tension, segments, resolutions, modulations };
}

// ------------------------------------------------------------- realtime

const RT_TAU = 0.35; // chroma window, seconds
const RT_EVAL = 0.125; // seconds between evaluations
const RT_MARGIN = 0.06; // score a new chord must win by
const RT_EVALS = 3; // consecutive wins before switching (~0.4 s)
const RT_MIN_SCORE = 0.45;

/**
 * Realtime-lite chord tracking for live input: a short exponential chroma window matched against
 * the triad templates every 1/8 s, switching only when another chord wins clearly for a few
 * evaluations. Tension and resolutions follow the same rules as the offline map, causally.
 */
export class HarmonyTracker {
  private readonly acc = new Float64Array(12);
  private readonly sc = new Float64Array(NUM_CHORDS);
  private fill = 0;
  private sinceEval = 0;
  private pending: ChordIndex = NO_CHORD;
  private pendingCount = 0;
  private peak = 0;
  private prev: ChordIndex = NO_CHORD;
  private key: KeyRef = { tonic: 0, mode: 'major' };

  chord: ChordIndex = NO_CHORD;
  tension = 0;
  /** Counters: incremented on each chord change / resolution / key change. */
  changes = 0;
  resolves = 0;
  modulations = 0;
  lastResolve = 0;
  lastMove: MoveKind = MOVE.none;
  lastFifths = 0;

  reset(): void {
    this.acc.fill(0);
    this.fill = 0;
    this.sinceEval = 0;
    this.pending = NO_CHORD;
    this.pendingCount = 0;
    this.peak = 0;
    this.prev = NO_CHORD;
    this.chord = NO_CHORD;
    this.tension = 0;
  }

  /** Tells the tracker the current key (from the key tracker); counts a modulation when it changes. */
  setKey(tonic: number, mode: 'major' | 'minor', valid = true): void {
    if (!valid) return;
    if (tonic !== this.key.tonic || mode !== this.key.mode) {
      if (this.changes > 0 || this.chord >= 0) this.modulations++;
      this.key = { tonic, mode };
      this.peak = 0;
    }
  }

  /** One chroma frame (normalized to max 1), its loudness weight 0..1 and duration. Returns true on a chord change. */
  push(chroma: ArrayLike<number>, weight: number, dt: number): boolean {
    const k = Math.exp(-dt / RT_TAU);
    for (let i = 0; i < 12; i++) this.acc[i] = this.acc[i] * k + weight * chroma[i] * dt;
    this.fill = this.fill * k + weight * dt;
    this.sinceEval += dt;
    if (this.sinceEval < RT_EVAL) return false;
    this.sinceEval = 0;
    const quiet = this.fill < RT_TAU * 0.15;
    const tot = quiet ? 0 : chordScores(this.acc, this.sc);
    let best = 0;
    for (let c = 1; c < NUM_CHORDS; c++) if (this.sc[c] > this.sc[best]) best = c;
    const cand = tot > 0 && this.sc[best] >= RT_MIN_SCORE ? best : NO_CHORD;
    const curScore = this.chord >= 0 ? this.sc[this.chord] : RT_MIN_SCORE;
    let changed = false;
    if (cand !== this.chord && (cand < 0 || this.sc[cand] - curScore > RT_MARGIN)) {
      if (cand === this.pending) this.pendingCount++;
      else {
        this.pending = cand;
        this.pendingCount = 1;
      }
      if (this.pendingCount >= RT_EVALS) {
        changed = this.switchTo(cand);
        this.pendingCount = 0;
      }
    } else this.pendingCount = 0;
    const rough = tot > 0 ? roughness(this.acc, this.chord) : 0;
    const target = quiet ? 0 : chordTension(this.chord, this.key, rough);
    this.tension += (target - this.tension) * 0.35;
    return changed;
  }

  private switchTo(c: ChordIndex): boolean {
    const old = this.chord;
    this.chord = c;
    if (c < 0) return false;
    this.changes++;
    this.lastMove = moveKind(this.prev, c);
    this.lastFifths = fifthDirection(this.prev, c);
    const home = c === tonicChord(this.key) || (chordRoot(c) === this.key.tonic && this.key.mode === 'minor');
    if (home) {
      const kind = cadenceKind(this.prev, c, this.key);
      const w = kind === CADENCE.authentic ? 1 : kind === CADENCE.plagal ? 0.8 : 0.6;
      const s = Math.min(1, this.peak * w);
      if (kind !== CADENCE.none && s > 0.2) {
        this.resolves++;
        this.lastResolve = s;
      }
      this.peak = 0;
    } else this.peak = Math.max(this.peak, chordTension(c, this.key, 0));
    this.prev = c;
    return old !== c;
  }
}

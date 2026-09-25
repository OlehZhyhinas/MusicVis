// Note tracking: the melody line as notes, and how they are played (articulation).
//
// A monophonic-ish tracker on the melodic region (the 'other' + 'vocals' bands, ~120 Hz-5 kHz):
//
//   pitch    per frame, a harmonic-sum estimator on the whitened magnitude spectrum (spectral
//            peaks above their local mean): every f0 candidate on a half-semitone grid from 130 Hz
//            to 1.05 kHz sums its first 8 harmonics with decaying weights (so the octave below,
//            which only sees every other harmonic, and the octave above, which misses the odd
//            ones, both score lower). The best few candidates are refined from the interpolated
//            harmonic peaks (a few cents), and scored by the share of the region's power their
//            harmonics carry.
//   line     offline, a Viterbi path over those candidates plus an unvoiced state (steps cost, big
//            jumps more, voicing flips a little) picks one smooth line; live, a greedy causal pick.
//   notes    the voiced line is cut into notes where voicing starts, where the pitch steps by more
//            than ~0.7 semitone and settles (a glide does not cut), and where the harmonics' level
//            dips and re-attacks at the same pitch ("tu tu tu" on one note). A note ends when the
//            line goes unvoiced or its level falls 15 dB under its peak.
//
// Per note: strength, a legato score (how long it sounds, up to where its level falls 9 dB, over
// the time to the next note, plus glide and vibrato) and a voice score (vibrato, pitch drift inside
// the note, a scoop into it and distance from the equal-tempered grid: a singer; a synth lead sits
// dead steady on the grid). Per frame: note-on pulse, held strength, legato around now, glide
// (pitch slope, vibrato removed), vibrato depth, pitch and its height within the song's range, and
// the voice cue.
//
// Offline, PitchFrames is fed every third frame of the main STFT analyzePcm already computes (no
// extra FFT; the frames between are interpolated); a frame costs one pass over ~450 bins plus ~600 table lookups. Realtime, NoteTracker is fed
// the live main spectrum.

import type { NoteEvent, NoteMark, NoteStats, NoteTrack } from '../types';

/** Latest notes kept as marks in NoteStats.recent. */
export const NOTE_RECENT = 12;
/** Candidates kept per frame. */
export const NOTE_K = 4;
/** Offline, the pitch estimator runs on every PITCH_EVERY-th STFT frame. */
const PITCH_EVERY = 3;

const F0_LO = 130;
const F0_HI = 1050;
const REG_LO = 120;
const REG_HI = 5000;
const STEP = 0.5; // semitones per coarse candidate
const H = 8; // harmonics summed
const HR = 6; // harmonics used to refine f0
const W = 6; // local-mean half width (bins) for whitening
const TINY = 1e-20;
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const hzToMidi = (f: number) => 69 + 12 * Math.log2(f / 440);

/** Per-frame f0 candidates (MIDI pitch, harmonic share of the region's power, harmonic level dB). */
export class PitchFrames {
  readonly pitch = new Float64Array(NOTE_K);
  readonly obs = new Float64Array(NOTE_K);
  readonly energy = new Float64Array(NOTE_K);
  /** Level of the candidate's harmonics in the novelty spectrum (sustained sounds fade out of it), dB. */
  readonly novel = new Float64Array(NOTE_K);
  /** Candidates found this frame (0 in silence). */
  count = 0;
  /** Level of the melodic region, dB. */
  level = -200;
  private readonly binHz: number;
  private readonly kLo: number;
  private readonly kHi: number;
  private readonly nc: number;
  private readonly ctr: Int32Array; // [c * H + h] centre bin of harmonic h+1, -1 outside the region
  private readonly rad: Int8Array;
  private readonly wt = new Float64Array(H);
  private readonly mag: Float64Array;
  private readonly pre: Float64Array;
  private readonly white: Float64Array;
  private readonly slow: Float64Array;
  private readonly nov: Float64Array;
  private readonly sal: Float64Array;
  private readonly dil: Float64Array;
  private readonly nb: number;
  private readonly kSlow: number;

  constructor(n: number, sr: number, frameRate = sr / (n / 8)) {
    this.kSlow = 1 - Math.exp(-1 / (1.5 * frameRate));
    const nb = n / 2 + 1;
    this.binHz = sr / n;
    this.kLo = Math.max(W + 1, Math.floor(REG_LO / this.binHz));
    this.kHi = Math.min(nb - W - 2, Math.ceil(Math.min(REG_HI, sr * 0.45) / this.binHz));
    this.nc = Math.floor((12 * Math.log2(F0_HI / F0_LO)) / STEP) + 1;
    this.ctr = new Int32Array(this.nc * H);
    this.rad = new Int8Array(this.nc * H);
    for (let h = 0; h < H; h++) this.wt[h] = Math.pow(0.84, h);
    for (let c = 0; c < this.nc; c++) {
      const b0 = (F0_LO * Math.pow(2, (c * STEP) / 12)) / this.binHz;
      for (let h = 0; h < H; h++) {
        const x = (h + 1) * b0;
        const k = Math.round(x);
        this.ctr[c * H + h] = k >= this.kLo && k <= this.kHi ? k : -1;
        // The grid is a quarter semitone off at worst: search that far around the harmonic.
        this.rad[c * H + h] = Math.min(2, Math.max(0, Math.ceil(x * 0.0146 - 0.3)));
      }
    }
    this.mag = new Float64Array(nb);
    this.pre = new Float64Array(nb + 1);
    this.white = new Float64Array(nb);
    this.slow = new Float64Array(nb);
    this.nov = new Float64Array(nb);
    this.nb = nb;
    this.dil = new Float64Array(3 * nb);
    this.sal = new Float64Array(this.nc);
  }

  push(pow: Float64Array): void {
    const { kLo, kHi, mag, pre, white, slow, nov, sal, ctr, rad, wt, kSlow } = this;
    const a = kLo - W;
    const b = kHi + W + 1;
    pre[a] = 0;
    let regPow = 0;
    for (let k = a; k < b; k++) {
      const m = Math.sqrt(pow[k]);
      mag[k] = m;
      pre[k + 1] = pre[k] + m;
    }
    for (let k = kLo; k <= kHi; k++) {
      regPow += pow[k];
      const mean = (pre[k + W + 1] - pre[k - W]) / (2 * W + 1);
      const d = mag[k] - 1.5 * mean;
      const w = d > 0 ? d : 0;
      white[k] = w;
      // Temporal novelty: peaks above 0.8 of their own ~1.5 s average, so sustained pads and chords
      // fade out and a moving melody stands out (a held note keeps a fifth of its height).
      const sl = slow[k];
      const v = w - 0.8 * sl;
      nov[k] = v > 0 ? v : 0;
      slow[k] = sl + (w - sl) * kSlow;
    }
    this.level = 10 * Math.log10(regPow + TINY);
    this.count = 0;
    if (!(regPow > 1e-12)) return;
    // The novelty spectrum dilated by one and two bins (the harmonic lookups' search radius).
    const nb = this.nb, dil = this.dil;
    for (let k = kLo; k <= kHi; k++) {
      const a1 = nov[k - 1], a0 = nov[k], b1 = nov[k + 1];
      const m1 = a1 > a0 ? (a1 > b1 ? a1 : b1) : a0 > b1 ? a0 : b1;
      const a2 = nov[k - 2], b2 = nov[k + 2];
      dil[k] = a0;
      dil[nb + k] = m1;
      dil[2 * nb + k] = m1 > a2 ? (m1 > b2 ? m1 : b2) : a2 > b2 ? a2 : b2;
    }
    // Harmonic sums on the coarse grid.
    const nc = this.nc;
    for (let c = 0; c < nc; c++) {
      let s = 0, m1 = 0, mx = 0;
      const o = c * H;
      for (let h = 0; h < H; h++) {
        const k = ctr[o + h];
        if (k < 0) {
          if (h > 0) break;
          continue;
        }
        const m = dil[rad[o + h] * nb + k];
        s += wt[h] * m;
        if (h === 0) m1 = m;
        if (m > mx) mx = m;
      }
      // A melody note has its fundamental: a candidate whose first harmonic is weak against its
      // others (the octave or twelfth below a note, gathering its partials and a chord's) scores less.
      sal[c] = mx > 0 ? s * Math.sqrt(0.1 + 0.9 * (m1 / mx)) : 0;
    }
    // The best local maxima.
    const P = this.pitch, O = this.obs, E = this.energy;
    let n = 0;
    const best = [-1, -1, -1, -1];
    const bs = [0, 0, 0, 0];
    for (let c = 0; c < nc; c++) {
      const v = sal[c];
      if (!(v > 0) || (c > 0 && sal[c - 1] >= v) || (c + 1 < nc && sal[c + 1] > v)) continue;
      let i = Math.min(n, NOTE_K - 1);
      if (n === NOTE_K && v <= bs[i]) continue;
      while (i > 0 && bs[i - 1] < v) {
        bs[i] = bs[i - 1];
        best[i] = best[i - 1];
        i--;
      }
      bs[i] = v;
      best[i] = c;
      if (n < NOTE_K) n++;
    }
    // Refine: f0 from the interpolated harmonic peaks, weighted by their whitened height.
    for (let i = 0; i < n; i++) {
      const c = best[i];
      const b0 = (F0_LO * Math.pow(2, (c * STEP) / 12)) / this.binHz;
      let sw = 0, sf = 0, e = 0, ev = 0;
      for (let h = 0; h < HR; h++) {
        const k0 = ctr[c * H + h];
        if (k0 < 0) {
          if (h > 0) break;
          continue;
        }
        const r = rad[c * H + h] + 1;
        let k = k0;
        for (let j = k0 - r; j <= k0 + r; j++) if (j > kLo && j < kHi && pow[j] > pow[k]) k = j;
        const w = nov[k] + 0.1 * white[k];
        if (!(w > 0) || k <= kLo || k >= kHi) continue;
        const al = Math.log(pow[k - 1] + TINY), be = Math.log(pow[k] + TINY), ga = Math.log(pow[k + 1] + TINY);
        const den = al - 2 * be + ga;
        const d = den < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (al - ga)) / den)) : 0;
        const fk = (k + d) / (h + 1);
        // Harmonics far off the candidate (another sound's peak) are not counted.
        if (Math.abs(fk / b0 - 1) > 0.03) continue;
        sw += w;
        sf += w * fk;
        e += pow[k - 1] + pow[k] + pow[k + 1];
        ev += nov[k] * nov[k];
      }
      if (!(sw > 0)) continue;
      P[this.count] = hzToMidi((sf / sw) * this.binHz);
      // The share of the region's power the harmonics carry, discounted by salience against the best
      // candidate (the octave below a note explains the same power, but with a weaker harmonic sum).
      O[this.count] = clamp01(e / regPow) * (bs[i] / bs[0]);
      E[this.count] = 10 * Math.log10(e + TINY);
      this.novel[this.count] = 10 * Math.log10(ev + TINY);
      this.count++;
    }
  }
}

// ------------------------------------------------------------------ line + notes (shared rules)

/** Cost of moving the line by d semitones between frames. */
function jumpCost(d: number): number {
  return d < 0.45 ? 0 : 0.12 + Math.min(1, (d - 0.45) / 7);
}

/** Frame score of a candidate: the harmonic share, gated by the region's level under its top. */
function voicedScore(obs: number, levelGate: number): number {
  return Math.sqrt(obs) * levelGate;
}

const UNVOICED = 0.3; // score of the unvoiced state per frame
const LAMBDA = 0.5; // weight of jump costs
const BETA = 0.35; // cost of a voicing flip
/** The live pick pays the jump cost every frame it stays away, so it is a light hysteresis. */
const LIVE_LAMBDA = 0.15;

/** Offline: the Viterbi line over the candidates. pitch NaN where unvoiced; energy dB of the pick. */
/** Per-frame candidates of a whole song (NOTE_K slots per frame). */
export interface NoteCands {
  p: Float32Array;
  o: Float32Array;
  e: Float32Array;
  v: Float32Array;
  n: Uint8Array;
  level: Float32Array;
}

export function viterbiLine(c: NoteCands, T: number): { pitch: Float32Array; energy: Float32Array; novel: Float32Array } {
  const { p: cp, o: co, e: ce, v: cv, n: cn, level } = c;
  const S = NOTE_K + 1;
  const back = new Uint8Array(T * S);
  let prev = new Float64Array(S);
  let cur = new Float64Array(S);
  const top = levelTop(level);
  for (let t = 0; t < T; t++) {
    const g = clamp01((level[t] - (top - 45)) / 12);
    const o = t * NOTE_K;
    for (let s = 0; s < S; s++) {
      const voiced = s < NOTE_K;
      if (voiced && s >= cn[t]) {
        cur[s] = -Infinity;
        continue;
      }
      const obs = voiced ? voicedScore(co[o + s], g) : UNVOICED;
      let bestV = -Infinity, bestI = NOTE_K;
      if (t === 0) {
        bestV = 0;
      } else {
        const po = o - NOTE_K;
        for (let q = 0; q < S; q++) {
          const pv = prev[q];
          if (pv === -Infinity) continue;
          const qv = q < NOTE_K;
          let tr = 0;
          if (voiced && qv) tr = LAMBDA * jumpCost(Math.abs(cp[o + s] - cp[po + q]));
          else if (voiced !== qv) tr = BETA;
          const v = pv - tr;
          if (v > bestV) {
            bestV = v;
            bestI = q;
          }
        }
      }
      cur[s] = bestV + obs;
      back[t * S + s] = bestI;
    }
    const tmp = prev;
    prev = cur;
    cur = tmp;
  }
  const pitch = new Float32Array(T).fill(NaN);
  const energy = new Float32Array(T).fill(-200);
  const novel = new Float32Array(T).fill(-200);
  let s = NOTE_K;
  for (let q = 0; q < S; q++) if (prev[q] > prev[s]) s = q;
  for (let t = T - 1; t >= 0; t--) {
    if (s < NOTE_K) {
      pitch[t] = cp[t * NOTE_K + s];
      energy[t] = ce[t * NOTE_K + s];
      novel[t] = cv[t * NOTE_K + s];
    }
    s = back[t * S + s];
  }
  return { pitch, energy, novel };
}

/** A robust top of the region level (the 97th percentile of frames above silence). */
function levelTop(level: Float32Array): number {
  const v: number[] = [];
  for (let i = 0; i < level.length; i += 4) if (level[i] > -150) v.push(level[i]);
  if (!v.length) return -200;
  v.sort((a, b) => a - b);
  return v[Math.min(v.length - 1, Math.floor(v.length * 0.97))];
}

function percentile(v: number[], q: number, def: number): number {
  if (!v.length) return def;
  const s = [...v].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))];
}

interface RawNote {
  a: number; // first frame
  b: number; // last frame + 1
}

/** Cuts the voiced line into notes (frame ranges). */
export function segmentNotes(pitch: Float32Array, energy: Float32Array, fr: number): RawNote[] {
  const T = pitch.length;
  const out: RawNote[] = [];
  const minLen = Math.max(2, Math.round(0.04 * fr));
  const settle = Math.max(2, Math.round(0.035 * fr));
  let a = -1;
  let peak = -200;
  let trough = -200;
  let dead = false; // the last note died by level: the next one needs a re-attack
  const close = (b: number) => {
    if (a >= 0 && b - a >= minLen) out.push({ a, b });
    a = -1;
  };
  for (let t = 0; t < T; t++) {
    const p = pitch[t];
    if (Number.isNaN(p)) {
      // A one-frame dropout inside a note does not end it.
      if (a >= 0 && t + 1 < T && !Number.isNaN(pitch[t + 1]) && Math.abs(pitch[t + 1] - pitch[t - 1]) < 0.5) {
        pitch[t] = pitch[t - 1];
        energy[t] = energy[t - 1];
      } else {
        close(t);
        dead = false;
        continue;
      }
    }
    const e = energy[t];
    if (a < 0) {
      if (dead) {
        // A pluck's tail: wait for the level to rise again.
        if (e - trough > 4.5 && t > 0 && e > energy[t - 1]) dead = false;
        else {
          trough = Math.min(trough, e);
          pitch[t] = NaN;
          continue;
        }
      }
      a = t;
      peak = trough = e;
      continue;
    }
    // Pitch step: a jump that settles (a glide moves on instead).
    let step = false;
    if (t - a >= minLen && t >= 2) {
      const ref = pitch[t - 2];
      if (Math.abs(pitch[t] - ref) > 0.7) {
        step = true;
        for (let j = 1; j <= settle && t + j < T; j++) {
          const q = pitch[t + j];
          if (Number.isNaN(q) || Math.abs(q - pitch[t]) > 0.45) {
            step = false;
            break;
          }
        }
      }
    }
    // Re-attack: the level fell from the note's peak and rises again.
    const re = !step && t - a >= minLen && peak - trough >= 4 && e - trough >= 4.5 && e > energy[t - 1];
    if (step || re) {
      close(t);
      a = t;
      peak = trough = e;
      continue;
    }
    if (e < peak - 15) {
      close(t);
      pitch[t] = NaN;
      dead = true;
      trough = e;
      continue;
    }
    if (e > peak) peak = trough = e;
    else if (e < trough) trough = e;
  }
  close(T);
  return out;
}

/** Background dB under the loudest nearby note (in the novelty spectrum) below which a note is dropped. */
const BACKGROUND_DB = 9;

/**
 * Drops background notes: a pad or chord tone the line falls onto between melody notes is much
 * weaker in the novelty spectrum than the melody notes within 1.5 s of it.
 */
export function rejectBackground(notes: RawNote[], novel: Float32Array, fr: number): RawNote[] {
  const pk = notes.map((n) => {
    let m = -200;
    for (let t = n.a; t < n.b; t++) m = Math.max(m, novel[t]);
    return m;
  });
  const w = 1.5 * fr;
  const out: RawNote[] = [];
  let j0 = 0;
  for (let i = 0; i < notes.length; i++) {
    while (notes[j0].b < notes[i].a - w) j0++;
    let ref = -200;
    for (let j = j0; j < notes.length && notes[j].a <= notes[i].b + w; j++) ref = Math.max(ref, pk[j]);
    if (pk[i] >= ref - BACKGROUND_DB) out.push(notes[i]);
  }
  return out;
}

/**
 * Drops revealed tones: a note the line steps onto that was already sounding just before at about
 * the same level (a pad or chord tone left over when a melody note ends) has no onset of its own.
 */
export function rejectRevealed(notes: RawNote[], c: NoteCands, pitch: Float32Array, energy: Float32Array, fr: number): RawNote[] {
  const back0 = Math.max(3, Math.round(0.05 * fr)), back1 = Math.max(back0 + 1, Math.round(0.1 * fr));
  return notes.filter((n) => {
    if (n.a < back1) return true;
    let pk = -200;
    for (let t = n.a; t < Math.min(n.b, n.a + back1); t++) pk = Math.max(pk, energy[t]);
    const p = pitch[n.a];
    let before = -200;
    for (let t = n.a - back1; t <= n.a - back0; t++) {
      for (let i = 0; i < c.n[t]; i++) {
        const j = t * NOTE_K + i;
        if (Math.abs(c.p[j] - p) < 0.5) before = Math.max(before, c.e[j]);
      }
    }
    return before < pk - 3;
  });
}

/** 5-tap median of x[a..b) (shrinking at the ends), into out[a..b). */
function medianRange(x: Float32Array, a: number, b: number, out: Float32Array): void {
  const w = [0, 0, 0, 0, 0];
  for (let t = a; t < b; t++) {
    const lo = Math.max(a, t - 2), hi = Math.min(b - 1, t + 2);
    const n = hi - lo + 1;
    for (let j = 0; j < n; j++) w[j] = x[lo + j];
    for (let i = 1; i < n; i++) for (let j = i; j > 0 && w[j - 1] > w[j]; j--) {
      const q = w[j];
      w[j] = w[j - 1];
      w[j - 1] = q;
    }
    out[t] = n & 1 ? w[n >> 1] : 0.5 * (w[(n >> 1) - 1] + w[n >> 1]);
  }
}

/** Centred moving average of x[a..b) (width w), into out[a..b). */
function smoothRange(x: Float32Array, a: number, b: number, w: number, out: Float32Array): void {
  const h = w >> 1;
  for (let t = a; t < b; t++) {
    const lo = Math.max(a, t - h), hi = Math.min(b - 1, t + h);
    let s = 0;
    for (let j = lo; j <= hi; j++) s += x[j];
    out[t] = s / (hi - lo + 1);
  }
}

/** The whole offline note track from the per-frame candidates. */
export function buildNoteTrack(
  c: NoteCands, T: number, fr: number,
): NoteTrack {
  const { pitch, energy, novel } = viterbiLine(c, T);
  const raw = rejectRevealed(rejectBackground(segmentNotes(pitch, energy, fr), novel, fr), c, pitch, energy, fr);
  const track: NoteTrack = {
    notes: [], on: new Float32Array(T), held: new Float32Array(T), legato: new Float32Array(T), glide: new Float32Array(T),
    vibrato: new Float32Array(T), pitch: new Float32Array(T), height: new Float32Array(T), voice: new Float32Array(T), lo: 55, hi: 79,
  };
  // Song-wide scales: harmonic level (strength) and tuning.
  const ev: number[] = [];
  for (const n of raw) for (let t = n.a; t < n.b; t += 2) ev.push(energy[t]);
  const eLo = percentile(ev, 0.1, -60), eHi = Math.max(eLo + 12, percentile(ev, 0.97, 0));
  const sm = new Float32Array(T);
  const vib = new Float32Array(T);
  const wSm = Math.max(3, Math.round(0.18 * fr) | 1);
  const wV = Math.max(3, Math.round(0.2 * fr));
  const pitches: number[] = [];
  const weights: number[] = [];
  let tunX = 0, tunY = 0;
  interface Info { a: number; b: number; med: number; eff: number; str: number; glide: number; vib: number; drift: number; scoop: number }
  const info: Info[] = [];
  for (const n of raw) {
    const { a, b } = n;
    // A 5-frame median first: a stray frame on another candidate (an octave off) is neither vibrato nor glide.
    medianRange(pitch, a, b, sm);
    for (let t = a; t < b; t++) pitch[t] = sm[t];
    smoothRange(pitch, a, b, wSm, sm);
    const ps: number[] = [];
    let pk = -200, pkAt = a;
    for (let t = a; t < b; t++) {
      ps.push(pitch[t]);
      if (energy[t] > pk) {
        pk = energy[t];
        pkAt = t;
      }
    }
    const med = percentile(ps, 0.5, 60);
    // Vibrato: the residual around the smoothed contour, when it oscillates at 3..10 Hz.
    let vsum = 0, vn = 0;
    for (let t = a; t < b; t++) {
      const lo = Math.max(a, t - wV), hi = Math.min(b - 1, t + wV);
      let s2 = 0, zc = 0;
      let last = pitch[lo] - sm[lo];
      for (let j = lo; j <= hi; j++) {
        const r = pitch[j] - sm[j];
        s2 += r * r;
        if ((r > 0) !== (last > 0)) zc++;
        last = r;
      }
      const span = (hi - lo + 1) / fr;
      const rate = zc / 2 / Math.max(span, 1e-3);
      const depth = Math.sqrt((2 * s2) / (hi - lo + 1));
      // A deeper wobble than 1.5 semitones is the line jumping to another sound, not vibrato.
      vib[t] = hi - lo + 1 >= wV && rate >= 3 && rate <= 10 && depth <= 1.5 ? depth : 0;
      vsum += vib[t];
      vn++;
    }
    // Effective length: up to where the level has fallen 9 dB under the peak (a pluck's tail is not held).
    let effB = b;
    for (let t = pkAt; t < b; t++) if (energy[t] < pk - 9) {
      effB = t;
      break;
    }
    // Glide: how far the smoothed contour travels from the note's pitch.
    let gl = 0;
    for (let t = a; t < b; t++) gl = Math.max(gl, Math.abs(sm[t] - med));
    // Drift: spread of the smoothed contour over the middle of the note.
    const m0 = a + Math.floor((b - a) * 0.2), m1 = b - Math.floor((b - a) * 0.2);
    const dev: number[] = [];
    for (let t = m0; t < m1; t++) dev.push(Math.abs(sm[t] - med));
    const drift = dev.length > 2 ? percentile(dev, 0.5, 0) * 1.4826 : 0;
    const s0 = Math.min(b, a + Math.max(2, Math.round(0.05 * fr)));
    let sc = 0;
    for (let t = a; t < s0; t++) sc += pitch[t];
    const scoop = Math.abs(sc / (s0 - a) - med);
    const str = clamp01((pk - eLo) / (eHi - eLo));
    info.push({ a, b, med, eff: (effB - a) / fr, str, glide: gl, vib: vn ? vsum / vn : 0, drift, scoop });
    const dur = (b - a) / fr;
    pitches.push(med);
    weights.push(dur);
    const ang = 2 * Math.PI * (med - Math.round(med));
    tunX += Math.cos(ang) * dur;
    tunY += Math.sin(ang) * dur;
  }
  const tuning = Math.atan2(tunY, tunX) / (2 * Math.PI); // semitones off A440's grid
  // Melodic range: the 5th..95th percentile of note pitches (by length), at least an octave.
  {
    const idx = pitches.map((_, i) => i).sort((x, y) => pitches[x] - pitches[y]);
    const tot = weights.reduce((s, w) => s + w, 0);
    let acc = 0, lo = NaN, hi = NaN;
    for (const i of idx) {
      acc += weights[i];
      if (Number.isNaN(lo) && acc >= tot * 0.05) lo = pitches[i];
      if (Number.isNaN(hi) && acc >= tot * 0.95) hi = pitches[i];
    }
    if (Number.isFinite(lo) && Number.isFinite(hi)) {
      lo -= 1;
      hi += 1;
      if (hi - lo < 12) {
        const c = (lo + hi) / 2;
        lo = c - 6;
        hi = c + 6;
      }
      track.lo = lo;
      track.hi = hi;
    }
  }
  const lo = track.lo, span = track.hi - track.lo;
  // Note events with legato and voice scores.
  for (let i = 0; i < info.length; i++) {
    const n = info[i];
    const start = n.a / fr, end = n.b / fr;
    const next = i + 1 < info.length ? info[i + 1].a / fr : Infinity;
    const ioi = Math.min(next - start, 0.6);
    const r = Math.min(n.eff, end - start) / Math.max(ioi, 1e-3);
    const glideP = clamp01((n.glide - 0.4) / 1.0);
    const vibP = clamp01((n.vib - 0.12) / 0.3);
    const legato = clamp01(Math.max((r - 0.4) / 0.45, glideP, vibP * 0.8));
    let voice = NaN;
    if (end - start >= 0.15) {
      let off = Math.abs(n.med - tuning - Math.round(n.med - tuning)) * 100; // cents off the grid
      off = clamp01((off - 8) / 25);
      voice = clamp01(0.45 * vibP + 0.35 * clamp01((n.drift - 0.04) / 0.2) + 0.2 * clamp01((n.scoop - 0.25) / 0.8) + 0.3 * off);
    }
    const contour = pitch.slice(n.a, n.b);
    track.notes.push({ start, end, pitch: n.med, contour, strength: n.str, legato, voice });
  }
  // Per frame.
  const notes = track.notes;
  const onTau = 0.15 * fr;
  const relK = Math.exp(-1 / (0.05 * fr));
  let held = 0;
  let lastP = notes.length ? notes[0].pitch : 60;
  let ni = 0;
  for (let t = 0; t < T; t++) {
    while (ni + 1 < info.length && info[ni + 1].a <= t) ni++;
    const n = info[ni];
    const inside = !!n && t >= n.a && t < n.b;
    if (inside) {
      const e = clamp01((energy[t] - eLo) / (eHi - eLo));
      held = Math.max(e, 0.15 * n.str);
      lastP = pitch[t];
      const gl = t > n.a + 1 && t < n.b - 2 ? (sm[Math.min(n.b - 1, t + 2)] - sm[Math.max(n.a, t - 2)]) * fr / (Math.min(n.b - 1, t + 2) - Math.max(n.a, t - 2)) : 0;
      track.glide[t] = Math.max(-30, Math.min(30, gl));
      track.vibrato[t] = vib[t];
    } else {
      held *= relK;
    }
    track.held[t] = held;
    track.pitch[t] = lastP;
    track.height[t] = clamp01((lastP - lo) / span);
    if (n && t >= n.a) track.on[t] = n.str * Math.exp(-(t - n.a) / onTau);
  }
  // Legato and voice around each frame: nearby notes weighted by distance (and length for voice).
  const sig = 0.6, sigV = 3;
  let lastL = 0.5, lastV = 0;
  let j0 = 0;
  // Every 4th frame (both change over seconds), filled in linearly.
  const EV = 4;
  for (let t = 0; t < T; t += EV) {
    const tt = t / fr;
    while (j0 < notes.length && notes[j0].end < tt - 3 * sigV) j0++;
    let wl = 0, sl = 0, wv = 0, sv = 0;
    for (let j = j0; j < notes.length && notes[j].start <= tt + 3 * sigV; j++) {
      const n = notes[j];
      const d = tt < n.start ? n.start - tt : tt > n.end ? tt - n.end : 0;
      if (d < 3 * sig) {
        const w = Math.exp(-((d / sig) ** 2)) * (0.3 + n.strength);
        wl += w;
        sl += w * n.legato;
      }
      if (!Number.isNaN(n.voice)) {
        const w = Math.exp(-((d / sigV) ** 2)) * (n.end - n.start);
        wv += w;
        sv += w * n.voice;
      }
    }
    if (wl > 0.05) lastL = sl / wl;
    else if (wl > 0) lastL += (sl / wl - lastL) * (wl / 0.05);
    if (wv > 0.05) lastV = sv / wv;
    track.legato[t] = lastL;
    track.voice[t] = lastV;
    if (t >= EV) {
      const l0 = track.legato[t - EV], v0 = track.voice[t - EV];
      for (let j = 1; j < EV; j++) {
        track.legato[t - EV + j] = l0 + ((lastL - l0) * j) / EV;
        track.voice[t - EV + j] = v0 + ((lastV - v0) * j) / EV;
      }
    }
  }
  for (let t = Math.max(1, T - ((T - 1) % EV)); t < T; t++) {
    track.legato[t] = track.legato[t - 1];
    track.voice[t] = track.voice[t - 1];
  }
  return track;
}

/** Offline: collects the candidates frame by frame (call frame() from the STFT callback), then build(). */
export class NoteRecorder {
  private readonly pf: PitchFrames;
  private readonly c: NoteCands;
  private readonly frameRate: number;
  private readonly T: number;

  constructor(n: number, sr: number, frameRate: number, T: number) {
    this.frameRate = frameRate;
    this.T = T;
    this.pf = new PitchFrames(n, sr, frameRate / PITCH_EVERY);
    const z = () => new Float32Array(T * NOTE_K);
    this.c = { p: z(), o: z(), e: z(), v: z(), n: new Uint8Array(T), level: new Float32Array(T) };
  }

  frame(f: number, pow: Float64Array): void {
    const pf = this.pf;
    const c = this.c;
    // Every third frame (~29 Hz is plenty for notes, glides and vibrato); build() fills the rest in.
    if (f % PITCH_EVERY !== 0) return;
    pf.push(pow);
    c.n[f] = pf.count;
    c.level[f] = pf.level;
    for (let i = 0; i < pf.count; i++) {
      const j = f * NOTE_K + i;
      c.p[j] = pf.pitch[i];
      c.o[j] = pf.obs[i];
      c.e[j] = pf.energy[i];
      c.v[j] = pf.novel[i];
    }
  }

  /**
   * The frames between analysed ones: each candidate of the frame before that continues (within half
   * a semitone) into the frame after is interpolated; the others are held from the nearer frame.
   */
  private fill(): void {
    const c = this.c, T = this.T, K = NOTE_K;
    for (let a = 0; a < T; a += PITCH_EVERY) {
      const b = a + PITCH_EVERY;
      for (let f = a + 1; f < Math.min(b, T); f++) {
        const hasB = b < T;
        const w = hasB ? (f - a) / PITCH_EVERY : 0;
        const src = !hasB || w < 0.5 ? a : b;
        c.level[f] = hasB ? c.level[a] + (c.level[b] - c.level[a]) * w : c.level[a];
        c.n[f] = c.n[src];
        for (let i = 0; i < c.n[src]; i++) {
          const js = src * K + i, jf = f * K + i;
          c.p[jf] = c.p[js];
          c.o[jf] = c.o[js];
          c.e[jf] = c.e[js];
          c.v[jf] = c.v[js];
          if (!hasB) continue;
          // Pair this candidate with its continuation on the other side.
          const other = src === a ? b : a;
          for (let q = 0; q < c.n[other]; q++) {
            const jo = other * K + q;
            if (Math.abs(c.p[jo] - c.p[js]) >= 0.5) continue;
            const ja = src === a ? js : jo, jb = src === a ? jo : js;
            c.p[jf] = c.p[ja] + (c.p[jb] - c.p[ja]) * w;
            c.o[jf] = c.o[ja] + (c.o[jb] - c.o[ja]) * w;
            c.e[jf] = c.e[ja] + (c.e[jb] - c.e[ja]) * w;
            c.v[jf] = c.v[ja] + (c.v[jb] - c.v[ja]) * w;
            break;
          }
        }
      }
    }
  }

  build(): NoteTrack {
    this.fill();
    return buildNoteTrack(this.c, this.T, this.frameRate);
  }
}

// ------------------------------------------------------------------ sampling

export const neutralNotes = (): NoteStats => ({ on: 0, held: 0, legato: 0.5, glide: 0, vibrato: 0, pitch: 60, height: 0.5, voice: 0, recent: [] });

/** First index with notes[i].start > t (binary search). */
function upperStart(notes: NoteEvent[], t: number): number {
  let lo = 0, hi = notes.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (notes[m].start <= t) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/** Fills `out` from an offline note track at `time` (frames interpolated; marks from the events). */
export function sampleNoteTrack(tr: NoteTrack, frameRate: number, time: number, out: NoteStats): void {
  const T = tr.on.length;
  if (!T) return;
  const fpos = Math.max(0, time * frameRate);
  const i0 = Math.min(T - 1, Math.floor(fpos));
  const i1 = Math.min(T - 1, i0 + 1);
  const f = i0 === i1 ? 0 : fpos - i0;
  const L = (a: Float32Array) => a[i0] + (a[i1] - a[i0]) * f;
  out.on = L(tr.on);
  out.held = L(tr.held);
  out.legato = L(tr.legato);
  out.glide = L(tr.glide);
  out.vibrato = L(tr.vibrato);
  out.voice = L(tr.voice);
  out.pitch = Math.abs(tr.pitch[i1] - tr.pitch[i0]) < 1 ? L(tr.pitch) : tr.pitch[i0];
  out.height = Math.abs(tr.pitch[i1] - tr.pitch[i0]) < 1 ? L(tr.height) : tr.height[i0];
  const end = upperStart(tr.notes, time);
  const start = Math.max(0, end - NOTE_RECENT);
  const span = Math.max(1, tr.hi - tr.lo);
  out.recent.length = end - start;
  for (let i = start; i < end; i++) {
    const n = tr.notes[i];
    const m = (out.recent[i - start] ??= { age: 0, len: 0, ended: false, height: 0, strength: 0 });
    m.age = time - n.start;
    m.ended = n.end <= time;
    m.len = Math.min(time, n.end) - n.start;
    m.height = clamp01(((n.contour.length ? n.contour[0] : n.pitch) - tr.lo) / span);
    m.strength = n.strength;
  }
}

// ------------------------------------------------------------------ realtime-lite

/**
 * Live: a greedy causal version of the same tracker, fed the main spectrum frame by frame.
 * `out` holds the running articulation after each frame (recent marks aged by the frame clock).
 */
export class NoteTracker {
  readonly out: NoteStats = neutralNotes();
  private readonly pf: PitchFrames;
  private readonly fr: number;
  private t = 0; // frames
  private top = -200;
  private voiced = false;
  private p = NaN; // current line pitch
  private readonly pHist = new Float64Array(4).fill(NaN);
  // The last RING frames' candidates (pitch, level) for the revealed-tone test.
  private readonly ringP = new Float64Array(RING * NOTE_K);
  private readonly ringE = new Float64Array(RING * NOTE_K);
  private readonly ringN = new Uint8Array(RING);
  private prevE = -200;
  private trough = -200;
  private dead = false;
  private novRef = -200;
  private note: LiveNote | null = null;
  private marks: { start: number; end: number; height: number; strength: number }[] = [];
  private markOpen = false;
  private sm = NaN;
  private vr2 = 0;
  private zc = 0;
  private lastR = 0;
  private slope = 0;
  private eLo = -60;
  private eHi = -20;
  private lo = 55;
  private hi = 79;
  private tunX = 0;
  private tunY = 0;
  private legatoN = 0.5;
  private pend: { a: number; eff: number; vibP: number } | null = null;
  private voiceN = 0;

  constructor(n: number, sr: number, frameRate: number) {
    this.pf = new PitchFrames(n, sr, frameRate);
    this.fr = frameRate;
  }

  push(pow: Float64Array): void {
    const fr = this.fr;
    const pf = this.pf;
    pf.push(pow);
    const t = ++this.t;
    this.top = Math.max(pf.level, this.top - 1 / (30 * fr));
    this.novRef -= 6 / fr;
    const g = clamp01((pf.level - (this.top - 45)) / 12);
    // Greedy pick: the best score after the jump cost from the current pitch.
    let bi = -1, bv = -Infinity;
    for (let i = 0; i < pf.count; i++) {
      const v = voicedScore(pf.obs[i], g) - (Number.isNaN(this.p) ? 0 : LIVE_LAMBDA * jumpCost(Math.abs(pf.pitch[i] - this.p)));
      if (v > bv) {
        bv = v;
        bi = i;
      }
    }
    const o = this.out;
    o.on *= Math.exp(-1 / (0.15 * fr));
    o.glide *= 0.8;
    o.vibrato *= 0.9;
    const voicedNow = bi >= 0 && bv > UNVOICED + (this.voiced ? -0.08 : 0.08);
    if (!voicedNow) {
      this.voiced = false;
      this.endNote();
      this.dead = false;
      this.prevE = -200;
      o.held *= Math.exp(-1 / (0.05 * fr));
    } else this.voicedFrame(pf.pitch[bi], pf.energy[bi], pf.novel[bi]);
    // Remember this frame's candidates.
    const r = t % RING;
    this.ringN[r] = pf.count;
    for (let i = 0; i < pf.count; i++) {
      this.ringP[r * NOTE_K + i] = pf.pitch[i];
      this.ringE[r * NOTE_K + i] = pf.energy[i];
    }
    this.settle(false);
    o.legato = this.legatoN;
    o.voice = this.voiceN;
    o.height = clamp01((o.pitch - this.lo) / (this.hi - this.lo));
    this.age();
  }

  private voicedFrame(p: number, e: number, nv: number): void {
    const fr = this.fr, t = this.t, o = this.out;
    this.voiced = true;
    this.eLo = Math.min(this.eLo + 0.3 / fr, e);
    this.eHi = Math.max(this.eHi - 0.3 / fr, e, this.eLo + 12);
    const p2 = this.pHist[(t + 2) & 3];
    this.pHist[t & 3] = p;
    const rising = e > this.prevE;
    this.prevE = e;
    const n = this.note;
    const minLen = Math.max(2, Math.round(0.04 * fr));
    const step = !!n && t - n.a >= minLen && !Number.isNaN(p2) && Math.abs(p - p2) > 0.8;
    // Re-attack: the level fell from the note's peak and rises again; a note whose level fell
    // 15 dB under its peak is over (a pluck's tail), and the next one needs such a rise.
    const re = !!n && t - n.a >= minLen && n.peak - this.trough >= 4 && e - this.trough >= 4.5 && rising;
    const died = !!n && e < n.peak - 15;
    if (died) {
      this.endNote();
      this.dead = true;
      this.trough = e;
    } else if (!n && this.dead) {
      if (e - this.trough > 4.5 && rising) this.dead = false;
      else this.trough = Math.min(this.trough, e);
    }
    if (!died && !this.dead && (!n || step || re)) {
      this.endNote();
      this.startNote(p, e);
      this.trough = e;
    }
    this.p = p;
    const cur = this.note;
    if (!cur) return;
    if (e > cur.peak) {
      cur.peak = e;
      this.trough = e;
    } else if (e < this.trough) this.trough = e;
    if (e >= cur.peak - 9) cur.eff = t - cur.a + 1;
    cur.nov = Math.max(cur.nov, nv);
    // Confirmed two frames in, or ~60 ms in when this pitch was already sounding before (its level
    // is still rising through the window): a melody note, or a background / revealed tone.
    const late = cur.before > -150;
    if (!cur.done && t >= cur.a + (late ? Math.max(3, Math.round(0.1 * fr)) : 2)) {
      cur.done = true;
      cur.bg = cur.before >= cur.peak - 3 || cur.nov < this.novRef - BACKGROUND_DB;
      if (!cur.bg) {
        this.novRef = Math.max(this.novRef, cur.nov);
        this.settle(true);
        const str = clamp01((cur.peak - this.eLo) / (this.eHi - this.eLo));
        o.on = Math.max(o.on, 0.3 + 0.7 * str);
        this.marks.push({ start: cur.a, end: t, height: clamp01((cur.first - this.lo) / (this.hi - this.lo)), strength: str });
        if (this.marks.length > NOTE_RECENT) this.marks.shift();
        this.markOpen = true;
      }
    }
    cur.sum += p;
    cur.n++;
    // Smoothed contour, residual (vibrato) and slope (glide).
    const prevSm = this.sm;
    this.sm = Number.isNaN(this.sm) ? p : this.sm + (p - this.sm) * (1 - Math.exp(-1 / (0.07 * fr)));
    const r = p - this.sm;
    const kV = 1 - Math.exp(-1 / (0.2 * fr));
    this.vr2 += (r * r - this.vr2) * kV;
    this.zc += (((r > 0) !== (this.lastR > 0) ? fr / 2 : 0) - this.zc) * kV;
    this.lastR = r;
    const vibNow = this.zc >= 3 && this.zc <= 10 ? Math.sqrt(2 * this.vr2) : 0;
    if (!Number.isNaN(prevSm)) this.slope += ((this.sm - prevSm) * fr - this.slope) * (1 - Math.exp(-1 / (0.06 * fr)));
    cur.vib += vibNow;
    cur.d1 += this.sm;
    cur.d2 += this.sm * this.sm;
    if (cur.n === Math.max(2, Math.round(0.05 * fr))) cur.scoop = cur.sum / cur.n;
    if (cur.bg) {
      o.held *= Math.exp(-1 / (0.05 * fr));
      return;
    }
    o.vibrato = vibNow;
    o.glide = this.slope;
    o.held = Math.max(clamp01((e - this.eLo) / (this.eHi - this.eLo)), 0.15);
    o.pitch = p;
    this.lo = Math.min(this.lo + 1 / (10 * fr), p);
    this.hi = Math.max(this.hi - 1 / (10 * fr), p, this.lo + 12);
    // A long held note keeps pulling the legato up while it sounds.
    if ((t - cur.a) / fr > 0.3 && cur.eff / fr > 0.3) this.legatoN += (1 - this.legatoN) * (1 - Math.exp(-1 / (0.5 * fr)));
    if (this.markOpen) this.marks[this.marks.length - 1].end = t;
  }

  /** Scores the last finished note's legato once the time to the next onset is known (or 0.6 s passed). */
  private settle(force: boolean): void {
    const q = this.pend;
    if (!q) return;
    const ioi = (this.t - q.a) / this.fr;
    if (!force && ioi < 0.6) return;
    this.pend = null;
    const r = q.eff / this.fr / Math.max(1e-3, Math.min(0.6, ioi));
    const leg = clamp01(Math.max((r - 0.4) / 0.45, q.vibP * 0.8));
    this.legatoN += (leg - this.legatoN) * 0.35;
  }

  private startNote(p: number, e: number): void {
    const t = this.t;
    // Revealed: this pitch was already sounding 50-100 ms ago at about this level.
    const b0 = Math.max(3, Math.round(0.05 * this.fr)), b1 = Math.min(RING - 1, Math.max(b0 + 1, Math.round(0.1 * this.fr)));
    let before = -200;
    for (let k = b0; k <= b1 && k < t; k++) {
      const r = (t - k) % RING;
      for (let i = 0; i < this.ringN[r]; i++) if (Math.abs(this.ringP[r * NOTE_K + i] - p) < 0.5) before = Math.max(before, this.ringE[r * NOTE_K + i]);
    }
    this.note = { a: t, peak: e, eff: 1, sum: 0, n: 0, d1: 0, d2: 0, vib: 0, first: p, scoop: p, nov: -200, bg: true, done: false, before };
    this.sm = p;
    this.vr2 = 0;
    this.slope = 0;
  }

  private endNote(): void {
    const n = this.note;
    if (!n) return;
    this.note = null;
    this.markOpen = false;
    if (n.bg) return;
    const fr = this.fr;
    const len = (this.t - n.a) / fr;
    const med = n.n ? n.sum / n.n : n.first;
    // Its legato waits for the next note's start (the inter-onset time), at most 0.6 s.
    const vibP = clamp01(((n.n ? n.vib / n.n : 0) - 0.12) / 0.3);
    this.pend = { a: n.a, eff: n.eff, vibP };
    if (len >= 0.15 && n.n > 2) {
      const ang = 2 * Math.PI * (med - Math.round(med));
      this.tunX += Math.cos(ang) * len;
      this.tunY += Math.sin(ang) * len;
      const tuning = Math.atan2(this.tunY, this.tunX) / (2 * Math.PI);
      const off = clamp01((Math.abs(med - tuning - Math.round(med - tuning)) * 100 - 8) / 25);
      const drift = Math.sqrt(Math.max(0, n.d2 / n.n - (n.d1 / n.n) ** 2));
      const v = clamp01(0.45 * vibP + 0.35 * clamp01((drift - 0.04) / 0.2) + 0.2 * clamp01((Math.abs(n.scoop - med) - 0.25) / 0.8) + 0.3 * off);
      this.voiceN += (v - this.voiceN) * Math.min(1, len / 3);
    }
  }

  /** Recent marks against the frame clock. */
  private age(): void {
    const o = this.out;
    const fr = this.fr;
    const now = this.t;
    o.recent.length = this.marks.length;
    for (let i = 0; i < this.marks.length; i++) {
      const m = this.marks[i];
      const r: NoteMark = (o.recent[i] ??= { age: 0, len: 0, ended: false, height: 0, strength: 0 });
      const live = this.markOpen && i === this.marks.length - 1;
      r.age = (now - m.start) / fr;
      r.ended = !live;
      r.len = ((live ? now : m.end) - m.start) / fr;
      r.height = m.height;
      r.strength = m.strength;
    }
  }
}

/** Frames of candidate history the live tracker keeps. */
const RING = 12;

interface LiveNote {
  a: number;
  peak: number;
  eff: number;
  sum: number;
  n: number;
  d1: number;
  d2: number;
  vib: number;
  first: number;
  scoop: number;
  /** Peak novelty level so far (dB). */
  nov: number;
  /** A background or revealed tone: tracked for the line, but not reported as a note. */
  bg: boolean;
  /** The background test has run. */
  done: boolean;
  /** Level of this pitch 50-100 ms before the note (a revealed tone was already sounding). */
  before: number;
}

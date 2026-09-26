// Live notes with a small visual lag: the notes are drawn a fixed V seconds
// behind the sound, and the note tracker uses those V seconds as look-ahead.
// Everything predictable or measured without look-ahead (the beat and bar
// clock, hits, stems, levels) stays at the sound's time.
//
// LookaheadNotes runs the offline note pipeline (notes.ts) in a sliding window
// on the undelayed input:
//   - the same pitch candidates as offline (PitchFrames on every third frame
//     of the main spectrum, the frames between interpolated; frames numbered
//     and centred as analyzePcm numbers them),
//   - a fixed-lag Viterbi line: every couple of frames viterbiLine runs over the
//     frames not yet committed, continuing from the last committed pitch, and
//     commits the frames up to the display time (the frames after it stay
//     provisional and are re-decided next time),
//   - over the last WINDOW_S seconds of that line, the offline segmentation,
//     rejectBackground / rejectRevealed and note scoring (noteTrackFromLine),
//   - articulation from what can be known at the display time: notes that
//     have ended score as offline (the next note's start is known), a note
//     still sounding counts as held once it has lasted, and the legato around
//     now is a phrase-level running estimate over the last few seconds.
//
// LiveLookahead feeds every block to the look-ahead notes and to the
// RealtimeAnalyzer at once; only the notes are sampled V seconds back, so a
// shared tab (whose sound cannot be delayed) gets its beats, hits and stems on
// time and its notes V late.

import type { NoteStats, NoteTrack } from '../types';
import { RealFFT } from './fft';
import { hann } from './dsp';
import { StreamDecimator } from './rtUtil';
import { NOTE_K, PITCH_EVERY, PitchFrames, fillFrame, noteTrackFromLine, sampleNoteTrack, viterbiLine, type NoteCands, type NoteLine } from './notes';
import type { RealtimeAnalyzer } from './RealtimeAnalyzer';

/** Visual lag used by the app, seconds. */
export const DEFAULT_VISUAL_LAG = 0.1;
/** Frames of candidates and line kept (power of two, ~47 s at 86 fps). */
const CAP = 4096;
/** Seconds of line the notes are rebuilt over. */
const WINDOW_S = 10;
/** Frames between rebuilds (~23 ms). */
const REBUILD_EVERY = 2;
/** Level histogram: 1 dB bins from LVL_LO. */
const LVL_LO = -160;
const LVL_BINS = 220;

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

function pow2Near(x: number): number {
  return Math.pow(2, Math.max(5, Math.round(Math.log2(Math.max(32, x)))));
}

/** Width of the frame legato's weighting of nearby notes, seconds (offline: 0.6, both sides). */
const LEGATO_SIGMA = 0.6;
/** Memory of the phrase-style running estimate, seconds. */
const PHRASE_TAU = 2;

export class LookaheadNotes {
  readonly factor: number;
  readonly sr: number;
  readonly hop: number;
  readonly n: number;
  readonly frameRate: number;
  /** Seconds from a frame's centre until the frame is complete (half the window). */
  readonly latency: number;
  /** Seconds the display runs behind the input. */
  lag: number;
  /** Seconds of input consumed. */
  streamTime = 0;

  private readonly inputRate: number;
  private readonly dec: StreamDecimator;
  private readonly ring: Float32Array;
  private ringPos = 0;
  private decCount = 0;
  private frames = 0;
  private readonly fft: RealFFT;
  private readonly win: Float32Array;
  private readonly buf: Float64Array;
  private readonly pow: Float64Array;
  private readonly pf: PitchFrames;
  private readonly lead: number; // frames between a live frame's end and its centre (n / 2 / hop)

  // Candidate history by offline frame index f (ring of CAP).
  private readonly h: NoteCands;
  private last = -1; // newest frame index stored
  // Committed line.
  private readonly lp = new Float32Array(CAP);
  private readonly le = new Float32Array(CAP);
  private readonly lv = new Float32Array(CAP);
  private committed = -1;
  private readonly lvlHist = new Float64Array(LVL_BINS);
  private lvlN = 0;
  private sinceBuild = 0;

  private track: NoteTrack | null = null;
  private trackStart = 0;

  constructor(inputRate: number, lag = DEFAULT_VISUAL_LAG) {
    this.inputRate = inputRate > 0 ? inputRate : 44100;
    this.factor = Math.max(1, Math.round(this.inputRate / 22050));
    this.sr = this.inputRate / this.factor;
    this.hop = pow2Near(this.sr / 86);
    this.n = this.hop * 8;
    this.frameRate = this.sr / this.hop;
    this.latency = this.n / 2 / this.sr;
    this.lead = this.n / 2 / this.hop;
    this.lag = lag;
    this.dec = new StreamDecimator(this.factor);
    this.ring = new Float32Array(this.n);
    this.fft = new RealFFT(this.n);
    this.win = hann(this.n);
    this.buf = new Float64Array(this.n);
    this.pow = new Float64Array(this.n / 2 + 1);
    this.pf = new PitchFrames(this.n, this.sr, this.frameRate / PITCH_EVERY);
    const z = () => new Float32Array(CAP * NOTE_K);
    this.h = { p: z(), o: z(), e: z(), v: z(), n: new Uint8Array(CAP), level: new Float32Array(CAP) };
  }

  /** Feed a block of input (right may be the same array as left). */
  process(left: Float32Array, right: Float32Array): void {
    const len = Math.min(left.length, right.length);
    if (len <= 0) return;
    const mid = new Float32Array(len);
    for (let i = 0; i < len; i++) mid[i] = 0.5 * (left[i] + right[i]);
    this.dec.process(mid, (v) => this.pushSample(v));
    this.streamTime += len / this.inputRate;
  }

  private pushSample(v: number): void {
    this.ring[this.ringPos] = Number.isFinite(v) ? v : 0;
    this.ringPos = (this.ringPos + 1) % this.n;
    if (++this.decCount % this.hop === 0) this.frame();
  }

  private frame(): void {
    // The live frame ending here is centred n/2 samples back: offline frame f = k - lead.
    const f = ++this.frames - this.lead;
    if (f < 0) return;
    const h = this.h;
    const slot = (x: number) => x & (CAP - 1);
    const r = slot(f);
    if (f % PITCH_EVERY === 0) {
      // As offline: the pitch estimator on every third frame, the frames between interpolated.
      const n = this.n;
      for (let i = 0; i < n; i++) this.buf[i] = this.ring[(this.ringPos + i) % n] * this.win[i];
      this.fft.power(this.buf, this.pow);
      const pf = this.pf;
      pf.push(this.pow);
      h.n[r] = pf.count;
      h.level[r] = pf.level;
      for (let i = 0; i < pf.count; i++) {
        const j = r * NOTE_K + i;
        h.p[j] = pf.pitch[i];
        h.o[j] = pf.obs[i];
        h.e[j] = pf.energy[i];
        h.v[j] = pf.novel[i];
      }
      const a = f - PITCH_EVERY;
      if (a >= 0) for (let g = a + 1; g < f; g++) fillFrame(h, a, f, g, slot);
      if (pf.level > -150) {
        this.lvlHist[Math.max(0, Math.min(LVL_BINS - 1, Math.floor(pf.level - LVL_LO)))]++;
        this.lvlN++;
      }
    } else {
      // Held from the last analysed frame until the next one arrives.
      fillFrame(h, f - (f % PITCH_EVERY), -1, f, slot);
    }
    this.last = f;
    if (++this.sinceBuild >= REBUILD_EVERY) {
      this.sinceBuild = 0;
      this.rebuild();
    }
  }

  /** The 97th percentile of the region level so far (as viterbiLine's levelTop over the song). */
  private levelTop(): number {
    if (!this.lvlN) return -200;
    const want = this.lvlN * 0.97;
    let acc = 0;
    for (let b = 0; b < LVL_BINS; b++) {
      acc += this.lvlHist[b];
      if (acc >= want) return LVL_LO + b + 0.5;
    }
    return LVL_LO + LVL_BINS;
  }

  /** Copies frames [a, a + T) of the candidate ring into contiguous arrays. */
  private cands(a: number, T: number): NoteCands {
    const K = NOTE_K, h = this.h;
    const c: NoteCands = {
      p: new Float32Array(T * K), o: new Float32Array(T * K), e: new Float32Array(T * K), v: new Float32Array(T * K),
      n: new Uint8Array(T), level: new Float32Array(T),
    };
    for (let t = 0; t < T; t++) {
      const r = (a + t) & (CAP - 1);
      c.n[t] = h.n[r];
      c.level[t] = h.level[r];
      const cnt = h.n[r];
      for (let i = 0; i < cnt; i++) {
        c.p[t * K + i] = h.p[r * K + i];
        c.o[t * K + i] = h.o[r * K + i];
        c.e[t * K + i] = h.e[r * K + i];
        c.v[t * K + i] = h.v[r * K + i];
      }
    }
    return c;
  }

  private rebuild(): void {
    const f = this.last;
    const fr = this.frameRate;
    // Frames centred at or before the display time are committed; the rest stay provisional.
    const future = Math.max(0, Math.floor((this.lag - this.latency) * fr));
    const cf = f - future;
    const c0 = Math.max(this.committed + 1, f - CAP + 1);
    const T1 = f - c0 + 1;
    if (T1 <= 0) return;
    const init = this.committed >= 0 && c0 === this.committed + 1 ? this.lp[this.committed & (CAP - 1)] : undefined;
    const path = viterbiLine(this.cands(c0, T1), T1, { top: this.levelTop(), init });
    for (let t = c0; t <= cf; t++) {
      const r = t & (CAP - 1);
      this.lp[r] = path.pitch[t - c0];
      this.le[r] = path.energy[t - c0];
      this.lv[r] = path.novel[t - c0];
    }
    if (cf >= c0) this.committed = cf;
    // The notes over the window: committed line, then the provisional tail.
    const w0 = Math.max(0, f - Math.round(WINDOW_S * fr) + 1, f - CAP + 1);
    const T = f - w0 + 1;
    const line: NoteLine = { pitch: new Float32Array(T), energy: new Float32Array(T), novel: new Float32Array(T) };
    for (let t = w0; t <= f; t++) {
      const i = t - w0;
      if (t <= this.committed) {
        const r = t & (CAP - 1);
        line.pitch[i] = this.lp[r];
        line.energy[i] = this.le[r];
        line.novel[i] = this.lv[r];
      } else {
        line.pitch[i] = path.pitch[t - c0];
        line.energy[i] = path.energy[t - c0];
        line.novel[i] = path.novel[t - c0];
      }
    }
    this.track = noteTrackFromLine(this.cands(w0, T), T, fr, line);
    this.trackStart = w0;
  }

  /**
   * Articulation at `time` (seconds on the input's clock; the display time,
   * normally streamTime - lag). Notes that start after `time` are not shown yet.
   */
  sample(time: number, out: NoteStats): void {
    const tr = this.track;
    if (!tr) return;
    const t0 = this.trackStart / this.frameRate;
    const local = time - t0;
    sampleNoteTrack(tr, this.frameRate, local, out);
    out.legato = this.phraseLegato(tr, local, out.legato);
  }

  /**
   * Legato around `t` from what is known by then (the newest frame, `lag` ahead of `t`). A note's
   * offline score needs the time to the next note; until that is known it lies between bounds:
   * a note still sounding is at least as legato as it has been held so far, a note that has ended
   * with no next note yet is at most as legato as its gap allows. Inside the bounds the phrase's
   * style decides: the running average of the final scores of the last few seconds of notes.
   * The frame legato is then the offline weighting of the notes around `t`, past side only.
   */
  private phraseLegato(tr: NoteTrack, t: number, fallback: number): number {
    const notes = tr.notes;
    const known = (tr.on.length - 1) / this.frameRate;
    const end = upper(notes, t);
    // Phrase style: final scores, weighted by recency (seconds) and strength.
    let pw = 0, ps = 0;
    for (let i = end - 1; i >= 0; i--) {
      const n = notes[i];
      const d = Math.max(0, t - n.end);
      if (d > 3 * PHRASE_TAU) break;
      if (!(i + 1 < notes.length || known - n.start >= 0.6)) continue;
      const w = Math.exp(-d / PHRASE_TAU) * (0.3 + n.strength);
      pw += w;
      ps += w * n.legato;
    }
    const prior = pw > 0.05 ? ps / pw : pw > 0 ? 0.5 + (ps / pw - 0.5) * (pw / 0.05) : 0.5;
    let w = 0, s = 0;
    for (let i = end - 1; i >= 0; i--) {
      const n = notes[i];
      const d = n.end >= t ? 0 : t - n.end;
      if (d > 3 * LEGATO_SIGMA) break;
      let leg = n.legato;
      if (!(i + 1 < notes.length || known - n.start >= 0.6)) {
        // n.legato was scored as if the next note came 0.6 s after this one: a lower bound.
        const sounding = n.end >= known - 1.5 / this.frameRate;
        const hi = sounding ? 1 : clamp01(((n.end - n.start) / Math.max(1e-3, known - n.start) - 0.4) / 0.45);
        leg = Math.min(Math.max(prior, n.legato), Math.max(hi, n.legato));
      }
      const wi = Math.exp(-((d / LEGATO_SIGMA) ** 2)) * (0.3 + n.strength);
      w += wi;
      s += wi * leg;
    }
    if (w > 0.05) return s / w;
    return w > 0 ? prior + (s / w - prior) * (w / 0.05) : Number.isFinite(fallback) ? fallback : prior;
  }
}

/** First index with notes[i].start > t. */
function upper(notes: NoteTrack['notes'], t: number): number {
  let lo = 0, hi = notes.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (notes[m].start <= t) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/**
 * The look-ahead notes beside the analyzer: both get every block at once; the
 * notes are sampled `lag` seconds behind the analyzer's clock (the one the
 * visuals are drawn on), which is their look-ahead.
 */
export class LiveLookahead {
  readonly analyzer: RealtimeAnalyzer;
  readonly notes: LookaheadNotes;
  /** Analyzer stream time when this started (the notes' clock starts there). */
  private readonly origin: number;

  constructor(analyzer: RealtimeAnalyzer, lag = DEFAULT_VISUAL_LAG) {
    this.analyzer = analyzer;
    this.origin = analyzer.streamTime;
    this.notes = new LookaheadNotes(analyzer.inputRate, lag);
    analyzer.notesLiveOn = false;
  }

  get lag(): number {
    return this.notes.lag;
  }

  process(left: Float32Array, right: Float32Array): void {
    this.notes.process(left, right);
    this.analyzer.process(left, right);
  }

  /** Nothing is held back any more (kept for callers switching the lag off). */
  flush(): void {}

  /** NoteStats source for RealtimeSampler.noteSource: the notes `lag` behind the analyzer's clock. */
  readonly sampleNotes = (time: number, out: NoteStats): void => this.notes.sample(time - this.notes.lag - this.origin, out);
}

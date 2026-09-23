// Real-time beat tracking for live input.
//
// Onset strength (spectral flux) arrives one frame at a time (~86 Hz).
// - Tempo: autocorrelation of the last ~8 s of onset strength with a
//   log-Gaussian prior around 120 BPM (same scoring as the offline
//   estimateTempo), folded into 70..180 BPM, with hysteresis.
// - Phase: a comb over the last ~4 s finds where the beats fall; a
//   phase-locked loop keeps a continuous beat clock aligned to it, so the
//   clock extrapolates smoothly between frames and through short gaps.
// - Downbeat: accent pattern per beat slot (kick + bass onsets + chord
//   changes, minus snare), assuming 4/4.

import { History, wrapHalf } from './rtUtil';

const BEATS_PER_BAR = 4;
const TEMPO_WINDOW_S = 8;
const TEMPO_MIN_S = 2.5;
const TEMPO_EVAL_S = 0.5;
const COMB_WINDOW_S = 4;
const PRIOR_BPM = 120;
const PRIOR_OCT_SD = 0.9;
const BPM_LO = 70;
const BPM_HI = 180;

export interface BeatAccent {
  kick: number;
  bass: number;
  snare: number;
}

export class BeatTracker {
  readonly frameRate: number;
  private readonly env: History;
  private readonly kick: History;
  private readonly bassOn: History;
  private readonly snare: History;
  private readonly acfBuf: Float64Array;
  private readonly acf: Float64Array;

  // Tempo state
  bpm = PRIOR_BPM;
  private haveTempo = false;
  private pendingBpm = 0;
  private pendingCount = 0;
  private stableEvals = 0;
  private evalCountdown = 0;

  // Beat clock: beatPos(t) = refPos + (t - refTime) / period
  period = 60 / PRIOR_BPM; // seconds
  refTime = 0;
  refPos = 0;

  // Lock / confidence
  locked = false;
  confidence = 0;
  combConfidence = 0;
  /** Stream time of the first lock since the last reset (NaN until then). */
  firstLockTime = NaN;
  private errEma = 0.5;
  /** Smoothed comb strength half a beat off the best phase, relative to it (1 = no beat reference). */
  private halfRatio = 0;
  private quietFrames = 0;

  // Downbeat
  downbeatSlot = 0;
  private readonly slotScore = new Float64Array(BEATS_PER_BAR);
  private lastBeatFloor = -Infinity;
  private pendingAccents: { beat: number; age: number }[] = [];
  private readonly beatChroma = new Float64Array(12);
  private readonly prevBeatChroma = new Float64Array(12);
  private beatChromaFrames = 0;
  private haveBeatChroma = false;

  private frames = 0;

  constructor(frameRate: number) {
    this.frameRate = frameRate;
    const len = Math.ceil(TEMPO_WINDOW_S * frameRate) + 8;
    this.env = new History(len);
    this.kick = new History(64);
    this.bassOn = new History(64);
    this.snare = new History(64);
    this.acfBuf = new Float64Array(len);
    this.acf = new Float64Array(len);
  }

  reset(): void {
    this.env.reset();
    this.kick.reset();
    this.bassOn.reset();
    this.snare.reset();
    this.bpm = PRIOR_BPM;
    this.haveTempo = false;
    this.pendingCount = 0;
    this.stableEvals = 0;
    this.evalCountdown = 0;
    this.period = 60 / PRIOR_BPM;
    this.locked = false;
    this.confidence = 0;
    this.combConfidence = 0;
    this.firstLockTime = NaN;
    this.errEma = 0.5;
    this.halfRatio = 0;
    this.slotScore.fill(0);
    this.downbeatSlot = 0;
    this.pendingAccents = [];
    this.beatChroma.fill(0);
    this.prevBeatChroma.fill(0);
    this.beatChromaFrames = 0;
    this.haveBeatChroma = false;
    this.lastBeatFloor = -Infinity;
    this.quietFrames = 0;
    this.frames = 0;
  }

  /** Continuous beat position at stream time t (seconds). */
  positionAt(t: number): number {
    return this.refPos + (t - this.refTime) / this.period;
  }

  /**
   * Push one analysis frame.
   * @param onset  onset strength (normalized spectral flux, >= 0)
   * @param t      stream time of the frame, seconds
   * @param active false during silence (no beat evidence)
   */
  push(onset: number, t: number, active: boolean, accent: BeatAccent, chroma: ArrayLike<number> | null): void {
    const fr = this.frameRate;
    this.frames++;
    if (this.frames === 1) {
      this.refTime = t;
      this.refPos = 0;
    }
    this.env.push(active ? onset : 0);
    this.kick.push(accent.kick);
    this.bassOn.push(accent.bass);
    this.snare.push(accent.snare);

    if (active) this.quietFrames = 0;
    else this.quietFrames++;
    if (this.quietFrames > 2 * fr) {
      // Long silence: forget the lock, keep the clock free-running.
      this.locked = false;
      this.stableEvals = 0;
      this.errEma = Math.min(0.5, this.errEma + 0.002);
    }

    // --- Tempo, a couple of times per second ---
    if (--this.evalCountdown <= 0) {
      this.evalCountdown = Math.max(1, Math.round(TEMPO_EVAL_S * fr));
      if (this.env.count >= TEMPO_MIN_S * fr && this.quietFrames < fr) this.estimateTempo();
    }

    // --- Phase (comb) + PLL ---
    const P = (60 * fr) / this.bpm; // frames per beat
    const span = Math.min(this.env.count, Math.round(COMB_WINDOW_S * fr));
    if (this.haveTempo && span > P * 2 && this.quietFrames < fr * 0.5) {
      let best = -Infinity;
      let bestPhi = 0;
      let sum = 0;
      let n = 0;
      const decay = 1 / (2 * fr);
      for (let phi = 0; phi < P; phi += 0.5) {
        let s = 0;
        for (let age = phi; age < span; age += P) {
          const w = Math.exp(-age * decay);
          s += w * (this.env.atFrac(age) + 0.5 * (this.env.atFrac(age - 1) + this.env.atFrac(age + 1)));
        }
        sum += s;
        n++;
        if (s > best) {
          best = s;
          bestPhi = phi;
        }
      }
      const meanS = n > 0 ? sum / n : 0;
      const cc = best > 1e-9 ? (best - meanS) / best : 0;
      this.combConfidence += (cc - this.combConfidence) * 0.05;
      // Sub-step refinement: parabola through the neighbours of the best phase.
      const sc = (phi: number) => {
        let s = 0;
        for (let age = phi; age < span; age += P) {
          const w = Math.exp(-age * decay);
          s += w * (this.env.atFrac(age) + 0.5 * (this.env.atFrac(age - 1) + this.env.atFrac(age + 1)));
        }
        return s;
      };
      const a = sc(bestPhi - 0.5 < 0 ? bestPhi - 0.5 + P : bestPhi - 0.5);
      const c = sc(bestPhi + 0.5 >= P ? bestPhi + 0.5 - P : bestPhi + 0.5);
      const den = a - 2 * best + c;
      if (den < 0) bestPhi += Math.max(-0.5, Math.min(0.5, (0.5 * 0.5 * (a - c)) / den));

      // Half-beat ambiguity (e.g. an even snare roll with no kick): the comb cannot
      // tell beats from off-beats, so a locked clock freewheels instead of slipping.
      const half = bestPhi + P / 2 >= P ? bestPhi + P / 2 - P : bestPhi + P / 2;
      const hr = best > 1e-9 ? sc(half) / best : 1;
      this.halfRatio += (hr - this.halfRatio) * 0.1;
      const ambiguous = this.locked && (hr > 0.75 || this.halfRatio > 0.75);
      if (cc > 0.15 && !ambiguous) {
        const measFrac = bestPhi / P; // phase since the last beat at time t
        const pos = this.positionAt(t);
        const e = wrapHalf(measFrac - (pos - Math.floor(pos)));
        const ae = Math.abs(e);
        this.errEma += (ae - this.errEma) * (1 - Math.exp(-1 / fr));
        // Fast pull-in before lock, gentle tracking after.
        const alpha = this.locked ? (ae > 0.2 ? 0.02 : 0.06) : 0.25;
        const newPos = pos + alpha * e;
        this.refPos = newPos;
        this.refTime = t;
        // Frequency: follow the tempo estimate, plus an integral term on the phase error.
        const target = 60 / this.bpm;
        let per = this.period + (target - this.period) * (this.locked ? 0.01 : 0.1);
        per *= 1 - 0.004 * e;
        // The integral term may only fine-tune: stay within 1 % of the tempo estimate.
        this.period = Math.max(target / 1.01, Math.min(target * 1.01, per));
      }
    } else {
      this.combConfidence *= 0.995;
      // Re-anchor so positionAt stays numerically tidy.
      const pos = this.positionAt(t);
      this.refPos = pos;
      this.refTime = t;
    }

    // --- Lock ---
    const wasLocked = this.locked;
    if (!this.locked) {
      if (this.haveTempo && this.stableEvals >= 2 && this.errEma < 0.07 && this.combConfidence > 0.3 && this.quietFrames < fr * 0.5) {
        this.locked = true;
        if (!Number.isFinite(this.firstLockTime)) this.firstLockTime = t;
      }
    } else if (this.errEma > 0.16 || this.combConfidence < 0.15) {
      this.locked = false;
    }
    if (this.locked && !wasLocked) this.errEma = Math.min(this.errEma, 0.05);
    const cTarget = this.locked ? Math.min(1, 0.4 + this.combConfidence) : Math.min(0.35, this.combConfidence);
    this.confidence += (cTarget - this.confidence) * 0.05;

    // --- Downbeat accents ---
    this.trackDownbeat(t, chroma);
  }

  private trackDownbeat(t: number, chroma: ArrayLike<number> | null): void {
    if (chroma) {
      for (let k = 0; k < 12; k++) this.beatChroma[k] += chroma[k];
      this.beatChromaFrames++;
    }
    const pos = this.positionAt(t);
    const fl = Math.floor(pos);
    if (fl !== this.lastBeatFloor) {
      const first = !Number.isFinite(this.lastBeatFloor);
      this.lastBeatFloor = fl;
      if (!first && this.quietFrames === 0) {
        this.pendingAccents.push({ beat: fl, age: 0 });
        // Chord change at the beat that just ended vs the one before it.
        if (this.beatChromaFrames > 0) {
          if (this.haveBeatChroma) {
            let dot = 0,
              na = 0,
              nb = 0;
            for (let k = 0; k < 12; k++) {
              const a = this.beatChroma[k] / this.beatChromaFrames;
              const b = this.prevBeatChroma[k];
              dot += a * b;
              na += a * a;
              nb += b * b;
            }
            const change = na > 0 && nb > 0 ? 1 - dot / Math.sqrt(na * nb) : 0;
            // `fl - 1` is the beat whose chroma we just closed; its start is the change point.
            this.addAccent(fl - 1, 3 * change);
          }
          for (let k = 0; k < 12; k++) this.prevBeatChroma[k] = this.beatChroma[k] / this.beatChromaFrames;
          this.haveBeatChroma = true;
        }
      }
      this.beatChroma.fill(0);
      this.beatChromaFrames = 0;
    }
    // Onset accents a few frames after each beat (so the attack is inside the window).
    for (let i = this.pendingAccents.length - 1; i >= 0; i--) {
      const pa = this.pendingAccents[i];
      pa.age++;
      if (pa.age < 4) continue;
      let k = 0,
        b = 0,
        s = 0;
      for (let a = 0; a < 8; a++) {
        k = Math.max(k, this.kick.at(a));
        b = Math.max(b, this.bassOn.at(a));
        s = Math.max(s, this.snare.at(a));
      }
      this.addAccent(pa.beat, k + 0.5 * b - 0.6 * s);
      this.pendingAccents.splice(i, 1);
    }
  }

  private addAccent(beat: number, v: number): void {
    const slot = ((beat % BEATS_PER_BAR) + BEATS_PER_BAR) % BEATS_PER_BAR;
    // Slow decay (per accent contribution), so the pattern of the last ~8 bars dominates.
    for (let k = 0; k < BEATS_PER_BAR; k++) this.slotScore[k] *= 0.985;
    this.slotScore[slot] += v;
    let best = this.downbeatSlot;
    for (let k = 0; k < BEATS_PER_BAR; k++) if (this.slotScore[k] > this.slotScore[best]) best = k;
    const cur = this.slotScore[this.downbeatSlot];
    if (best !== this.downbeatSlot && this.slotScore[best] > cur + 0.3 * Math.abs(cur) + 0.5) this.downbeatSlot = best;
  }

  private estimateTempo(): void {
    const fr = this.frameRate;
    const e = this.acfBuf;
    const T = this.env.copyOrdered(e);
    if (T < 8) return;
    let mean = 0;
    for (let i = 0; i < T; i++) mean += e[i];
    mean /= T;
    for (let i = 0; i < T; i++) e[i] -= mean;
    const maxLag = Math.min(T - 1, Math.ceil((4 * 60 * fr) / 60) + 2);
    const acf = this.acf;
    for (let l = 0; l <= maxLag; l++) {
      let s = 0;
      for (let i = 0; i + l < T; i++) s += e[i] * e[i + l];
      acf[l] = s / (T - l);
    }
    const a0 = acf[0] > 0 ? acf[0] : 0;
    if (!(a0 > 1e-12)) return;
    const at = (lag: number): number => {
      if (lag >= maxLag) return 0;
      const i = Math.floor(lag);
      const f = lag - i;
      return (acf[i] * (1 - f) + acf[i + 1] * f) / a0;
    };
    let bestBpm = 0;
    let bestScore = -Infinity;
    for (let bpm = 60; bpm <= 200; bpm += 0.1) {
      const L = (60 * fr) / bpm;
      const s = at(L) + 0.5 * at(2 * L) + 0.25 * at(4 * L);
      const oct = Math.log2(bpm / PRIOR_BPM) / PRIOR_OCT_SD;
      const v = s * Math.exp(-0.5 * oct * oct);
      if (v > bestScore) {
        bestScore = v;
        bestBpm = bpm;
      }
    }
    if (!(bestScore > 0.02)) return;
    // Parabolic refinement on the grid.
    const scoreAt = (bpm: number) => {
      const L = (60 * fr) / bpm;
      const oct = Math.log2(bpm / PRIOR_BPM) / PRIOR_OCT_SD;
      return (at(L) + 0.5 * at(2 * L) + 0.25 * at(4 * L)) * Math.exp(-0.5 * oct * oct);
    };
    {
      const a = scoreAt(bestBpm - 0.1);
      const c = scoreAt(bestBpm + 0.1);
      const den = a - 2 * bestScore + c;
      if (den < 0) bestBpm += Math.max(-0.1, Math.min(0.1, (0.1 * 0.5 * (a - c)) / den));
    }
    let bpm = bestBpm;
    while (bpm < BPM_LO) bpm *= 2;
    while (bpm > BPM_HI) bpm /= 2;
    // Octave check: a slow candidate whose off-beats are about as strong as its
    // beats is really the faster tempo (e.g. 87 vs 174 BPM with a kick on every beat).
    if (bpm < 100 && bpm * 2 <= BPM_HI && this.halfBeatRatio(bpm) > 0.55) bpm *= 2;

    if (!this.haveTempo) {
      this.bpm = bpm;
      this.period = 60 / bpm;
      this.haveTempo = true;
      this.stableEvals = 0;
      return;
    }
    const ratio = bpm / this.bpm;
    if (this.locked) {
      // Locked: refine locally (+-2 %) and only switch to a genuinely different
      // tempo after sustained, clearly stronger evidence. Candidates at simple
      // ratios of the current tempo (half / double time, triplet feel, snare
      // rolls in a build) are the same groove and never cause a switch.
      let localBpm = this.bpm;
      let localScore = -Infinity;
      for (let b = this.bpm * 0.98; b <= this.bpm * 1.02; b += 0.05) {
        const v = scoreAt(b);
        if (v > localScore) {
          localScore = v;
          localBpm = b;
        }
      }
      // Only follow a clear interior peak: a maximum at the edge of the window is
      // evidence for some other periodicity (e.g. a snare roll), not a tempo drift.
      const edge = localBpm < this.bpm * 0.985 || localBpm > this.bpm * 1.015;
      if (localScore > 0 && !edge && this.halfRatio < 0.75) this.bpm += (localBpm - this.bpm) * 0.1;
      this.stableEvals++;
      const related = [0.5, 2 / 3, 0.75, 4 / 3, 1.5, 2, 3, 1 / 3].some((r) => Math.abs(ratio / r - 1) < 0.03);
      if (Math.abs(ratio - 1) < 0.03 || related) {
        this.pendingCount = 0;
        return;
      }
      if (this.pendingCount > 0 && Math.abs(bpm / this.pendingBpm - 1) < 0.03) this.pendingCount++;
      else {
        this.pendingBpm = bpm;
        this.pendingCount = 1;
      }
      if (this.pendingCount >= 8 && bestScore > 1.5 * Math.max(0, localScore)) {
        this.bpm = this.pendingBpm;
        this.pendingCount = 0;
        this.stableEvals = 0;
        this.locked = false;
        this.errEma = 0.3;
      }
      return;
    }
    if (Math.abs(ratio - 1) < 0.04) {
      this.bpm += (bpm - this.bpm) * 0.35;
      this.stableEvals++;
      this.pendingCount = 0;
    } else {
      if (this.pendingCount > 0 && Math.abs(bpm / this.pendingBpm - 1) < 0.04) this.pendingCount++;
      else {
        this.pendingBpm = bpm;
        this.pendingCount = 1;
      }
      // A different tempo must persist (~1.5 s) before we switch.
      if (this.pendingCount >= 3) {
        this.bpm = this.pendingBpm;
        this.pendingCount = 0;
        this.stableEvals = 0;
      }
    }
  }

  /** Strength at the midpoints between beats relative to the beats themselves, for tempo `bpm`. */
  private halfBeatRatio(bpm: number): number {
    const fr = this.frameRate;
    const P = (60 * fr) / bpm;
    const span = Math.min(this.env.count, Math.round(TEMPO_WINDOW_S * fr));
    let best = -Infinity;
    let bestPhi = 0;
    for (let phi = 0; phi < P; phi += 1) {
      let s = 0;
      for (let age = phi; age < span; age += P) s += this.env.atFrac(age) + 0.5 * (this.env.atFrac(age - 1) + this.env.atFrac(age + 1));
      if (s > best) {
        best = s;
        bestPhi = phi;
      }
    }
    let mid = 0;
    for (let age = bestPhi + P / 2; age < span; age += P) mid += this.env.atFrac(age) + 0.5 * (this.env.atFrac(age - 1) + this.env.atFrac(age + 1));
    return best > 1e-9 ? mid / best : 0;
  }
}
